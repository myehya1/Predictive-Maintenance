import json
import os
from pathlib import Path

import pandas as pd
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import PROJECT_ROOT, retrieve_passages

MODEL_DIR = PROJECT_ROOT / "models" / "farm_a"
REPORT_DIR = PROJECT_ROOT / "reports" / "farm_a"
METRICS_FILE = REPORT_DIR / "window_metrics.csv"
EVENT_RESULTS_FILE = REPORT_DIR / "event_level_scores.csv"
CONFIG_FILE = MODEL_DIR / "training_config.json"
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")
app = FastAPI(title="Farm A Wind Turbine Assistant", version="3.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def get_farm_a_metrics(metric: str) -> dict:
    """Return a saved evaluation metric. Use 'all' to return the metric table."""
    if not METRICS_FILE.is_file():
        raise FileNotFoundError(f"Evaluation metrics are missing: {METRICS_FILE}")
    table = pd.read_csv(METRICS_FILE)
    values = {str(row.metric): float(row.value) for row in table.itertuples(index=False)}
    if metric.strip().lower() in {"all", "summary", "all metrics"}:
        return {"farm": "Farm A", "evaluation_unit": "prediction windows", "metrics": values}

    key = metric.strip().lower().replace("_", "-")
    aliases = {
        "f1": "Anomaly F1",
        "f1-score": "Anomaly F1",
        "recall": "Anomaly recall",
        "anomaly-recall": "Anomaly recall",
        "precision": "Anomaly precision",
        "pr-auc": "Average precision (PR-AUC)",
        "average-precision": "Average precision (PR-AUC)",
        "roc-auc": "ROC-AUC",
        "accuracy": "Accuracy",
    }
    selected = aliases.get(key)
    if selected is None or selected not in values:
        return {"farm": "Farm A", "requested_metric": metric, "available_metrics": values}
    return {"farm": "Farm A", "metric": selected, "value": values[selected]}


def get_farm_a_event_result(event_id: str) -> dict:
    """Look up a saved Farm A event prediction; this does not run new inference."""
    if not EVENT_RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Event evaluation results are missing: {EVENT_RESULTS_FILE}")
    table = pd.read_csv(EVENT_RESULTS_FILE, dtype={"event_id": str, "turbine_id": str})
    matches = table.loc[table.event_id.eq(str(event_id))]
    if matches.empty:
        return {
            "farm": "Farm A",
            "event_id": str(event_id),
            "available": False,
            "message": "No saved prediction result for this event. The report only covers the held-out prediction events.",
        }
    row = matches.iloc[0]
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.is_file() else {}
    flagged = bool(int(row.predicted_anomaly))
    return {
        "farm": "Farm A",
        "event_id": str(event_id),
        "available": True,
        "turbine_id": str(row.turbine_id),
        "saved_prediction": "anomaly-like pattern" if flagged else "not flagged by this demo threshold",
        "mean_reconstruction_error": round(float(row.mean_error), 6),
        "window_threshold_used_by_event_rule": round(float(config.get("threshold", 0.0)), 6),
        "event_rule_note": "The saved event flag compares the event mean error with the threshold selected for individual windows. This is an exploratory rule, not an event-calibrated threshold.",
        "windows_in_event": int(row.windows),
        "fraction_of_windows_flagged": round(float(row.anomaly_window_fraction), 4),
        "important_note": "This is a saved experimental prediction, not a confirmed fault or live inference.",
    }


TOOLS = [get_farm_a_metrics, get_farm_a_event_result]
TOOL_FUNCTIONS = {
    get_farm_a_metrics.__name__: get_farm_a_metrics,
    get_farm_a_event_result.__name__: get_farm_a_event_result,
}


def answer_question(question: str) -> dict:
    passages = retrieve_passages(question, top_k=3)
    context = "\n\n".join(
        f"Source: {item['source']}\nPassage: {item['text']}"
        for item in passages
    )
    system_text = (
        "You are a beginner-friendly Farm A project assistant. Use retrieved passages for "
        "documentation questions. Use tools for exact metrics or saved event results. "
        "Never invent a number. Clearly distinguish a saved anomaly-like prediction from a "
        "confirmed turbine fault. This demo does not perform live inference. Retrieved context:\n"
        + context
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    tools_used = []

    # Limit the number of tool rounds so the local demo cannot loop indefinitely.
    for _ in range(3):
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            # Ollama returns any reasoning in a separate field; only content is sent to the user.
            think=True,
            options={"temperature": 0, "num_predict": 1200},
        )
        messages.append(response.message)
        tool_calls = response.message.tool_calls or []
        if not tool_calls:
            answer = response.message.content or "I could not form an answer from the available information."
            if "</think>" in answer:
                answer = answer.rsplit("</think>", 1)[-1].strip()
            elif answer.startswith("<think>"):
                answer = "I could not form a final answer. Please try again."
            return {
                "answer": answer,
                "sources": [{"source": x["source"], "relevance": x["relevance"]} for x in passages],
                "tools_used": tools_used,
                "model": CHAT_MODEL,
            }

        for call in tool_calls:
            name = call.function.name
            function = TOOL_FUNCTIONS.get(name)
            if function is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = function(**dict(call.function.arguments))
                except (FileNotFoundError, ValueError, KeyError) as exc:
                    result = {"error": str(exc)}
            tools_used.append(name)
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return {
        "answer": "I could not finish the tool lookup. Please ask a simpler question.",
        "sources": [{"source": x["source"], "relevance": x["relevance"]} for x in passages],
        "tools_used": tools_used,
        "model": CHAT_MODEL,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "farm": "Farm A",
        "chat_model": CHAT_MODEL,
        "embedding_model": "embeddinggemma",
        "metrics_available": METRICS_FILE.is_file(),
        "event_results_available": EVENT_RESULTS_FILE.is_file(),
    }


@app.get("/metrics")
def metrics():
    try:
        return get_farm_a_metrics("all")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/events/{event_id}/prediction")
def event_prediction(event_id: str):
    try:
        result = get_farm_a_event_result(event_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not result["available"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local assistant unavailable: {exc}. Check that Ollama is running and both models are installed.",
        ) from exc
