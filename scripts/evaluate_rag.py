"""Compare Qwen answers with and without the project's retrieved documentation.

Uses only Python's standard library and the local Ollama HTTP API. The score is a
transparent keyword checklist, not a human judgment or a semantic-quality score.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
API = "http://localhost:11434/api"
EMBED_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen3:4b"
QUESTION_FILE = ROOT / "docs" / "rag_eval_questions.json"
OUTPUT_DIR = ROOT / "reports" / "rag_eval"
TOPIC_TERMS = {
    "unknown units": ("anonym", "unit", "sensor_"),
    "ten-minute aggregates": ("ten-minute", "aggregate", "average"),
    "zero readings": ("zero", "missing-value", "stopped-state"),
    "flat readings": ("flat", "repeated", "stale"),
    "cross-farm comparisons": ("farm", "calibration", "availability"),
    "event labels": ("event", "weak label", "approximate"),
    "gaps and status": ("timestamp", "gap", "sparse"),
    "wind and power context": ("power", "curtailment", "operating mode"),
    "channel scales": ("reactive power", "scale", "aggregation"),
    "temperature readings": ("temperature", "spike", "warm-up"),
}


def post(path, payload, timeout=300):
    request = Request(
        f"{API}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not call Ollama at {API}/{path}: {exc}. Check that Ollama is running and both models are available."
        ) from exc


def embed(text):
    response = post("embed", {"model": EMBED_MODEL, "input": text})
    return response["embeddings"][0]


def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / max(left_norm * right_norm, 1e-12)


def split_markdown(source, text):
    result, heading, lines = [], "", []

    def save_paragraph():
        paragraph = " ".join(lines).strip()
        if len(paragraph) > 40:
            result.append({"source": source, "text": f"{heading}\n{paragraph}".strip()})
        lines.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            save_paragraph()
            heading = line.lstrip("# ").strip()
        elif line.startswith("- "):
            save_paragraph()
            bullet = f"{heading}\n{line[2:].strip()}".strip()
            if len(bullet) > 40:
                result.append({"source": source, "text": bullet})
        elif line:
            lines.append(line)
        else:
            save_paragraph()
    save_paragraph()
    return result


def load_passages():
    paths = (("README.md", ROOT / "README.md"),
             ("docs/sensor_caveats.md", ROOT / "docs" / "sensor_caveats.md"))
    passages = []
    for source, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"RAG source document is missing: {path}")
        passages.extend(split_markdown(source, path.read_text(encoding="utf-8")))
    vectors = [embed(item["text"]) for item in passages]
    return passages, vectors


def chat(question, passage=None):
    system = (
        "Answer the user's question in simple language, in at most 3 sentences. "
        "Be careful with uncertain dataset-specific claims. Do not invent facts."
    )
    if passage is not None:
        system += (
            " Use the supplied retrieved CARE project passage as evidence. If it does not answer the question, say so. "
            "Do not add unsupported dataset-specific details.\n\n"
            f"Retrieved passage (source: {passage['source']}):\n{passage['text']}"
        )
    response = post("chat", {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
        "stream": False,
        # Ask Ollama to separate Qwen's private reasoning from its final answer.
        # The evaluator saves and scores only message.content, never message.thinking.
        "think": True,
        "options": {"temperature": 0, "num_predict": 900},
    }, timeout=600)
    message = response["message"]
    content = message.get("content", "").strip()
    # A few Ollama versions can put thinking markers in the content field even
    # when thinking is enabled. Never save the reasoning in the report.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    elif content.startswith("<think>"):
        content = ""
    if "thinking" not in message and any(
        marker in content.casefold()
        for marker in ("the user is asking", "let me think", "i should answer", "the user wants")
    ):
        raise RuntimeError("Ollama did not separate Qwen's reasoning from its final answer; refusing to save it.")
    return content


def check_answer(text, expected_groups):
    normalized = text.casefold()
    missing = [
        " / ".join(group) for group in expected_groups
        if not any(term.casefold() in normalized for term in group)
    ]
    return not missing, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30,
                        help="Run the first N questions (default: all 30).")
    args = parser.parse_args()
    questions = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))[:args.limit]
    if not questions:
        raise SystemExit("No evaluation questions selected.")

    print(f"Preparing local RAG index for {len(questions)} questions...", flush=True)
    passages, vectors = load_passages()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "rag_comparison.csv"
    fields = ["id", "topic", "question", "retrieved_source", "retrieval_similarity",
              "top1_topic_evidence", "without_rag_pass", "without_rag_missing_groups",
              "without_rag_answer", "with_rag_pass", "with_rag_missing_groups", "with_rag_answer"]
    totals = {"without_rag": 0, "with_rag": 0, "top1_topic_evidence": 0}
    by_topic = {}
    start = time.time()

    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, item in enumerate(questions, start=1):
            q_vector = embed(item["question"])
            ranked = sorted(
                ((cosine(q_vector, vector), passage)
                 for passage, vector in zip(passages, vectors)),
                key=lambda pair: pair[0], reverse=True,
            )
            similarity, best = ranked[0]
            text = best["text"].casefold()
            topic_hit = any(term in text for term in TOPIC_TERMS[item["topic"]])

            answer_without = chat(item["question"])
            answer_with = chat(item["question"], best)
            pass_without, missing_without = check_answer(answer_without, item["expected_groups"])
            pass_with, missing_with = check_answer(answer_with, item["expected_groups"])
            totals["without_rag"] += int(pass_without)
            totals["with_rag"] += int(pass_with)
            totals["top1_topic_evidence"] += int(topic_hit)
            group = by_topic.setdefault(item["topic"], {"questions": 0, "without_rag": 0, "with_rag": 0})
            group["questions"] += 1
            group["without_rag"] += int(pass_without)
            group["with_rag"] += int(pass_with)
            writer.writerow({
                "id": item["id"], "topic": item["topic"], "question": item["question"],
                "retrieved_source": best["source"], "retrieval_similarity": round(similarity, 4),
                "top1_topic_evidence": topic_hit, "without_rag_pass": pass_without,
                "without_rag_missing_groups": "; ".join(missing_without),
                "without_rag_answer": answer_without, "with_rag_pass": pass_with,
                "with_rag_missing_groups": "; ".join(missing_with), "with_rag_answer": answer_with,
            })
            handle.flush()
            print(f"[{number}/{len(questions)}] {item['id']}: without RAG={'PASS' if pass_without else 'MISS'}, "
                  f"with RAG={'PASS' if pass_with else 'MISS'}, top-1 retrieval={'hit' if topic_hit else 'miss'}",
                  flush=True)

    summary = {
        "questions": len(questions), "chat_model": CHAT_MODEL, "embedding_model": EMBED_MODEL,
        "without_rag_pass": totals["without_rag"], "with_rag_pass": totals["with_rag"],
        "top1_retrieval_topic_evidence": totals["top1_topic_evidence"],
        "by_topic": by_topic,
        "scoring": "Pass only when the answer contains at least one phrase from every expected phrase group. Top-1 retrieval is a small topic-keyword check. These are transparent keyword checks, not human or semantic judgments.",
        "files": {"details": str(detail_path.relative_to(ROOT))},
        "elapsed_seconds": round(time.time() - start, 1),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary")
    print(f"Answers passing checklist without RAG: {totals['without_rag']}/{len(questions)}")
    print(f"Answers passing checklist with RAG:    {totals['with_rag']}/{len(questions)}")
    print(f"Top-1 retrieval contained topic terms:  {totals['top1_topic_evidence']}/{len(questions)}")
    print(f"Detailed answers: {detail_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
