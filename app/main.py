import os
import re
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data import ROOT,FARMS,event_ids,load_event
from src.model import load_artifact,score_latest
app=FastAPI(title="Wind Turbine Predictive Maintenance Assistant",version="2.0.0")
DOC=ROOT/"docs"/"sensor_caveats.md"
class AskRequest(BaseModel): question:str=Field(min_length=3,max_length=2000)
def _chunks(text): return [p.strip() for p in re.split(r"\n(?=##? )",text) if p.strip()]
@lru_cache(maxsize=1)
def _kb():
    c=_chunks(DOC.read_text(encoding="utf-8")); v=TfidfVectorizer(stop_words="english"); return c,v,v.fit_transform(c)
def retrieve_caveats(q,limit=3):
    c,v,m=_kb(); scores=cosine_similarity(v.transform([q]),m).ravel(); ix=scores.argsort()[::-1][:limit]
    return [{"text":c[i],"relevance":round(float(scores[i]),3)} for i in ix if scores[i]>0]
def event_status(farm,event_id):
    if farm not in FARMS: raise ValueError(f"farm must be one of {', '.join(FARMS)}")
    model_path=ROOT/"models"/f"fault_lstm_{farm}.pt"
    if not model_path.exists(): raise FileNotFoundError("Farm model is missing. Train it with python -m src.train.")
    f=load_event(farm,str(event_id)); model,imputer,scaler,a=load_artifact(model_path)
    score=score_latest(model,imputer,scaler,a,f)
    cov=float(f.tail(a["window"])[a["features"]].notna().mean().mean())
    return {"farm":farm,"event_id":str(event_id),"turbine_id":str(f.turbine_id.iloc[0]),"as_of":f.time_stamp.iloc[-1].isoformat(),
      "anomaly_score":round(score,6),"threshold":round(a["threshold"],6),"status":"anomaly-like pattern" if score>=a["threshold"] else "normal-like pattern",
      "recent_data_coverage":round(cov,3),"window_rows":a["window"],"caveats":retrieve_caveats("sensor missing values stale SCADA data",2)}
TOOLS=[{"type":"function","function":{"name":"get_event_status","description":"Score one CARE v6 event file; event ID is different from turbine ID.","parameters":{"type":"object","properties":{"farm":{"type":"string","enum":list(FARMS)},"event_id":{"type":"string"}},"required":["farm","event_id"],"additionalProperties":False}}},
{"type":"function","function":{"name":"search_sensor_caveats","description":"Search sensor documentation for data quality caveats.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":False}}}]
def _call_tool(name,args):
    if name=="get_event_status": return event_status(args["farm"],args["event_id"])
    if name=="search_sensor_caveats": return {"caveats":retrieve_caveats(args["query"])}
    raise ValueError("Unknown tool: "+name)
def answer(question):
    caves=retrieve_caveats(question); key=os.getenv("OPENAI_API_KEY")
    if key:
        try:
            import json
            from openai import OpenAI
            client=OpenAI(api_key=key); model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
            msgs=[{"role":"system","content":"Use event_id to inspect a CARE event file, and report its turbine_id separately. Never infer sensor meanings. Retrieved caveats: "+"\n".join(x["text"] for x in caves)},{"role":"user","content":question}]
            res=client.chat.completions.create(model=model,messages=msgs,tools=TOOLS,tool_choice="auto").choices[0].message
            if res.tool_calls:
                msgs.append(res)
                for call in res.tool_calls: msgs.append({"role":"tool","tool_call_id":call.id,"content":json.dumps(_call_tool(call.function.name,json.loads(call.function.arguments)))})
                out=client.chat.completions.create(model=model,messages=msgs)
                return {"answer":out.choices[0].message.content,"caveats":caves,"mode":"llm_function_calling"}
            return {"answer":res.content,"caveats":caves,"mode":"llm"}
        except ImportError: pass
    m=re.search(r"windfarm[abc]",question,re.I); i=re.search(r"event\s*#?\s*(\d+)",question,re.I)
    if m and i:
        try:
            s=event_status(m.group(),i.group(1)); return {"answer":f"{s['farm']} event {s['event_id']} (turbine {s['turbine_id']}) shows a {s['status']} as of {s['as_of']}. This is an anomaly score, not a confirmed fault.","status":s,"caveats":caves,"mode":"local_tools"}
        except (ValueError,FileNotFoundError) as e: return {"answer":str(e),"caveats":caves,"mode":"local_tools"}
    return {"answer":"Ask about a specific event, for example: What is windfarmA event 68 status? Event ID and turbine ID are different.","caveats":caves,"mode":"local_tools"}
@app.get("/health")
def health(): return {"ok":True,"farms":list(FARMS),"event_ids":{f:len(event_ids(f)) for f in FARMS}}
@app.get("/events/{farm}/{event_id}/status")
def status(farm,event_id):
    try: return event_status(farm,event_id)
    except (ValueError,FileNotFoundError) as e: raise HTTPException(status_code=404,detail=str(e)) from e
@app.post("/ask")
def ask(request:AskRequest): return answer(request.question)
