from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FARMS=("windfarmA","windfarmB","windfarmC")
NAMES={"windfarmA":"Wind Farm A","windfarmB":"Wind Farm B","windfarmC":"Wind Farm C"}
def farm_dir(farm,root=ROOT):
    if farm not in FARMS: raise ValueError(f"Unknown farm: {farm}")
    return root/"data_v6"/"CARE_To_Compare"/NAMES[farm]
def event_ids(farm,root=ROOT): return sorted((p.stem for p in (farm_dir(farm,root)/"datasets").glob("*.csv")),key=int)
def _metadata(farm,root):
    d=pd.read_csv(farm_dir(farm,root)/"event_info.csv",sep=";")
    col="asset_id" if "asset_id" in d else "asset" if "asset" in d else None
    if col is None or "event_id" not in d: raise ValueError("Metadata needs event_id and asset/asset_id")
    return d.rename(columns={col:"turbine_id"})
def load_event(farm,event_id,root=ROOT):
    d=farm_dir(farm,root); p=d/"datasets"/f"{event_id}.csv"
    if not p.is_file(): raise FileNotFoundError(f"No event {event_id} for {farm}")
    f=pd.read_csv(p,sep=";"); m=_metadata(farm,root); m=m[m.event_id.astype(str)==str(event_id)]
    if len(m)!=1: raise ValueError(f"Expected one metadata row for event {event_id}")
    e=m.iloc[0]
    if not f.asset_id.astype(str).eq(str(e.turbine_id)).all(): raise ValueError("CSV asset_id does not match metadata turbine_id")
    f["time_stamp"]=pd.to_datetime(f.time_stamp,errors="coerce"); f=f.dropna(subset=["time_stamp"]).sort_values("time_stamp").reset_index(drop=True)
    f["event_label"]=str(e.event_label).lower(); f["event_start"]=pd.to_datetime(e.event_start,errors="coerce"); f["event_end"]=pd.to_datetime(e.event_end,errors="coerce")
    f["label"]=np.int8(0)
    if f.event_label.iloc[0]=="anomaly" and pd.notna(f.event_start.iloc[0]) and pd.notna(f.event_end.iloc[0]):
        mask=f.time_stamp.between(f.event_start.iloc[0],f.event_end.iloc[0],inclusive="both") & f.train_test.eq("prediction")
        f.loc[mask,"label"]=1
    f["farm"]=farm; f["event_id"]=str(event_id); f["turbine_id"]=str(e.turbine_id)
    return f
def iter_events(root=ROOT):
    for farm in FARMS:
        for eid in event_ids(farm,root): yield farm,eid
def feature_columns(farm,frame,root=ROOT):
    d=pd.read_csv(farm_dir(farm,root)/"feature_description.csv",sep=";")
    if not {"sensor_name","statistics_type"}.issubset(d.columns): raise ValueError("Invalid feature description columns")
    out=[]
    for r in d.itertuples(index=False):
        if "average" not in {s.strip().lower() for s in str(r.statistics_type).split(",")}: continue
        name=str(r.sensor_name); col=next((x for x in (name+"_avg",name) if x in frame.columns),None)
        if col is None: raise ValueError(f"Average feature {name} missing in {farm}")
        if col not in out: out.append(col)
    if not out: raise ValueError(f"No average features for {farm}")
    return out
