import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from src.data import ROOT,FARMS,event_ids,load_event,feature_columns
from src.model import NormalLSTMAutoencoder,save_artifact
NORMAL_STATUS={0}
def split_training_rows(frame,validation_fraction=.2):
    rows=frame[frame.train_test.eq("train")]
    if len(rows)<2: raise ValueError("Not enough training rows")
    k=int(len(rows)*(1-validation_fraction))
    if not 0<k<len(rows): raise ValueError("Invalid validation split")
    return rows.iloc[:k].copy(),rows.iloc[k:].copy()
def make_windows(frame,features,window,stride=1):
    if len(frame)<window: return np.empty((0,window,len(features)),np.float32)
    x=frame[features].replace([np.inf,-np.inf],np.nan).to_numpy()
    return np.stack([x[i:i+window] for i in range(0,len(x)-window+1,stride)]).astype(np.float32)
def fit_preprocessing(training_frames,features):
    values=pd.concat(training_frames,ignore_index=True)[features].replace([np.inf,-np.inf],np.nan)
    if values.empty: raise ValueError("Empty training data")
    imputer=SimpleImputer(strategy="median").fit(values)
    scaler=StandardScaler().fit(imputer.transform(values))
    return imputer,scaler
def _transform(frame,features,imputer,scaler):
    values=frame[features].replace([np.inf,-np.inf],np.nan)
    return scaler.transform(imputer.transform(values)).astype(np.float32)
def train(epochs=8,window=12,output_dir=ROOT/"models",stride=120,scaler_sample_step=30):
    for farm in FARMS:
        ids=event_ids(farm)
        if not ids: raise RuntimeError(f"No v6 event files for {farm}")
        first=load_event(farm,ids[0]); features=feature_columns(farm,first); del first
        # Use a deterministic sample from training-prefix, normal-status rows to bound RAM.
        # Imputation and scaling never see validation or prediction rows.
        fit_samples=[]
        for eid in ids:
            frame=load_event(farm,eid)
            if feature_columns(farm,frame)!=features: raise ValueError(f"Inconsistent average feature schema for {farm}/{eid}")
            tr,_=split_training_rows(frame)
            normal=tr[tr.status_type_id.isin(NORMAL_STATUS)]
            fit_samples.append(normal.iloc[::scaler_sample_step][features])
            del frame
        imputer,scaler=fit_preprocessing(fit_samples,features); del fit_samples
        train_windows=[]; validation_windows=[]; prediction_rows=0
        # Each event and each partition is windowed separately. No joined or overlapping split windows.
        for eid in ids:
            frame=load_event(farm,eid); tr,va=split_training_rows(frame)
            tr=tr[tr.status_type_id.isin(NORMAL_STATUS)]; va=va[va.status_type_id.isin(NORMAL_STATUS)]
            for rows,destination in ((tr,train_windows),(va,validation_windows)):
                x=_transform(rows,features,imputer,scaler)
                if len(x)>=window:
                    destination.append(np.stack([x[i:i+window] for i in range(0,len(x)-window+1,stride)]))
            prediction_rows+=int(frame.train_test.eq("prediction").sum())
            del frame
        if not train_windows or not validation_windows: raise RuntimeError(f"{farm}: empty training or validation windows")
        xtrain=np.concatenate(train_windows); xval=np.concatenate(validation_windows)
        del train_windows,validation_windows
        loader=DataLoader(TensorDataset(torch.from_numpy(xtrain)),batch_size=256,shuffle=True)
        model=NormalLSTMAutoencoder(len(features)); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        for epoch in range(epochs):
            model.train(); losses=[]
            for (batch,) in loader:
                optimizer.zero_grad(); loss=((model(batch)-batch)**2).mean(); loss.backward(); optimizer.step(); losses.append(loss.item())
            print(f"{farm} epoch {epoch+1}/{epochs} reconstruction_mse={np.mean(losses):.6f}")
        model.eval()
        with torch.no_grad(): errors=((model(torch.from_numpy(xval))-torch.from_numpy(xval))**2).mean((1,2)).numpy()
        threshold=float(np.quantile(errors,.99)); path=output_dir/f"fault_lstm_{farm}.pt"
        save_artifact(path,model,imputer,scaler,features,window,threshold)
        print(f"Saved {path}; validation threshold={threshold:.6f}; prediction rows isolated={prediction_rows:,}")
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=8); parser.add_argument("--window",type=int,default=12); parser.add_argument("--stride",type=int,default=120); parser.add_argument("--scaler-sample-step",type=int,default=30)
    args=parser.parse_args(); train(args.epochs,args.window,stride=args.stride,scaler_sample_step=args.scaler_sample_step)
