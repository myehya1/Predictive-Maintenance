import joblib
import numpy as np
import torch
from torch import nn
class NormalLSTMAutoencoder(nn.Module):
    def __init__(self,n_features,hidden_size=64):
        super().__init__(); self.encoder=nn.LSTM(n_features,hidden_size,batch_first=True); self.decoder=nn.LSTM(hidden_size,hidden_size,batch_first=True); self.head=nn.Linear(hidden_size,n_features)
    def forward(self,x):
        _,(h,_)=self.encoder(x); z=h[-1].unsqueeze(1).expand(-1,x.shape[1],-1); y,_=self.decoder(z); return self.head(y)
FaultLSTM=NormalLSTMAutoencoder
def save_artifact(path,model,imputer,scaler,features,window,threshold):
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"n_features":len(features),"features":features,"window":window,"threshold":float(threshold)},path); joblib.dump({"imputer":imputer,"scaler":scaler},path.with_suffix(".preprocess.joblib"))
def load_artifact(path):
    a=torch.load(path,map_location="cpu",weights_only=False); m=NormalLSTMAutoencoder(a["n_features"]); m.load_state_dict(a["state_dict"]); m.eval(); p=joblib.load(path.with_suffix(".preprocess.joblib")); return m,p["imputer"],p["scaler"],a
def score_latest(model,imputer,scaler,artifact,frame):
    n=artifact["window"]
    if len(frame)<n: raise ValueError(f"Need at least {n} recent rows")
    x=frame[artifact["features"]].replace([np.inf,-np.inf],np.nan).tail(n); x=scaler.transform(imputer.transform(x)).astype(np.float32)
    with torch.no_grad():
        t=torch.tensor(x).unsqueeze(0); return float(((model(t)-t)**2).mean().item())
