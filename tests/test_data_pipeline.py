import numpy as np
import pandas as pd
from src.data import feature_columns,load_event
from src.train import make_windows,split_training_rows,fit_preprocessing
def test_average_selection_uses_metadata_and_bare_column(tmp_path):
    d=tmp_path/"data_v6/CARE_To_Compare/Wind Farm A"; d.mkdir(parents=True)
    (d/"feature_description.csv").write_text("sensor_name;statistics_type\na;average\nb;average,maximum\nc;maximum\n")
    assert feature_columns("windfarmA",pd.DataFrame(columns=["a_avg","b","c_avg"]),tmp_path)==["a_avg","b"]
def test_windows_do_not_cross_partition():
    a=make_windows(pd.DataFrame({"x":[0,1,2,3]}),["x"],3); b=make_windows(pd.DataFrame({"x":[100,101,102,103]}),["x"],3)
    assert a.max()<b.min() and a.tolist()==[[[0],[1],[2]],[[1],[2],[3]]]
def test_event_labels_and_ids(tmp_path):
    d=tmp_path/"data_v6/CARE_To_Compare/Wind Farm A"; (d/"datasets").mkdir(parents=True)
    (d/"event_info.csv").write_text("asset;event_id;event_label;event_start;event_end\n44;7;anomaly;2020-01-03;2020-01-04\n")
    pd.DataFrame({"time_stamp":pd.date_range("2020-01-01",periods=4),"asset_id":[44]*4,"train_test":["train","train","prediction","prediction"]}).to_csv(d/"datasets/7.csv",sep=";",index=False)
    f=load_event("windfarmA","7",tmp_path)
    assert set(f.event_id)=={"7"} and set(f.turbine_id)=={"44"} and f.label.tolist()==[0,0,1,1]
def test_normal_event_never_positive(tmp_path):
    d=tmp_path/"data_v6/CARE_To_Compare/Wind Farm A"; (d/"datasets").mkdir(parents=True)
    (d/"event_info.csv").write_text("asset;event_id;event_label;event_start;event_end\n44;7;normal;2020-01-01;2020-01-04\n")
    pd.DataFrame({"time_stamp":pd.date_range("2020-01-01",periods=4),"asset_id":[44]*4,"train_test":["train","train","prediction","prediction"]}).to_csv(d/"datasets/7.csv",sep=";",index=False)
    assert load_event("windfarmA","7",tmp_path).label.sum()==0
def test_training_only_preprocessing():
    im,sc=fit_preprocessing([pd.DataFrame({"x":[0.,1.,2.,3.]})],["x"])
    assert im.statistics_[0]==1.5 and sc.mean_[0]==1.5
def test_validation_is_chronological_train_tail():
    f=pd.DataFrame({"train_test":["train"]*8+["prediction"]*2})
    tr,va=split_training_rows(f)
    assert tr.index.max()<va.index.min() and va.train_test.eq("train").all()
