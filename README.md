# Wind Turbine Predictive Maintenance Assistant

An experimental CARE to Compare v6 SCADA project. The data loader reads the verified files from `data_v6/CARE_To_Compare`; the original dataset folders are left untouched.

## Data and model

CSV filenames are **event IDs**. Metadata columns `asset` or `asset_id` identify the **turbine ID**. The loader normalizes these names and checks that the CSV's turbine ID matches event metadata.

Sensor inputs come from each farm's `feature_description.csv` rows that list `average`. Farm feature sets differ, so training builds one model per farm. The trainer uses only rows marked `train`, takes a chronological validation tail from those rows, and keeps `prediction` rows separate. Imputation and scaling are fitted on normal-status (status 0) training rows only. The LSTM autoencoder learns to reconstruct normal sensor windows. Its alert threshold is the 99th percentile of normal validation reconstruction error. Event labels are reserved for checking predictions and never used to fit the model.

Run tests with `python -m pytest`. Training, which is intentionally not run as part of the current project update, can be started with `python -m src.train --epochs 8 --window 12`. It writes one model and preprocessing artifact per farm to `models/`.

## API

Start with `uvicorn app.main:app --reload`.

- `GET /health` lists farms and event counts.
- `GET /events/{farm}/{event_id}/status` scores that event and returns both its event ID and turbine ID.
- `POST /ask` accepts a question with a farm and event ID. Function calling and the sensor caveat retrieval remain available.

Anomaly scores are experimental signals, not confirmed diagnoses. Evaluate thresholds and event detection quality against the untouched prediction rows before operational use.
