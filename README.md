Time Series Modelling — Guayas Daily Demand (2014 Q1)

Forecasting daily unit sales for Ecuador’s Guayas region using classic feature-based time-series (XGBoost), with lightweight tuning, MLflow tracking, and a simple post-hoc bias correction.

**What’s inside**
- Data prep: clamp to 2014-01-01 → 2014-03-31, aggregate to daily totals, add calendar + lag/rolling features.
- Models: XGBoost baseline → tuned XGBoost → (optional) bias-corrected forecast.
- Tracking: MLflow logs for params, metrics, and forecast plots.
- Artifacts: saved feature list, scaler (for future NN work), bias-correction factor.

**Repo structure**
--notebooks/
-   1_Time_Series_W1_Data_Prep.ipynb   # EDA (holidays, perishable, oil, etc.)output: build guayas_prepared.csv
-   2_Time_Series_W2_Analysis.ipynb    # XGBoost baseline
-   3_Time_Series_W3_Analysis.ipynb    # XGBoost baseline, tuning, bias correction, MLflow
README.md


**Data**

- Base files:     
    "holiday_events": "1RMjSuqHXHTwAw_PGD5XVjhA3agaAGHDH",
    "items": "1ogMRixVhNY6XOJtIRtkRllyOyzw1nqya",
    "oil": "1Q59vk2v4WQ-Rpc9t2nqHcsZM3QWGFje_",
    "stores": "1Ei0MUXmNhmOcmrlPad8oklnFEDM95cDi",
    "train": "1oEX8NEJPY7wPmSJ0n7lO1JUFYyZjFBRv",
    "transactions": "1PW5LnAEAiL43fI5CRDn_h6pgDG5rtBW_"
- Working dataset: guayas_prepared.csv (daily sales at item/store level filtered to Guayas; Jan–Mar 2014 window is applied in the notebook).
- The modeling notebook aggregates to one row per day for the region.
- Focus on top-3 families (Grocery I, Beverages, Cleaning) is supported for to 2014-01-01 → 2014-03-31
- Current runs use daily totals after any upstream filtering.


**Environment**
- Use Colab (recommended) or local Python 3.10+.

**How to run (quick start)**
- Open notebooks in Colab.
- Mount Drive and point to guayas_prepared.csv (already in your Drive).
- Run all cells


**Results (snapshot)**
- Model	MAE	RMSE	Notes
- XGB baseline	~1219	~1409	Under-forecasts peaks
- XGB tuned	~1023	~1158	Better peak tracking
- Tuned + bias (k≈1.14)	≈600	≈788	Simple multiplicative scale improves amplitude

**Key takeaways:** Weekly seasonality is learned; peak magnitudes improve with tuning and are best after bias correction. Remaining error concentrates on the largest spikes.


**MLflow**

- Experiment name: guayas_xgb_experiment
- Each run logs:
--- Params: tree depth, estimators, learning rate, subsampling, #features (and bias_k when applicable)
--- Metrics: MAE, RMSE, Bias, MAD, rMAD (%), MAPE (%)
--- Artifacts: forecast plot(s), feature_cols.json
--- Local store (Colab): file:/content/mlruns.
--- To inspect locally: mlflow ui --backend-store-uri file:mlruns




**Artifacts saved (for reuse)**
- artifacts/feature_cols.json — list of model features used.
- artifacts/scaler.joblib — StandardScaler fit on train features (handy for future LSTM/NN work; not needed by trees).
- artifacts/df_ml_janmar2014.parquet — final modeling frame used in Week 3.
- artifacts/bias_correction.json — current k (e.g., 1.14) and optional validation window.


**Roadmap**
- Add holiday & promo intensity features.
- Try a simple LSTM/Temporal CNN baseline (log to MLflow).
- Add time-series cross-validation across February for more stable tuning.

License

MIT (or your choice). Add a LICENSE file if needed.
