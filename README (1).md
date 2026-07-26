# Energy Demand Forecast

An interactive dashboard forecasting next-day electricity demand using a
Gradient Boosting model trained on real PJM Interconnection grid data
(2002–2018, US Eastern Interconnection).

**[Live app link — add after deploying]**

## What it does

- Trains a Gradient Boosting Regressor on calendar features (day of week,
  month, season, weekend flag) and lagged demand features (yesterday, same
  day last week, 7-day rolling average).
- Uses `TimeSeriesSplit` cross-validation and a chronological train/test
  split — standard practice for time series, unlike a shuffled K-Fold which
  would leak future data into training.
- Benchmarks against a naive "yesterday = today" baseline so the reported
  error numbers have context.
- Serves everything through an interactive Streamlit dashboard: actual vs.
  predicted demand over a selectable date range, a predicted-vs-actual
  scatter plot, feature importances, and a form to generate a custom
  prediction.

## Data source

PJM Interconnection hourly grid demand (megawatts), originally published on
Kaggle as ["Hourly Energy Consumption"](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
by robikscube.

## Project structure

```
energy_app/
├── app.py                    # Streamlit app (loads precomputed artifacts only)
├── train_model.py            # Offline training script — run this to regenerate artifacts
├── model.joblib               # Trained pipeline (preprocessor + Gradient Boosting model)
├── test_results.csv           # Actual vs. predicted values for the test period
├── processed_history.csv      # Full daily series with engineered features
├── feature_importance.csv     # Feature importance rankings
├── metrics.json                # RMSE / MAE / R² for model and baseline
└── requirements.txt
```

The app **does not retrain on load** — it reads the artifacts above. This
keeps the deployed app fast and avoids needing Kaggle credentials at runtime.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retraining with fresh data

`train_model.py` pulls the dataset directly from Kaggle via `kagglehub`,
so it always uses the canonical source rather than a third-party mirror.

1. Create a free Kaggle account and generate an API token (Account Settings
   → API → "Create New Token"). This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or
   `C:\Users\<you>\.kaggle\kaggle.json` (Windows), or set the
   `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables instead.
3. Install training dependencies: `pip install -r requirements-train.txt`
4. Run `python train_model.py` — `kagglehub` downloads and caches the
   dataset automatically, then the script regenerates all five artifact
   files (`model.joblib`, `test_results.csv`, `processed_history.csv`,
   `feature_importance.csv`, `metrics.json`).
5. Restart the Streamlit app (or redeploy) to pick up the new artifacts.

Note: `requirements-train.txt` (training-only deps) is separate from
`requirements.txt` (deployed app deps) since the deployed app never needs
`kagglehub` — it only reads the artifacts above.

## Deploying (Streamlit Community Cloud — free)

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app."
3. Point it at your repo, branch, and `app.py`.
4. Deploy — it installs `requirements.txt` automatically and gives you a
   public URL.

No secrets or API keys are needed for deployment since the model and data
are pre-baked into the repo as static files.
