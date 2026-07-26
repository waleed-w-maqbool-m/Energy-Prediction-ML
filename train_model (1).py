# ==================================================
# OFFLINE TRAINING SCRIPT
# Trains the model once and saves artifacts for the
# Streamlit app to load - the app itself never retrains.
# ==================================================

import json
import os

import numpy as np
import pandas as pd
import joblib
import kagglehub

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------
# LOAD DATASET (DIRECT FROM KAGGLE - ORIGINAL SOURCE)
# --------------------------------------------------
# "Hourly Energy Consumption" dataset by robikscube on Kaggle.
# Contains hourly demand (MW) for several US grid regions
# (PJME, PJMW, AEP, COMED, DAYTON, etc). We use PJME here.
#
# ONE-TIME SETUP REQUIRED:
# 1. Create a free Kaggle account: https://www.kaggle.com
# 2. Go to Account Settings -> API -> "Create New Token"
#    This downloads a file called kaggle.json
# 3. Place it at ~/.kaggle/kaggle.json (Linux/Mac) or
#    C:\Users\<you>\.kaggle\kaggle.json (Windows)
#    OR set the environment variables KAGGLE_USERNAME and
#    KAGGLE_KEY before running this script.
#
# kagglehub handles the download and caches it locally after
# the first run - no manual unzipping needed.

dataset_path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
print("Dataset downloaded to:", dataset_path)

csv_path = os.path.join(dataset_path, "PJME_hourly.csv")

df = pd.read_csv(csv_path, parse_dates=["Datetime"])
df = df.rename(columns={"PJME_MW": "demand_mw"})
df = df.sort_values("Datetime")

# --------------------------------------------------
# DAILY AGGREGATION
# --------------------------------------------------

df = (
    df.set_index("Datetime")["demand_mw"]
    .resample("D")
    .mean()
    .dropna()
    .reset_index()
    .rename(columns={"Datetime": "date"})
)

# --------------------------------------------------
# CALENDAR FEATURES
# --------------------------------------------------

df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["is_weekend"] = df["date"].dt.dayofweek >= 5


def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


df["season"] = df["month"].apply(month_to_season)

# --------------------------------------------------
# LAG FEATURES
# --------------------------------------------------

df["demand_lag_1"] = df["demand_mw"].shift(1)
df["demand_lag_7"] = df["demand_mw"].shift(7)
df["demand_roll7_mean"] = df["demand_mw"].shift(1).rolling(window=7).mean()
df = df.dropna().reset_index(drop=True)

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------

feature_cols = [
    "day_of_week", "month", "is_weekend", "season",
    "demand_lag_1", "demand_lag_7", "demand_roll7_mean"
]

X = df[feature_cols]
y = df["demand_mw"]

numerical_features = list(X.select_dtypes(include=["int64", "float64", "bool"]).columns)
categorical_features = list(X.select_dtypes(include=["object"]).columns)

# --------------------------------------------------
# PREPROCESSING + MODEL PIPELINE
# --------------------------------------------------

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

# --------------------------------------------------
# CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

test_fraction = 0.15
split_idx = int(len(df) * (1 - test_fraction))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_dates = df["date"].iloc[split_idx:]

print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

# --------------------------------------------------
# GRID SEARCH (TIME-SERIES-AWARE CV)
# --------------------------------------------------

param_grid = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [2, 3, 4],
    "regressor__learning_rate": [0.03, 0.1],
}

cv = TimeSeriesSplit(n_splits=5)

search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

gb_predictions = search.predict(X_test)
baseline_predictions = X_test["demand_lag_1"].values


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


gb_metrics = compute_metrics(y_test, gb_predictions)
baseline_metrics = compute_metrics(y_test, baseline_predictions)

print("GB metrics:", gb_metrics)
print("Baseline metrics:", baseline_metrics)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

cat_encoder = search.best_estimator_.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, encoded_cat_names])
importances = search.best_estimator_.named_steps["regressor"].feature_importances_

importance_df = (
    pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# --------------------------------------------------
# SAVE ARTIFACTS
# --------------------------------------------------

joblib.dump(search.best_estimator_, "model.joblib")

# Test-period results for the dashboard (actual vs predicted over time)
results_df = pd.DataFrame({
    "date": test_dates.values,
    "actual_mw": y_test.values,
    "predicted_mw": gb_predictions,
    "baseline_mw": baseline_predictions,
})
results_df.to_csv("test_results.csv", index=False)

# Full processed history (used for the "recent demand" chart + prediction defaults)
df.to_csv("processed_history.csv", index=False)

importance_df.to_csv("feature_importance.csv", index=False)

with open("metrics.json", "w") as f:
    json.dump({
        "gb": gb_metrics,
        "baseline": baseline_metrics,
        "best_params": search.best_params_,
        "improvement_pct": 100 * (baseline_metrics["rmse"] - gb_metrics["rmse"]) / baseline_metrics["rmse"],
    }, f, indent=2)

print("\nArtifacts saved: model.joblib, test_results.csv, processed_history.csv, feature_importance.csv, metrics.json")
