"""Project-wide constants: paths, data sources, split dates and the feature contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

DEMAND_CSV = DATA_DIR / "pjme_daily.csv"
WEATHER_CSV = DATA_DIR / "weather_daily.csv"

# Project links shown in the UI. Fill these in after pushing / deploying.
REPO_URL = ""
DEMO_URL = ""

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
KAGGLE_DATASET = "robikscube/hourly-energy-consumption"
KAGGLE_FILE = "PJME_hourly.csv"

# Weather: Open-Meteo ERA5 reanalysis (free, no API key).
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# PJM East spans the Mid-Atlantic corridor. Temperature is a demand-weighted blend
# of five metro areas; weights are approximate metro populations (millions).
WEATHER_STATIONS: dict[str, dict[str, float]] = {
    "Philadelphia": {"lat": 39.95, "lon": -75.16, "weight": 6.2},
    "Washington DC": {"lat": 38.90, "lon": -77.04, "weight": 6.3},
    "Baltimore": {"lat": 39.29, "lon": -76.61, "weight": 2.8},
    "Newark": {"lat": 40.74, "lon": -74.17, "weight": 2.0},
    "Harrisburg": {"lat": 40.27, "lon": -76.88, "weight": 0.6},
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
TARGET = "demand_mw"
TEST_FRACTION = 0.15  # chronological hold-out, same split as v1 for like-for-like comparison
CALIBRATION_DAYS = 730  # window just before the test set, used to calibrate intervals
MAX_HORIZON = 7  # days ahead supported by the recursive forecaster
INTERVAL_COVERAGE = 0.80  # nominal coverage of prediction intervals
RANDOM_STATE = 42

# Degree-day balance point: 65 F = 18.3 C, the industry convention.
BALANCE_TEMP_C = 18.3

# History rows the feature builder needs behind a forecast origin: the longest look-back is the 28-day rolling mean.
FEATURE_WARMUP_DAYS = 28

# Features by group. `FEATURES` is the contract between training, API and the app.
CALENDAR_FEATURES = [
    "dow", "month", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "pre_holiday", "post_holiday", "is_nonworking",
]
WEATHER_FEATURES = [
    "temp_mean", "temp_max", "temp_min", "hdd", "cdd", "temp_mean_roll3", "temp_delta_1",
]
LAG_FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14", "roll7_mean", "roll28_mean",
    "lag_1_nonworking", "lag_7_nonworking",
]
FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES
FEATURES_NO_WEATHER = CALENDAR_FEATURES + LAG_FEATURES

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
