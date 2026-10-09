"""Model definitions: the production gradient-boosting model and comparison candidates."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# Features the 2018 v1 model used (it had a text "season" column, which is a function of month).
V1_FEATURES = ["dow", "month", "is_weekend", "lag_1", "lag_7", "roll7_mean"]

# Linear baseline: degree-days capture the V-shaped temperature response.
RIDGE_FEATURES = [
    "dow", "is_holiday", "pre_holiday", "post_holiday", "doy_sin", "doy_cos",
    "hdd", "cdd", "temp_mean_roll3", "lag_1", "lag_7", "roll7_mean", "roll28_mean",
]

# Best hyper-parameters of the v1 grid search, kept so the v1 candidate is a faithful re-run.
V1_PARAMS = {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.1}

HGB_SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_iter": [200, 300, 500],
    "max_depth": [3, 4, 6, None],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
}


@dataclass
class Candidate:
    name: str
    estimator: object
    features: list[str]
    note: str = ""


def make_hgb(**params) -> HistGradientBoostingRegressor:
    """Production model family. Early stopping is off so fits are deterministic."""
    defaults = {"early_stopping": False, "random_state": config.RANDOM_STATE}
    return HistGradientBoostingRegressor(**{**defaults, **params})


def make_ridge() -> Pipeline:
    numeric = [f for f in RIDGE_FEATURES if f != "dow"]
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
        ])),
        ("model", Ridge(alpha=1.0)),
    ])


def make_v1() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), [f for f in V1_FEATURES if f != "dow"]),
            ("dow", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
        ])),
        ("model", GradientBoostingRegressor(random_state=config.RANDOM_STATE, **V1_PARAMS)),
    ])


def comparison_candidates(best_params: dict) -> list[Candidate]:
    """Learned models for the comparison table (naive baselines are handled separately)."""
    return [
        Candidate("Ridge regression", make_ridge(), RIDGE_FEATURES,
                  "Linear, with degree-day weather terms"),
        Candidate("Gradient boosting, calendar + lags (v1)", make_v1(), V1_FEATURES,
                  "The original model: no weather, no holidays"),
        Candidate("Hist. gradient boosting, no weather", make_hgb(**best_params), config.FEATURES_NO_WEATHER,
                  "Ablation: final model with the weather features removed"),
        Candidate("Hist. gradient boosting + weather (final)", make_hgb(**best_params), config.FEATURES,
                  "Production model"),
    ]
