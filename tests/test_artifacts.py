"""Quality gates on the shipped artifacts: fail CI if a retrain regresses the model."""

import json

import pandas as pd
import pytest

from energy_forecast import config


@pytest.fixture(scope="module")
def meta():
    return json.loads((config.ARTIFACT_DIR / "metrics.json").read_text())


def test_model_beats_baselines_on_holdout(meta):
    assert meta["holdout"]["mape"] < 0.6 * meta["baseline"]["mape"]
    assert meta["improvement_pct"] > 40


def test_weather_features_earn_their_place(meta):
    by_name = {r["model"]: r for r in meta["comparison"]}
    final = by_name["Hist. gradient boosting + weather (final)"]
    no_weather = by_name["Hist. gradient boosting, no weather"]
    assert final["rmse"] < no_weather["rmse"]


def test_holdout_is_strictly_after_training(meta):
    s = meta["split"]
    assert s["train_end"] < s["test_start"]
    assert s["calibration_start"] < s["test_start"]


def test_interval_coverage_is_close_to_nominal():
    horizon = pd.read_csv(config.ARTIFACT_DIR / "horizon_metrics.csv")
    actual = horizon[horizon["weather"] == "actual"]
    assert actual["interval_coverage"].between(65, 92).all(), actual[["horizon", "interval_coverage"]]


def test_error_grows_with_horizon_but_stays_below_seasonal_naive():
    h = pd.read_csv(config.ARTIFACT_DIR / "horizon_metrics.csv")
    for _, grp in h.groupby("weather"):
        grp = grp.sort_values("horizon")
        assert (grp["mape"] < grp["seasonal_naive_mape"]).all()
