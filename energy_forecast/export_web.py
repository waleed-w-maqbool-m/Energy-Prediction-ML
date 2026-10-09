"""Export the trained model and all dashboard data as static JSON for the Cloudflare app.

    python -m energy_forecast.export_web

Writes ``web/data/``:

* ``model.json``   the gradient-boosting trees (flat arrays), evaluated in JavaScript
* ``meta.json``    metrics, comparison, intervals, importance, backtests, holidays, climatology
* ``daily.json``   the daily demand + temperature history
* ``holdout.json`` day-ahead predictions on the hold-out period

The JavaScript forecaster (``web/js/forecast.js``) reproduces ``energy_forecast.forecast`` exactly;
``tests/test_js_parity.py`` checks the two agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .data import load_daily, temperature_climatology
from .features import _holidays

WEB_DATA = config.ROOT / "web" / "data"


def export_model(model) -> dict:
    """Flatten a fitted ``HistGradientBoostingRegressor`` into plain arrays."""
    if model.n_trees_per_iteration_ != 1:
        raise ValueError("only single-output regressors are supported")
    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported by the JS evaluator")
        leaf = nodes["is_leaf"].astype(bool)
        trees.append({
            "f": [-1 if is_leaf else int(f) for is_leaf, f in zip(leaf, nodes["feature_idx"], strict=True)],
            "t": [float(x) for x in nodes["num_threshold"]],
            "l": [int(x) for x in nodes["left"]],
            "r": [int(x) for x in nodes["right"]],
            "v": [float(x) for x in nodes["value"]],
        })
    return {
        "features": config.FEATURES,
        "baseline": float(model._baseline_prediction.ravel()[0]),
        "trees": trees,
    }


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def export_meta(metrics: dict, daily: pd.DataFrame) -> dict:
    art = config.ARTIFACT_DIR
    holdout = pd.read_csv(art / "test_results.csv", parse_dates=["date"])
    train_end = pd.Timestamp(metrics["split"]["train_end"])
    clim = temperature_climatology(daily[daily["date"] <= train_end])
    peak_day = holdout.loc[holdout["actual_mw"].idxmax(), "date"]

    return {
        **{k: metrics[k] for k in (
            "model", "split", "holdout", "baseline", "improvement_pct", "comparison",
            "intervals", "interval_nominal_coverage", "environment", "trained_at",
        )},
        "links": {"repo": config.REPO_URL, "demo": config.DEMO_URL},
        "importance": _records(pd.read_csv(art / "feature_importance.csv")),
        "backtest_by_year": _records(pd.read_csv(art / "backtest_by_year.csv")),
        "horizon": _records(pd.read_csv(art / "horizon_metrics.csv")),
        "holidays": [d.strftime("%Y-%m-%d") for d in _holidays()],
        "climatology": {col: [float(v) for v in clim[col]] for col in ("temp_mean", "temp_max", "temp_min")},
        "constants": {
            "max_horizon": config.MAX_HORIZON,
            "warmup_days": config.FEATURE_WARMUP_DAYS,
            "balance_temp_c": config.BALANCE_TEMP_C,
            "first_date": daily["date"].min().strftime("%Y-%m-%d"),
            "last_date": daily["date"].max().strftime("%Y-%m-%d"),
            "default_origin": (peak_day - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        },
    }


def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    art = config.ARTIFACT_DIR
    model = joblib.load(art / "model.joblib")
    metrics = json.loads((art / "metrics.json").read_text())
    daily = load_daily()

    outputs: dict[str, object] = {
        "model.json": export_model(model),
        "meta.json": export_meta(metrics, daily),
        "daily.json": {
            "start": daily["date"].min().strftime("%Y-%m-%d"),
            "demand_mw": daily[config.TARGET].tolist(),
            "temp_mean": daily["temp_mean"].tolist(),
            "temp_max": daily["temp_max"].tolist(),
            "temp_min": daily["temp_min"].tolist(),
        },
    }
    holdout = pd.read_csv(art / "test_results.csv", parse_dates=["date"])
    outputs["holdout.json"] = {
        "date": holdout["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_mw": holdout["actual_mw"].tolist(),
        "predicted_mw": holdout["predicted_mw"].tolist(),
        "temp_mean_c": holdout["temp_mean_c"].tolist(),
    }

    for name, payload in outputs.items():
        path: Path = WEB_DATA / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"{name:14s} {path.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
