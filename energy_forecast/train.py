"""Offline training + evaluation pipeline.

    python -m energy_forecast.train

Reads the committed daily dataset in ``data/`` (no network or credentials needed) and
writes every artifact the API and Streamlit app serve into ``artifacts/``.

Protocol
--------
1. Chronological hold-out: the last 15 % of days are never used for fitting or tuning.
2. Hyper-parameters: randomized search scored with ``TimeSeriesSplit`` on the training span.
3. Comparison: naive baselines, ridge, the original v1 model and a no-weather ablation are
   fitted on the same training span and scored on the same hold-out.
4. Walk-forward backtest: retrain before each calendar year, score that year.
5. Prediction intervals: a model trained *before* a calibration window produces recursive
   1-7 day forecasts over that window; per-horizon relative-error quantiles become the
   interval. Coverage is then measured on the untouched hold-out.
"""

from __future__ import annotations

import os

# One OpenMP thread per process: gradient boosting on ~5k rows is faster (and far more
# predictable) than oversubscribed threads; parallelism comes from the search's processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # noqa: E402

from . import config  # noqa: E402
from .data import load_daily, temperature_climatology  # noqa: E402
from .evaluate import (  # noqa: E402
    fit_predict,
    horizon_summary,
    interval_quantiles,
    recursive_backtest,
    regression_metrics,
    walk_forward_by_year,
)
from .features import training_frame  # noqa: E402
from .models import HGB_SEARCH_SPACE, comparison_candidates, make_hgb

log = logging.getLogger("train")
Y = config.TARGET
FEATURE_GROUPS = {
    **dict.fromkeys(config.CALENDAR_FEATURES, "Calendar"),
    **dict.fromkeys(config.WEATHER_FEATURES, "Weather"),
    **dict.fromkeys(config.LAG_FEATURES, "Demand history"),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    config.ARTIFACT_DIR.mkdir(exist_ok=True)

    daily = load_daily()
    frame = training_frame(daily)
    split = int(len(frame) * (1 - config.TEST_FRACTION))
    train, test = frame.iloc[:split], frame.iloc[split:]
    test_start, train_end = test["date"].min(), train["date"].max()
    cal_start = test_start - pd.Timedelta(days=config.CALIBRATION_DAYS)
    log.info("Data %s -> %s | train %d days | test %d days (from %s)",
             frame["date"].min().date(), frame["date"].max().date(), len(train), len(test), test_start.date())

    # 1. Hyper-parameter search -------------------------------------------------
    search = RandomizedSearchCV(
        make_hgb(), HGB_SEARCH_SPACE, n_iter=30, cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error", n_jobs=4, random_state=config.RANDOM_STATE,
    )
    search.fit(train[config.FEATURES], train[Y])
    best = {k: (None if v is None else (v.item() if hasattr(v, "item") else v))
            for k, v in search.best_params_.items()}
    log.info("Best params %s | CV RMSE %.0f MW", best, -search.best_score_)

    # 2. Model comparison on the hold-out -----------------------------------------
    comparison = [
        {"model": "Naive: same as yesterday", "note": "Baseline", **regression_metrics(test[Y], test["lag_1"])},
        {"model": "Seasonal naive: same weekday last week", "note": "Baseline",
         **regression_metrics(test[Y], test["lag_7"])},
    ]
    predictions: dict[str, np.ndarray] = {}
    for cand in comparison_candidates(best):
        pred = fit_predict(cand.estimator, cand.features, train, test)
        predictions[cand.name] = pred
        comparison.append({"model": cand.name, "note": cand.note, **regression_metrics(test[Y], pred)})
        log.info("%-48s MAPE %.2f%%  RMSE %.0f", cand.name, comparison[-1]["mape"], comparison[-1]["rmse"])

    final = clone(make_hgb(**best)).fit(train[config.FEATURES], train[Y])
    final_pred = final.predict(test[config.FEATURES])
    holdout = regression_metrics(test[Y], final_pred)
    naive = regression_metrics(test[Y], test["lag_1"])

    # 3. Walk-forward backtest by year ----------------------------------------------
    by_year = walk_forward_by_year(frame, make_hgb(**best))
    by_year.round(3).to_csv(config.ARTIFACT_DIR / "backtest_by_year.csv", index=False)
    log.info("Walk-forward backtest:\n%s", by_year.round(2).to_string(index=False))

    # 4. Multi-horizon intervals ------------------------------------------------------
    cal_end = test_start - pd.Timedelta(days=1)
    cal_model = make_hgb(**best).fit(frame.loc[frame["date"] < cal_start, config.FEATURES],
                                     frame.loc[frame["date"] < cal_start, Y])
    clim_cal = temperature_climatology(daily[daily["date"] < cal_start])
    clim_test = temperature_climatology(daily[daily["date"] <= train_end])

    intervals, horizon_rows = {}, []
    for mode in ("actual", "climatology"):
        cal_bt = recursive_backtest(cal_model, daily, (cal_start, cal_end), mode, clim_cal)
        intervals[mode] = interval_quantiles(cal_bt)
        test_bt = recursive_backtest(final, daily, (test_start, daily["date"].max()), mode, clim_test)
        summary = horizon_summary(test_bt, intervals[mode]).assign(weather=mode)
        horizon_rows.append(summary)
        log.info("Horizon summary (%s weather):\n%s", mode, summary.round(2).to_string(index=False))
    horizon_df = pd.concat(horizon_rows, ignore_index=True)
    horizon_df.round(3).to_csv(config.ARTIFACT_DIR / "horizon_metrics.csv", index=False)

    # 5. Explainability ---------------------------------------------------------------------
    perm = permutation_importance(
        final, test[config.FEATURES], test[Y], n_repeats=10,
        scoring="neg_root_mean_squared_error", random_state=config.RANDOM_STATE, n_jobs=4,
    )
    importance = (
        pd.DataFrame({
            "feature": config.FEATURES,
            "group": [FEATURE_GROUPS[f] for f in config.FEATURES],
            "rmse_increase_mw": perm.importances_mean,
            "std": perm.importances_std,
        })
        .sort_values("rmse_increase_mw", ascending=False)
        .reset_index(drop=True)
    )
    importance.round(3).to_csv(config.ARTIFACT_DIR / "feature_importance.csv", index=False)

    # 6. Persist ------------------------------------------------------------------------------
    joblib.dump(final, config.ARTIFACT_DIR / "model.joblib", compress=3)
    results = pd.DataFrame({
        "actual_mw": test[Y].to_numpy(),
        "predicted_mw": final_pred,
        "baseline_mw": test["lag_1"].to_numpy(),
        "temp_mean_c": test["temp_mean"].to_numpy(),
    }).round(3)
    results.insert(0, "date", test["date"].to_numpy())
    results.to_csv(config.ARTIFACT_DIR / "test_results.csv", index=False)

    metrics = {
        "model": {"name": "HistGradientBoostingRegressor", "params": best, "n_features": len(config.FEATURES),
                  "features": config.FEATURES, "cv_rmse": float(-search.best_score_)},
        "split": {
            "train_start": str(train["date"].min().date()), "train_end": str(train_end.date()),
            "calibration_start": str(cal_start.date()),
            "test_start": str(test_start.date()), "test_end": str(test["date"].max().date()),
            "n_train": len(train), "n_test": len(test),
        },
        "holdout": holdout,
        "baseline": naive,
        "improvement_pct": 100 * (naive["rmse"] - holdout["rmse"]) / naive["rmse"],
        "comparison": sorted(comparison, key=lambda r: -r["mape"]),
        "intervals": intervals,
        "interval_nominal_coverage": config.INTERVAL_COVERAGE,
        "environment": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (config.ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Done in %.0fs. Hold-out MAPE %.2f%% (naive %.2f%%).", time.time() - t0, holdout["mape"], naive["mape"])


if __name__ == "__main__":
    main()
