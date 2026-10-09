"""Evaluation: metrics, walk-forward backtests, and multi-horizon interval calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .forecast import ONE_DAY, recursive_forecast_many, weather_for

Y = config.TARGET


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_predict(estimator, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = clone(estimator).fit(train[features], train[Y])
    return model.predict(test[features])


# --------------------------------------------------------------------------
# One-step-ahead, expanding window, one fold per calendar year
# --------------------------------------------------------------------------
def walk_forward_by_year(frame: pd.DataFrame, estimator, first_year: int = 2008) -> pd.DataFrame:
    """Retrain on all data before each year and score that year (one-step-ahead)."""
    rows = []
    for year in range(first_year, int(frame["date"].dt.year.max()) + 1):
        train = frame[frame["date"] < f"{year}-01-01"]
        test = frame[frame["date"].dt.year == year]
        pred = fit_predict(estimator, config.FEATURES, train, test)
        model_m = regression_metrics(test[Y], pred)
        naive_m = regression_metrics(test[Y], test["lag_1"])
        rows.append({
            "year": year, "days": len(test),
            "model_mape": model_m["mape"], "naive_mape": naive_m["mape"],
            "model_rmse": model_m["rmse"], "naive_rmse": naive_m["rmse"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Multi-horizon recursive backtest
# --------------------------------------------------------------------------
def recursive_backtest(
    model,
    daily: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    weather_mode: str,
    climatology: pd.DataFrame,
    horizon: int = config.MAX_HORIZON,
) -> pd.DataFrame:
    """Forecast ``horizon`` days from every origin whose targets fall inside ``window``.

    weather_mode ``actual`` feeds observed temperatures (a perfect weather forecast);
    ``climatology`` feeds normal weather for the day of year (no weather forecast at all).
    """
    indexed = daily.set_index("date")
    start, end = window
    origins = pd.date_range(start - ONE_DAY, end - horizon * ONE_DAY, freq="D")
    date_grid = [pd.date_range(o + ONE_DAY, periods=horizon, freq="D") for o in origins]

    histories = [daily[daily["date"] <= o] for o in origins]
    futures = [weather_for(d, daily, climatology, weather_mode) for d in date_grid]
    pred = recursive_forecast_many(model, histories, futures)

    rows = []
    for i, (origin, dates) in enumerate(zip(origins, date_grid, strict=True)):
        rows.append(pd.DataFrame({
            "origin": origin, "date": dates, "horizon": np.arange(1, horizon + 1),
            "actual": indexed.loc[dates, Y].to_numpy(), "pred": pred[i],
            "persistence": indexed.loc[origin, Y],
            "seasonal_naive": indexed[Y].reindex(dates - 7 * ONE_DAY).to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def interval_quantiles(backtest: pd.DataFrame, coverage: float = config.INTERVAL_COVERAGE) -> dict[str, list[float]]:
    """Per-horizon relative-error quantiles ``(actual/pred - 1)`` -> multiplicative interval."""
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    rel = backtest.assign(rel=backtest["actual"] / backtest["pred"] - 1)
    out = {}
    for h, grp in rel.groupby("horizon"):
        out[str(int(h))] = [float(grp["rel"].quantile(lo_q)), float(grp["rel"].quantile(hi_q))]
    return out


def horizon_summary(backtest: pd.DataFrame, intervals: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for h, grp in backtest.groupby("horizon"):
        lo, hi = intervals[str(int(h))]
        inside = (grp["actual"] >= grp["pred"] * (1 + lo)) & (grp["actual"] <= grp["pred"] * (1 + hi))
        rows.append({
            "horizon": int(h),
            "mape": regression_metrics(grp["actual"], grp["pred"])["mape"],
            "rmse": regression_metrics(grp["actual"], grp["pred"])["rmse"],
            "persistence_mape": regression_metrics(grp["actual"], grp["persistence"])["mape"],
            "seasonal_naive_mape": regression_metrics(grp["actual"], grp["seasonal_naive"])["mape"],
            "interval_coverage": float(inside.mean() * 100),
            "n": len(grp),
        })
    return pd.DataFrame(rows)
