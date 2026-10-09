"""Inference: recursive multi-day forecasting with calibrated prediction intervals.

``recursive_forecast`` is the engine. It feeds each day's prediction back in as the next
day's lag features, using the same ``build_features`` as training. ``Forecaster`` wraps it
with the saved artifacts, the weather options (observed / climatology / scenario shift) and
the interval calibration learned during training.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import WEATHER_COLUMNS, load_daily, temperature_climatology
from .features import build_features

ONE_DAY = pd.Timedelta(days=1)


_PAD_ROWS = 30  # NaN rows between stacked segments; > longest rolling window, so nothing leaks across


def recursive_forecast_many(
    model, histories: list[pd.DataFrame], futures: list[pd.DataFrame]
) -> np.ndarray:
    """Recursive multi-day forecasts for many origins at once -> array ``(origins, horizon)``.

    Each origin is a segment of ``FEATURE_WARMUP_DAYS`` history rows followed by the days to
    forecast. Segments are stacked with NaN padding so a single vectorised ``build_features``
    call (and one ``predict``) serves every origin at every step. Padding is wider than any
    rolling window, so no feature ever sees across segments.
    """
    horizon = len(futures[0])
    segments, starts = [], []
    offset = 0
    for hist, fut in zip(histories, futures, strict=True):
        tail = hist[["date", config.TARGET, *WEATHER_COLUMNS]].tail(config.FEATURE_WARMUP_DAYS)
        ahead = fut[["date", *WEATHER_COLUMNS]].assign(**{config.TARGET: np.nan})
        seg = pd.concat([tail, ahead], ignore_index=True)
        if len(tail) < config.FEATURE_WARMUP_DAYS or len(ahead) != horizon:
            raise ValueError("histories need the full warm-up window; futures must share one horizon")
        if seg["date"].diff().dropna().ne(ONE_DAY).any():
            raise ValueError("history and future_weather must form one consecutive daily series")
        pad = pd.DataFrame({"date": [seg["date"].iloc[0]] * _PAD_ROWS})
        segments += [pad, seg]
        starts.append(offset + _PAD_ROWS + len(tail))
        offset += _PAD_ROWS + len(seg)

    stacked = pd.concat(segments, ignore_index=True)
    first = np.asarray(starts)
    out = np.empty((len(histories), horizon))
    for step in range(horizon):
        rows = first + step
        feats = build_features(stacked)
        out[:, step] = model.predict(feats.iloc[rows][config.FEATURES])
        stacked.loc[rows, config.TARGET] = out[:, step]
    return out


def recursive_forecast(model, history: pd.DataFrame, future_weather: pd.DataFrame) -> np.ndarray:
    """Predict ``len(future_weather)`` consecutive days after the last row of ``history``.

    ``history`` needs ``date``, demand and weather columns; only its last
    ``FEATURE_WARMUP_DAYS`` rows are used. ``future_weather`` has ``date`` + weather columns
    for the days to forecast, starting the day after ``history`` ends.
    """
    return recursive_forecast_many(model, [history], [future_weather])[0]


def weather_for(
    dates: pd.DatetimeIndex,
    daily: pd.DataFrame,
    climatology: pd.DataFrame,
    mode: str,
    temp_shift: float = 0.0,
) -> pd.DataFrame:
    """Weather columns for ``dates``: observed where available (mode ``actual``) else climatology."""
    doy = dates.dayofyear.to_numpy().clip(max=365)
    clim = climatology.loc[doy, WEATHER_COLUMNS].reset_index(drop=True)
    observed = daily.set_index("date")[WEATHER_COLUMNS].reindex(dates).reset_index(drop=True)
    if mode == "actual":
        weather = observed.fillna(clim)
        source = np.where(observed["temp_mean"].notna(), "observed", "climatology")
    elif mode == "climatology":
        weather, source = clim, np.full(len(dates), "climatology")
    else:
        raise ValueError("weather must be 'actual' or 'climatology'")
    weather = weather + temp_shift
    weather.insert(0, "date", dates)
    weather["weather_source"] = source
    return weather


class Forecaster:
    """Loads the trained model + data and serves forecasts for the API and the app."""

    def __init__(self, artifact_dir: str | Path = config.ARTIFACT_DIR, daily: pd.DataFrame | None = None):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.meta: dict = json.loads((artifact_dir / "metrics.json").read_text())
        self.daily = load_daily() if daily is None else daily
        self.last_date: pd.Timestamp = self.daily["date"].max()
        self.first_origin: pd.Timestamp = self.daily["date"].min() + (config.FEATURE_WARMUP_DAYS - 1) * ONE_DAY

        # "Normal weather" is learned from the training period only.
        train_end = pd.Timestamp(self.meta["split"]["train_end"])
        self.climatology = temperature_climatology(self.daily[self.daily["date"] <= train_end])
        self._intervals: dict = self.meta["intervals"]

    # ------------------------------------------------------------------
    def forecast(
        self,
        origin: str | pd.Timestamp,
        horizon: int = 7,
        weather: str = "actual",
        temp_shift: float = 0.0,
    ) -> pd.DataFrame:
        """Forecast ``horizon`` days after ``origin`` (the last day of known demand).

        weather: ``"actual"`` uses observed temperatures where the dataset has them
        (what a perfect weather forecast would give) and climatology beyond;
        ``"climatology"`` always uses normal weather for the day of year.
        temp_shift: degrees C added to every day's temperature (scenario analysis).
        """
        origin = pd.Timestamp(origin).normalize()
        if not 1 <= horizon <= config.MAX_HORIZON:
            raise ValueError(f"horizon must be between 1 and {config.MAX_HORIZON}")
        if not self.first_origin <= origin <= self.last_date:
            raise ValueError(
                f"origin must be between {self.first_origin:%Y-%m-%d} and {self.last_date:%Y-%m-%d}"
            )

        dates = pd.date_range(origin + ONE_DAY, periods=horizon, freq="D")
        wx = weather_for(dates, self.daily, self.climatology, weather, temp_shift)
        history = self.daily[self.daily["date"] <= origin]
        pred = recursive_forecast(self.model, history, wx)

        band = np.array([self._intervals[weather][str(h)] for h in range(1, horizon + 1)])
        actual = self.daily.set_index("date")[config.TARGET].reindex(dates).to_numpy()
        return pd.DataFrame({
            "date": dates,
            "predicted_mw": pred,
            "lower_mw": pred * (1 + band[:, 0]),
            "upper_mw": pred * (1 + band[:, 1]),
            "actual_mw": actual,
            "temp_mean_c": wx["temp_mean"].to_numpy(),
            "weather_source": wx["weather_source"].to_numpy(),
        })

    def predict_single(
        self,
        date: str | pd.Timestamp,
        recent_demand_mw: list[float],
        temp_mean: float,
        temp_max: float | None = None,
        temp_min: float | None = None,
        prev_temp_mean: float | None = None,
    ) -> dict:
        """One-day-ahead prediction from caller-supplied history (used by ``POST /predict``).

        ``recent_demand_mw`` are consecutive daily means ending the day before ``date``
        (oldest first); at least 28 values are required.
        """
        if len(recent_demand_mw) < 28:
            raise ValueError("recent_demand_mw needs at least 28 daily values")
        date = pd.Timestamp(date).normalize()
        recent = np.asarray(recent_demand_mw[-config.FEATURE_WARMUP_DAYS:], dtype=float)
        prev = temp_mean if prev_temp_mean is None else prev_temp_mean
        past_dates = pd.date_range(end=date - ONE_DAY, periods=len(recent), freq="D")
        history = pd.DataFrame({
            "date": past_dates, config.TARGET: recent,
            "temp_mean": prev, "temp_max": prev, "temp_min": prev,
        })
        today = pd.DataFrame({
            "date": [date], "temp_mean": [temp_mean],
            "temp_max": [temp_mean if temp_max is None else temp_max],
            "temp_min": [temp_mean if temp_min is None else temp_min],
        })
        pred = float(recursive_forecast(self.model, history, today)[0])
        lo, hi = self._intervals["actual"]["1"]
        return {"date": date, "predicted_mw": pred, "lower_mw": pred * (1 + lo), "upper_mw": pred * (1 + hi)}
