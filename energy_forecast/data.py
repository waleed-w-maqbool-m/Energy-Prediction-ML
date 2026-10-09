"""Data layer: build, load and validate the daily demand + weather dataset.

The repository ships the two small daily CSVs in ``data/`` so that training and the
app work offline. ``build_data`` (``python -m energy_forecast.build_data``) rebuilds
them from the original sources (Kaggle + Open-Meteo).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

WEATHER_COLUMNS = ["temp_mean", "temp_max", "temp_min"]
MIN_HOURS_PER_DAY = 20  # drop days with large gaps; DST days legitimately have 23 or 25 rows


# --------------------------------------------------------------------------
# Building the dataset from source
# --------------------------------------------------------------------------
def build_daily_demand(hourly_csv: str | Path) -> pd.DataFrame:
    """Aggregate the hourly PJME file to daily mean demand (MW)."""
    hourly = pd.read_csv(hourly_csv, parse_dates=["Datetime"])
    hourly = hourly.rename(columns={"PJME_MW": config.TARGET}).sort_values("Datetime")

    grouped = hourly.set_index("Datetime")[config.TARGET].resample("D")
    daily = pd.DataFrame({config.TARGET: grouped.mean(), "hours": grouped.count()})
    complete = daily["hours"] >= MIN_HOURS_PER_DAY
    log.info("Dropping %d incomplete days", int((~complete).sum()))
    daily = daily[complete].drop(columns="hours")
    daily.index.name = "date"
    return daily.reset_index()


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Population-weighted daily temperatures (deg C) over the PJM East stations."""
    stations = config.WEATHER_STATIONS
    total_weight = sum(s["weight"] for s in stations.values())
    blended: pd.DataFrame | None = None

    for name, s in stations.items():
        query = urllib.parse.urlencode({
            "latitude": s["lat"], "longitude": s["lon"],
            "start_date": start, "end_date": end,
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
            "timezone": "America/New_York",
        })
        with urllib.request.urlopen(f"{config.OPEN_METEO_ARCHIVE}?{query}", timeout=60) as resp:
            payload = json.load(resp)["daily"]
        frame = pd.DataFrame({
            "date": pd.to_datetime(payload["time"]),
            "temp_mean": payload["temperature_2m_mean"],
            "temp_max": payload["temperature_2m_max"],
            "temp_min": payload["temperature_2m_min"],
        }).set_index("date")
        weighted = frame * (s["weight"] / total_weight)
        blended = weighted if blended is None else blended.add(weighted)
        log.info("Fetched weather for %s (%d days)", name, len(frame))

    assert blended is not None
    return blended.round(2).reset_index()


def build_data(hourly_csv: str | Path) -> pd.DataFrame:
    """Rebuild both CSVs in ``data/`` from source and return the merged frame."""
    config.DATA_DIR.mkdir(exist_ok=True)
    demand = build_daily_demand(hourly_csv)
    demand.assign(**{config.TARGET: demand[config.TARGET].round(3)}).to_csv(config.DEMAND_CSV, index=False)

    start, end = demand["date"].min().strftime("%Y-%m-%d"), demand["date"].max().strftime("%Y-%m-%d")
    weather = fetch_weather(start, end)
    weather.to_csv(config.WEATHER_CSV, index=False)
    return load_daily()


# --------------------------------------------------------------------------
# Loading + validation
# --------------------------------------------------------------------------
def validate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ``ValueError`` if the merged frame violates basic data-quality contracts."""
    problems: list[str] = []
    if df["date"].duplicated().any():
        problems.append("duplicate dates")
    if not df["date"].is_monotonic_increasing:
        problems.append("dates not sorted")
    gaps = df["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        problems.append(f"{int((gaps != pd.Timedelta(days=1)).sum())} gaps in the daily index")
    if df[[config.TARGET, *WEATHER_COLUMNS]].isna().any().any():
        problems.append("missing values")
    if not df[config.TARGET].between(5_000, 100_000).all():
        problems.append("demand outside the plausible 5-100 GW range")
    if not df["temp_mean"].between(-40, 50).all():
        problems.append("temperature outside the plausible range")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    return df


def load_daily() -> pd.DataFrame:
    """Daily demand merged with weather, validated and indexed by consecutive dates."""
    demand = pd.read_csv(config.DEMAND_CSV, parse_dates=["date"])
    weather = pd.read_csv(config.WEATHER_CSV, parse_dates=["date"])
    df = demand.merge(weather, on="date", how="left").sort_values("date").reset_index(drop=True)

    # Rare missing demand days are interpolated so lag features stay aligned to calendar days.
    full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_index)
    df[config.TARGET] = df[config.TARGET].interpolate(limit=3)
    df[WEATHER_COLUMNS] = df[WEATHER_COLUMNS].interpolate(limit=3)
    df.index.name = "date"
    df = df.reset_index().astype({"date": "datetime64[ns]"})
    return validate_daily(df)


def temperature_climatology(df: pd.DataFrame, smooth_days: int = 7) -> pd.DataFrame:
    """Mean temperature by day-of-year (circularly smoothed): the 'normal weather' fallback."""
    doy = df["date"].dt.dayofyear.clip(upper=365)
    clim = df.groupby(doy)[WEATHER_COLUMNS].mean().reindex(range(1, 366))
    wrapped = pd.concat([clim.iloc[-smooth_days:], clim, clim.iloc[:smooth_days]])
    smooth = wrapped.rolling(2 * smooth_days + 1, center=True).mean().iloc[smooth_days:-smooth_days]
    smooth.index = clim.index
    return smooth.astype(np.float64)
