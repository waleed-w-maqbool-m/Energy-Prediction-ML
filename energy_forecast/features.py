"""Feature engineering shared by training, evaluation, the API and the Streamlit app.

Single source of truth: because every consumer calls ``build_features``, there is no
train/serve skew. The contract for a row at date *t* (forecast issued end-of-day *t-1*):

* calendar + holiday features depend only on the date,
* weather features use temperature for day *t* (observed in backtests; a forecast or
  climatology in production),
* demand features use demand through day *t-1* only. ``tests/test_features.py`` checks
  this by perturbing future demand and asserting the features do not move.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config

ONE_DAY = pd.Timedelta(days=1)

# Tree splits can sit exactly on a value that recurs in the data (the same calendar day every year, or
# a rolling mean of fixed-precision inputs). Libraries differ in the last bit (numpy vs V8 sin, pandas vs
# a naive mean), which would flip such a split. Rounding makes every platform compute identical values.
SIN_COS_DECIMALS = 10
ROLL_TEMP_DECIMALS = 8
ROLL_DEMAND_DECIMALS = 6


@lru_cache(maxsize=1)
def _holidays() -> pd.DatetimeIndex:
    return USFederalHolidayCalendar().holidays(start="1995-01-01", end="2045-12-31")


def nonworking(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """True for weekends and U.S. federal holidays (a pure function of the date)."""
    idx = pd.DatetimeIndex(dates)
    return np.asarray((idx.dayofweek >= 5) | idx.isin(_holidays()))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every column in ``config.FEATURES`` added.

    ``df`` needs ``date``, ``demand_mw`` (NaN allowed for days still to be forecast) and
    the weather columns, on a consecutive daily index. Rows without enough history for
    the longest lag contain NaN features; use ``training_frame`` to drop them.
    """
    out = df.copy()
    dates = pd.DatetimeIndex(out["date"])
    y = out[config.TARGET]

    # --- calendar -----------------------------------------------------------
    holiday = dates.isin(_holidays())
    angle = 2 * np.pi * dates.dayofyear / 365.25
    out["dow"] = dates.dayofweek
    out["month"] = dates.month
    out["doy_sin"] = np.round(np.sin(angle), SIN_COS_DECIMALS)
    out["doy_cos"] = np.round(np.cos(angle), SIN_COS_DECIMALS)
    out["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    out["is_holiday"] = holiday.astype(int)
    out["pre_holiday"] = (dates + ONE_DAY).isin(_holidays()).astype(int)
    out["post_holiday"] = (dates - ONE_DAY).isin(_holidays()).astype(int)
    out["is_nonworking"] = nonworking(dates).astype(int)

    # --- weather (day t) ----------------------------------------------------
    out["hdd"] = (config.BALANCE_TEMP_C - out["temp_mean"]).clip(lower=0)
    out["cdd"] = (out["temp_mean"] - config.BALANCE_TEMP_C).clip(lower=0)
    out["temp_mean_roll3"] = out["temp_mean"].rolling(3).mean().round(ROLL_TEMP_DECIMALS)
    out["temp_delta_1"] = out["temp_mean"].diff()

    # --- demand lags (strictly before day t) ---------------------------------
    shifted = y.shift(1)
    out["lag_1"] = shifted
    out["lag_2"] = y.shift(2)
    out["lag_7"] = y.shift(7)
    out["lag_14"] = y.shift(14)
    out["roll7_mean"] = shifted.rolling(7).mean().round(ROLL_DEMAND_DECIMALS)
    out["roll28_mean"] = shifted.rolling(28).mean().round(ROLL_DEMAND_DECIMALS)
    out["lag_1_nonworking"] = nonworking(dates - ONE_DAY).astype(int)
    out["lag_7_nonworking"] = nonworking(dates - 7 * ONE_DAY).astype(int)
    return out


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame with warm-up rows (insufficient history) removed."""
    feats = build_features(df)
    return feats.dropna(subset=[*config.FEATURES, config.TARGET]).reset_index(drop=True)
