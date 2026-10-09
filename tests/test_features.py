import numpy as np
import pandas as pd

from energy_forecast import config
from energy_forecast.features import build_features, training_frame


def test_features_never_see_the_target_day_or_later(daily):
    """Leakage guard: scrambling demand from day t onward must not change day t's features."""
    t = 3000
    window = daily.iloc[t - 60 : t + 10].reset_index(drop=True)
    cut = 60  # row of day t inside the window
    baseline = build_features(window).iloc[cut][config.FEATURES]

    perturbed = window.copy()
    perturbed.loc[cut:, config.TARGET] = perturbed.loc[cut:, config.TARGET] * 3 + 12345
    after = build_features(perturbed).iloc[cut][config.FEATURES]
    pd.testing.assert_series_equal(baseline, after)


def test_holiday_features():
    dates = pd.date_range("2017-07-02", "2017-07-06")
    df = pd.DataFrame({
        "date": dates, config.TARGET: 30_000.0, "temp_mean": 25.0, "temp_max": 30.0, "temp_min": 20.0,
    })
    f = build_features(df).set_index("date")
    assert f.loc["2017-07-04", "is_holiday"] == 1
    assert f.loc["2017-07-03", "pre_holiday"] == 1
    assert f.loc["2017-07-05", "post_holiday"] == 1
    assert f.loc["2017-07-02", "is_weekend"] == 1  # Sunday
    assert f.loc["2017-07-05", "lag_1_nonworking"] == 1  # yesterday was July 4th


def test_degree_days_are_complementary(daily):
    f = build_features(daily)
    assert (f["hdd"] >= 0).all() and (f["cdd"] >= 0).all()
    assert (f["hdd"] * f["cdd"] == 0).all()


def test_training_frame_is_complete(daily):
    frame = training_frame(daily)
    assert not frame[config.FEATURES].isna().any().any()
    assert set(config.FEATURES) <= set(frame.columns)
    assert len(frame) == len(daily) - 28  # the 28-day rolling mean needs 28 prior days


def test_cyclical_day_of_year_is_continuous():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-12-31", "2018-01-01"]), config.TARGET: 1.0,
        "temp_mean": 0.0, "temp_max": 0.0, "temp_min": 0.0,
    })
    f = build_features(df)
    assert np.hypot(f["doy_sin"].diff().iloc[1], f["doy_cos"].diff().iloc[1]) < 0.05
