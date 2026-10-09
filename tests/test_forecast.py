import numpy as np
import pandas as pd
import pytest

from energy_forecast import config
from energy_forecast.data import temperature_climatology
from energy_forecast.features import training_frame
from energy_forecast.forecast import recursive_forecast, recursive_forecast_many, weather_for
from energy_forecast.models import make_hgb


@pytest.fixture(scope="module")
def tiny_model(daily):
    frame = training_frame(daily).iloc[:3000]
    return make_hgb(max_iter=30).fit(frame[config.FEATURES], frame[config.TARGET])


def test_batched_recursion_matches_single_origin(daily, tiny_model):
    """Stacking origins with NaN padding must give exactly the per-origin answer."""
    clim = temperature_climatology(daily)
    origins = pd.to_datetime(["2012-03-01", "2013-07-15", "2015-12-24"])
    grids = [pd.date_range(o + pd.Timedelta(days=1), periods=7) for o in origins]
    histories = [daily[daily["date"] <= o] for o in origins]
    futures = [weather_for(g, daily, clim, "actual") for g in grids]

    batched = recursive_forecast_many(tiny_model, histories, futures)
    for i in range(len(origins)):
        np.testing.assert_allclose(batched[i], recursive_forecast(tiny_model, histories[i], futures[i]))


def test_recursion_rejects_non_consecutive_weather(daily, tiny_model):
    clim = temperature_climatology(daily)
    dates = pd.DatetimeIndex(["2012-03-02", "2012-03-05"])  # gap
    history = daily[daily["date"] <= "2012-03-01"]
    with pytest.raises(ValueError, match="consecutive"):
        recursive_forecast(tiny_model, history, weather_for(dates, daily, clim, "actual"))


def test_forecast_shape_and_bands(forecaster):
    out = forecaster.forecast("2017-07-10", horizon=7)
    assert len(out) == 7
    assert out["date"].iloc[0] == pd.Timestamp("2017-07-11")
    assert (out["lower_mw"] < out["predicted_mw"]).all() and (out["predicted_mw"] < out["upper_mw"]).all()
    assert out["actual_mw"].notna().all()  # inside the dataset


def test_forecast_beyond_data_uses_climatology(forecaster):
    out = forecaster.forecast(forecaster.last_date, horizon=3)
    assert (out["weather_source"] == "climatology").all()
    assert out["actual_mw"].isna().all()


def test_heatwave_scenario_raises_summer_demand(forecaster):
    normal = forecaster.forecast("2017-07-10", 5, temp_shift=0.0)
    hot = forecaster.forecast("2017-07-10", 5, temp_shift=5.0)
    assert (hot["predicted_mw"] > normal["predicted_mw"]).all()


def test_wider_intervals_without_a_weather_forecast(forecaster):
    known = forecaster.forecast("2017-07-10", 7, weather="actual")
    unknown = forecaster.forecast("2017-07-10", 7, weather="climatology")

    def width(d):
        return ((d["upper_mw"] - d["lower_mw"]) / d["predicted_mw"]).mean()

    assert width(unknown) > width(known)


@pytest.mark.parametrize("kwargs", [
    {"origin": "1999-01-01"},
    {"origin": "2030-01-01"},
    {"origin": "2017-07-10", "horizon": 0},
    {"origin": "2017-07-10", "horizon": 30},
    {"origin": "2017-07-10", "weather": "sunny"},
])
def test_forecast_validates_inputs(forecaster, kwargs):
    with pytest.raises(ValueError):
        forecaster.forecast(**kwargs)


def test_predict_single_matches_dataset_forecast(forecaster, daily):
    """Caller-supplied history should reproduce the dataset-backed one-day prediction."""
    origin = pd.Timestamp("2017-07-10")
    hist = daily[daily["date"] <= origin]
    target = daily[daily["date"] == origin + pd.Timedelta(days=1)].iloc[0]
    out = forecaster.predict_single(
        target["date"], hist[config.TARGET].tail(35).tolist(),
        target["temp_mean"], target["temp_max"], target["temp_min"],
        prev_temp_mean=hist["temp_mean"].iloc[-1],
    )
    reference = forecaster.forecast(origin, 1).iloc[0]
    # Prior-day temperatures are approximated by one value, so allow a small difference.
    assert abs(out["predicted_mw"] - reference["predicted_mw"]) / reference["predicted_mw"] < 0.03
    with pytest.raises(ValueError, match="28"):
        forecaster.predict_single("2017-07-11", [30_000.0] * 10, 25.0)
