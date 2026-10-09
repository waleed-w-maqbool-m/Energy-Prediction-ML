"""The JavaScript forecaster (Cloudflare Worker + browser) must match the Python one.

Python is the source of truth: it trains the model and defines the features. The exported JSON
model and ``web/js/forecast.js`` re-implement inference, so any drift would silently change
the numbers users see. These tests run both on the same inputs and compare.
"""

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from energy_forecast import config

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

RUNNER = config.ROOT / "web" / "tests" / "run_cases.mjs"


def run_js(cases: list[dict]) -> list:
    proc = subprocess.run(
        [NODE, str(RUNNER)], input=json.dumps(cases), capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(proc.stdout)


def test_exported_data_is_fresh():
    """web/data must come from the model in artifacts/ (re-run export_web after training)."""
    web_meta = json.loads((config.ROOT / "web" / "data" / "meta.json").read_text())
    trained = json.loads((config.ARTIFACT_DIR / "metrics.json").read_text())["trained_at"]
    assert web_meta["trained_at"] == trained, "run `python -m energy_forecast.export_web`"


def test_multi_day_forecasts_match_python(forecaster, daily):
    origins = list(daily["date"].iloc[27::311])
    origins += [forecaster.first_origin, forecaster.last_date, pd.Timestamp("2017-07-17")]
    cases, expected = [], []
    for origin in origins:
        for weather in ("actual", "climatology"):
            for shift in (-5.0, 0.0, 3.5):
                horizon = 7 if shift != 3.5 else 3
                args = {"origin": origin.strftime("%Y-%m-%d"), "horizon": horizon, "weather": weather,
                        "tempShift": shift}
                cases.append({"kind": "forecast", "args": args})
                expected.append(forecaster.forecast(origin, horizon, weather, shift))

    got = run_js(cases)
    assert len(got) == len(expected) > 40
    for case, js_rows, py in zip(cases, got, expected, strict=True):
        assert isinstance(js_rows, list), (case, js_rows)
        for js, (_, row) in zip(js_rows, py.iterrows(), strict=True):
            tag = (case["args"], row["date"])
            assert js["date"] == row["date"].strftime("%Y-%m-%d"), tag
            for key in ("predicted_mw", "lower_mw", "upper_mw", "temp_mean_c"):
                np.testing.assert_allclose(js[key], row[key], rtol=1e-9, err_msg=str((key, tag)))
            assert js["weather_source"] == row["weather_source"], tag
            if np.isnan(row["actual_mw"]):
                assert js["actual_mw"] is None, tag
            else:
                assert js["actual_mw"] == pytest.approx(row["actual_mw"]), tag


def test_single_day_prediction_matches_python(forecaster, daily):
    cases, expected = [], []
    for end in ("2012-03-14", "2016-08-09", "2018-01-20"):
        hist = daily[daily["date"] <= end]
        target = pd.Timestamp(end) + pd.Timedelta(days=1)
        recent = hist[config.TARGET].tail(40).tolist()
        kwargs = {"temp_mean": 21.5, "temp_max": 27.0, "temp_min": 15.0, "prev_temp_mean": 19.0}
        expected.append(forecaster.predict_single(target, recent, **kwargs))
        cases.append({"kind": "single", "args": {
            "date": target.strftime("%Y-%m-%d"), "recentDemand": recent, "tempMean": 21.5,
            "tempMax": 27.0, "tempMin": 15.0, "prevTempMean": 19.0,
        }})
    for js, py in zip(run_js(cases), expected, strict=True):
        for key in ("predicted_mw", "lower_mw", "upper_mw"):
            np.testing.assert_allclose(js[key], py[key], rtol=1e-9)


@pytest.mark.parametrize("args", [
    {"origin": "1999-01-01"},
    {"origin": "2030-01-01"},
    {"origin": "2017-07-10", "horizon": 0},
    {"origin": "2017-07-10", "horizon": 30},
    {"origin": "2017-07-10", "weather": "sunny"},
    {"origin": "garbage"},
])
def test_js_rejects_the_same_bad_input(args):
    (result,) = run_js([{"kind": "forecast", "args": args}])
    assert "error" in result
