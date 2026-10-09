"""Smoke tests: the Streamlit app renders and responds to its controls without errors."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file("app.py", default_timeout=120).run()


def test_app_renders_without_exceptions(app):
    assert not app.exception
    assert len(app.tabs) == 4


def test_scenario_slider_changes_forecast(app):
    baseline_md = " ".join(m.value for m in app.markdown)
    assert "Scenario" not in baseline_md.replace("Temperature scenario", "")

    scenario = next(s for s in app.slider if s.label.startswith("Temperature scenario"))
    scenario.set_value(4.0).run()
    assert not app.exception
    assert any("Scenario effect" in m.value for m in app.markdown)


def test_weather_assumption_toggle(app):
    radio = next(r for r in app.radio if r.label == "Weather assumption")
    radio.set_value("Normal weather for the season").run()
    assert not app.exception
