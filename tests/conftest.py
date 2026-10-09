import pytest

from energy_forecast.data import load_daily
from energy_forecast.forecast import Forecaster


@pytest.fixture(scope="session")
def daily():
    return load_daily()


@pytest.fixture(scope="session")
def forecaster(daily):
    return Forecaster(daily=daily)
