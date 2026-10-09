import pandas as pd
import pytest

from energy_forecast import config
from energy_forecast.data import temperature_climatology, validate_daily


def test_daily_dataset_is_clean(daily):
    assert len(daily) > 6000
    assert daily["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert not daily.isna().any().any()


def test_validation_rejects_gaps(daily):
    gappy = daily.drop(index=[100, 101]).reset_index(drop=True)
    with pytest.raises(ValueError, match="gaps"):
        validate_daily(gappy)


def test_validation_rejects_implausible_demand(daily):
    bad = daily.copy()
    bad.loc[10, config.TARGET] = 1.0
    with pytest.raises(ValueError, match="demand"):
        validate_daily(bad)


def test_climatology_is_seasonal(daily):
    clim = temperature_climatology(daily)
    assert len(clim) == 365
    assert clim.loc[200, "temp_mean"] > clim.loc[15, "temp_mean"] + 15  # July vs January
    assert not clim.isna().any().any()
