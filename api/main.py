"""REST API for the PJM East demand forecaster.

    uvicorn api.main:app --reload        # interactive docs at /docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as Date
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from energy_forecast import config
from energy_forecast.forecast import Forecaster

state: dict[str, Forecaster] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["forecaster"] = Forecaster()  # model + data are loaded once at startup
    yield
    state.clear()


app = FastAPI(
    title="PJM East Demand Forecast API",
    version="1.0.0",
    description=(
        "Day-ahead and 7-day electricity demand forecasts for the PJM East region "
        "(Gradient Boosting on weather, calendar and lagged-demand features) with "
        "calibrated 80 % prediction intervals."
    ),
    lifespan=lifespan,
)


def forecaster() -> Forecaster:
    return state["forecaster"]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    date: Date
    predicted_mw: float
    lower_mw: float = Field(description="Lower bound of the 80 % prediction interval")
    upper_mw: float = Field(description="Upper bound of the 80 % prediction interval")
    actual_mw: float | None = Field(None, description="Observed demand, when the date is in the dataset")
    temp_mean_c: float = Field(description="Daily mean temperature assumed for this day")
    weather_source: Literal["observed", "climatology"]


class ForecastResponse(BaseModel):
    origin: Date
    horizon: int
    weather: Literal["actual", "climatology"]
    temp_shift_c: float
    points: list[ForecastPoint]


class PredictRequest(BaseModel):
    date: Date = Field(description="Day to predict")
    recent_demand_mw: list[float] = Field(
        min_length=28, max_length=60,
        description="Consecutive daily mean demand (MW) ending the day before `date`, oldest first",
    )
    temp_mean: float = Field(ge=-40, le=50, description="Daily mean temperature on `date` (deg C)")
    temp_max: float | None = Field(None, ge=-40, le=55)
    temp_min: float | None = Field(None, ge=-50, le=50)
    prev_temp_mean: float | None = Field(None, ge=-40, le=50, description="Mean temperature the days before")

    @model_validator(mode="after")
    def _plausible_demand(self):
        if not all(5_000 <= v <= 100_000 for v in self.recent_demand_mw):
            raise ValueError("recent_demand_mw values must be daily means between 5,000 and 100,000 MW")
        return self


class PredictResponse(BaseModel):
    date: Date
    predicted_mw: float
    lower_mw: float
    upper_mw: float


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health() -> dict:
    f = forecaster()
    return {
        "status": "ok",
        "data_through": f.last_date.date().isoformat(),
        "trained_at": f.meta["trained_at"],
    }


@app.get("/model", tags=["meta"])
def model_card() -> dict:
    """Hold-out accuracy, model comparison and training configuration."""
    meta = forecaster().meta
    return {k: meta[k] for k in ("model", "split", "holdout", "baseline", "improvement_pct", "comparison")}


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    origin: Date = Query(description="Last day of known demand; the forecast starts the day after"),
    horizon: int = Query(7, ge=1, le=config.MAX_HORIZON),
    weather: Literal["actual", "climatology"] = Query(
        "actual", description="'actual': observed temperatures where available (a perfect weather forecast); "
        "'climatology': normal weather for the time of year",
    ),
    temp_shift_c: float = Query(0.0, ge=-15, le=15, description="Scenario: add this many deg C to every day"),
) -> ForecastResponse:
    """Recursive multi-day forecast from a historical origin, with 80 % intervals."""
    try:
        df = forecaster().forecast(origin, horizon, weather, temp_shift_c)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    points = [
        ForecastPoint(
            date=r.date.date(), predicted_mw=r.predicted_mw, lower_mw=r.lower_mw, upper_mw=r.upper_mw,
            actual_mw=None if r.actual_mw != r.actual_mw else r.actual_mw,  # NaN -> null
            temp_mean_c=r.temp_mean_c, weather_source=r.weather_source,
        )
        for r in df.itertuples()
    ]
    return ForecastResponse(origin=origin, horizon=horizon, weather=weather, temp_shift_c=temp_shift_c, points=points)


@app.post("/predict", response_model=PredictResponse, tags=["forecast"])
def predict(req: PredictRequest) -> PredictResponse:
    """Day-ahead prediction from caller-supplied recent demand and that day's temperature."""
    try:
        out = forecaster().predict_single(
            req.date, req.recent_demand_mw, req.temp_mean, req.temp_max, req.temp_min, req.prev_temp_mean,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(
        date=out["date"].date(), predicted_mw=out["predicted_mw"],
        lower_mw=out["lower_mw"], upper_mw=out["upper_mw"],
    )
