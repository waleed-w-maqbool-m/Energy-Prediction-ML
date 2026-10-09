import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_through"] == "2018-08-02"


def test_model_card(client):
    body = client.get("/model").json()
    assert body["holdout"]["mape"] < body["baseline"]["mape"]
    assert len(body["comparison"]) >= 5


def test_forecast(client):
    r = client.get("/forecast", params={"origin": "2017-07-10", "horizon": 3, "temp_shift_c": 2})
    assert r.status_code == 200
    pts = r.json()["points"]
    assert len(pts) == 3
    assert all(p["lower_mw"] < p["predicted_mw"] < p["upper_mw"] for p in pts)


def test_forecast_beyond_data_serialises_null_actuals(client):
    pts = client.get("/forecast", params={"origin": "2018-08-02", "horizon": 2}).json()["points"]
    assert all(p["actual_mw"] is None for p in pts)


@pytest.mark.parametrize("params", [
    {"origin": "1990-01-01"},
    {"origin": "2017-07-10", "horizon": 99},
    {"origin": "not-a-date"},
])
def test_forecast_rejects_bad_requests(client, params):
    assert client.get("/forecast", params=params).status_code == 422


def test_predict(client):
    body = {
        "date": "2017-07-11", "recent_demand_mw": [38_000 + 100 * i for i in range(28)],
        "temp_mean": 27.0, "temp_max": 32.0, "temp_min": 22.0,
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["lower_mw"] < out["predicted_mw"] < out["upper_mw"]


def test_predict_rejects_short_or_implausible_history(client):
    base = {"date": "2017-07-11", "temp_mean": 27.0}
    assert client.post("/predict", json={**base, "recent_demand_mw": [30_000] * 5}).status_code == 422
    assert client.post("/predict", json={**base, "recent_demand_mw": [1.0] * 28}).status_code == 422
