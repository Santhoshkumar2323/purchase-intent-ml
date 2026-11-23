import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import settings
from src.paths import artifact_path

needs_model = pytest.mark.skipif(
    not artifact_path(settings.model.version, "model.joblib").exists(),
    reason="Train the model first (see README).",
)
GOOD = {"ProductRelated": 30, "ProductRelated_Duration": 1200, "PageValues": 8.5,
        "ExitRates": 0.02, "BounceRates": 0.01, "VisitorType": "Returning_Visitor"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c: 
        yield c


@needs_model
def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == settings.model.version


@needs_model
def test_predict_returns_valid_probability(client):
    resp = client.post("/predict", json=GOOD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["action"] in {"show_offer", "no_action"}


@needs_model
@pytest.mark.parametrize("bad", [
    {"BounceRates": 5},                  
    {"VisitorType": "Alien"},            
    {"ProductRelated": -3},             
    {"NotAField": 1},                   
])
def test_bad_input_is_rejected(client, bad):
    assert client.post("/predict", json={**GOOD, **bad}).status_code == 422


@needs_model
def test_explain_returns_reasons(client):
    resp = client.post("/explain", json=GOOD)
    assert resp.status_code in (200, 503)  
    if resp.status_code == 200:
        reasons = resp.json()["reasons"]
        assert 1 <= len(reasons) <= 5
        assert {"feature", "effect", "direction"} <= set(reasons[0])


