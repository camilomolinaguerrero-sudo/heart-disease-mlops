from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)
PACIENTE = {"Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,
            "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "Y",
            "Oldpeak": 1.5, "ST_Slope": "Flat"}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_riesgo_alto():
    r = client.post("/predict", json=PACIENTE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["heart_disease_probability"] <= 1
    assert body["prediction"] == 1


def test_predict_riesgo_bajo():
    sano = {**PACIENTE, "Age": 35, "ChestPainType": "ATA", "MaxHR": 175, "ExerciseAngina": "N",
            "Oldpeak": 0, "ST_Slope": "Up"}
    assert client.post("/predict", json=sano).json()["prediction"] == 0


def test_colesterol_cero_es_ausente():
    assert client.post("/predict", json={**PACIENTE, "Cholesterol": 0}).status_code == 200


def test_entrada_invalida():
    assert client.post("/predict", json={**PACIENTE, "Sex": "X"}).status_code == 422
