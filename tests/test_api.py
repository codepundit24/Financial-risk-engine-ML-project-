import os
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "risk-scoring-engine"}

def test_predict_approved():
    payload = {
        "Time": 500,
        "Amount": 1200,
        "V1": 0.1, "V2": -0.2, "V3": 0.5, "V4": -0.1,
        "V5": 0.3, "V6": 0.0, "V7": -0.4, "V8": 0.2,
        "V9": 0.1, "V10": -0.3, "V11": 0.4, "V12": -0.1,
        "V13": 0.2, "V14": 0.0, "V15": -0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVED"
    assert data["requires_manual_review"] is False
    assert len(data["top_risk_factors"]) == 0

def test_predict_flagged():
    payload = {
        "Time": 1500.0,
        "Amount": 85000.0,
        "V1": 3.2, "V2": 2.8, "V3": 3.1, "V4": 2.9,
        "V5": 3.0, "V6": 2.7, "V7": 3.4, "V8": 2.5,
        "V9": 3.1, "V10": 2.8, "V11": 3.0, "V12": 3.3,
        "V13": 2.9, "V14": 3.5, "V15": 2.6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "FLAGGED"
    assert data["requires_manual_review"] is True
    assert len(data["top_risk_factors"]) > 0