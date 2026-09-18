"""Tests for Demo Mode endpoints."""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app

client = TestClient(app)


def test_list_demo_samples():
    response = client.get("/api/demo-samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 1
    sample_ids = [s["id"] for s in samples]
    assert "sample_001" in sample_ids


def test_predict_demo_sample():
    response = client.post("/api/predict-demo/sample_001")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ground_truth"] == "pot"
    assert "prediction" in data
    assert "confidence" in data["prediction"]
    assert "communication" in data
    assert "audio" in data
    assert "visualization" in data
    assert "waveforms" in data["visualization"]
