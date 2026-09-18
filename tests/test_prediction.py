"""Tests for Full Inference Pipeline and Predict API."""
import sys
import tempfile
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.services.inference_service import inference_service

client = TestClient(app)


def test_inference_service_direct():
    # Create sample harmonized tensor
    tensor = np.random.randn(1, 122, 1280).astype(np.float32)
    mask = np.ones((1, 122), dtype=np.float32)

    result = inference_service.run_pipeline(
        preprocessed_data={
            "tensor": tensor,
            "channel_mask": mask,
            "dataset_name": "kara_one",
            "subject_name": "MM05",
            "ground_truth": "pot",
        },
        dataset_name="kara_one",
    )

    assert result["success"] is True
    assert "prediction" in result
    assert "word" in result["prediction"]
    assert 0.0 <= result["prediction"]["confidence"] <= 1.0
    assert len(result["top_predictions"]) >= 1
    assert "communication" in result
    assert "sentence" in result["communication"]
    assert "audio" in result
    assert "processing" in result
    assert result["processing"]["inference_time_ms"] > 0
    assert "visualization" in result
    assert result["evaluation"]["correct"] in [True, False]


def test_predict_api_file_upload():
    # Create temporary npz file
    eeg = np.random.randn(122, 1280).astype(np.float32)
    mask = np.ones(122, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        np.savez(tmp_path, eeg=eeg, channel_mask=mask, subject="MM05", dataset="Kara One", label="pat")

    try:
        with open(tmp_path, "rb") as f:
            response = client.post(
                "/api/predict",
                files={"file": ("test_trial.npz", f, "application/octet-stream")},
                data={"dataset": "kara_one", "subject": "MM05"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["input"]["dataset"] == "Kara One"
        assert data["ground_truth"] == "pat"
        assert "prediction" in data
        assert "confidence" in data["prediction"]
        assert "communication" in data
        assert "audio" in data
    finally:
        tmp_path.unlink(missing_ok=True)


def test_predict_vector_endpoint():
    # 2D list of numbers: 62 channels x 500 samples
    vec = [[0.1 * (i + j) for j in range(500)] for i in range(62)]
    response = client.post(
        "/api/predict-vector",
        json={"eeg": vec, "dataset": "kara_one", "subject": "MM05", "ground_truth": "knew"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["ground_truth"] == "knew"
    assert "prediction" in data
    assert "word" in data["prediction"]
    assert "communication" in data
    assert "audio" in data
    assert "visualization" in data

