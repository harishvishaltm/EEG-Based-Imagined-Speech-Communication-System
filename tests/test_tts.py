"""Tests for Text-To-Speech Synthesis and Audio Delivery."""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.services.tts_service import tts_service

client = TestClient(app)


def test_tts_service_synthesis():
    result = tts_service.synthesize("I need water.", word_hint="water")
    assert "audio_url" in result
    assert "backend_used" in result


def test_tts_api_endpoint():
    response = client.post("/api/text-to-speech", json={"text": "I need help.", "word_hint": "help"})
    assert response.status_code == 200
    data = response.json()
    assert "audio_url" in data
    assert "backend_used" in data
