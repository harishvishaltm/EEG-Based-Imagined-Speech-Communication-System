"""Tests for Assistive LLM Sentence Generation."""
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from backend.app.services.llm_service import TemplateFallbackGenerator, llm_service

client = TestClient(app)


def test_template_fallback_categories():
    assert TemplateFallbackGenerator.generate("water") == "I need water."
    assert TemplateFallbackGenerator.generate("help") == "I need help."
    assert TemplateFallbackGenerator.generate("hungry") == "I am hungry."
    assert TemplateFallbackGenerator.generate("up") == "I want to go up."
    assert TemplateFallbackGenerator.generate("yes") == "Yes."
    assert TemplateFallbackGenerator.generate("pot") == "I want a pot."
    assert TemplateFallbackGenerator.generate("/diy/") == "I'm interested in DIY."


def test_llm_service_generate_sentence():
    result = llm_service.generate_sentence("water")
    assert "sentence" in result
    assert "generation_mode" in result
    assert len(result["sentence"]) > 0


def test_llm_api_endpoint():
    response = client.post("/api/generate-sentence", json={"word": "help"})
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "help"
    assert "I need help" in data["sentence"]
