"""Tests for Model Loading and In-Memory Singleton."""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.model_loader import model_manager


def test_model_manager_initialization():
    model_manager.initialize()
    assert model_manager.model_loaded is True
    assert model_manager.model is not None


def test_model_eval_mode():
    model_manager.initialize()
    # Check that model is in eval mode
    assert model_manager.model.training is False
    # Check that all parameters have requires_grad set to False
    for param in model_manager.model.parameters():
        assert param.requires_grad is False


def test_model_exact_parameters():
    model_manager.initialize()
    total_params = sum(p.numel() for p in model_manager.model.parameters())
    assert total_params == 698347
