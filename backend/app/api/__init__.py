"""API package routes."""
from .routes_demo import router as demo_router
from .routes_health import router as health_router
from .routes_llm import router as llm_router
from .routes_prediction import router as prediction_router
from .routes_tts import router as tts_router

__all__ = [
    "demo_router",
    "health_router",
    "llm_router",
    "prediction_router",
    "tts_router",
]
