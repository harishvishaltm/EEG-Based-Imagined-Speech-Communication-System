"""Main FastAPI Application Entrypoint.

Handles:
- Lifespan model loading (strictly loaded ONCE into memory)
- CORS middleware configuration
- Router registration
- Controlled exception handling
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import (
    demo_router,
    health_router,
    llm_router,
    prediction_router,
    tts_router,
)
from backend.app.config import settings
from backend.app.models.model_loader import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("eeg_bci_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Loads the trained PyTorch model ONCE at startup."""
    logger.info("=" * 70)
    logger.info("STARTING EEG-BASED IMAGINED SPEECH COMMUNICATION SYSTEM")
    logger.info("=" * 70)
    try:
        model_manager.initialize()
        logger.info("Model loaded and memory-resident for real-time inference.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
    yield
    logger.info("Shutting down application backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production backend for EEG-based imagined speech decoding, "
        "assistive sentence generation, and text-to-speech communication."
    ),
    lifespan=lifespan,
)

# CORS middleware for React/Vite frontend (e.g. http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler to prevent leaking internal stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An internal server error occurred while processing your request.",
            "path": request.url.path,
        },
    )


# Register all API routers
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(demo_router)
app.include_router(llm_router)
app.include_router(tts_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
