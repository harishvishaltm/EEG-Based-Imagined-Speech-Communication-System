"""Application configuration settings."""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
KARA_ONE_DIR = DATA_DIR / "kara_one"
DEMO_SAMPLES_DIR = KARA_ONE_DIR / "demo_samples"
STATIC_AUDIO_DIR = BACKEND_DIR / "static" / "audio"

# Ensure runtime directories exist
STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


class Settings:
    PROJECT_NAME: str = "EEG-Based Imagined Speech Communication System"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    MODELS_DIR: Path = MODELS_DIR
    MODEL_PATH: Path = MODELS_DIR / os.getenv("MODEL_FILENAME", "kara_one_model.pth")
    MODEL_CONFIG_PATH: Path = MODELS_DIR / "model_config.json"
    LABEL_MAPPING_PATH: Path = MODELS_DIR / "label_mapping.json"
    CHANNEL_MAPPING_PATH: Path = MODELS_DIR / "channel_mapping.json"
    PREPROCESSING_CONFIG_PATH: Path = MODELS_DIR / "preprocessing_config.json"

    DEMO_SAMPLES_DIR: Path = DEMO_SAMPLES_DIR
    STATIC_AUDIO_DIR: Path = STATIC_AUDIO_DIR

    # Device
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu")

    # LLM Settings
    ACTIVE_LLM_BACKEND: str = os.getenv("ACTIVE_LLM_BACKEND", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "40"))

    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]

    # Limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))


settings = Settings()
