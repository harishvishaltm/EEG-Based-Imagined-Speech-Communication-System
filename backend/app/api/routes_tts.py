"""Text-to-Speech Endpoints and Audio File Delivery."""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.schemas.prediction import TextToSpeechRequest, TextToSpeechResponse
from backend.app.services.tts_service import tts_service

logger = logging.getLogger("routes_tts")
router = APIRouter(tags=["Text-To-Speech"])


@router.post(
    "/api/text-to-speech",
    response_model=TextToSpeechResponse,
    summary="Synthesize speech audio from text",
)
def text_to_speech(req: TextToSpeechRequest):
    result = tts_service.synthesize(text=req.text, word_hint=req.word_hint)
    return TextToSpeechResponse(
        text=req.text,
        audio_url=result["audio_url"],
        backend_used=result["backend_used"],
    )


@router.get("/audio/{filename}", summary="Stream generated audio file")
def get_audio_file(filename: str):
    # Sanitize filename against directory traversal
    clean_name = Path(filename).name
    file_path = settings.STATIC_AUDIO_DIR / clean_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{clean_name}' not found.",
        )

    media_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(path=file_path, media_type=media_type, filename=clean_name)
