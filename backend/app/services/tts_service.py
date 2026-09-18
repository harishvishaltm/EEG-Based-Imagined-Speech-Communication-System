"""Text-To-Speech Service.

Matches Section 16.7a of the research notebook:
- Tries gTTS first (cloud-based natural neural voice, outputs .mp3)
- Falls back to pyttsx3 (local offline speech engine, outputs .wav)
- Provides safe audio URL for frontend playback: /audio/{filename}
"""
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.app.config import settings

logger = logging.getLogger("tts_service")


class TTSService:
    def __init__(self, output_dir: Optional[Path] = None, lang: str = "en"):
        self.output_dir = output_dir or settings.STATIC_AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lang = lang

    @staticmethod
    def _sanitize_filename(word: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "", word) or "speech"

    def synthesize(self, text: str, word_hint: Optional[str] = None) -> Dict[str, str]:
        """Synthesizes speech audio for `text` and returns audio URL and backend used."""
        if not text or not text.strip():
            return {"audio_url": "", "backend_used": "none"}

        clean_text = text.strip()
        safe_word = self._sanitize_filename(word_hint or "sentence")
        # Unique hash based on text content
        text_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()[:8]
        mp3_filename = f"{safe_word}_{text_hash}.mp3"
        mp3_path = self.output_dir / mp3_filename

        # If audio already cached, return existing URL
        if mp3_path.exists() and mp3_path.stat().st_size > 0:
            return {"audio_url": f"/audio/{mp3_filename}", "backend_used": "cache"}

        # Attempt 1: gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=self.lang)
            tts.save(str(mp3_path))
            logger.info(f"[TTS] Synthesized with gTTS -> {mp3_filename}")
            return {"audio_url": f"/audio/{mp3_filename}", "backend_used": "gtts"}
        except Exception as e:
            logger.warning(f"[TTS] gTTS synthesis failed ({e}). Attempting pyttsx3 fallback...")

        # Attempt 2: pyttsx3 offline fallback
        wav_filename = f"{safe_word}_{text_hash}.wav"
        wav_path = self.output_dir / wav_filename
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(clean_text, str(wav_path))
            engine.runAndWait()
            if wav_path.exists() and wav_path.stat().st_size > 0:
                logger.info(f"[TTS] Synthesized with pyttsx3 -> {wav_filename}")
                return {"audio_url": f"/audio/{wav_filename}", "backend_used": "pyttsx3"}
        except Exception as e2:
            logger.warning(f"[TTS] pyttsx3 offline fallback failed ({e2}).")

        # Fallback: empty audio url
        return {"audio_url": "", "backend_used": "unavailable"}


tts_service = TTSService()
