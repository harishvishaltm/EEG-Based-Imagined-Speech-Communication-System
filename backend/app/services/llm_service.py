"""Assistive LLM Service for imagined-speech communication.

Separation of Concerns:
- EEG model: EEG signal -> predicted word
- LLM service: predicted word -> natural first-person assistive sentence

Supports OpenAI API with automatic fallback to deterministic rules if offline or unconfigured.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional

from backend.app.config import settings

logger = logging.getLogger("llm_service")


class TemplateFallbackGenerator:
    """Deterministic templates for assistive communication matching Section 16.2 of the notebook."""

    NEED_WORDS = {"water", "help", "food", "rest", "bathroom", "pain", "medicine"}
    STATE_WORDS = {"hungry", "thirsty", "tired", "cold", "hot", "sick"}
    DIRECTION_WORDS = {"in", "out", "up", "down", "left", "right"}
    AFFIRM_WORDS = {"yes", "no"}

    # Kara One vocabulary mappings
    KARA_ONE_WORDS = {
        "pot": "I want a pot.",
        "pat": "I want a pat.",
        "gnaw": "I feel like chewing.",
        "knew": "I knew that.",
        "/diy/": "I'm interested in DIY.",
        "/iy/": "I need to communicate.",
        "/m/": "I want more.",
        "/n/": "No.",
        "/piy/": "I want peace.",
        "/tiy/": "I would like some tea.",
        "/uw/": "I want to move up.",
        "diy": "I'm interested in DIY.",
        "iy": "I need to communicate.",
        "m": "I want more.",
        "n": "No.",
        "piy": "I want peace.",
        "tiy": "I would like some tea.",
        "uw": "I want to move up.",
    }

    @classmethod
    def generate(cls, word_or_prompt: str) -> str:
        # Strip phoneme slashes, quotes, spaces
        cleaned = word_or_prompt.strip().strip('"').strip("'")
        match = re.search(r'"([^"]+)"', cleaned)
        if match:
            raw_word = match.group(1)
        else:
            raw_word = cleaned.split()[-1] if cleaned.split() else ""

        normalized = raw_word.strip().lower()
        unslashed = normalized.strip("/")

        # Check explicit Kara One mappings
        if normalized in cls.KARA_ONE_WORDS:
            return cls.KARA_ONE_WORDS[normalized]
        if unslashed in cls.KARA_ONE_WORDS:
            return cls.KARA_ONE_WORDS[unslashed]

        # Check category mappings
        if unslashed in cls.NEED_WORDS:
            return f"I need {unslashed}."
        elif unslashed in cls.STATE_WORDS:
            return f"I am {unslashed}."
        elif unslashed in cls.DIRECTION_WORDS:
            return f"I want to go {unslashed}."
        elif unslashed in cls.AFFIRM_WORDS:
            return f"{unslashed.capitalize()}."
        elif unslashed:
            return f"I mean: {unslashed}."
        else:
            return "I am trying to speak."


class LLMService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.history: List[str] = []
        self.memory_window = 5

        self.system_prompt = (
            "You are an assistive communication device for a person who cannot speak. "
            "They think of ONE word, and you must speak FOR THEM, in first person, "
            "expressing what they most likely want or mean by that word. "
            "Do not describe or define the word. Do not state facts about the word. "
            "Output ONLY the sentence they would say, nothing else — no explanation, "
            "no quotation marks, no preamble.\n\n"
            "Examples:\n"
            "Word: water -> I need water.\n"
            "Word: help -> I need help.\n"
            "Word: hungry -> I am hungry.\n"
            "Word: up -> I want to go up.\n"
            "Word: yes -> Yes.\n"
        )

        self._client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self._client = None
        else:
            logger.info("No OPENAI_API_KEY provided; using deterministic template fallback.")

    def generate_sentence(self, predicted_word: str, use_memory: bool = False) -> Dict[str, str]:
        """Converts predicted word to a natural first-person assistive sentence."""
        clean_word = predicted_word.strip()

        # Try OpenAI API if client is available
        if self._client is not None:
            try:
                if use_memory and self.history:
                    recent = self.history[-self.memory_window :]
                    user_prompt = (
                        f"The person thought these words in order: {recent + [clean_word]}. "
                        "Speak for them as ONE short first-person sentence expressing what they want. "
                        "Output only the sentence."
                    )
                else:
                    user_prompt = f'Word: "{clean_word}" -> '

                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                sentence = response.choices[0].message.content.strip().strip('"')

                if use_memory:
                    self.history.append(clean_word)
                    self.history = self.history[-self.memory_window :]

                return {"sentence": sentence, "generation_mode": "llm"}
            except Exception as e:
                logger.warning(f"OpenAI API call failed ({e}); falling back to deterministic template.")

        # Deterministic rule-based fallback
        fallback_sentence = TemplateFallbackGenerator.generate(clean_word)
        if use_memory:
            self.history.append(clean_word)
            self.history = self.history[-self.memory_window :]

        return {"sentence": fallback_sentence, "generation_mode": "fallback"}

    def reset_memory(self):
        self.history = []


llm_service = LLMService()
