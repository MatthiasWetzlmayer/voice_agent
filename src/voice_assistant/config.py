"""Konfiguration aus Umgebungsvariablen bzw. `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    voice_model: str = "openai:gpt-5-mini"
    voice_reasoning_effort: str | None = None
    task_model: str = "openai:gpt-5-mini"
    stt_model: str = "gpt-4o-mini-transcribe"
    stt_language: str = "de"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_instructions: str = "Sprich freundlich, natürlich und zügig auf Deutsch."
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()

        def env(name: str, default: str | None) -> str | None:
            value = os.getenv(name)
            return value if value not in (None, "") else default

        return cls(
            voice_model=env("VOICE_MODEL", cls.voice_model),
            voice_reasoning_effort=env("VOICE_REASONING_EFFORT", None),
            task_model=env("TASK_MODEL", cls.task_model),
            stt_model=env("STT_MODEL", cls.stt_model),
            stt_language=env("STT_LANGUAGE", cls.stt_language),
            tts_model=env("TTS_MODEL", cls.tts_model),
            tts_voice=env("TTS_VOICE", cls.tts_voice),
            tts_instructions=env("TTS_INSTRUCTIONS", cls.tts_instructions),
            log_level=env("LOG_LEVEL", cls.log_level).upper(),
        )
