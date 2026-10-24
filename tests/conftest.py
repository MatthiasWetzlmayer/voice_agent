from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import pytest

from voice_assistant.app import App, build_app
from voice_assistant.fakes import make_fake_models


class RecordingTTS:
    """TTS-Attrappe: merkt sich jede Ausgabe (ein Eintrag pro Voice-Lauf) mit Zeitstempeln."""

    def __init__(self, seconds_per_sentence: float = 0.0) -> None:
        self.seconds_per_sentence = seconds_per_sentence
        self.turns: list[list[str]] = []
        self.intervals: list[tuple[float, float]] = []

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        start = time.monotonic()
        turn: list[str] = []
        self.turns.append(turn)
        async for sentence in text_stream:
            if self.seconds_per_sentence:
                await asyncio.sleep(self.seconds_per_sentence)
            turn.append(sentence)
        self.intervals.append((start, time.monotonic()))

    @property
    def texts(self) -> list[str]:
        return [" ".join(t) for t in self.turns]


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Bedingung nicht rechtzeitig erfüllt")
        await asyncio.sleep(0.01)


@pytest.fixture
async def make_app():
    apps: list[App] = []

    def factory(work_seconds: float = 0.2, voice_delay: float = 0.0, tts_delay: float = 0.0, provider: str = "openai"):
        voice, task = make_fake_models(work_seconds=work_seconds, voice_delay=voice_delay)
        tts = RecordingTTS(tts_delay)
        app = build_app(voice, task, tts, provider=provider)
        apps.append(app)
        return app, tts

    yield factory
    for app in apps:
        await app.task_manager.shutdown()
