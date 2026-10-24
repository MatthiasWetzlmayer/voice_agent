"""TTS-Ersatz für `--no-audio`: Text erscheint nur auf der Konsole."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class ConsoleTTS:
    """Gibt nichts aus, sondern verbraucht nur den Satz-Stream.

    Die Sätze selbst loggt die VoiceSession (Label VOICE-AGENT), damit sie in jedem
    Modus genau einmal auf der Konsole stehen. `seconds_per_char` simuliert optional
    die Sprechdauer, um das Timing des Locks im Textmodus nachvollziehen zu können.
    """

    def __init__(self, seconds_per_char: float = 0.0) -> None:
        self.seconds_per_char = seconds_per_char

    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        async for sentence in text_stream:
            if self.seconds_per_char:
                await asyncio.sleep(len(sentence) * self.seconds_per_char)
