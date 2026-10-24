"""Kleine Interfaces für Spracherkennung (STT) und Sprachausgabe (TTS)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class STT(Protocol):
    async def transcribe(self, audio: bytes) -> str:
        """WAV-Audio -> Text."""
        ...


@runtime_checkable
class TTS(Protocol):
    async def speak(self, text_stream: AsyncIterator[str]) -> None:
        """Spricht die Sätze aus `text_stream`, sobald sie eintreffen.

        Kehrt erst zurück, wenn die Ausgabe vollständig beendet ist.
        """
        ...
