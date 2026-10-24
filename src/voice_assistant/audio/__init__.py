"""Austauschbare STT/TTS-Adapter und Hilfen für die Sprach-Pipeline."""

from .base import STT, TTS
from .console import ConsoleTTS
from .sentences import SentenceSplitter, buffered

__all__ = ["STT", "TTS", "ConsoleTTS", "SentenceSplitter", "buffered"]
