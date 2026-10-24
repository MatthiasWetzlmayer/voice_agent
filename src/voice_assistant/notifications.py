"""Nachrichten, mit denen der TaskManager den Voice-Agenten anstößt."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

NOTIFICATION_MARKER = "[TASK-BENACHRICHTIGUNG]"

# Anbieter, die System-Messages nur am Anfang des Verlaufs erlauben.
_SYSTEM_ONLY_AT_START = {"anthropic", "google", "google_genai", "google_vertexai", "gemini"}


def provider_of(model: str | BaseChatModel) -> str:
    """Ermittelt den Anbieter aus `"anbieter:modell"` oder einer Modellinstanz."""
    if isinstance(model, str):
        return model.split(":", 1)[0].lower() if ":" in model else "openai"
    name = type(model).__name__.lower()
    for key in ("anthropic", "google", "gemini", "openai"):
        if key in name:
            return key
    return "openai"


def make_notification(text: str, provider: str = "openai") -> BaseMessage:
    """Baut die Benachrichtigung passend zum Anbieter des Voice-Modells.

    OpenAI akzeptiert System-Messages an beliebiger Stelle, Anthropic/Gemini nicht.
    Dort wird eine klar markierte HumanMessage verwendet.
    """
    content = f"{NOTIFICATION_MARKER} {text}"
    if provider.lower() in _SYSTEM_ONLY_AT_START:
        return HumanMessage(content=content)
    return SystemMessage(content=content)


def is_notification(message: BaseMessage) -> bool:
    return isinstance(message.content, str) and message.content.startswith(NOTIFICATION_MARKER)
