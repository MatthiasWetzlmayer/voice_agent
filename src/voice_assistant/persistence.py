"""Fabriken für Checkpointer und Store.

Für den Prototyp In-Memory. Weil alle anderen Module nur diese Funktionen benutzen,
lässt sich später z.B. `AsyncPostgresSaver` / `AsyncPostgresStore` hier einsetzen,
ohne den Rest anzufassen.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


def make_checkpointer() -> BaseCheckpointSaver:
    """Neuer Checkpointer. Voice- und Task-Agent bekommen jeweils einen eigenen."""
    return InMemorySaver()


def make_store() -> BaseStore:
    """Gemeinsamer Langzeitspeicher, den beide Agenten teilen."""
    return InMemoryStore()
