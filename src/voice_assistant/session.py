"""VoiceSession: Sandwich-Pipeline Agent -> TTS und Synchronisation des Voice-Threads."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from .audio import TTS, SentenceSplitter, buffered
from .notifications import make_notification

log = logging.getLogger("va.voice")
user_log = logging.getLogger("va.user")
notify_log = logging.getLogger("va.notify")

FALLBACK_ERROR = "Entschuldigung, da ist gerade etwas schiefgelaufen. Versuch es bitte noch einmal."


class VoiceSession:
    """Ein Gespräch: ein Voice-Thread, geschützt durch einen Lock.

    - Nutzerbeiträge und Benachrichtigungen laufen nie gleichzeitig auf dem Thread.
    - Der Lock wird erst nach dem Ende der TTS-Ausgabe freigegeben.
    - Während der Nutzer aufnimmt, warten Benachrichtigungen.
    """

    def __init__(self, agent: CompiledStateGraph, tts: TTS, provider: str = "openai", thread_id: str | None = None) -> None:
        self.agent = agent
        self.tts = tts
        self.provider = provider
        self.thread_id = thread_id or f"voice-{uuid.uuid4().hex[:8]}"
        self.config: dict[str, Any] = {"configurable": {"thread_id": self.thread_id}}
        self._lock = asyncio.Lock()
        self._not_recording = asyncio.Event()
        self._not_recording.set()

    # ------------------------------------------------------------------ Aufnahme-Status

    def recording_started(self) -> None:
        self._not_recording.clear()

    def recording_stopped(self) -> None:
        self._not_recording.set()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ------------------------------------------------------------------ Eingänge

    async def handle_user_text(self, text: str) -> str:
        """Nutzerbeitrag (getippt oder transkribiert)."""
        user_log.info("%s", text)
        async with self._lock:
            return await self._run_turn(HumanMessage(content=text))

    async def notify_voice(self, text: str) -> str:
        """Task -> Voice: startet direkt einen neuen Lauf auf dem bestehenden Voice-Thread."""
        notify_log.info("Task-Agent -> Voice-Agent: %s", text)
        await self._acquire_when_user_idle()
        try:
            return await self._run_turn(make_notification(text, self.provider))
        finally:
            self._lock.release()

    async def _acquire_when_user_idle(self) -> None:
        while True:
            await self._not_recording.wait()
            await self._lock.acquire()
            if self._not_recording.is_set():
                return
            # Nutzer hat inzwischen mit der Aufnahme begonnen: wieder freigeben und warten.
            self._lock.release()

    # ------------------------------------------------------------------ Pipeline

    async def _run_turn(self, message) -> str:
        """Agent -> Sätze -> TTS. Kehrt erst nach dem Ende der Sprachausgabe zurück."""
        spoken: list[str] = []

        async def sentences() -> AsyncIterator[str]:
            try:
                async for sentence in self._agent_sentences(message):
                    spoken.append(sentence)
                    log.info("» %s", sentence)
                    yield sentence
            except Exception:  # noqa: BLE001 - ein Fehler darf das Gespräch nicht beenden
                log.exception("Fehler im Voice-Agenten")
                spoken.append(FALLBACK_ERROR)
                log.info("» %s", FALLBACK_ERROR)
                yield FALLBACK_ERROR

        try:
            await self.tts.speak(buffered(sentences()))
        except Exception:  # noqa: BLE001
            logging.getLogger("va.audio").exception("TTS-Fehler, Text: %s", " ".join(spoken))
        return " ".join(spoken)

    async def _agent_sentences(self, message) -> AsyncIterator[str]:
        """Streamt die Tokens des Voice-Agenten und schneidet sie in Sätze."""
        splitter = SentenceSplitter()
        current_id: str | None = None
        async for part in self.agent.astream(
            {"messages": [message]}, self.config, stream_mode=["messages", "updates"], version="v2"
        ):
            if part["type"] == "updates":
                self._log_tools(part["data"])
                continue
            chunk, meta = part["data"]
            if meta.get("langgraph_node") != "model" or not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue
            if chunk.id != current_id:  # neue Modellantwort: Rest der vorherigen ausgeben
                current_id = chunk.id
                for sentence in splitter.flush():
                    yield sentence
            for sentence in splitter.feed(chunk.text):
                yield sentence
        for sentence in splitter.flush():
            yield sentence

    @staticmethod
    def _log_tools(update: dict[str, Any]) -> None:
        for node, data in update.items():
            if not isinstance(data, dict):
                continue
            for msg in data.get("messages", []) or []:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        log.info("ruft Tool %s(%s)", call["name"], call["args"])
                elif isinstance(msg, ToolMessage):
                    log.info("Tool-Ergebnis %s: %s", msg.name, msg.text)
