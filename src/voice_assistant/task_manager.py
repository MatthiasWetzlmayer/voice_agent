"""TaskManager: verwaltet Hintergrund-Tasks des DeepAgents.

Voice -> Task: `start`, `status`, `cancel`, `answer` kehren sofort zurück, der DeepAgent
läuft in eigenen asyncio-Tasks.

Task -> Voice: bei Ergebnis, Fehler, Abbruch oder Rückfrage ruft der TaskManager den
`notify`-Callback auf (in der App `VoiceSession.notify_voice`). Das startet direkt einen
neuen Lauf des Voice-Agenten, ohne Queue dazwischen.
"""

from __future__ import annotations

import asyncio
import difflib
import enum
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

log = logging.getLogger("va.task")

NotifyFn = Callable[[str], Awaitable[None]]

MAX_RESULT_CHARS = 1500


class TaskStatus(enum.StrEnum):
    RUNNING = "läuft"
    WAITING = "wartet auf Antwort"
    DONE = "fertig"
    FAILED = "fehlgeschlagen"
    CANCELLED = "abgebrochen"


@dataclass
class TaskRecord:
    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.RUNNING
    created: float = field(default_factory=time.monotonic)
    progress: str = "gestartet"
    steps: int = 0
    result: str | None = None
    error: str | None = None
    question: str | None = None
    pending_interrupts: tuple[Interrupt, ...] = ()
    runner: asyncio.Task | None = None
    cancel_requested: bool = False

    @property
    def config(self) -> dict[str, Any]:
        # Eigener Thread pro Task, strikt getrennt vom Voice-Thread.
        return {"configurable": {"thread_id": self.id}}

    def summary(self) -> str:
        elapsed = int(time.monotonic() - self.created)
        text = f"'{self.name}': {self.status.value} (seit {elapsed} s"
        if self.status is TaskStatus.RUNNING:
            text += f", {self.steps} Schritte, aktuell: {self.progress}"
        text += ")"
        if self.status is TaskStatus.WAITING and self.question:
            text += f". Offene Frage: {self.question}"
        if self.status is TaskStatus.DONE and self.result:
            text += f". Ergebnis: {_shorten(self.result, 300)}"
        if self.status is TaskStatus.FAILED and self.error:
            text += f". Fehler: {self.error}"
        return text


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _last_ai_text(values: dict[str, Any]) -> str:
    for message in reversed(values.get("messages", [])):
        if isinstance(message, AIMessage) and message.text.strip():
            return message.text.strip()
    return "Die Aufgabe ist erledigt, es liegt aber kein Text-Ergebnis vor."


def _question_from(interrupts: tuple[Interrupt, ...]) -> str:
    """Macht aus den HITL-Requests eine vorlesbare Frage."""
    actions: list[str] = []
    for intr in interrupts:
        value = intr.value if isinstance(intr.value, dict) else {}
        for request in value.get("action_requests", []):
            actions.append(request.get("description") or f"{request['name']} {request['args']}")
    if not actions:
        return "Der Task braucht eine Freigabe, um weiterzumachen."
    return "Darf ich folgende Aktion ausführen: " + "; ".join(actions) + "?"


def _resume_value(interrupts: tuple[Interrupt, ...], decision: dict[str, Any]) -> Any:
    """Baut den Wert für `Command(resume=...)` im HITL-Format (`{"decisions": [...]}`).

    Eine Entscheidung gilt für alle wartenden Aktionen. Bei mehreren gleichzeitigen
    Interrupts verlangt LangGraph ein Mapping Interrupt-ID -> Antwort.
    """

    def for_interrupt(intr: Interrupt) -> dict[str, Any]:
        value = intr.value if isinstance(intr.value, dict) else {}
        count = max(1, len(value.get("action_requests", [])))
        return {"decisions": [dict(decision) for _ in range(count)]}

    if len(interrupts) == 1:
        return for_interrupt(interrupts[0])
    return {intr.id: for_interrupt(intr) for intr in interrupts}


class TaskLookupError(LookupError):
    pass


class TaskManager:
    def __init__(self, agent: CompiledStateGraph, notify: NotifyFn | None = None) -> None:
        self.agent = agent
        self.notify = notify
        self.tasks: dict[str, TaskRecord] = {}
        self._notifications: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ Voice -> Task

    def start(self, name: str, description: str) -> TaskRecord:
        """Startet einen Task im Hintergrund und kehrt sofort zurück."""
        record = TaskRecord(id=f"task-{uuid.uuid4().hex[:8]}", name=self._unique_name(name), description=description)
        self.tasks[record.id] = record
        log.info("Voice-Agent -> Task-Agent: starte '%s' (%s): %s", record.name, record.id, description)
        initial = {"messages": [HumanMessage(content=description)]}
        record.runner = asyncio.create_task(self._drive(record, initial), name=record.id)
        return record

    def all_tasks(self) -> list[TaskRecord]:
        return list(self.tasks.values())

    def find(self, name: str) -> TaskRecord:
        """Findet einen Task über seinen (ungefähren) Namen."""
        if not self.tasks:
            raise TaskLookupError("Es gibt keine Tasks.")
        query = name.strip().lower()
        for prefix in ("die ", "der ", "das ", "den "):
            if query.startswith(prefix):
                query = query[len(prefix) :]
        records = list(self.tasks.values())
        exact = [r for r in records if r.name.lower() == query or r.id == name]
        if exact:
            return exact[-1]
        partial = [r for r in records if query in r.name.lower() or r.name.lower() in query]
        if not partial:
            names = {r.name.lower(): r for r in records}
            close = difflib.get_close_matches(query, list(names), n=1, cutoff=0.5)
            partial = [names[close[0]]] if close else []
        if not partial:
            raise TaskLookupError(f"Kein Task namens '{name}'. Vorhanden: {', '.join(r.name for r in records)}.")
        # Bei mehreren Treffern bevorzugt den aktiven, sonst den jüngsten.
        active = [r for r in partial if r.status in (TaskStatus.RUNNING, TaskStatus.WAITING)]
        if len(active) > 1:
            raise TaskLookupError(f"'{name}' ist nicht eindeutig: {', '.join(r.name for r in active)}.")
        return (active or partial)[-1]

    async def cancel(self, name: str) -> TaskRecord:
        """Bricht einen Task ab. Der Voice-Agent bestätigt den Abbruch selbst."""
        record = self.find(name)
        if record.status not in (TaskStatus.RUNNING, TaskStatus.WAITING):
            return record
        record.cancel_requested = True
        log.info("Voice-Agent -> Task-Agent: breche '%s' ab", record.name)
        if record.runner is not None and not record.runner.done():
            record.runner.cancel()
            await asyncio.wait({record.runner}, timeout=5)
        record.status = TaskStatus.CANCELLED
        record.pending_interrupts = ()
        return record

    def answer(
        self, name: str, decision: Literal["approve", "reject"], message: str | None = None
    ) -> TaskRecord:
        """Gibt die Antwort des Nutzers an einen wartenden Task und setzt ihn fort."""
        record = self.find(name)
        if record.status is not TaskStatus.WAITING:
            raise TaskLookupError(f"Der Task '{record.name}' wartet auf keine Antwort (Status: {record.status.value}).")
        payload: dict[str, Any] = {"type": decision}
        if decision == "reject" and message:
            payload["message"] = message
        resume = _resume_value(record.pending_interrupts, payload)
        log.info("Voice-Agent -> Task-Agent: Antwort an '%s': %s %s", record.name, decision, message or "")
        record.status = TaskStatus.RUNNING
        record.question = None
        record.pending_interrupts = ()
        record.progress = "setzt nach Freigabe fort" if decision == "approve" else "verarbeitet Ablehnung"
        record.runner = asyncio.create_task(self._drive(record, Command(resume=resume)), name=record.id)
        return record

    # ------------------------------------------------------------------ Task -> Voice

    async def _drive(self, record: TaskRecord, agent_input: Any) -> None:
        """Lässt den DeepAgent bis zum Ende oder zum nächsten Interrupt laufen."""
        try:
            async for part in self.agent.astream(agent_input, record.config, stream_mode="updates", version="v2"):
                self._track_progress(record, part["data"])
            state = await self.agent.aget_state(record.config)
            if state.interrupts:
                record.status = TaskStatus.WAITING
                record.pending_interrupts = tuple(state.interrupts)
                record.question = _question_from(record.pending_interrupts)
                log.info("Task-Agent '%s' pausiert, Rückfrage: %s", record.name, record.question)
                self._emit(
                    f"Rückfrage von Task '{record.name}': {record.question} "
                    f"Stelle dem Nutzer diese Frage und gib seine Antwort mit answer_task weiter."
                )
                return
            record.result = _last_ai_text(state.values)
            record.status = TaskStatus.DONE
            log.info("Task-Agent '%s' fertig: %s", record.name, _shorten(record.result, 200))
            self._emit(f"Task '{record.name}' ist fertig. Ergebnis: {_shorten(record.result, MAX_RESULT_CHARS)}")
        except asyncio.CancelledError:
            record.status = TaskStatus.CANCELLED
            log.info("Task-Agent '%s' abgebrochen", record.name)
            if not record.cancel_requested:
                # Vom Nutzer veranlasste Abbrüche bestätigt der Voice-Agent bereits im
                # cancel_task-Aufruf, nur unerwartete Abbrüche werden gemeldet.
                self._emit(f"Task '{record.name}' wurde unerwartet abgebrochen.")
            raise
        except Exception as exc:  # noqa: BLE001 - Fehler dürfen das Gespräch nie beenden
            record.status = TaskStatus.FAILED
            record.error = _shorten(f"{type(exc).__name__}: {exc}", 300)
            log.error("Task-Agent '%s' fehlgeschlagen: %s", record.name, record.error)
            log.debug("Traceback", exc_info=True)
            self._emit(
                f"Task '{record.name}' ist leider fehlgeschlagen. Technischer Grund: {record.error}"
            )

    def _track_progress(self, record: TaskRecord, update: dict[str, Any]) -> None:
        for node, data in update.items():
            if node.startswith("__") or not isinstance(data, dict):
                continue
            for message in data.get("messages", []) or []:
                if isinstance(message, AIMessage) and message.tool_calls:
                    record.steps += 1
                    calls = ", ".join(c["name"] for c in message.tool_calls)
                    record.progress = f"führt {calls} aus"
                    log.info("Task-Agent '%s' ruft Tool: %s", record.name, calls)
                elif isinstance(message, AIMessage):
                    record.steps += 1
                    record.progress = "schreibt Ergebnis"

    def _emit(self, text: str) -> None:
        """Startet die Benachrichtigung entkoppelt vom Task-Lauf.

        Als eigener asyncio-Task, damit ein Task-Abbruch nie eine bereits laufende
        Ansage abbricht und der DeepAgent nicht auf den Voice-Lock wartet.
        """
        if self.notify is None:
            log.warning("Keine Benachrichtigungsfunktion gesetzt: %s", text)
            return
        task = asyncio.create_task(self._safe_notify(text))
        self._notifications.add(task)
        task.add_done_callback(self._notifications.discard)

    async def _safe_notify(self, text: str) -> None:
        try:
            await self.notify(text)
        except Exception:  # noqa: BLE001
            log.exception("Benachrichtigung an den Voice-Agenten fehlgeschlagen")

    # ------------------------------------------------------------------ Hilfen

    def _unique_name(self, name: str) -> str:
        base = " ".join(name.split()) or "Aufgabe"
        existing = {r.name.lower() for r in self.tasks.values()}
        candidate, n = base, 2
        while candidate.lower() in existing:
            candidate, n = f"{base} {n}", n + 1
        return candidate

    async def wait_idle(self, timeout: float | None = None) -> None:
        """Wartet, bis keine Task-Läufe und Benachrichtigungen mehr aktiv sind (Tests)."""

        async def loop() -> None:
            while True:
                pending = [r.runner for r in self.tasks.values() if r.runner and not r.runner.done()]
                pending += list(self._notifications)
                if not pending:
                    return
                await asyncio.wait(pending)

        await asyncio.wait_for(loop(), timeout)

    async def shutdown(self) -> None:
        for record in self.tasks.values():
            if record.runner and not record.runner.done():
                record.cancel_requested = True
                record.runner.cancel()
        for task in list(self._notifications):
            task.cancel()
        pending = [r.runner for r in self.tasks.values() if r.runner] + list(self._notifications)
        await asyncio.gather(*pending, return_exceptions=True)
