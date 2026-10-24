"""Voice-Agent: schneller LangChain-Agent, der das Gespräch führt und Tasks delegiert."""

from __future__ import annotations

import logging
from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from .notifications import NOTIFICATION_MARKER
from .task_manager import TaskLookupError, TaskManager

log = logging.getLogger("va.voice")

VOICE_SYSTEM_PROMPT = f"""\
Du bist ein freundlicher Sprachassistent. Alles, was du schreibst, wird vorgelesen.
Antworte deshalb auf Deutsch, kurz (meist ein bis zwei Sätze), in natürlicher
gesprochener Sprache, ohne Emojis, ohne Markdown, ohne Aufzählungszeichen und ohne IDs.

Delegieren:
- Einfache Fragen, Smalltalk, Wissen und kurze Rechnungen beantwortest du selbst und sofort.
- Nur echte, längere Aufgaben (z.B. recherchieren, etwas ausarbeiten, planen, buchen,
  etwas erledigen lassen) startest du mit start_task. Gib dem Task einen kurzen,
  sprechbaren Namen wie "Recherche Elektroautos" und eine vollständige Beschreibung.
  Sag dem Nutzer danach in einem Satz, dass du dich meldest, sobald es fertig ist.
- Nach dem Status fragst du mit list_tasks, abbrechen mit cancel_task. Sprich Tasks immer
  über ihren Namen an, nie über IDs.

Benachrichtigungen:
- Nachrichten, die mit {NOTIFICATION_MARKER} beginnen, kommen nicht vom Nutzer, sondern
  vom Hintergrundsystem. Der Nutzer hat sie nicht gehört.
- Ist ein Task fertig, sag dem Nutzer von dir aus kurz, dass er fertig ist, und fasse das
  Ergebnis in ein bis drei Sätzen zusammen.
- Ist ein Task fehlgeschlagen, erkläre es verständlich in einem Satz, ohne technische Details.
- Enthält die Benachrichtigung eine Rückfrage, stelle dem Nutzer genau diese Frage.
  Seine Antwort gibst du mit answer_task weiter: decision "approve" bei Zustimmung,
  "reject" bei Ablehnung, optional mit seinem Kommentar als message.
"""


def build_voice_tools(task_manager: TaskManager) -> list:
    @tool
    async def start_task(name: str, description: str) -> str:
        """Startet eine längere Aufgabe im Hintergrund. Kehrt sofort zurück.

        Args:
            name: Kurzer, sprechbarer Name, z.B. "Recherche Elektroautos".
            description: Vollständige Beschreibung der Aufgabe für den Hintergrund-Agenten.
        """
        record = task_manager.start(name, description)
        return f"Task '{record.name}' wurde gestartet und läuft im Hintergrund."

    @tool
    async def list_tasks() -> str:
        """Liefert Status und Fortschritt aller Tasks."""
        records = task_manager.all_tasks()
        if not records:
            return "Es gibt keine Tasks."
        return "\n".join(r.summary() for r in records)

    @tool
    async def cancel_task(name: str) -> str:
        """Bricht einen laufenden Task ab.

        Args:
            name: Name des Tasks, z.B. "die Recherche".
        """
        try:
            record = await task_manager.cancel(name)
        except TaskLookupError as exc:
            return str(exc)
        if record.cancel_requested:
            return f"Task '{record.name}' wurde abgebrochen."
        return f"Task '{record.name}' läuft nicht mehr (Status: {record.status.value})."

    @tool
    async def answer_task(
        name: str, decision: Literal["approve", "reject"], message: str | None = None
    ) -> str:
        """Gibt die Antwort des Nutzers auf eine Rückfrage an einen wartenden Task weiter.

        Args:
            name: Name des Tasks, der die Rückfrage gestellt hat.
            decision: "approve" wenn der Nutzer zustimmt, "reject" wenn er ablehnt.
            message: Optionaler Kommentar oder Grund des Nutzers.
        """
        try:
            record = task_manager.answer(name, decision, message)
        except TaskLookupError as exc:
            return str(exc)
        if decision == "approve":
            return f"Freigabe an '{record.name}' weitergegeben, der Task läuft weiter."
        return f"Ablehnung an '{record.name}' weitergegeben, die Aktion wird nicht ausgeführt."

    return [start_task, list_tasks, cancel_task, answer_task]


def build_voice_agent(
    model: str | BaseChatModel,
    task_manager: TaskManager,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    return create_agent(
        model,
        tools=build_voice_tools(task_manager),
        system_prompt=VOICE_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        store=store,
        name="voice_agent",
    )
