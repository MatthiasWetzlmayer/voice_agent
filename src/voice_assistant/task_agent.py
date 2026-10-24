"""Task-Agent: ein DeepAgent mit Dummy-Tools für längere Hintergrundaufgaben."""

from __future__ import annotations

import asyncio
import logging
import time

from deepagents import create_deep_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

log = logging.getLogger("va.task")

RESULTS_NAMESPACE = ("task_results",)

TASK_SYSTEM_PROMPT = """\
Du bist ein Hintergrund-Agent, der Aufgaben für einen Sprachassistenten erledigt.
Arbeite die Aufgabe selbstständig ab. Für Recherchen und längere Arbeiten nutze das
Tool `slow_work`. Für Aktionen mit Außenwirkung (etwas buchen, senden, kaufen, löschen)
nutze `dummy_action`, der Nutzer muss diese Aktion freigeben.
Lehnt der Nutzer eine Aktion ab, führe sie nicht erneut aus, sondern beende die
Aufgabe und fasse kurz zusammen, was erledigt wurde und was nicht.
Deine letzte Nachricht ist das Ergebnis. Es wird dem Nutzer vorgelesen: schreibe
deutsch, knapp (höchstens 3-4 Sätze), ohne Markdown und ohne Emojis.
"""


@tool
async def slow_work(seconds: float, topic: str, runtime: ToolRuntime) -> str:
    """Erledigt eine länger dauernde Arbeit (z.B. eine Recherche) zum Thema `topic`.

    Args:
        seconds: Wie lange die Arbeit ungefähr dauert (1 bis 30 Sekunden).
        topic: Das Thema der Arbeit.
    """
    seconds = max(0.0, min(float(seconds), 30.0))
    log.info("slow_work: arbeite %.1fs an '%s' ...", seconds, topic)
    await asyncio.sleep(seconds)
    result = f"Fake-Ergebnis zu '{topic}': drei wichtige Punkte gefunden, alles im grünen Bereich."
    # Ergebnis im gemeinsamen Store ablegen, damit auch der Voice-Agent es lesen kann.
    if runtime.store is not None:
        await runtime.store.aput(RESULTS_NAMESPACE, f"{time.time():.3f}", {"topic": topic, "result": result})
    return result


@tool
async def dummy_action(description: str) -> str:
    """Führt eine kritische Aktion mit Außenwirkung aus (z.B. buchen, senden, kaufen).

    Braucht eine Freigabe durch den Nutzer.

    Args:
        description: Kurze, verständliche Beschreibung der Aktion.
    """
    log.info("dummy_action AUSGEFÜHRT: %s", description)
    return f"Aktion ausgeführt: {description}"


def _describe_action(tool_call, state, runtime) -> str:
    return tool_call["args"].get("description", str(tool_call["args"]))


def build_task_agent(
    model: str | BaseChatModel,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    return create_deep_agent(
        model=model,
        tools=[slow_work, dummy_action],
        system_prompt=TASK_SYSTEM_PROMPT,
        interrupt_on={
            "dummy_action": {
                "allowed_decisions": ["approve", "reject"],
                "description": _describe_action,
            }
        },
        checkpointer=checkpointer,
        store=store,
        name="task_agent",
    )
