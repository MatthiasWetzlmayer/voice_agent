"""Die beiden Agenten: Voice-Agent (create_agent) und Task-Agent (DeepAgent)."""

import asyncio
import logging

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

log = logging.getLogger("task-agent")

VOICE_PROMPT = """\
Du bist ein Sprachassistent. Alles, was du schreibst, wird vorgelesen: antworte auf Deutsch,
kurz (ein bis zwei Sätze), ohne Emojis, ohne Markdown, ohne Aufzählungen.

Einfache Fragen und Smalltalk beantwortest du selbst. Nur echte, längere Aufgaben
(recherchieren, ausarbeiten, buchen, erledigen) startest du mit start_task und gibst dem Task
einen kurzen sprechbaren Namen wie "Recherche Elektroautos". Status: list_tasks.
Abbrechen: cancel_task. Sprich Tasks immer über ihren Namen an.

Nachrichten, die mit [TASK-BENACHRICHTIGUNG] beginnen, kommen vom Hintergrundsystem, nicht vom
Nutzer. Ist ein Task fertig, sag es dem Nutzer von dir aus und fasse das Ergebnis kurz zusammen.
Ist er fehlgeschlagen, erkläre das in einem Satz. Stellt ein Task eine Rückfrage, stelle sie dem
Nutzer und gib seine Antwort mit answer_task weiter ("approve" oder "reject").
"""

TASK_PROMPT = """\
Du erledigst Aufgaben im Hintergrund. Für Recherchen und längere Arbeit nutze slow_work, für
Aktionen mit Außenwirkung (buchen, senden, kaufen) dummy_action. Lehnt der Nutzer eine Aktion
ab, beende die Aufgabe. Deine letzte Nachricht ist das Ergebnis: deutsch, höchstens drei Sätze.
"""


@tool
async def slow_work(seconds: float, topic: str) -> str:
    """Erledigt eine länger dauernde Arbeit (z.B. eine Recherche) zum Thema `topic`."""
    log.info("slow_work: %.0f s zu '%s'", seconds, topic)
    await asyncio.sleep(min(seconds, 30))
    return f"Fake-Ergebnis zu '{topic}': drei wichtige Punkte gefunden."


@tool
async def dummy_action(description: str) -> str:
    """Führt eine kritische Aktion aus (z.B. buchen). Braucht die Freigabe des Nutzers."""
    log.info("dummy_action ausgeführt: %s", description)
    return f"Aktion ausgeführt: {description}"


def make_model(spec: str):
    """Modell aus "anbieter:name", z.B. "ollama:qwen3:4b" oder "openai:gpt-5-mini"."""
    if spec.startswith("ollama:"):
        import ollama
        from langchain_ollama import ChatOllama

        name = spec.removeprefix("ollama:")
        try:
            capabilities = ollama.Client().show(name).capabilities or []
        except ollama.ResponseError as e:
            raise SystemExit(f"Ollama-Modell '{name}' nicht verfügbar ({e.error}). Erst: ollama pull {name}")
        except ConnectionError:
            raise SystemExit("Ollama läuft nicht. Bitte die Ollama-App starten (oder: ollama serve).")
        # Kann das Modell denken, trennt Ollama mit reasoning=True das Denken von der Antwort ab.
        # reasoning=False genügt nicht: reine Thinking-Modelle (z.B. Qwen3-Thinking-2507) denken
        # trotzdem, und der Denktext würde vorgelesen.
        # num_ctx: Ollamas Standard ist zu klein für den DeepAgent-Prompt (ca. 3.500 Tokens plus
        # Verlauf). Für beide Agenten gleich lassen, sonst lädt Ollama das Modell bei jedem
        # Wechsel neu.
        return ChatOllama(model=name, reasoning=True if "thinking" in capabilities else None, num_ctx=16384)
    return init_chat_model(spec)


def create_voice_agent(model, tools, store):
    return create_agent(model, tools, system_prompt=VOICE_PROMPT, checkpointer=InMemorySaver(), store=store)


def create_task_agent(model, store):
    return create_deep_agent(
        model=model,
        tools=[slow_work, dummy_action],
        system_prompt=TASK_PROMPT,
        interrupt_on={"dummy_action": {"allowed_decisions": ["approve", "reject"]}},
        checkpointer=InMemorySaver(),
        store=store,
    )
