"""Gespräch und Hintergrund-Tasks: hier läuft die Kommunikation zwischen beiden Agenten.

Voice -> Task: Die Tools des Voice-Agenten starten den DeepAgent per asyncio.create_task
               und kehren sofort zurück.
Task -> Voice: Ist ein Task fertig (oder hat eine Rückfrage), ruft er notify_voice() auf. Das
               startet einen neuen Lauf des Voice-Agenten auf dessen Thread.
"""

import asyncio
import logging
import time
import uuid
from typing import Literal

from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .agents import create_task_agent, create_voice_agent
from .audio import sentences

log = logging.getLogger("voice-agent")


def make_notification(text: str, provider: str):
    """OpenAI erlaubt System-Messages mitten im Verlauf, Anthropic/Gemini/Ollama nicht zuverlässig."""
    text = f"[TASK-BENACHRICHTIGUNG] {text}"
    return SystemMessage(text) if provider == "openai" else HumanMessage(text)


class Assistant:
    def __init__(self, voice_model, task_model, tts, provider: str):
        store = InMemoryStore()  # gemeinsames Gedächtnis beider Agenten
        self.voice_agent = create_voice_agent(voice_model, self._tools(), store)
        self.task_agent = create_task_agent(task_model, store)
        self.voice_config = {"configurable": {"thread_id": "voice"}}
        self.tts = tts
        self.provider = provider
        self.lock = asyncio.Lock()  # nur ein Lauf auf dem Voice-Thread, inkl. Sprachausgabe
        self.user_idle = asyncio.Event()  # gelöscht, solange der Nutzer aufnimmt
        self.user_idle.set()
        self.speaking = False  # Mikrofon ist stumm, solange der Assistent spricht
        self.tasks: dict[str, dict] = {}  # Name -> status, runner, thread, ...

    # --- Gespräch -----------------------------------------------------------------------

    async def user_says(self, text: str) -> str:
        logging.getLogger("nutzer").info(text)
        async with self.lock:
            return await self._voice_turn(HumanMessage(text))

    async def notify_voice(self, text: str) -> str:
        logging.getLogger("benachrichtigung").info(text)
        async with self.lock:
            await self.user_idle.wait()  # nicht dazwischenreden, während der Nutzer aufnimmt
            return await self._voice_turn(make_notification(text, self.provider))

    async def _voice_turn(self, message) -> str:
        """Voice-Agent streamen -> Sätze -> TTS. Endet erst, wenn die Ausgabe fertig ist."""
        spoken = []

        async def tokens():
            try:
                async for chunk, meta in self.voice_agent.astream(
                    {"messages": [message]}, self.voice_config, stream_mode="messages"
                ):
                    if meta["langgraph_node"] == "model" and isinstance(chunk, (AIMessage, AIMessageChunk)):
                        yield chunk.text
            except Exception:
                log.exception("Fehler im Voice-Agenten")
                yield " Entschuldigung, da ist etwas schiefgelaufen."

        async def spoken_sentences():
            async for sentence in sentences(tokens()):
                spoken.append(sentence)
                yield sentence

        self.speaking = True
        try:
            await self.tts.speak(spoken_sentences())
        finally:
            self.speaking = False
        return " ".join(spoken)

    # --- Tasks --------------------------------------------------------------------------

    def _find(self, name: str) -> str | None:
        """Findet einen Task über ein gemeinsames Wort im Namen ("die Recherche")."""
        words = {w for w in name.lower().split() if len(w) > 3}
        matches = [n for n in self.tasks if n.lower() == name.lower() or words & set(n.lower().split())]
        return matches[-1] if matches else None

    def _start(self, name: str, agent_input) -> None:
        task = self.tasks[name]
        task["runner"] = asyncio.create_task(self._run_task(name, task, agent_input))

    async def _run_task(self, name: str, task: dict, agent_input) -> None:
        config = {"configurable": {"thread_id": task["thread"]}}  # eigener Thread pro Task
        try:
            await self.task_agent.ainvoke(agent_input, config)
            state = await self.task_agent.aget_state(config)
            if state.interrupts:  # dummy_action wartet auf Freigabe
                actions = state.interrupts[0].value["action_requests"]
                task.update(status="wartet auf Antwort", actions=len(actions))
                question = "; ".join(f"{a['name']}({a['args']})" for a in actions)
                text = f"Rückfrage von Task '{name}': Darf ich {question} ausführen?"
            else:
                task.update(status="fertig", result=state.values["messages"][-1].text)
                text = f"Task '{name}' ist fertig. Ergebnis: {task['result']}"
        except asyncio.CancelledError:
            task["status"] = "abgebrochen"  # bestätigt der Voice-Agent selbst in cancel_task
            raise
        except Exception as e:
            task["status"] = "fehlgeschlagen"
            text = f"Task '{name}' ist fehlgeschlagen: {e}"
        await self.notify_voice(text)

    def _tools(self):
        task_log = logging.getLogger("task-agent")

        @tool
        async def start_task(name: str, description: str) -> str:
            """Startet eine längere Aufgabe im Hintergrund. `name` ist kurz und sprechbar."""
            while name in self.tasks:
                name += " 2"
            task_log.info("starte '%s': %s", name, description)
            self.tasks[name] = {"status": "läuft", "since": time.monotonic(), "thread": uuid.uuid4().hex}
            self._start(name, {"messages": [HumanMessage(description)]})
            return f"Task '{name}' läuft im Hintergrund."

        @tool
        async def list_tasks() -> str:
            """Status aller Tasks."""
            return "\n".join(
                f"{n}: {t['status']} (seit {time.monotonic() - t['since']:.0f} s)" for n, t in self.tasks.items()
            ) or "Keine Tasks."

        @tool
        async def cancel_task(name: str) -> str:
            """Bricht einen laufenden Task ab."""
            found = self._find(name)
            if not found or self.tasks[found]["status"] not in ("läuft", "wartet auf Antwort"):
                return f"Kein laufender Task '{name}'."
            task_log.info("breche '%s' ab", found)
            self.tasks[found]["runner"].cancel()
            self.tasks[found]["status"] = "abgebrochen"
            return f"Task '{found}' wurde abgebrochen."

        @tool
        async def answer_task(name: str, decision: Literal["approve", "reject"], message: str = "") -> str:
            """Gibt die Antwort des Nutzers auf eine Rückfrage an den Task weiter."""
            found = self._find(name)
            if not found or self.tasks[found]["status"] != "wartet auf Antwort":
                return f"Task '{name}' wartet auf keine Antwort."
            task_log.info("Antwort an '%s': %s %s", found, decision, message)
            answer = {"type": decision, **({"message": message} if decision == "reject" and message else {})}
            self.tasks[found]["status"] = "läuft"
            self._start(found, Command(resume={"decisions": [answer] * self.tasks[found]["actions"]}))
            return f"Antwort an '{found}' weitergegeben."

        return [start_task, list_tasks, cancel_task, answer_task]

    async def shutdown(self) -> None:
        for task in self.tasks.values():
            task["runner"].cancel()
