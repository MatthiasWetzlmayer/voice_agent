"""Regelbasierte Fake-Chatmodelle: Tests und Textmodus ohne API-Kosten (`--fake`)."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .notifications import NOTIFICATION_MARKER

Responder = Callable[[Sequence[BaseMessage]], AIMessage]


class FakeChatModel(BaseChatModel):
    """Chatmodell, dessen Antworten eine Python-Funktion bestimmt.

    Streamt Text wortweise, damit die Satz-Pipeline realistisch getestet wird.
    `delay` simuliert die Latenz pro Modellaufruf.
    """

    responder: Responder
    delay: float = 0.0
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-rule-based"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        return self

    def _respond(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.calls += 1
        message = self.responder(messages)
        if message.id is None:
            message.id = f"fake-{uuid.uuid4().hex[:8]}"
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._generate(messages)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        if self.delay:
            await asyncio.sleep(self.delay)
        message = self._respond(messages)
        if message.tool_calls:
            chunk = AIMessageChunk(
                content=message.text,
                id=message.id,
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(message.tool_calls)
                ],
            )
            yield ChatGenerationChunk(message=chunk)
            return
        words = re.findall(r"\S+\s*", message.text) or [""]
        for word in words:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word, id=message.id))
            if run_manager:
                await run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk
            await asyncio.sleep(0)


def _call(tool_name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": f"call-{uuid.uuid4().hex[:8]}"}])


def _strip_marker(text: str) -> str:
    return text.replace(NOTIFICATION_MARKER, "").strip()


def _is_notification(message: BaseMessage) -> bool:
    return isinstance(message, (SystemMessage, HumanMessage)) and str(message.content).startswith(NOTIFICATION_MARKER)


def _pending_question_task(messages: Sequence[BaseMessage]) -> str | None:
    """Name des Tasks aus der letzten Rückfrage-Benachrichtigung."""
    for message in reversed(messages):
        if _is_notification(message):
            match = re.search(r"Rückfrage von Task '([^']+)'", str(message.content))
            return match.group(1) if match else None
        if isinstance(message, AIMessage) and any(c["name"] == "answer_task" for c in message.tool_calls):
            return None
    return None


def _task_name(text: str, default: str) -> str:
    match = re.search(r"(?:zu|über|ueber|nach)\s+([\wäöüÄÖÜß\- ]+)", text)
    topic = match.group(1).strip() if match else ""
    return f"{default} {topic}".strip() if topic else default


def voice_responder(messages: Sequence[BaseMessage]) -> AIMessage:
    """Imitiert den Voice-Agenten mit einfachen Schlüsselwort-Regeln."""
    last = messages[-1]

    if isinstance(last, ToolMessage):
        return AIMessage(content=_voice_reply_to_tool(last))

    text = str(last.content)
    if _is_notification(last):
        body = _strip_marker(text)
        if body.startswith("Rückfrage"):
            question = body.split(":", 1)[1].split("Stelle dem Nutzer")[0].strip()
            return AIMessage(content=f"Kurze Rückfrage. {question}")
        if "fehlgeschlagen" in body:
            name = re.search(r"Task '([^']+)'", body)
            return AIMessage(content=f"Leider hat die Aufgabe {name.group(1) if name else ''} nicht geklappt.")
        result = body.split("Ergebnis:", 1)[-1].strip()
        name = re.search(r"Task '([^']+)'", body)
        return AIMessage(content=f"Die Aufgabe {name.group(1) if name else ''} ist fertig. {result}")

    lower = text.lower()
    pending = _pending_question_task(messages)
    if pending and re.search(r"\b(ja|mach|ok|okay|einverstanden)\b", lower):
        return _call("answer_task", name=pending, decision="approve")
    if pending and re.search(r"\b(nein|nicht|lass)\b", lower):
        return _call("answer_task", name=pending, decision="reject", message=text)
    if re.search(r"\b(brich|abbrechen|stopp)", lower):
        target = re.search(r"(?:die|den|das)\s+([\wäöüÄÖÜß\- ]+?)\s+(?:ab|abbrechen)\b", text)
        return _call("cancel_task", name=target.group(1) if target else text)
    if re.search(r"wie weit|status|was macht", lower):
        return _call("list_tasks")
    if "recherch" in lower:
        return _call("start_task", name=_task_name(text, "Recherche"), description=text)
    if re.search(r"buch|bestell|kauf|schick", lower):
        return _call("start_task", name=_task_name(text, "Buchung"), description=text + " (kritische Aktion)")
    if re.search(r"erledige|aufgabe|fehler", lower):
        return _call("start_task", name=_task_name(text, "Aufgabe"), description=text)
    return AIMessage(content="Mir geht es gut, danke der Nachfrage. Wie kann ich helfen?")


def _voice_reply_to_tool(message: ToolMessage) -> str:
    content = message.text
    if message.name == "start_task":
        return "Alles klar, ich kümmere mich im Hintergrund darum. Ich melde mich, sobald es fertig ist."
    if message.name == "list_tasks":
        return f"Hier der Stand. {content}"
    if message.name == "cancel_task":
        return f"Erledigt. {content}"
    if message.name == "answer_task":
        return f"Verstanden. {content}"
    return content


def task_responder(messages: Sequence[BaseMessage], work_seconds: float = 3.0) -> AIMessage:
    """Imitiert den DeepAgent: erst ein Tool, dann eine Zusammenfassung."""
    task = next(str(m.content) for m in messages if isinstance(m, HumanMessage))
    last = messages[-1]
    lower = task.lower()
    if "fehler" in lower:
        raise RuntimeError("Simulierter Fehler im Task-Modell")
    if isinstance(last, HumanMessage):
        if re.search(r"kritisch|buch|bestell|kauf|schick", lower):
            return _call("dummy_action", description=f"Ausführen: {task}")
        seconds = float(m.group(1)) if (m := re.search(r"(\d+(?:\.\d+)?)\s*s(?:ek)?", lower)) else work_seconds
        return _call("slow_work", seconds=seconds, topic=task)
    if isinstance(last, ToolMessage):
        if last.status == "error" or "rejected" in last.text:
            return AIMessage(content="Die Aktion wurde nicht ausgeführt, weil der Nutzer sie abgelehnt hat.")
        return AIMessage(content=f"Erledigt. {last.text}")
    return AIMessage(content="Erledigt.")


def make_fake_models(work_seconds: float = 3.0, voice_delay: float = 0.0) -> tuple[FakeChatModel, FakeChatModel]:
    voice = FakeChatModel(responder=voice_responder, delay=voice_delay)
    task = FakeChatModel(responder=lambda msgs: task_responder(msgs, work_seconds))
    return voice, task
