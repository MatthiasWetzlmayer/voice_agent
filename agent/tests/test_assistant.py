"""Akzeptanztests mit Fake-Modellen (keine API-Kosten)."""

import asyncio
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from voice_assistant.assistant import Assistant


class FakeModel(BaseChatModel):
    respond: object
    delay: float = 0.0

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(self.delay)
        return self._generate(messages)


def call(tool, **args):
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": f"{tool}-{time.monotonic_ns()}"}])


def voice(messages):
    last = messages[-1]
    text = str(last.content).lower()
    if isinstance(last, ToolMessage):
        return AIMessage(f"Okay. {last.content}")
    if text.startswith("[task-benachrichtigung]"):
        return AIMessage(f"Info: {last.content.split('] ', 1)[1]}")
    if "wie weit" in text:
        return call("list_tasks")
    if "brich" in text:
        return call("cancel_task", name="die Recherche")
    if "recherch" in text:
        return call("start_task", name="Recherche " + text.split()[-1], description=text)
    if "buch" in text:
        return call("start_task", name="Buchung", description=text)
    if text.startswith("ja"):
        return call("answer_task", name="Buchung", decision="approve")
    if text.startswith("nein"):
        return call("answer_task", name="Buchung", decision="reject", message="lieber nicht")
    return AIMessage("Mir geht es gut. Und dir?")


def task(messages):
    job = next(m.content for m in messages if isinstance(m, HumanMessage))
    if "fehler" in job:
        raise RuntimeError("kaputt")
    if isinstance(messages[-1], ToolMessage):
        return AIMessage(f"Ergebnis: {messages[-1].content}")
    if "buch" in job:
        return call("dummy_action", description="Tisch buchen")
    return call("slow_work", seconds=5 if "lang" in job else 0.3, topic=job)


class RecordingTTS:
    def __init__(self, delay=0.0):
        self.delay, self.turns, self.times = delay, [], []

    async def speak(self, text_stream):
        start, turn = time.monotonic(), []
        async for sentence in text_stream:
            await asyncio.sleep(self.delay)
            turn.append(sentence)
        self.turns.append(" ".join(turn))
        self.times.append((start, time.monotonic()))


def make(voice_delay=0.0, tts_delay=0.0, provider="openai"):
    tts = RecordingTTS(tts_delay)
    return Assistant(FakeModel(respond=voice, delay=voice_delay), FakeModel(respond=task), tts, provider), tts


async def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "Timeout"
        await asyncio.sleep(0.01)


async def test_1_smalltalk_without_task():
    a, tts = make()
    assert await a.user_says("Hallo, wie geht's?") == "Mir geht es gut. Und dir?"
    assert a.tasks == {}


async def test_2_3_task_runs_in_background_and_reports_back():
    a, tts = make()
    start = time.monotonic()
    reply = await a.user_says("Recherchier mir etwas zu Solar")
    assert time.monotonic() - start < 0.2 and "läuft im Hintergrund" in reply
    assert a.tasks["Recherche solar"]["status"] == "läuft"
    assert "gut" in await a.user_says("Wie geht's?")  # weiter ansprechbar
    await until(lambda: len(tts.turns) == 3)
    assert tts.turns[-1].startswith("Info: Task 'Recherche solar' ist fertig. Ergebnis: Ergebnis: Fake-Ergebnis")
    voice_thread = (await a.voice_agent.aget_state(a.voice_config)).values["messages"]
    assert any(isinstance(m, SystemMessage) and "TASK-BENACHRICHTIGUNG" in m.content for m in voice_thread)


async def test_4_two_tasks_in_parallel_and_status():
    a, tts = make()
    await a.user_says("Recherchier etwas zu Solar")
    await a.user_says("Recherchier etwas zu Wind")
    status = await a.user_says("Wie weit ist die Recherche?")
    assert "Recherche solar: läuft" in status and "Recherche wind: läuft" in status
    start = time.monotonic()
    await until(lambda: all(t["status"] == "fertig" for t in a.tasks.values()))
    assert time.monotonic() - start < 0.6  # parallel, nicht nacheinander


async def test_5_question_approved():
    a, tts = make()
    await a.user_says("Buch mir einen Tisch")
    await until(lambda: a.tasks["Buchung"]["status"] == "wartet auf Antwort" and "Rückfrage" in tts.turns[-1])
    await a.user_says("Ja, mach")
    await until(lambda: a.tasks["Buchung"]["status"] == "fertig")
    assert "Aktion ausgeführt" in a.tasks["Buchung"]["result"]


async def test_5_question_rejected():
    a, tts = make()
    await a.user_says("Buch mir einen Tisch")
    await until(lambda: a.tasks["Buchung"]["status"] == "wartet auf Antwort")
    await a.user_says("Nein")
    await until(lambda: a.tasks["Buchung"]["status"] == "fertig")
    assert "rejected" in a.tasks["Buchung"]["result"] and "Aktion ausgeführt" not in a.tasks["Buchung"]["result"]


async def test_6_cancel():
    a, tts = make()
    await a.user_says("Recherchier etwas lang")
    reply = await a.user_says("Brich die Recherche ab")
    assert "abgebrochen" in reply
    await asyncio.sleep(0.05)
    assert a.tasks["Recherche lang"]["status"] == "abgebrochen" and a.tasks["Recherche lang"]["runner"].cancelled()


async def test_7_notification_waits_for_running_turn():
    a, tts = make(voice_delay=0.2, tts_delay=0.05)
    user = asyncio.create_task(a.user_says("Hallo"))
    await until(a.lock.locked)
    note = asyncio.create_task(a.notify_voice("Task 'X' ist fertig."))
    await asyncio.gather(user, note)
    assert tts.turns == ["Mir geht es gut. Und dir?", "Info: Task 'X' ist fertig."]
    assert tts.times[1][0] >= tts.times[0][1]  # keine Überlagerung


async def test_7_notification_waits_while_user_speaks():
    a, tts = make()
    a.user_idle.clear()
    note = asyncio.create_task(a.notify_voice("Task 'X' ist fertig."))
    await asyncio.sleep(0.1)
    assert tts.turns == []
    a.user_idle.set()
    await note
    assert tts.turns == ["Info: Task 'X' ist fertig."]


async def test_task_error_becomes_notification():
    a, tts = make()
    await a.user_says("Recherchier etwas zu fehler")
    await until(lambda: len(tts.turns) == 2)
    assert a.tasks["Recherche fehler"]["status"] == "fehlgeschlagen" and "fehlgeschlagen" in tts.turns[-1]


async def test_ollama_gets_notification_as_human_message():
    a, tts = make(provider="ollama")
    await a.notify_voice("Task 'X' ist fertig.")
    messages = (await a.voice_agent.aget_state(a.voice_config)).values["messages"]
    assert isinstance(messages[0], HumanMessage) and messages[0].content.startswith("[TASK-BENACHRICHTIGUNG]")


async def test_snapshot_for_dashboard():
    a, tts = make()
    await a.user_says("Buch mir einen Tisch")
    await until(lambda: a.tasks["Buchung"]["status"] == "wartet auf Antwort")
    (task,) = a.snapshot()["tasks"]
    assert task["name"] == "Buchung" and task["status"] == "wartet auf Antwort"
    assert task["steps"][0]["tool"] == "dummy_action" and "Tisch buchen" in task["question"]
    await a.user_says("Ja, mach")
    await until(lambda: a.tasks["Buchung"]["status"] == "fertig")
    (task,) = a.snapshot()["tasks"]
    assert task["question"] is None and "Aktion ausgeführt" in task["result"] and task["finished"] >= task["started"]


async def test_dashboard_api_serves_state():
    import json
    import urllib.request

    from voice_assistant.dashboard import start_dashboard_api

    a, tts = make()
    server = start_dashboard_api(a, {"voice_model": "fake"}, port=0)  # Port 0: freien Port wählen
    try:
        await a.user_says("Recherchier etwas lang")
        url = f"http://127.0.0.1:{server.server_address[1]}/api/state"
        response = await asyncio.to_thread(urllib.request.urlopen, url)
        state = json.loads(response.read())
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert state["agent"]["voice_model"] == "fake"
        assert [t["status"] for t in state["tasks"]] == ["läuft"]
    finally:
        server.shutdown()
        await a.shutdown()


async def test_plan_of_deep_agent_appears_in_snapshot():
    def planning_task(messages):
        if not any(isinstance(m, ToolMessage) for m in messages):
            return call("write_todos", todos=[{"content": "Quellen sammeln", "status": "in_progress"}])
        return AIMessage("Fertig.")

    tts = RecordingTTS()
    a = Assistant(FakeModel(respond=voice), FakeModel(respond=planning_task), tts, "openai")
    await a.user_says("Recherchier etwas zu Solar")
    await until(lambda: a.tasks["Recherche solar"]["status"] == "fertig")
    (task,) = a.snapshot()["tasks"]
    assert task["todos"] == [{"content": "Quellen sammeln", "status": "in_progress"}]
