"""Akzeptanztests 1-7 mit Fake-Modellen (keine API-Kosten)."""

from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from voice_assistant.notifications import NOTIFICATION_MARKER
from voice_assistant.task_agent import RESULTS_NAMESPACE
from voice_assistant.task_manager import TaskStatus

from .conftest import wait_until


async def voice_messages(app):
    state = await app.session.agent.aget_state(app.session.config)
    return state.values["messages"]


async def task_messages(app, record):
    state = await app.task_manager.agent.aget_state(record.config)
    return state.values["messages"]


# 1 ---------------------------------------------------------------------------------


async def test_smalltalk_is_answered_directly(make_app):
    app, tts = make_app()
    reply = await app.session.handle_user_text("Hallo, wie geht es dir?")
    assert "gut" in reply
    assert app.task_manager.all_tasks() == []
    assert tts.turns == [["Mir geht es gut, danke der Nachfrage.", "Wie kann ich helfen?"]]


# 2 + 3 -----------------------------------------------------------------------------


async def test_research_starts_task_returns_immediately_and_reports_back(make_app):
    app, tts = make_app(work_seconds=0.5)
    started = time.monotonic()
    reply = await app.session.handle_user_text("Recherchier mir bitte etwas zu Wärmepumpen")
    assert time.monotonic() - started < 0.4, "Voice-Agent darf nicht auf den Task warten"
    assert "melde mich" in reply
    (record,) = app.task_manager.all_tasks()
    assert record.name == "Recherche Wärmepumpen"
    assert record.status is TaskStatus.RUNNING

    # weiter ansprechbar, während der Task läuft
    assert "gut" in await app.session.handle_user_text("Wie geht's?")
    assert record.status is TaskStatus.RUNNING

    # 3: ungefragte Rückmeldung mit dem Ergebnis
    await app.task_manager.wait_idle(timeout=5)
    assert record.status is TaskStatus.DONE
    assert "Die Aufgabe Recherche Wärmepumpen ist fertig." in tts.texts[-1]
    assert "drei wichtige Punkte" in tts.texts[-1]

    # Benachrichtigung landet als SystemMessage auf dem Voice-Thread ...
    notes = [m for m in await voice_messages(app) if str(m.content).startswith(NOTIFICATION_MARKER)]
    assert len(notes) == 1 and isinstance(notes[0], SystemMessage)
    # ... und die Threads bleiben getrennt.
    assert all(NOTIFICATION_MARKER not in str(m.content) for m in await task_messages(app, record))
    assert app.session.thread_id != record.id
    # Gemeinsamer Store: das Ergebnis des Task-Agenten ist für alle sichtbar.
    assert len(await app.store.asearch(RESULTS_NAMESPACE)) == 1


# 4 ---------------------------------------------------------------------------------


async def test_two_parallel_tasks_and_status_query(make_app):
    app, tts = make_app(work_seconds=0.6)
    await app.session.handle_user_text("Recherchier etwas zu Wärmepumpen")
    await app.session.handle_user_text("Erledige die Aufgabe zu Steuern")
    records = app.task_manager.all_tasks()
    assert [r.name for r in records] == ["Recherche Wärmepumpen", "Aufgabe Steuern"]
    await wait_until(lambda: all(r.progress.startswith("führt slow_work") for r in records))

    reply = await app.session.handle_user_text("Wie weit ist die Recherche?")
    assert "'Recherche Wärmepumpen': läuft" in reply
    assert "slow_work" in reply
    assert all(r.status is TaskStatus.RUNNING for r in records)

    started = time.monotonic()
    await app.task_manager.wait_idle(timeout=5)
    assert time.monotonic() - started < 1.0, "Tasks müssen parallel laufen"
    assert all(r.status is TaskStatus.DONE for r in records)
    finished = [t for t in tts.texts if "ist fertig" in t]
    assert len(finished) == 2


# 5 ---------------------------------------------------------------------------------


async def _start_booking(app, tts):
    await app.session.handle_user_text("Buch mir einen Tisch beim Italiener")
    (record,) = app.task_manager.all_tasks()
    await wait_until(lambda: record.status is TaskStatus.WAITING and any("Rückfrage" in t for t in tts.texts))
    assert "Tisch beim Italiener" in record.question
    return record


def _dummy_action_results(messages):
    return [m for m in messages if isinstance(m, ToolMessage) and m.name == "dummy_action"]


async def test_question_approved_continues_and_reports(make_app):
    app, tts = make_app()
    record = await _start_booking(app, tts)

    reply = await app.session.handle_user_text("Ja, mach")
    assert "Freigabe" in reply
    await app.task_manager.wait_idle(timeout=5)

    assert record.status is TaskStatus.DONE
    (result,) = _dummy_action_results(await task_messages(app, record))
    assert result.status == "success" and "Aktion ausgeführt" in result.text
    assert "Die Aufgabe Buchung ist fertig." in tts.texts[-1]


async def test_question_rejected_ends_cleanly_without_action(make_app):
    app, tts = make_app()
    record = await _start_booking(app, tts)

    reply = await app.session.handle_user_text("Nein, lieber nicht")
    assert "Ablehnung" in reply
    await app.task_manager.wait_idle(timeout=5)

    assert record.status is TaskStatus.DONE
    (result,) = _dummy_action_results(await task_messages(app, record))
    assert result.status == "error" and "rejected" in result.text
    assert "nicht ausgeführt" in tts.texts[-1]


# 6 ---------------------------------------------------------------------------------


async def test_cancel_running_task(make_app):
    app, tts = make_app(work_seconds=5)
    await app.session.handle_user_text("Recherchier etwas zu Wärmepumpen")
    (record,) = app.task_manager.all_tasks()
    await wait_until(lambda: record.progress.startswith("führt slow_work"))

    reply = await app.session.handle_user_text("Brich die Recherche ab")
    assert "abgebrochen" in reply
    assert record.status is TaskStatus.CANCELLED
    assert record.runner.cancelled()
    turns_before = len(tts.turns)
    await app.task_manager.wait_idle(timeout=2)
    assert len(tts.turns) == turns_before, "Keine doppelte Ansage nach bestätigtem Abbruch"


# 7 ---------------------------------------------------------------------------------


async def test_notification_waits_for_running_voice_turn(make_app):
    app, tts = make_app(voice_delay=0.2, tts_delay=0.05)
    user_turn = asyncio.create_task(app.session.handle_user_text("Hallo, wie geht es dir?"))
    await wait_until(lambda: app.session.busy)
    note = asyncio.create_task(app.session.notify_voice("Task 'Test' ist fertig. Ergebnis: Alles gut."))
    await asyncio.sleep(0.05)
    assert not note.done()
    await asyncio.gather(user_turn, note)

    assert tts.texts[0].startswith("Mir geht es gut")
    assert tts.texts[1].startswith("Die Aufgabe Test ist fertig.")
    (_, end_first), (start_second, _) = tts.intervals
    assert start_second >= end_first, "Ansagen dürfen sich nicht überlagern"
    messages = await voice_messages(app)
    human = next(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    system = next(i for i, m in enumerate(messages) if isinstance(m, SystemMessage))
    assert human < system


async def test_notification_waits_while_user_records(make_app):
    app, tts = make_app()
    app.session.recording_started()
    note = asyncio.create_task(app.session.notify_voice("Task 'Test' ist fertig. Ergebnis: Ok."))
    await asyncio.sleep(0.1)
    assert not note.done() and tts.turns == []
    app.session.recording_stopped()
    await asyncio.wait_for(note, 2)
    assert "Test ist fertig" in tts.texts[0]


# Robustheit ------------------------------------------------------------------------


async def test_task_failure_becomes_notification_and_conversation_survives(make_app):
    app, tts = make_app()
    await app.session.handle_user_text("Erledige die Aufgabe zu Fehler")
    (record,) = app.task_manager.all_tasks()
    await app.task_manager.wait_idle(timeout=5)
    assert record.status is TaskStatus.FAILED
    assert "nicht geklappt" in tts.texts[-1]
    assert "gut" in await app.session.handle_user_text("Hallo")


async def test_anthropic_style_notification_is_marked_human_message(make_app):
    app, tts = make_app(provider="anthropic")
    await app.session.notify_voice("Task 'Test' ist fertig. Ergebnis: Ok.")
    last_human = [m for m in await voice_messages(app) if isinstance(m, HumanMessage)][-1]
    assert last_human.content.startswith(NOTIFICATION_MARKER)
    assert not any(isinstance(m, SystemMessage) for m in await voice_messages(app))
