"""Verdrahtung: Store, Checkpointer, beide Agenten, TaskManager und VoiceSession."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore

from .audio import TTS
from .config import Settings
from .notifications import provider_of
from .persistence import make_checkpointer, make_store
from .session import VoiceSession
from .task_agent import build_task_agent
from .task_manager import TaskManager
from .voice_agent import build_voice_agent


@dataclass
class App:
    session: VoiceSession
    task_manager: TaskManager
    store: BaseStore


def init_models(settings: Settings) -> tuple[BaseChatModel, BaseChatModel]:
    voice_kwargs = {}
    if settings.voice_reasoning_effort:
        voice_kwargs["reasoning_effort"] = settings.voice_reasoning_effort
    return init_chat_model(settings.voice_model, **voice_kwargs), init_chat_model(settings.task_model)


def build_app(
    voice_model: str | BaseChatModel,
    task_model: str | BaseChatModel,
    tts: TTS,
    provider: str | None = None,
) -> App:
    store = make_store()  # gemeinsam für beide Agenten
    task_agent = build_task_agent(task_model, checkpointer=make_checkpointer(), store=store)
    task_manager = TaskManager(task_agent)
    voice_agent = build_voice_agent(voice_model, task_manager, checkpointer=make_checkpointer(), store=store)
    session = VoiceSession(voice_agent, tts, provider=provider or provider_of(voice_model))
    task_manager.notify = session.notify_voice  # Rückkanal Task -> Voice
    return App(session=session, task_manager=task_manager, store=store)
