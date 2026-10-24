import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from voice_assistant.fakes import FakeChatModel
from voice_assistant.notifications import NOTIFICATION_MARKER, make_notification, provider_of


@pytest.mark.parametrize(
    ("provider", "cls"),
    [("openai", SystemMessage), ("anthropic", HumanMessage), ("google_genai", HumanMessage), ("gemini", HumanMessage)],
)
def test_make_notification_per_provider(provider, cls):
    msg = make_notification("Task fertig", provider)
    assert type(msg) is cls
    assert msg.content == f"{NOTIFICATION_MARKER} Task fertig"


def test_provider_of():
    assert provider_of("openai:gpt-5-mini") == "openai"
    assert provider_of("anthropic:claude-sonnet-5-5") == "anthropic"
    assert provider_of("google_genai:gemini-2.5-flash") == "google_genai"
    assert provider_of(FakeChatModel(responder=lambda m: None)) == "openai"
