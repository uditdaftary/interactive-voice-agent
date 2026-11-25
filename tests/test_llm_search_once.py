"""A barge-in question costs exactly one web search, and no scripted answer exists to add another (H7)."""

import httpx
import pytest

from fakes import FakeSearch, FakeTts
from voice_agent.llm import client as llm_client_module
from voice_agent.llm.client import OFFLINE_LINE, LLMClient
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings

BOOK = [
    "The rain fell on the empty street.",
    "A courier crossed the bridge at dawn.",
]


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


async def test_one_search_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_calls: list[str] = []
    monkeypatch.setattr(llm_client_module, "search_web_sync", lambda query, max_results=8: sync_calls.append(query) or "")
    search = FakeSearch()
    tts = FakeTts()
    engine = NarrationEngine(
        ConnectionManager(),
        llm_client=LLMClient(transport=httpx.MockTransport(refuse)),
        initial_sentences=BOOK,
        settings=Settings.fast(),
        tts=tts,
        search=search,
    )

    try:
        await engine.handle_user_interruption("who is the narrator")
    finally:
        engine.manager.tasks.cancel("narration")

    assert len(search.calls) == 1
    assert sync_calls == []  # the old scripted answer's second, blocking search is gone
    assert tts.texts[0] == OFFLINE_LINE  # with no model, the user hears one neutral line, not scripted lore
    assert not hasattr(LLMClient, "_synthesize_fallback_answer")
