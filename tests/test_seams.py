"""Injection seams: NarrationEngine and create_app take settings, LLM, TTS and search."""

import asyncio

import pytest

from fakes import FakeLLM, FakeSearch, FakeTts, wait_for_state
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings


@pytest.fixture
def cancel_narration():
    """Collect engines whose background narration loop must be cancelled at teardown."""
    engines: list[NarrationEngine] = []
    yield engines.append
    for engine in engines:
        engine.manager.tasks.cancel("narration")


async def test_engine_uses_injected_tts_and_search(cancel_narration) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    search = FakeSearch()
    llm = FakeLLM(tokens=["Lucy ", "is ", "a ", "netrunner."], web_search_enabled=True)
    engine = NarrationEngine(manager, llm_client=llm, settings=Settings.fast(), tts=tts, search=search)
    cancel_narration(engine)

    await engine.handle_user_interruption("who is lucy")

    assert "Lucy is a netrunner." in tts.texts
    assert [query for query, _ in search.calls] == ["who is lucy"]
    # The search result reached the prompt the LLM saw.
    assert search.result in llm.calls[0][-1]["content"]
    await wait_for_state(manager.session, "PAUSED")  # asked while paused: the follow-up window ends back there


async def test_search_seam_is_skipped_when_web_search_is_off(cancel_narration) -> None:
    search = FakeSearch()
    engine = NarrationEngine(
        ConnectionManager(),
        llm_client=FakeLLM(tokens=["Fine."], web_search_enabled=False),
        settings=Settings.fast(),
        tts=FakeTts(),
        search=search,
    )
    cancel_narration(engine)

    await engine.handle_user_interruption("who is lucy")

    assert search.calls == []


async def test_narration_loop_synthesizes_sentences_through_injected_tts(cancel_narration) -> None:
    tts = FakeTts()
    engine = NarrationEngine(
        ConnectionManager(),
        llm_client=FakeLLM(),
        initial_sentences=["First sentence here.", "Second one."],
        settings=Settings.fast(),
        tts=tts,
        search=FakeSearch(),
    )
    cancel_narration(engine)

    engine.start_narration()
    # The loop wraps around, so a third call proves it advanced past the last sentence.
    async with asyncio.timeout(5):
        await tts.wait_for_calls(3)
    assert tts.texts[:3] == ["First sentence here.", "Second one.", "First sentence here."]


def test_default_llm_is_built_from_settings() -> None:
    settings = Settings(llm_model="custom-model", llm_base_url="http://example.invalid/v1", web_search_enabled=False)
    engine = NarrationEngine(ConnectionManager(), settings=settings)

    config = engine.llm.get_config()
    assert config["model"] == "custom-model"
    assert config["base_url"] == "http://example.invalid/v1"
    assert config["web_search_enabled"] is False
    assert config["provider"] == "ollama"  # D2: the default provider is the local Ollama daemon


def test_create_app_threads_seams() -> None:
    tts = FakeTts()
    search = FakeSearch()
    llm = FakeLLM()
    settings = Settings.fast()

    app = create_app(connection_manager=ConnectionManager(), settings=settings, llm_client=llm, tts=tts, search=search)

    engine = app.state.narration_engine
    assert engine.tts is tts
    assert engine.search is search
    assert engine.llm is llm
    assert engine.settings is settings
