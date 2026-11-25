"""Next, previous and load go through the engine, so no sentence is skipped and a turn in progress is not left
talking about a document that is gone."""

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import BlockingLLM, FakeLLM, FakeSearch, FakeTts, RecordingSocket, StreamingLLM, wait_for_state, wait_until
from voice_agent.document.parser import DocumentContent
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings

SENTENCES = ["Zero is first.", "One is second.", "Two is third.", "Three is last."]
NEW_SENTENCES = ["Alpha opens the book.", "Beta follows it."]


class StepTts(FakeTts):
    """Records each synthesis, then holds it until the test lets it through."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes, str]:
        result = await super().__call__(text, voice_id, cadence)
        await self.gate.wait()
        return result


def make_engine(manager: ConnectionManager, *, llm: Any = None, tts: Any = None) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=llm if llm is not None else StreamingLLM(),
        initial_sentences=list(SENTENCES),
        settings=Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
    )


async def connected(manager: ConnectionManager) -> RecordingSocket:
    socket = RecordingSocket()
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


def document(sentences: list[str]) -> DocumentContent:
    return DocumentContent(
        title="New book",
        filename="new.txt",
        paragraphs=[" ".join(sentences)],
        sentences=sentences,
        total_words=sum(len(s.split()) for s in sentences),
        total_sentences=len(sentences),
    )


async def reach(engine: NarrationEngine, manager: ConnectionManager, state: str) -> None:
    """Put the session in `state` (reached by legal edges, with a reply that stays in flight for REPLYING)."""
    if state == "NARRATING":
        engine.start_narration()
    elif state == "LISTENING":
        manager.trigger_barge_in()
    elif state == "REPLYING":
        engine.ask("who is lucy")
        assert isinstance(engine.llm, BlockingLLM)
        async with asyncio.timeout(5):
            await engine.llm.started.wait()
    assert manager.state == state


# --- no sentence is skipped -----------------------------------------------------------------------------------------------


async def test_next_from_the_first_sentence_narrates_the_second_and_not_the_third() -> None:
    manager = ConnectionManager()
    tts = StepTts()
    engine = make_engine(manager, tts=tts)
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)  # the loop is inside sentence 0

    engine.next_sentence()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(2)

    assert tts.texts[:2] == [SENTENCES[0], SENTENCES[1]]
    assert engine.current_sentence_idx == 1
    tts.gate.set()
    await manager.tasks.cancel_all()


async def test_previous_from_the_first_sentence_wraps_to_the_last() -> None:
    manager = ConnectionManager()
    tts = StepTts()
    engine = make_engine(manager, tts=tts)
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)

    engine.previous_sentence()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(2)

    assert tts.texts[1] == SENTENCES[-1]
    assert engine.current_sentence_idx == len(SENTENCES) - 1
    tts.gate.set()
    await manager.tasks.cancel_all()


async def test_next_wraps_after_the_last_sentence() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    engine.current_sentence_idx = len(SENTENCES) - 1

    engine.next_sentence()

    assert engine.current_sentence_idx == 0


@pytest.mark.parametrize("state", ["PAUSED", "LISTENING", "REPLYING"])
@pytest.mark.parametrize("direction", ["next", "prev"])
async def test_moving_while_not_narrating_moves_the_cursor_without_starting_narration(state: str, direction: str) -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, llm=BlockingLLM())
    socket = await connected(manager)
    await reach(engine, manager, state)
    engine.current_sentence_idx = 2
    before = socket.seen().count(("cursor_update", ""))

    if direction == "next":
        engine.next_sentence()
    else:
        engine.previous_sentence()

    expected = 3 if direction == "next" else 1
    assert engine.current_sentence_idx == expected
    assert engine.current_word_idx == 0
    assert manager.state == state
    assert manager.tasks.get("narration") is None
    await socket.wait_until(lambda: socket.seen().count(("cursor_update", "")) == before + 1)
    assert socket.messages("cursor_update")[-1]["sentence_idx"] == expected
    assert manager.get_state_sync_message().sentence_idx == expected
    await manager.tasks.cancel_all()


# --- loading a document ---------------------------------------------------------------------------------------------------


async def test_loading_a_document_while_paused_resets_the_cursor_and_the_history() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    engine.current_sentence_idx = 2
    engine._conversation_history.append({"role": "user", "content": "about the old book"})
    socket = await connected(manager)

    engine.load_document(document(NEW_SENTENCES))

    assert engine.sentences == NEW_SENTENCES
    assert engine.current_sentence_idx == 0
    assert engine._conversation_history == []
    assert manager.state == "PAUSED"
    assert manager.tasks.get("narration") is None
    await socket.wait_until(lambda: bool(socket.messages("state_sync")))
    assert socket.messages("state_sync")[-1]["sentence_idx"] == 0


async def test_loading_a_document_while_narrating_restarts_at_its_first_sentence() -> None:
    manager = ConnectionManager()
    tts = StepTts()
    engine = make_engine(manager, tts=tts)
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)
    engine.next_sentence()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(2)

    engine.load_document(document(NEW_SENTENCES))
    async with asyncio.timeout(5):
        await tts.wait_for_calls(3)

    assert tts.texts[2] == NEW_SENTENCES[0]
    assert engine.current_sentence_idx == 0
    tts.gate.set()
    await manager.tasks.cancel_all()


async def test_a_document_without_sentences_changes_nothing() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    engine.current_sentence_idx = 2

    engine.load_document(document([]))

    assert engine.sentences == SENTENCES
    assert engine.current_sentence_idx == 2


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_loading_a_document_abandons_a_reply_in_progress(resting: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, llm=BlockingLLM(), tts=tts)
    socket = await connected(manager)
    if resting == "NARRATING":
        engine.start_narration()
        async with asyncio.timeout(5):
            await tts.wait_for_calls(1)
    await reach(engine, manager, "REPLYING")
    reply = manager.tasks.get("reply")
    assert reply is not None
    engine._conversation_history.append({"role": "user", "content": "about the old book"})

    engine.load_document(document(NEW_SENTENCES))
    await asyncio.wait([reply], timeout=5)

    assert reply.cancelled()  # abandoned, not failed
    assert manager.state == resting
    assert manager.tasks.get("reply") is None
    assert engine._conversation_history == []
    if resting == "PAUSED":
        assert engine.current_sentence_idx == 0
    assert (manager.tasks.get("narration") is not None) == (resting == "NARRATING")
    assert socket.messages("reply_error") == []
    await manager.tasks.cancel_all()


async def test_narration_restarted_by_a_document_load_starts_at_the_new_first_sentence() -> None:
    manager = ConnectionManager()
    tts = StepTts()
    engine = make_engine(manager, llm=BlockingLLM(), tts=tts)
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)
    engine.next_sentence()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(2)
    engine.current_sentence_idx = 3
    await reach(engine, manager, "REPLYING")

    engine.load_document(document(NEW_SENTENCES))  # the reply is abandoned and narration resumes
    async with asyncio.timeout(5):
        await tts.wait_for_calls(3)

    assert tts.texts[2] == NEW_SENTENCES[0]  # not the old index, and not a sentence of the old book
    tts.gate.set()
    await manager.tasks.cancel_all()


async def test_loading_a_document_while_listening_drops_the_pending_transcription() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    manager.trigger_barge_in()
    pending = manager.tasks.spawn(asyncio.sleep(3600), "stt")
    await asyncio.sleep(0)

    engine.load_document(document(NEW_SENTENCES))
    await asyncio.wait([pending], timeout=5)

    assert pending.cancelled()
    await wait_for_state(manager.session, "PAUSED")  # the listening turn was about the old document


# --- through the endpoints --------------------------------------------------------------------------------------------------


def test_the_websocket_navigates_through_the_engine() -> None:
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    engine: NarrationEngine = app.state.narration_engine
    start = engine.current_sentence_idx

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "playback_control", "action": "next"})
        assert wait_until(lambda: engine.current_sentence_idx == (start + 1) % len(engine.sentences))
        ws.send_json({"type": "playback_control", "action": "prev"})
        ws.send_json({"type": "playback_control", "action": "prev"})
        assert wait_until(lambda: engine.current_sentence_idx == (start - 1) % len(engine.sentences))

    assert manager.state == "PAUSED"


def test_the_rest_endpoint_navigates_through_the_engine() -> None:
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, settings=replace(Settings.fast()), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    engine: NarrationEngine = app.state.narration_engine

    with TestClient(app) as client:
        resp = client.post("/api/narration/control", json={"action": "prev"})

    assert resp.status_code == 200
    assert resp.json()["current_sentence_idx"] == len(engine.sentences) - 1
