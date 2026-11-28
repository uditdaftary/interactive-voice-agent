"""The session state is owned by SessionState and driven through the server: one broadcaster, tracked tasks,
a LISTENING deadline, and a pause when narration fails or the last client leaves."""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import (
    BlockingLLM,
    FakeLLM,
    FakeSearch,
    FakeStt,
    FakeTts,
    FunctionTts,
    PlayerSocket,
    wait_for_state,
    wait_until,
)
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

SAMPLE_RATE = 16000
LOUD_20MS = (np.sin(2 * np.pi * 440 * np.arange(320) / SAMPLE_RATE) * 16000).astype(np.int16).tobytes()


async def wait_for(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached in time")


def make_engine(manager: ConnectionManager, **overrides: Any) -> NarrationEngine:
    settings = overrides.pop("settings", Settings())
    return NarrationEngine(
        manager,
        llm_client=overrides.pop("llm", FakeLLM(tokens=["Fine."])),
        settings=settings,
        tts=overrides.pop("tts", FakeTts()),
        search=FakeSearch(),
    )


async def connected(manager: ConnectionManager) -> PlayerSocket:
    """A tab that plays every clip it is sent and says so at once, as a browser does."""
    socket = PlayerSocket(manager, auto_ack=True)
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


# --- one session, one broadcaster ---------------------------------------------------------------------------------


def test_initial_state_sync_is_paused() -> None:
    app = create_app(connection_manager=ConnectionManager(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        first = ws.receive_json()

    assert first["type"] == "state_sync"
    assert first["state"] == "PAUSED"


async def test_state_sync_once_per_transition() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    socket = await connected(manager)

    engine.start_narration()
    await wait_for(lambda: socket.states() == ["NARRATING"])
    engine.pause_narration()
    await asyncio.sleep(0.1)

    assert socket.states() == ["NARRATING", "PAUSED"]
    await manager.tasks.cancel_all()


async def test_a_noop_transition_is_not_broadcast() -> None:
    manager = ConnectionManager()
    make_engine(manager)
    socket = await connected(manager)

    manager.session.transition("PAUSED", "again")
    await asyncio.sleep(0.05)

    assert socket.states() == []


async def test_state_property_follows_the_session() -> None:
    manager = ConnectionManager()

    manager.session.transition("NARRATING", "test")

    assert manager.state == "NARRATING"
    assert manager.get_state_sync_message().state == "NARRATING"


async def test_leaving_listening_disarms_the_accumulator() -> None:
    manager = ConnectionManager()
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    manager.accumulator = acc
    manager.session.transition("LISTENING", "test")
    acc.arm()
    for _ in range(10):
        acc.feed(LOUD_20MS)
    assert acc.is_speaking

    manager.session.transition("REPLYING", "question")

    assert not acc.is_speaking
    assert sum(len(chunk) for chunk in acc._session_buffer) == 0


# --- pausing ------------------------------------------------------------------------------------------------------


async def test_pause_cancels_narration_reply_and_stt_tasks() -> None:
    manager = ConnectionManager()
    make_engine(manager)
    manager.session.transition("NARRATING", "test")
    manager.session.transition("REPLYING", "test")
    tasks = [manager.tasks.spawn(asyncio.sleep(3600), name) for name in ("narration", "reply", "stt")]
    other = manager.tasks.spawn(asyncio.sleep(3600), "unrelated")
    await asyncio.sleep(0)

    manager.session.transition("PAUSED", "pause")
    await asyncio.sleep(0.05)

    assert all(task.cancelled() for task in tasks)
    assert not other.done()
    await manager.tasks.cancel_all()


async def test_tts_error_pauses_with_error() -> None:
    async def broken_tts(text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
        raise RuntimeError("synthesis exploded")

    manager = ConnectionManager()
    engine = make_engine(manager, tts=FunctionTts(broken_tts))
    socket = await connected(manager)

    engine.start_narration()
    await wait_for(lambda: manager.state == "PAUSED")
    await asyncio.sleep(0.05)

    errors = socket.messages("error")
    assert [error["detail"] for error in errors] == ["narration stopped: RuntimeError"]
    assert socket.states() == ["NARRATING", "PAUSED"]
    assert manager.tasks.get("narration") is None


async def test_last_disconnect_pauses() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    first, second = await connected(manager), await connected(manager)
    engine.start_narration()
    await wait_for(lambda: manager.tasks.get("narration") is not None)

    manager.disconnect(first)  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    assert manager.state == "NARRATING"

    manager.disconnect(second)  # type: ignore[arg-type]
    await asyncio.sleep(0.05)

    assert manager.state == "PAUSED"
    assert len(manager.tasks) == 0
    # A client that connects later is told the truth.
    assert manager.get_state_sync_message().state == "PAUSED"


async def test_barge_in_from_any_source_stops_narration() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    engine.start_narration()
    await wait_for(lambda: manager.tasks.get("narration") is not None)

    manager.trigger_barge_in()  # what the audio bridge and the websocket handler both call
    await asyncio.sleep(0.05)

    assert manager.state == "LISTENING"
    assert manager.tasks.get("narration") is None
    await manager.tasks.cancel_all()


# --- the LISTENING deadline, through the real endpoint -------------------------------------------------------------


def fast_settings(**overrides: Any) -> Settings:
    return replace(Settings.fast(), **overrides)


def read_until_state(ws: Any, state: str, limit: int = 2000) -> None:
    """Skip narration traffic until a state_sync announces `state`."""
    for _ in range(limit):
        frame = ws.receive()
        if frame.get("text") is None:
            continue
        message = json.loads(frame["text"])
        if message["type"] == "state_sync" and message["state"] == state:
            return
    raise AssertionError(f"no state_sync {state} within {limit} frames")


def test_barge_in_deadline_resumes() -> None:
    manager = ConnectionManager()
    app = create_app(
        connection_manager=manager,
        settings=replace(Settings(), barge_in_listen_s=0.05),
        llm_client=FakeLLM(),
        tts=FakeTts(),
        search=FakeSearch(),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "playback_control", "action": "resume"})
        read_until_state(ws, "NARRATING")

        client.portal.call(manager.trigger_barge_in)  # server-detected speech; a push-to-talk click holds the deadline
        read_until_state(ws, "LISTENING")
        read_until_state(ws, "NARRATING")  # nobody spoke: the deadline puts the session back where it was

        assert wait_until(lambda: manager.tasks.get("narration") is not None)


def test_barge_in_from_paused_returns_to_paused() -> None:
    manager = ConnectionManager()
    app = create_app(
        connection_manager=manager,
        settings=replace(Settings(), barge_in_listen_s=0.05),
        llm_client=FakeLLM(),
        tts=FakeTts(),
        search=FakeSearch(),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.portal.call(manager.trigger_barge_in)
        read_until_state(ws, "LISTENING")
        read_until_state(ws, "PAUSED")

        assert manager.tasks.get("narration") is None


def test_speaking_extends_deadline() -> None:
    manager = ConnectionManager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    deadline_s = 0.15
    app = create_app(
        connection_manager=manager,
        settings=replace(Settings(), barge_in_listen_s=deadline_s),
        llm_client=FakeLLM(),
        tts=FakeTts(),
        search=FakeSearch(),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        # Server-detected speech (not a push-to-talk click, which holds the deadline until voice_end).
        client.portal.call(manager.trigger_barge_in)
        read_until_state(ws, "LISTENING")

        started = time.monotonic()
        while time.monotonic() - started < deadline_s * 2:
            ws.send_bytes(LOUD_20MS)
            time.sleep(0.03)
            assert manager.state == "LISTENING"

        # The user stopped talking: the deadline now runs out.
        assert wait_until(lambda: manager.state != "LISTENING", timeout_s=5)


# --- the reply flow, through the session ---------------------------------------------------------------------------


async def test_question_while_narrating_resumes_narration() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, settings=Settings.fast(), tts=tts)
    engine.start_narration()
    await wait_for(lambda: manager.tasks.get("narration") is not None)

    await engine.handle_user_interruption("who is lucy")

    await wait_for_state(manager.session, "NARRATING")  # the follow-up window closes, and narration goes on
    await wait_for(lambda: manager.tasks.get("narration") is not None)
    await manager.tasks.cancel_all()


async def test_question_while_paused_stays_paused_afterwards() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, settings=Settings.fast(), tts=tts)

    await engine.handle_user_interruption("who is lucy")
    await asyncio.sleep(0.05)

    assert tts.texts == ["Fine."]  # only the reply was spoken
    assert manager.state == "PAUSED"
    assert manager.tasks.get("narration") is None


async def test_unrecognized_utterance_returns_to_where_the_session_was() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, settings=Settings.fast())
    manager.session.transition("LISTENING", "push_to_talk")

    engine.report_unrecognized()
    reply = manager.tasks.get("reply")
    assert reply is not None
    await asyncio.wait([reply], timeout=5)

    assert manager.state == "PAUSED"


async def test_shutdown_leaves_no_tasks() -> None:
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    engine: NarrationEngine = app.state.narration_engine

    async with app.router.lifespan_context(app):
        engine.start_narration()
        reply = manager.tasks.spawn(asyncio.sleep(3600), "reply")
        await wait_for(lambda: manager.tasks.get("narration") is not None)
        narration = manager.tasks.get("narration")
        assert narration is not None

    assert len(manager.tasks) == 0
    assert narration.done()
    assert reply.done()


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_barge_in_during_a_reply_cancels_the_reply(resting: str) -> None:
    llm = BlockingLLM()
    tts = FakeTts()
    manager = ConnectionManager()
    engine = make_engine(manager, settings=replace(Settings.fast(), barge_in_listen_s=0.1), llm=llm, tts=tts)
    if resting == "NARRATING":
        engine.start_narration()
        await wait_for(lambda: len(tts.calls) >= 1)
    reply = manager.tasks.spawn(engine.handle_user_interruption("who is lucy"), "reply", replace=True)
    async with asyncio.timeout(5):
        await llm.started.wait()
    assert manager.state == "REPLYING"
    assert manager.session.resume_to == resting

    manager.trigger_barge_in()  # the user speaks over the answer
    await asyncio.wait([reply], timeout=5)
    assert reply.cancelled()
    assert manager.state == "LISTENING"

    await wait_for(lambda: manager.state == resting)  # nobody spoke: the deadline puts the session back
    assert manager.state == resting  # and the cancelled reply (asserted above) cannot pull it back into LISTENING
    assert manager.tasks.get("reply") is None
    assert (manager.tasks.get("narration") is not None) == (resting == "NARRATING")
    await manager.tasks.cancel_all()


async def test_a_reply_survives_its_own_followup_window() -> None:
    """Entering LISTENING for the follow-up is not a barge-in, so the reply is not cancelled by it."""
    manager = ConnectionManager()
    engine = make_engine(manager, settings=replace(Settings.fast(), followup_window_s=0.1))

    reply = manager.tasks.spawn(engine.handle_user_interruption("who is lucy"), "reply", replace=True)
    await asyncio.wait([reply], timeout=5)

    assert reply.done() and not reply.cancelled()
    assert reply.exception() is None
    await manager.tasks.cancel_all()


async def test_a_replacement_reply_is_not_cancelled_by_its_own_barge_in() -> None:
    """A second question takes over from the first: its trigger_barge_in leaves REPLYING, but it is the new reply."""
    llm = BlockingLLM()
    manager = ConnectionManager()
    engine = make_engine(manager, settings=Settings.fast(), llm=llm)

    first = manager.tasks.spawn(engine.handle_user_interruption("one"), "reply", replace=True)
    async with asyncio.timeout(5):
        await llm.started.wait()
    second = manager.tasks.spawn(engine.handle_user_interruption("two"), "reply", replace=True)
    await wait_for(lambda: len(llm.calls) == 2)

    assert first.cancelled()
    assert not second.done()
    assert manager.state == "REPLYING"
    await manager.tasks.cancel_all()


# --- one engine per session ---------------------------------------------------------------------------------------


async def test_each_app_owns_its_session_and_engine() -> None:
    """Importing the module-level app and then building another must not stack listeners or share narration."""
    from voice_agent.server.app import app as module_app

    tts = FakeTts()
    other = create_app(settings=Settings.fast(), llm_client=FakeLLM(), tts=tts, search=FakeSearch())
    module_manager: ConnectionManager = module_app.state.connection_manager
    other_manager: ConnectionManager = other.state.connection_manager

    assert other_manager is not module_manager
    assert other_manager.session is not module_manager.session
    # Disarm, announce, and the one engine: no matter how many apps were built before.
    assert len(module_manager.session._listeners) == 3
    assert len(other_manager.session._listeners) == 3
    assert other_manager.session.followup_window_s == 0.0  # Settings.fast()
    assert module_manager.session.followup_window_s == Settings().followup_window_s

    other.state.narration_engine.start_narration()
    await wait_for(lambda: len(tts.calls) >= 1)  # the other app's own engine narrates, with its own TTS
    assert module_manager.tasks.get("narration") is None
    await other_manager.tasks.cancel_all()


def test_quiet_push_to_talk_is_answered_after_the_deadline_would_have_run_out() -> None:
    """Speech below the server's threshold never extends the deadline; the client's voice_end ends the turn."""
    deadline_s = 0.1
    manager = ConnectionManager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    manager.stt_engine = FakeStt("what is arasaka")
    quiet = (np.sin(2 * np.pi * 440 * np.arange(320) / SAMPLE_RATE) * 60).astype(np.int16).tobytes()  # RMS ~0.0013
    engine = MagicMock()
    app = create_app(connection_manager=manager, narration_engine=engine)
    manager.session.listen_deadline_s = deadline_s

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until_state(ws, "LISTENING")
        started = time.monotonic()
        while time.monotonic() - started < deadline_s * 2.5:
            ws.send_bytes(quiet)
            time.sleep(0.02)
            assert manager.state == "LISTENING"  # three deadlines of quiet speech later, still listening

        ws.send_json({"type": "voice_end"})
        assert wait_until(lambda: engine.ask.called, timeout_s=5)

    engine.ask.assert_called_once_with("what is arasaka")


def test_a_barge_in_without_voice_end_still_times_out_after_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.server import websocket as websocket_module

    monkeypatch.setattr(websocket_module, "MAX_UTTERANCE_S", 0.1)
    manager = ConnectionManager()
    app = create_app(
        connection_manager=manager,
        settings=replace(Settings(), barge_in_listen_s=0.1),
        llm_client=FakeLLM(),
        tts=FakeTts(),
        search=FakeSearch(),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until_state(ws, "LISTENING")
        read_until_state(ws, "PAUSED")  # cap, then the normal deadline


# --- message order ------------------------------------------------------------------------------------------------


async def test_state_sync_precedes_the_messages_the_state_causes() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    socket = await connected(manager)

    engine.start_narration()
    await wait_for(lambda: any(kind == "cursor_update" for kind, _ in socket.seen()))

    order = socket.seen()
    assert order.index(("state_sync", "NARRATING")) < order.index(("cursor_update", ""))
    await manager.tasks.cancel_all()


async def test_barge_in_then_deadline_resume_arrives_in_order() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, settings=replace(Settings(), barge_in_listen_s=0.05))
    socket = await connected(manager)
    engine.start_narration()
    await wait_for(lambda: any(kind == "cursor_update" for kind, _ in socket.seen()))
    socket.texts.clear()

    manager.trigger_barge_in()
    await wait_for(lambda: ("state_sync", "NARRATING") in socket.seen())
    await wait_for(lambda: any(kind == "cursor_update" for kind, _ in socket.seen()[socket.seen().index(("state_sync", "NARRATING")):]))

    order = socket.seen()
    barge_in = order.index(("barge_in", ""))
    listening = order.index(("state_sync", "LISTENING"))
    narrating = order.index(("state_sync", "NARRATING"))
    cursor_after_resume = next(i for i, (kind, _) in enumerate(order) if kind == "cursor_update" and i > listening)
    assert barge_in < listening < narrating < cursor_after_resume
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_unrecognized_speech_resumes_to_where_the_session_was(resting: str) -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, settings=Settings.fast())
    if resting == "NARRATING":
        engine.start_narration()
        await wait_for(lambda: manager.tasks.get("narration") is not None)
    manager.trigger_barge_in()

    engine.report_unrecognized()
    reply = manager.tasks.get("reply")
    assert reply is not None
    await asyncio.wait([reply], timeout=5)

    assert manager.state == resting
    assert (manager.tasks.get("narration") is not None) == (resting == "NARRATING")
    await manager.tasks.cancel_all()
