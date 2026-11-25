"""The session state is owned by SessionState and driven through the server: one broadcaster, tracked tasks,
a LISTENING deadline, and a pause when narration fails or the last client leaves."""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeStt, FakeTts, wait_until
from voice_agent.orchestrator.state_machine import SessionState
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.websocket import process_stt_utterance
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

SAMPLE_RATE = 16000
LOUD_20MS = (np.sin(2 * np.pi * 440 * np.arange(320) / SAMPLE_RATE) * 16000).astype(np.int16).tobytes()


class RecordingSocket:
    """Stands in for a WebSocket: records everything the manager sends it."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def accept(self) -> None:
        return None

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        return None

    def messages(self, message_type: str) -> list[dict[str, Any]]:
        parsed = [json.loads(text) for text in self.texts]
        return [message for message in parsed if message["type"] == message_type]

    def states(self) -> list[str]:
        return [message["state"] for message in self.messages("state_sync")]


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


async def connected(manager: ConnectionManager) -> RecordingSocket:
    socket = RecordingSocket()
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

        ws.send_json({"type": "barge_in_trigger"})
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
        ws.send_json({"type": "barge_in_trigger"})
        read_until_state(ws, "LISTENING")
        read_until_state(ws, "PAUSED")

        assert manager.tasks.get("narration") is None


def test_speaking_extends_deadline() -> None:
    manager = ConnectionManager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    deadline_s = 0.4
    app = create_app(
        connection_manager=manager,
        settings=replace(Settings(), barge_in_listen_s=deadline_s),
        llm_client=FakeLLM(),
        tts=FakeTts(),
        search=FakeSearch(),
    )

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until_state(ws, "LISTENING")

        started = time.monotonic()
        while time.monotonic() - started < deadline_s * 2:
            ws.send_bytes(LOUD_20MS)
            time.sleep(0.05)
            assert manager.state == "LISTENING"

        # The user stopped talking: the deadline now runs out.
        assert wait_until(lambda: manager.state != "LISTENING", timeout_s=deadline_s * 10)


async def test_deadline_waits_while_stt_runs() -> None:
    class SlowStt:
        def transcribe(self, audio: bytes) -> str:
            time.sleep(0.4)  # three deadlines long
            return ""

    manager = ConnectionManager(session=SessionState(listen_deadline_s=0.1, followup_window_s=0.1))
    manager.session.enter_listening("barge_in")

    work = asyncio.create_task(process_stt_utterance(b"\x01\x00" * 1600, manager, SlowStt(), engine=None))
    await asyncio.sleep(0.25)
    assert manager.state == "LISTENING"  # the utterance is still with the recognizer

    await work
    assert manager.state == "PAUSED"  # rejected transcript: back to where the session was


async def test_deadline_waits_while_the_stt_model_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first utterance builds the recognizer, which can take seconds. That time is not the user's to lose."""
    from voice_agent.server import websocket as websocket_module

    load_s = 0.4

    async def slow_load(app: Any) -> FakeStt:
        await asyncio.sleep(load_s)
        return FakeStt("")

    monkeypatch.setattr(websocket_module, "_ensure_stt_engine", slow_load)
    manager = ConnectionManager(session=SessionState(listen_deadline_s=0.1, followup_window_s=0.1))
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    manager.session.enter_listening("barge_in")
    manager.accumulator.arm()
    for _ in range(5):
        manager.accumulator.feed(LOUD_20MS)

    class Socket:
        app = object()

    handle = websocket_module.handle_client_text_message(
        '{"type": "voice_end"}', Socket(), manager  # type: ignore[arg-type]
    )
    work = asyncio.create_task(handle)
    await asyncio.sleep(load_s * 0.6)  # six deadlines in, the model is still loading
    assert manager.state == "LISTENING"

    await work
    assert manager.state == "PAUSED"  # empty transcript: back to where the session was


async def test_a_failed_stt_load_does_not_leave_the_deadline_suspended(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.server import websocket as websocket_module

    async def failing_load(app: Any) -> None:
        raise RuntimeError("no model")

    monkeypatch.setattr(websocket_module, "_ensure_stt_engine", failing_load)
    manager = ConnectionManager(session=SessionState(listen_deadline_s=0.1, followup_window_s=0.1))
    manager.session.enter_listening("barge_in")

    class Socket:
        app = object()

    with pytest.raises(RuntimeError):
        await websocket_module._active_stt_engine(Socket(), manager, None)  # type: ignore[arg-type]
    await asyncio.sleep(0.4)

    assert manager.state == "PAUSED"  # the deadline still ran out


# --- the reply flow, through the session ---------------------------------------------------------------------------


async def test_question_while_narrating_resumes_narration() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, settings=Settings.fast(), tts=tts)
    engine.start_narration()
    await wait_for(lambda: manager.tasks.get("narration") is not None)

    await engine.handle_user_interruption("who is lucy")

    assert manager.state == "NARRATING"
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

    await engine.handle_unrecognized_interruption()

    assert manager.state == "PAUSED"
