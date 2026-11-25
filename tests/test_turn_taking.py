"""One user turn gives exactly one reply: the first end of a turn claims it, transcription runs outside the receive
loop, and a transcript is answered only if its turn is still the one being listened to."""

import asyncio
import json
import logging
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
    GatedStt,
    StreamingLLM,
    wait_for_state,
    wait_until,
)
from voice_agent.orchestrator.state_machine import SessionState
from voice_agent.server import websocket as websocket_module
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.websocket import handle_client_text_message, submit_turn
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

SAMPLE_RATE = 16000
PCM = b"\x01\x00" * 1600
LOUD_20MS = (np.sin(2 * np.pi * 440 * np.arange(320) / SAMPLE_RATE) * 16000).astype(np.int16).tobytes()


def tone(seconds: float, hz: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (np.sin(2 * np.pi * hz * t) * 16000).astype(np.int16).tobytes()


def silence(seconds: float) -> bytes:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16).tobytes()


class StubSocket:
    """What the turn code needs of a WebSocket: the app it belongs to, and somewhere to send a message."""

    app = object()

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def quick_manager(deadline_s: float = 0.03) -> ConnectionManager:
    return ConnectionManager(session=SessionState(listen_deadline_s=deadline_s, followup_window_s=deadline_s))


def make_engine(manager: ConnectionManager, *, llm: Any = None, tts: Any = None, settings: Settings | None = None) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=llm if llm is not None else StreamingLLM(),
        settings=settings if settings is not None else Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
    )


def submit(
    manager: ConnectionManager,
    engine: Any,
    stt: Any,
    utterance: bytes | None = PCM,
    fallback: str | None = None,
) -> None:
    submit_turn(StubSocket(), manager, engine, stt, utterance, fallback)  # type: ignore[arg-type]


async def stt_finished(task: asyncio.Task[Any] | None) -> None:
    assert task is not None, "no stt task"
    await asyncio.wait([task], timeout=5)
    assert task.done()


# --- the deadline waits for the recognizer, then the turn is answered ----------------------------------------------------


async def test_the_deadline_waits_while_the_recognizer_works() -> None:
    manager = quick_manager()
    manager.session.enter_listening("barge_in")
    stt = GatedStt("")

    submit(manager, None, stt)
    await stt.wait_started()
    await asyncio.sleep(0.09)  # three deadlines, with the utterance still at the recognizer
    assert manager.state == "LISTENING"

    stt.release.set()
    await wait_for_state(manager.session, "PAUSED")  # nothing was understood: back to where the session was


async def test_the_deadline_waits_while_the_stt_model_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first utterance builds the recognizer, which can take seconds. That time is not the user's to lose."""
    loaded = asyncio.Event()

    async def slow_load(app: Any) -> FakeStt:
        await loaded.wait()
        return FakeStt("")

    monkeypatch.setattr(websocket_module, "_ensure_stt_engine", slow_load)
    manager = quick_manager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    manager.session.enter_listening("barge_in")
    manager.accumulator.arm()
    for _ in range(5):
        manager.accumulator.feed(LOUD_20MS)

    await handle_client_text_message('{"type": "voice_end"}', StubSocket(), manager)  # type: ignore[arg-type]
    await asyncio.sleep(0.09)  # three deadlines in, the model is still loading
    assert manager.state == "LISTENING"

    loaded.set()
    await wait_for_state(manager.session, "PAUSED")  # empty transcript: back to where the session was


async def test_a_failed_stt_load_is_an_unrecognized_turn_and_the_deadline_still_runs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing_load(app: Any) -> None:
        raise RuntimeError("no model")

    monkeypatch.setattr(websocket_module, "_ensure_stt_engine", failing_load)
    manager = quick_manager()
    engine = MagicMock()  # leaves the state alone, so only the deadline can end the turn
    manager.session.enter_listening("barge_in")

    with caplog.at_level(logging.ERROR, logger="voice_agent.server.websocket"):
        submit(manager, engine, None)
        await stt_finished(manager.tasks.get("stt"))

    engine.report_unrecognized.assert_called_once_with()
    engine.ask.assert_not_called()
    assert any(record.exc_info and record.exc_info[0] is RuntimeError for record in caplog.records)  # with traceback
    await wait_for_state(manager.session, "PAUSED")  # the deadline was not left suspended by the failure


async def test_a_transcript_is_answered_by_a_tracked_reply_task() -> None:
    manager = ConnectionManager()
    llm = BlockingLLM()
    engine = make_engine(manager, llm=llm)
    manager.session.enter_listening("push_to_talk")

    submit(manager, engine, FakeStt("what is arasaka"))
    await stt_finished(manager.tasks.get("stt"))
    reply = manager.tasks.get("reply")

    assert reply is not None  # the answer is a tracked task, not something the transcription waits inside
    llm.release.set()
    await asyncio.wait([reply], timeout=5)
    await wait_for_state(manager.session, "PAUSED")  # the follow-up window closes back to where the session was


@pytest.mark.parametrize("transcript", ["what is arasaka", ""])
@pytest.mark.parametrize("left_for", ["PAUSED", "REPLYING"])
async def test_a_transcript_is_dropped_when_the_session_left_listening(transcript: str, left_for: str) -> None:
    """A pause (or a typed question) while Whisper works must not be answered by the late transcript."""
    manager = ConnectionManager()
    engine = MagicMock()
    stt = GatedStt(transcript)
    manager.session.enter_listening("barge_in")

    submit(manager, engine, stt)
    task = manager.tasks.get("stt")
    await stt.wait_started()
    manager.session.transition(left_for, "meanwhile")
    stt.release.set()
    await stt_finished(task)

    engine.ask.assert_not_called()
    engine.report_unrecognized.assert_not_called()
    assert manager.tasks.get("reply") is None
    assert manager.state == left_for


async def test_a_transcript_is_answered_when_the_session_is_still_listening() -> None:
    manager = ConnectionManager()
    engine = MagicMock()
    stt = GatedStt("what is arasaka")
    manager.session.enter_listening("barge_in")

    submit(manager, engine, stt)
    task = manager.tasks.get("stt")
    await stt.wait_started()
    stt.release.set()
    await stt_finished(task)

    engine.ask.assert_called_once_with("what is arasaka")
    engine.report_unrecognized.assert_not_called()


# --- one turn, one reply -------------------------------------------------------------------------------------------------


async def test_two_turn_ends_for_one_listening_entry_give_one_reply() -> None:
    manager = ConnectionManager()
    engine = MagicMock()
    stt = GatedStt("what is arasaka")
    manager.session.enter_listening("barge_in")

    submit(manager, engine, stt)  # the accumulator heard the silence
    await stt.wait_started()
    submit(manager, engine, stt)  # and the client's own stop arrives while that is being transcribed
    stt.release.set()
    await stt_finished(manager.tasks.get("stt"))

    assert len(stt.inputs) == 1
    engine.ask.assert_called_once_with("what is arasaka")
    engine.report_unrecognized.assert_not_called()  # the second turn-end is not "I didn't catch that"


async def test_a_typed_question_beats_a_pending_transcript() -> None:
    manager = ConnectionManager()
    llm = BlockingLLM()
    engine = make_engine(manager, llm=llm)
    stt = GatedStt("spoken question")
    manager.session.enter_listening("barge_in")

    submit(manager, engine, stt)
    pending = manager.tasks.get("stt")
    assert pending is not None
    await stt.wait_started()
    engine.ask("typed question")
    stt.release.set()
    await asyncio.wait([pending], timeout=5)

    assert pending.cancelled()
    async with asyncio.timeout(5):
        await llm.started.wait()
    assert len(llm.calls) == 1
    assert "typed question" in llm.calls[0][-1]["content"]

    # A voice_end that arrives afterwards belongs to a turn that is over.
    await handle_client_text_message('{"type": "voice_end"}', StubSocket(), manager, engine, stt)  # type: ignore[arg-type]
    assert len(stt.inputs) == 1
    await manager.tasks.cancel_all()


async def test_pause_then_a_new_barge_in_never_answers_the_old_question() -> None:
    manager = ConnectionManager()
    llm = BlockingLLM()
    engine = make_engine(manager, llm=llm)
    old = GatedStt("old question")
    manager.session.enter_listening("barge_in")
    submit(manager, engine, old)
    await old.wait_started()

    engine.pause_narration()
    manager.trigger_barge_in()  # a newer turn begins while the recognizer is still on the old one
    old.release.set()
    await asyncio.sleep(0)  # let the abandoned recognizer finish, if it is going to

    assert manager.state == "LISTENING"
    assert not manager.session.turn_claimed
    assert llm.calls == []

    submit(manager, engine, FakeStt("new question"))
    await stt_finished(manager.tasks.get("stt"))
    async with asyncio.timeout(5):
        await llm.started.wait()
    assert [call[-1]["content"].rsplit("User Question: ", 1)[-1] for call in llm.calls] == ["new question"]
    await manager.tasks.cancel_all()


async def test_an_abandoned_transcription_does_not_lift_the_new_turns_suspension() -> None:
    manager = quick_manager()
    engine = MagicMock()
    first, second = GatedStt("first"), GatedStt("second")
    manager.session.enter_listening("barge_in")
    submit(manager, engine, first)
    await first.wait_started()

    manager.trigger_barge_in()  # a newer turn: the first transcription is cancelled
    submit(manager, engine, second)
    await second.wait_started()
    first.release.set()
    await asyncio.sleep(0.09)  # three deadlines; the cancelled task has cleaned up by now

    assert manager.state == "LISTENING"  # the second turn's clock is still stopped
    second.release.set()
    await stt_finished(manager.tasks.get("stt"))
    engine.ask.assert_called_once_with("second")


async def test_slow_stt_in_the_followup_window_gives_one_reply_and_never_starts_narration() -> None:
    manager = ConnectionManager()
    llm = StreamingLLM()
    tts = FakeTts()
    engine = make_engine(manager, llm=llm, tts=tts, settings=replace(Settings.fast(), followup_window_s=0.1))
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)
    engine.ask("first question")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await asyncio.wait([reply], timeout=5)
    assert manager.state == "LISTENING"  # the follow-up window is open
    states: list[str] = []
    manager.session.add_listener(lambda old, new, reason: states.append(new))

    stt = GatedStt("and her friend")
    submit(manager, engine, stt)
    await stt.wait_started()
    await asyncio.sleep(0.3)  # three windows, with the follow-up still being transcribed
    assert manager.state == "LISTENING"
    assert manager.tasks.get("narration") is None

    stt.release.set()
    await wait_for_state(manager.session, "NARRATING")

    assert len(llm.calls) == 2  # the first question and exactly one follow-up reply
    assert states == ["REPLYING", "LISTENING", "NARRATING"]
    await manager.tasks.cancel_all()


# --- a Stop that arrives after the turn is over ---------------------------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED", "REPLYING"])
async def test_a_voice_end_after_the_turn_is_over_is_answered_with_the_state(resting: str) -> None:
    """The client shows "Transcribing" after Stop and clears it on the next state change; none may be coming."""
    manager = ConnectionManager()
    llm = BlockingLLM()
    engine = make_engine(manager, llm=llm)
    if resting == "NARRATING":
        engine.start_narration()
    elif resting == "REPLYING":
        engine.ask("who is lucy")
        async with asyncio.timeout(5):
            await llm.started.wait()
    client = StubSocket()

    await handle_client_text_message('{"type": "voice_end"}', client, manager, engine, FakeStt("never used"))  # type: ignore[arg-type]

    assert [message["type"] for message in client.sent] == ["state_sync"]
    assert client.sent[0]["state"] == resting
    assert manager.state == resting  # and nothing else happened
    await manager.tasks.cancel_all()


async def test_a_voice_end_during_transcription_is_left_to_the_turn_in_flight() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, llm=BlockingLLM())
    stt = GatedStt("spoken question")
    manager.session.enter_listening("barge_in")
    submit(manager, engine, stt)
    await stt.wait_started()
    client = StubSocket()

    await handle_client_text_message('{"type": "voice_end"}', client, manager, engine, stt)  # type: ignore[arg-type]

    assert client.sent == []  # still LISTENING: the transcript and the reply will say what became of the turn
    assert len(stt.inputs) == 1
    stt.release.set()
    await manager.tasks.cancel_all()


# --- a turn without audio ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_an_empty_voice_end_returns_to_where_the_session_was_at_once(resting: str) -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    stt = FakeStt("never used")
    if resting == "NARRATING":
        engine.start_narration()
    manager.trigger_barge_in()  # the default deadline is seconds away
    assert manager.state == "LISTENING"
    client = StubSocket()

    await handle_client_text_message('{"type": "voice_end"}', client, manager, engine, stt)  # type: ignore[arg-type]

    assert client.sent == []  # the turn is ending now: the state change that follows tells the client
    assert manager.state == resting
    assert stt.calls == []
    assert manager.tasks.get("stt") is None
    assert manager.tasks.get("reply") is None  # no "I didn't catch that" for a turn with nothing in it
    await manager.tasks.cancel_all()


async def test_a_turn_without_audio_is_answered_from_its_fallback_text() -> None:
    manager = ConnectionManager()
    engine = MagicMock()
    manager.session.enter_listening("barge_in")

    submit(manager, engine, None, utterance=None, fallback="  typed words  ")

    engine.ask.assert_called_once_with("typed words")
    assert manager.tasks.get("stt") is None


async def test_the_fallback_text_stands_in_for_an_empty_transcript() -> None:
    manager = ConnectionManager()
    engine = MagicMock()
    manager.session.enter_listening("barge_in")

    submit(manager, engine, FakeStt(""), fallback="typed words")
    await stt_finished(manager.tasks.get("stt"))

    engine.ask.assert_called_once_with("typed words")
    engine.report_unrecognized.assert_not_called()


# --- through the real endpoint ---------------------------------------------------------------------------------------------


def read_until(ws: Any, predicate: Callable[[dict[str, Any]], bool], limit: int = 2000) -> dict[str, Any]:
    """Skip other traffic until a JSON message satisfies `predicate`."""
    for _ in range(limit):
        frame = ws.receive()
        if frame.get("text") is None:
            continue
        message: dict[str, Any] = json.loads(frame["text"])
        if predicate(message):
            return message
    raise AssertionError(f"no matching message within {limit} frames")


def listening(message: dict[str, Any]) -> bool:
    return message["type"] == "state_sync" and message["state"] == "LISTENING"


def short_silence_accumulator() -> SpeechTurnAccumulator:
    return SpeechTurnAccumulator(
        sample_rate=SAMPLE_RATE, energy_threshold=0.02, silence_timeout_ms=100, min_speech_duration_ms=100
    )


def test_a_pause_and_a_cadence_change_are_handled_while_a_recording_is_transcribed() -> None:
    manager = ConnectionManager()
    stt = GatedStt("what is arasaka")
    manager.stt_engine = stt
    llm = StreamingLLM()
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=llm, tts=FakeTts(), search=FakeSearch())

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until(ws, listening)
        ws.send_bytes(tone(0.3))
        ws.send_json({"type": "voice_end"})
        assert stt.started.wait(5)  # the recognizer is busy; the receive loop must not be

        ws.send_json({"type": "set_cadence", "cadence": 1.5})
        read_until(ws, lambda m: m["type"] == "state_sync" and m["cadence"] == 1.5)
        ws.send_json({"type": "playback_control", "action": "pause"})
        read_until(ws, lambda m: m["type"] == "state_sync" and m["state"] == "PAUSED")

        stt.release.set()
        assert wait_until(lambda: manager.tasks.get("stt") is None)

    assert llm.calls == []  # the question was dropped with the pause
    assert manager.state == "PAUSED"


def test_silence_and_voice_end_for_one_turn_give_one_reply() -> None:
    manager = ConnectionManager()
    stt = GatedStt("what is arasaka")
    manager.stt_engine = stt
    manager.accumulator = short_silence_accumulator()
    engine = MagicMock()
    app = create_app(connection_manager=manager, narration_engine=engine)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        client.portal.call(manager.trigger_barge_in)  # server-detected speech: silence can end this turn
        read_until(ws, listening)
        ws.send_bytes(tone(0.2))
        ws.send_bytes(silence(0.15))  # longer than the silence timeout: the turn ends
        assert stt.started.wait(5)

        ws.send_json({"type": "voice_end"})  # the client's own stop, for the same turn
        ws.send_json({"type": "set_cadence", "cadence": 1.2})
        read_until(ws, lambda m: m["type"] == "state_sync" and m["cadence"] == 1.2)  # both have been handled
        stt.release.set()
        assert wait_until(lambda: engine.ask.called)

    assert len(stt.inputs) == 1
    engine.ask.assert_called_once_with("what is arasaka")
    engine.report_unrecognized.assert_not_called()


def test_a_held_turn_records_across_a_pause_and_ends_on_voice_end() -> None:
    manager = ConnectionManager()
    stt = FakeStt("what is arasaka")
    manager.stt_engine = stt
    manager.accumulator = short_silence_accumulator()
    engine = MagicMock()
    app = create_app(connection_manager=manager, narration_engine=engine)
    first, gap, second = tone(0.2), silence(0.5), tone(0.2, 330.0)  # the gap is five times the silence timeout

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})  # push-to-talk: the turn is held
        read_until(ws, listening)
        ws.send_bytes(first)
        ws.send_bytes(gap)
        ws.send_bytes(second)
        ws.send_json({"type": "voice_end"})
        assert wait_until(lambda: engine.ask.called)

    engine.ask.assert_called_once_with("what is arasaka")
    assert stt.calls == [len(first) + len(gap) + len(second)]  # one recognition, both halves and the pause between
    engine.report_unrecognized.assert_not_called()


def test_silence_ends_the_turn_again_after_the_hold_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(websocket_module, "MAX_UTTERANCE_S", 0.05)
    manager = ConnectionManager()
    stt = FakeStt("what is arasaka")
    manager.stt_engine = stt
    manager.accumulator = short_silence_accumulator()
    engine = MagicMock()
    app = create_app(connection_manager=manager, narration_engine=engine)

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until(ws, listening)
        assert wait_until(lambda: not manager.session.deadline_held)  # the 30 s cap, shortened for the test
        ws.send_bytes(tone(0.2))
        ws.send_bytes(silence(0.15))  # no voice_end: the silence is the turn's end again
        assert wait_until(lambda: engine.ask.called)

    engine.ask.assert_called_once_with("what is arasaka")


def test_the_unrecognized_line_follows_an_empty_transcript() -> None:
    manager = ConnectionManager()
    manager.stt_engine = FakeStt("")
    tts = FakeTts()
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=FakeLLM(), tts=tts, search=FakeSearch())

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "barge_in_trigger"})
        read_until(ws, listening)
        ws.send_bytes(tone(0.3))
        ws.send_json({"type": "voice_end"})
        transcript = read_until(ws, lambda m: m["type"] == "user_speech_transcript")
        assert transcript["transcript"] == ""
        assert wait_until(lambda: "I didn't catch that." in tts.texts)
