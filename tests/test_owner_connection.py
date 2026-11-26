"""One connection owns the microphone, and the other tabs get state and text but no say over the turn (M11, N19, #86)."""

import json
import logging
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeTts, RecordingSocket, wait_for_state, wait_until
from voice_agent.orchestrator.state_machine import SessionState
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.websocket import handle_client_text_message
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

LOUD = (np.sin(2 * np.pi * 440 * np.arange(320) / 16000) * 16000).astype(np.int16).tobytes()


class FailingSocket(RecordingSocket):
    """A tab whose connection has died: every send fails."""

    async def send_text(self, text: str) -> None:
        raise RuntimeError("connection lost")


async def tabs(manager: ConnectionManager, count: int) -> list[RecordingSocket]:
    sockets = [RecordingSocket() for _ in range(count)]
    for socket in sockets:
        await manager.connect(socket)  # type: ignore[arg-type]
    return sockets


async def say(manager: ConnectionManager, socket: RecordingSocket, message: dict[str, Any]) -> None:
    await handle_client_text_message(json.dumps(message), socket, manager)  # type: ignore[arg-type]


# --- who owns the microphone -----------------------------------------------------------------------------------------


async def test_the_oldest_connection_owns_the_microphone_until_one_claims_it() -> None:
    manager = ConnectionManager()
    first, second = await tabs(manager, 2)

    assert manager.owner is first  # a tab with no microphone permission still hears the narration
    assert manager.is_owner(first) and not manager.is_owner(second)


async def test_the_first_tab_to_send_audio_keeps_the_microphone() -> None:
    manager = ConnectionManager()
    first, second = await tabs(manager, 2)

    assert manager.claim_ownership(second, explicit=False)  # nobody had claimed it: the first tab with audio does
    assert manager.owner is second
    assert not manager.claim_ownership(first, explicit=False)  # later audio does not take it back
    assert manager.owner is second


async def test_a_later_set_mic_source_takes_the_microphone_over() -> None:
    manager = ConnectionManager()
    first, second = await tabs(manager, 2)
    manager.claim_ownership(first, explicit=False)

    await say(manager, second, {"type": "set_mic_source", "source": "browser"})

    assert manager.owner is second


async def test_the_oldest_remaining_connection_takes_over_when_the_owner_leaves() -> None:
    manager = ConnectionManager()
    oldest, owner, newest = await tabs(manager, 3)
    manager.claim_ownership(owner, explicit=True)

    manager.disconnect(owner)  # type: ignore[arg-type]

    assert manager.owner is oldest
    assert newest in manager.active_connections


async def test_a_connection_dropped_by_a_failed_send_hands_the_microphone_on() -> None:
    manager = ConnectionManager()
    healthy = RecordingSocket()
    dying = FailingSocket()
    await manager.connect(dying)  # type: ignore[arg-type]
    await manager.connect(healthy)  # type: ignore[arg-type]
    manager.claim_ownership(dying, explicit=True)  # type: ignore[arg-type]

    await manager.broadcast_json({"type": "ping"})

    assert manager.active_connections == [healthy]
    assert manager.owner is healthy
    await wait_for_owner_announcement(manager, healthy)


async def wait_for_owner_announcement(manager: ConnectionManager, socket: RecordingSocket) -> None:
    await socket.wait_until(lambda: any(message["mic_owner"] for message in socket.messages("state_sync")))
    await manager.tasks.cancel_all()


async def test_each_tab_is_told_whether_it_owns_the_microphone_in_one_broadcast() -> None:
    manager = ConnectionManager()
    owner, other = await tabs(manager, 2)

    await manager.broadcast_state()

    assert [message["mic_owner"] for message in owner.messages("state_sync")] == [True]
    assert [message["mic_owner"] for message in other.messages("state_sync")] == [False]


async def test_the_handshake_snapshot_says_whether_the_tab_owns_the_microphone() -> None:
    manager = ConnectionManager()
    owner, other = await tabs(manager, 2)

    await manager.send_state_sync(owner)  # type: ignore[arg-type]
    await manager.send_state_sync(other)  # type: ignore[arg-type]

    assert owner.messages("state_sync")[0]["mic_owner"] is True
    assert other.messages("state_sync")[0]["mic_owner"] is False


async def test_a_new_owner_is_announced_to_every_tab_and_to_the_listeners() -> None:
    manager = ConnectionManager()
    first, second = await tabs(manager, 2)
    changes: list[int] = []
    manager.add_owner_listener(lambda: changes.append(1))

    manager.claim_ownership(second, explicit=True)
    await first.wait_until(lambda: bool(first.messages("state_sync")))
    await second.wait_until(lambda: bool(second.messages("state_sync")))

    assert changes == [1]
    assert first.messages("state_sync")[-1]["mic_owner"] is False
    assert second.messages("state_sync")[-1]["mic_owner"] is True
    manager.claim_ownership(second, explicit=True)  # already the owner: nothing changed, nobody is told again
    assert changes == [1]
    await manager.tasks.cancel_all()


# --- what a tab that is not the owner may do -------------------------------------------------------------------------


async def test_a_tab_that_does_not_own_the_microphone_cannot_start_or_end_the_turn() -> None:
    manager = ConnectionManager()
    owner, other = await tabs(manager, 2)

    await say(manager, other, {"type": "barge_in_trigger"})
    assert manager.state == "PAUSED"  # not LISTENING: the other tab's Speak did nothing

    await say(manager, owner, {"type": "barge_in_trigger"})
    assert manager.state == "LISTENING" and manager.session.deadline_held
    await say(manager, other, {"type": "voice_end"})
    assert manager.session.deadline_held  # the owner's push-to-talk is still going; the other tab cannot end it
    assert manager.state == "LISTENING"
    manager.session.transition("PAUSED", "test over")
    await manager.tasks.cancel_all()


async def test_each_dropped_connection_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()
    _owner, other = await tabs(manager, 2)

    with caplog.at_level(logging.INFO, logger="voice_agent.server.connection_manager"):
        for message in ({"type": "barge_in_trigger"}, {"type": "voice_end"}, {"type": "barge_in_trigger"}):
            await say(manager, other, message)
        manager.ignore_from_non_owner(other, "microphone audio")  # type: ignore[arg-type]

    lines = [record for record in caplog.records if "not the microphone owner" in record.getMessage()]
    assert len(lines) == 1
    assert "barge_in_trigger" in lines[0].getMessage()


async def test_a_tab_that_does_not_own_the_microphone_can_still_ask_pause_and_choose() -> None:
    manager = ConnectionManager()
    engine = NarrationEngine(
        manager, llm_client=FakeLLM(), settings=Settings.fast(), tts=FakeTts(), search=FakeSearch()
    )
    _owner, other = await tabs(manager, 2)

    await handle_client_text_message(
        json.dumps({"type": "query_interruption", "query": "who is lucy"}), other, manager, engine  # type: ignore[arg-type]
    )
    assert manager.state == "REPLYING"  # a typed question works from any tab
    await manager.tasks.cancel_all()

    await say(manager, other, {"type": "set_cadence", "cadence": 1.5})
    await say(manager, other, {"type": "select_voice", "voice_id": "lucy-kushinada"})
    assert manager.cadence == 1.5 and manager.active_voice == "lucy-kushinada"


def test_a_second_tabs_microphone_frames_never_reach_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()
    frames: list[bytes] = []
    manager.register_audio_sink(frames.append)
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())

    with TestClient(app) as client, caplog.at_level(logging.INFO, logger="voice_agent.server.connection_manager"):
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first.receive_json()
            second.receive_json()
            first.send_bytes(b"\x01\x00" * 10)
            assert wait_until(lambda: len(frames) == 1)
            for _ in range(3):
                second.send_bytes(b"\x02\x00" * 10)
            second.send_json({"type": "set_cadence", "cadence": 1.4})  # a marker: the frames before it were handled
            assert second.receive_json()["cadence"] == 1.4

    assert frames == [b"\x01\x00" * 10]
    lines = [record for record in caplog.records if "microphone audio" in record.getMessage()]
    assert len(lines) == 1  # three frames, one line


# --- a hold that outlives its client (#86) -----------------------------------------------------------------------------


async def test_a_push_to_talk_hold_ends_with_the_tab_that_made_it() -> None:
    manager = ConnectionManager(session=SessionState(listen_deadline_s=0.05, followup_window_s=0.05))
    manager.accumulator = SpeechTurnAccumulator(sample_rate=16000)
    owner, _other = await tabs(manager, 2)
    await say(manager, owner, {"type": "barge_in_trigger"})
    manager.accumulator.arm()
    for _ in range(5):
        manager.accumulator.feed(LOUD)
    assert manager.state == "LISTENING" and manager.session.deadline_held

    manager.disconnect(owner)  # type: ignore[arg-type]

    assert not manager.session.deadline_held
    assert manager.accumulator.flush() is None  # the departed tab's half-recorded turn is not the next owner's
    await wait_for_state(manager.session, "PAUSED")  # the normal deadline runs, instead of 30 s of listening to nobody
    await manager.tasks.cancel_all()
