"""The speech accumulator never holds more than one bounded turn, and only captures while armed (H3)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeTts, wait_until
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.stt.accumulator import MAX_UTTERANCE_S, PRE_ROLL_CHUNKS, SpeechTurnAccumulator

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
CAP_BYTES = int(MAX_UTTERANCE_S * BYTES_PER_SECOND)


def tone(duration_s: float, amplitude: int) -> bytes:
    samples = int(duration_s * SAMPLE_RATE)
    wave = np.sin(2 * np.pi * 440 * np.arange(samples) / SAMPLE_RATE) * amplitude
    return wave.astype(np.int16).tobytes()


LOUD_20MS = tone(0.02, 16000)
LOUD_100MS = tone(0.1, 16000)
QUIET_20MS = tone(0.02, 20)  # RMS far below the 0.010 speech threshold


def buffered_bytes(acc: SpeechTurnAccumulator) -> dict[str, int]:
    return {
        "session": sum(len(chunk) for chunk in acc._session_buffer),
        "speech": sum(len(chunk) for chunk in acc._speech_buffer),
        "pre_roll": len(acc._pre_roll_buffer),
    }


def test_constants_match_the_documented_budgets() -> None:
    assert MAX_UTTERANCE_S == 30.0
    assert PRE_ROLL_CHUNKS == 25


def test_buffers_bounded_over_an_hour_while_disarmed() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.disarm()

    for _ in range(int(3600 / 0.02)):
        assert acc.feed(LOUD_20MS) is None

    sizes = buffered_bytes(acc)
    assert sizes["session"] == 0
    assert sizes["speech"] <= CAP_BYTES
    assert sizes["pre_roll"] <= PRE_ROLL_CHUNKS
    assert not acc.is_speaking


def test_buffers_bounded_over_an_hour_while_armed() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.arm()
    utterances: list[bytes] = []
    largest_session = largest_speech = 0

    for _ in range(int(3600 / 0.1)):
        utterance = acc.feed(LOUD_100MS)
        if utterance is not None:
            utterances.append(utterance)
        sizes = buffered_bytes(acc)
        largest_session, largest_speech = max(largest_session, sizes["session"]), max(largest_speech, sizes["speech"])

    assert largest_session <= CAP_BYTES
    assert largest_speech <= CAP_BYTES
    assert len(utterances) == 120  # one hour of unbroken speech is 120 cut turns
    assert {len(utterance) for utterance in utterances} == {CAP_BYTES}


def test_speech_cut_at_30s() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.arm()

    utterances = [u for u in (acc.feed(LOUD_20MS) for _ in range(int(31 / 0.02))) if u is not None]

    assert len(utterances) == 1
    assert len(utterances[0]) == CAP_BYTES
    assert acc.is_speaking  # the last second started the next turn
    assert buffered_bytes(acc)["speech"] == int(1 * BYTES_PER_SECOND)


def test_speech_cut_during_trailing_silence() -> None:
    """The cap also holds when the 30 s mark falls on a pause inside the turn."""
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=10_000)
    acc.arm()
    acc.feed(LOUD_100MS)

    cut = None
    for _ in range(int(31 / 0.1)):
        cut = acc.feed(tone(0.1, 20)) or cut

    assert cut is not None
    assert len(cut) == CAP_BYTES


def test_quiet_ptt_still_returns_whole_turn() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.arm()

    for _ in range(int(5 / 0.02)):
        assert acc.feed(QUIET_20MS) is None

    assert not acc.is_speaking
    flushed = acc.flush()
    assert flushed is not None
    assert len(flushed) == int(5 * BYTES_PER_SECOND)


def test_session_buffer_keeps_only_the_latest_30s() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.arm()
    older, newer = tone(0.02, 20), tone(0.02, 40)

    for _ in range(int(10 / 0.02)):
        acc.feed(older)
    for _ in range(int(30 / 0.02)):
        acc.feed(newer)

    flushed = acc.flush()
    assert flushed is not None
    assert len(flushed) == CAP_BYTES
    assert flushed == newer * int(30 / 0.02)


def test_disarmed_feed_only_keeps_the_pre_roll() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.disarm()

    for _ in range(3):
        assert acc.feed(LOUD_20MS) is None

    assert not acc.is_speaking
    assert buffered_bytes(acc) == {"session": 0, "speech": 0, "pre_roll": 3}

    acc.arm()
    acc.feed(LOUD_20MS)
    assert acc.is_speaking
    assert len(acc._speech_buffer) == 4  # the three remembered chunks plus the one that started speech


def test_disarm_drops_the_turn_but_keeps_the_pre_roll() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.arm()
    for _ in range(5):
        acc.feed(QUIET_20MS)
    assert buffered_bytes(acc)["session"] > 0

    acc.disarm()

    assert buffered_bytes(acc) == {"session": 0, "speech": 0, "pre_roll": 5}
    assert acc.flush() == QUIET_20MS * 5


def test_disarm_ends_a_turn_in_progress() -> None:
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    acc.feed(LOUD_100MS)
    assert acc.is_speaking

    acc.disarm()

    assert not acc.is_speaking
    assert buffered_bytes(acc)["speech"] == 0


def test_a_new_accumulator_is_armed() -> None:
    """Standalone use (no WebSocket driving arm/disarm) keeps capturing turns."""
    acc = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)

    acc.feed(LOUD_100MS)

    assert acc.is_speaking


def test_websocket_arms_the_accumulator_only_while_listening() -> None:
    manager = ConnectionManager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    app = create_app(connection_manager=manager, llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    acc: SpeechTurnAccumulator = manager.accumulator

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        # NARRATING: audio only refreshes the pre-roll.
        for _ in range(20):
            ws.send_bytes(LOUD_20MS)
        assert wait_until(lambda: len(acc._pre_roll_buffer) == 20)
        assert buffered_bytes(acc)["session"] == 0
        assert not acc.is_speaking

        # LISTENING: the same audio is captured as a turn.
        ws.send_json({"type": "barge_in_trigger"})
        ws.receive_json()  # barge_in
        ws.receive_json()  # state_sync LISTENING
        for _ in range(10):
            ws.send_bytes(LOUD_20MS)
        assert wait_until(lambda: buffered_bytes(acc)["session"] == 10 * len(LOUD_20MS))
        assert acc.is_speaking

        # Back to NARRATING: the half-built turn is dropped.
        manager.set_state("NARRATING")
        ws.send_bytes(LOUD_20MS)
        assert wait_until(lambda: buffered_bytes(acc)["session"] == 0 and not acc.is_speaking)


@pytest.mark.parametrize("silent_state", ["NARRATING", "REPLYING", "PAUSED"])
def test_non_listening_states_never_capture(silent_state: str) -> None:
    manager = ConnectionManager()
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE)
    manager.set_state(silent_state)
    app = create_app(connection_manager=manager, llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    acc: SpeechTurnAccumulator = manager.accumulator

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(8):
            ws.send_bytes(LOUD_20MS)
        assert wait_until(lambda: len(acc._pre_roll_buffer) == 8)

    assert buffered_bytes(acc)["session"] == 0
    assert not acc.is_speaking
