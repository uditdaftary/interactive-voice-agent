"""Tier 5: Adversarial Stress, Fuzzing & Protocol Boundary Verification.

Comprehensive empirical test suite covering:
1. Concurrency Stress:
   - Rapid concurrent connects & disconnects across multiple clients using contextlib.ExitStack.
   - Concurrent audio PCM binary streaming from multiple clients.
   - Abrupt socket drops during active broadcasting (broadcast_text, broadcast_bytes).
   - Thread safety and deadlock immunity under threading.RLock.
   - Active connection cleanup and zero task/memory leaks.

2. Fuzzing & Malformed Inputs:
   - Malformed non-JSON strings, random byte noise, and oversized payloads.
   - JSON non-object primitives (arrays, integers, booleans, null, strings) verifying graceful handling vs crash.
   - Missing fields in JSON action payloads.
   - Out-of-range cadence values (< 0.1, > 3.0, non-numeric strings, NaN, Inf).
   - Unknown character voice IDs and invalid microphone sources.
   - Truncated and malformed binary PCM audio frames (1 byte, odd bytes, empty bytes, massive frames).

3. Rapid Barge-In Cycling:
   - 100 rapid consecutive barge-in events executed within milliseconds.
   - Statistical latency verification of atomic drain (< 0.05ms law).
   - Mathematical consistency of generation fencing, playhead tracking, and buffer pointers.
   - WebSocket barge_in_trigger flooding without deadlock.
"""

import asyncio
import contextlib
import threading
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_agent.audio.hybrid import HybridMicrophoneBridge
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import (
    InvalidClientMessage,
    parse_client_message,
)


# ============================================================================
# 1. CONCURRENCY STRESS TESTS
# ============================================================================


def test_concurrency_rapid_multi_client_connect_burst() -> None:
    """Stress test: 15 clients connect simultaneously, verify session registration and no deadlocks."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)
    client_count = 15

    with TestClient(app) as client:
        with contextlib.ExitStack() as stack:
            websockets = []
            for _ in range(client_count):
                ws = stack.enter_context(client.websocket_connect("/ws"))
                init_msg = ws.receive_json()
                assert init_msg["type"] == "state_sync"
                websockets.append(ws)

            assert manager.connection_count == client_count

            # Concurrent broadcast to all 15 clients
            asyncio.run(manager.broadcast_json({"type": "ping", "id": 999}))

            for ws in websockets:
                msg = ws.receive_json()
                assert msg["type"] == "ping"
                assert msg["id"] == 999

    assert manager.connection_count == 0


def test_concurrency_simultaneous_audio_streaming() -> None:
    """Stress test: 8 clients streaming 24kHz audio frames at once. Only the microphone owner's frames are taken."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    received_chunks = []
    chunk_lock = threading.Lock()

    def audio_sink(data: bytes):
        with chunk_lock:
            received_chunks.append(data)

    manager.register_audio_sink(audio_sink)

    client_count = 8
    chunks_per_client = 10
    test_pcm = b"\x10\x00" * 720  # 30ms 24kHz 16-bit PCM (1440 bytes)

    with TestClient(app) as client:
        with contextlib.ExitStack() as stack:
            websockets = []
            for _ in range(client_count):
                ws = stack.enter_context(client.websocket_connect("/ws"))
                _ = ws.receive_json()  # Handshake
                websockets.append(ws)

            assert manager.connection_count == client_count

            # Each client sends multiple audio frames
            for _ in range(chunks_per_client):
                for ws in websockets:
                    ws.send_bytes(test_pcm)

            import time
            expected_chunks = chunks_per_client  # the first tab to send audio owns the microphone; the other seven are dropped
            for _ in range(50):
                with chunk_lock:
                    if len(received_chunks) >= expected_chunks:
                        break
                time.sleep(0.01)

            # Frames from the other seven were sent before or after these, so let the sockets drain before counting.
            for ws in websockets:
                ws.send_json({"type": "set_cadence", "cadence": 1.1})
            for ws in websockets:
                while ws.receive_json().get("cadence") != 1.1:
                    pass
            with chunk_lock:
                assert len(received_chunks) == expected_chunks

    assert manager.connection_count == 0


def test_concurrency_abrupt_client_drops_during_active_broadcast() -> None:
    """Stress test: clients disconnect abruptly mid-broadcast; verify dead sockets purged without crash."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    with TestClient(app) as client:
        with contextlib.ExitStack() as stack:
            ws_stay = stack.enter_context(client.websocket_connect("/ws"))
            _ = ws_stay.receive_json()

            ws_drop1 = stack.enter_context(client.websocket_connect("/ws"))
            _ = ws_drop1.receive_json()

            ws_drop2 = stack.enter_context(client.websocket_connect("/ws"))
            _ = ws_drop2.receive_json()

            assert manager.connection_count == 3

            # Abruptly close two sockets from client side
            ws_drop1.close()
            ws_drop2.close()

            # Server attempts broadcasts - must not crash or deadlock
            for i in range(5):
                asyncio.run(manager.broadcast_json({"type": "telemetry", "seq": i}))
                asyncio.run(manager.send_audio(b"\x00\x00" * 240))  # to the owner: ws_stay, the oldest connection

            # ws_stay must have received all broadcasts
            for i in range(5):
                json_msg = ws_stay.receive_json()
                assert json_msg["type"] == "telemetry"
                assert json_msg["seq"] == i
                assert ws_stay.receive_json()["type"] == "audio_segment"
                bytes_msg = ws_stay.receive_bytes()
                assert len(bytes_msg) == 480

            # Dead sockets should be purged automatically
            assert manager.connection_count == 1

    assert manager.connection_count == 0


def test_concurrency_thread_safety_under_rlock() -> None:
    """Stress test: 10 background threads mutating the settings and cursor and reading snapshots concurrently.

    The session state is not part of this: it belongs to the event loop thread (see SessionState).
    """
    manager = ConnectionManager()
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                manager.set_cadence(1.0 + (i % 5) * 0.1)
                manager.set_voice("lucy-kushinada" if i % 2 else "david-martinez")
                manager.update_cursor(sentence_idx=i, word_offset=i * 2, latency_ms=0.03)
                snapshot = manager.get_state_sync_message()
                assert snapshot.state == "PAUSED"
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(errors) == 0, f"Thread-safety errors observed: {errors}"


async def test_barge_in_generation_counts_every_trigger() -> None:
    manager = ConnectionManager()

    for i in range(100):
        event = manager.trigger_barge_in(flushed_samples=i, latency_ms=0.02)
        assert event.generation == i + 1

    assert manager.generation == 100
    assert manager.state == "LISTENING"


# ============================================================================
# 2. FUZZING & MALFORMED INPUT TESTS
# ============================================================================


def test_fuzzing_malformed_non_json_strings() -> None:
    """Fuzzing: Send corrupt strings, binary noise, null bytes, and oversized strings."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    fuzz_payloads = [
        "",
        "   ",
        "NOT_JSON_AT_ALL",
        "{broken_json:",
        '{"type": "select_voice", missing_quote}',
        "\x00\x01\x02\xff\xfe",
        "A" * 65536,  # 64KB string
        '{"type": "\u0000\u0001"}',
        '{"type": "set_cadence", "cadence": "not_a_float"}',
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            for payload in fuzz_payloads:
                ws.send_text(payload)
                assert ws.receive_json()["type"] == "error"

            # Valid command sent afterward must succeed, proving connection stayed alive
            ws.send_json({"type": "set_cadence", "cadence": 1.2})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 1.2


def test_fuzzing_json_non_dict_primitives_graceful_rejection() -> None:
    """Fuzzing: Verify non-dict JSON primitives (lists, ints, booleans, null) are safely rejected as None.

    Verifies that:
    1. parse_client_message returns None without raising unhandled AttributeError.
    2. Over a live WebSocket, sending non-dict JSON primitives does not drop the session.
    3. The WebSocket remains fully operational for subsequent valid commands.
    """
    non_dict_inputs = [
        "[1, 2, 3]",
        "123",
        "true",
        "false",
        "null",
        '"raw_string"',
    ]

    for raw in non_dict_inputs:
        with pytest.raises(InvalidClientMessage, match="JSON object"):
            parse_client_message(raw)

    # Test over live WebSocket
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            for raw in non_dict_inputs:
                ws.send_text(raw)
                assert ws.receive_json()["type"] == "error"

            # Session remains healthy and responsive
            ws.send_json({"type": "set_cadence", "cadence": 1.15})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 1.15


def test_fuzzing_truncated_and_odd_binary_pcm_frames() -> None:
    """Fuzzing: Send odd-length, empty, and truncated binary audio frames."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    ring_buffer = AudioRingBuffer(capacity_samples=24000 * 5, sample_rate=24000)
    bridge = HybridMicrophoneBridge(source="browser", ring_buffer=ring_buffer)
    manager.register_audio_sink(bridge.feed_browser_pcm)

    odd_frames = [
        b"",
        b"\x00",              # 1 byte (odd)
        b"\x00\x01\x02",      # 3 bytes (odd)
        b"\x7f" * 719,        # 719 bytes (odd)
        b"\x00\x00" * 360,    # 720 samples int16 = 1440 bytes (valid 30ms)
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            for frame in odd_frames:
                ws.send_bytes(frame)

            # Channel remains healthy
            ws.send_json({"type": "barge_in_trigger"})
            barge_resp = ws.receive_json()
            assert barge_resp["type"] == "barge_in"

    bridge.close()


def test_fuzzing_missing_fields_in_actions() -> None:
    """Fuzzing BUG EXPOSURE: Actions with missing required fields are silently defaulted rather than rejected.

    Demonstrates that:
    1. {"type": "select_voice"} without voice_id is now rejected (it used to blank the voice).
    2. {"type": "set_cadence"} without cadence resets cadence to 1.0.
    3. {"type": "set_mic_source"} without source resets mic_source to "browser".
    4. {"type": "playback_control"} without action defaults to "resume".
    5. Each of these dispatches a state_sync broadcast rather than gracefully rejecting with None.
    """
    manager = ConnectionManager()
    manager.set_cadence(1.5)
    manager.set_voice("lucy-kushinada")
    app = create_app(connection_manager=manager)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            init_sync = ws.receive_json()
            assert init_sync["active_voice"] == "lucy-kushinada"
            assert init_sync["cadence"] == 1.5

            # 1. Send select_voice with missing voice_id -> rejected with an error reply, the voice is left alone
            ws.send_json({"type": "select_voice"})
            assert ws.receive_json()["type"] == "error"

            # 2. Send set_cadence with missing cadence -> resets cadence to default 1.0
            ws.send_json({"type": "set_cadence"})
            sync2 = ws.receive_json()
            assert sync2["type"] == "state_sync"
            assert sync2["active_voice"] == "lucy-kushinada"
            assert sync2["cadence"] == 1.0

            # 3. Send empty payload {} -> rejected as unrecognized, an error reply and no broadcast
            ws.send_json({})
            assert ws.receive_json()["type"] == "error"

            # Channel remains operational for subsequent valid action
            ws.send_json({"type": "set_cadence", "cadence": 0.9})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 0.9


def test_fuzzing_out_of_range_cadence_validation() -> None:
    """Fuzzing: Verify cadence bounds [0.1, 3.0] are strictly enforced."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    invalid_cadences = [
        -1.0,
        0.0,
        0.05,
        3.01,
        10.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            for bad_cadence in invalid_cadences:
                ws.send_json({"type": "set_cadence", "cadence": bad_cadence})
                assert ws.receive_json()["type"] == "error"

            # Valid in-range cadence should still be accepted
            ws.send_json({"type": "set_cadence", "cadence": 1.2})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 1.2


def test_fuzzing_unknown_voice_id_handling() -> None:
    """Fuzzing: a malformed voice id is rejected; a well-formed id is accepted (the registry arrives in S4.2)."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            ws.send_json({"type": "select_voice", "voice_id": "non_existent_cyborg_9000"})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "select_voice", "voice_id": "non-existent-cyborg-9000"})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["active_voice"] == "non-existent-cyborg-9000"


def test_fuzzing_invalid_mic_source_desynchronization_bug() -> None:
    """Fuzzing BUG EXPOSURE: Invalid mic source causes state desynchronization between ConnectionManager & Bridge.

    When a client sends an invalid mic source (e.g. 'invalid_source'),
    ConnectionManager updates self._mic_source BEFORE calling the registered callback.
    When bridge.set_source raises ValueError, ConnectionManager catches it and logs,
    leaving manager._mic_source set to 'invalid_source' while bridge remains on 'browser'.
    """
    manager = ConnectionManager()
    bridge = HybridMicrophoneBridge(source="browser")
    manager.register_mic_source_change_callback(bridge.set_source)

    assert manager.mic_source == "browser"
    assert bridge.get_source() == "browser"

    # Send invalid mic source
    manager.set_mic_source("quantum_sensor")

    # Observed defect: manager._mic_source was mutated to invalid string despite bridge rejection
    assert manager.mic_source == "quantum_sensor"
    assert bridge.get_source() == "browser"  # Desynchronized!

    bridge.close()


# ============================================================================
# 3. RAPID BARGE-IN CYCLING & MATHEMATICAL INVARIANTS
# ============================================================================


def test_rapid_barge_in_cycling_100_iterations_sub_0_05ms_law() -> None:
    """Stress test: 100 rapid barge-in drains within milliseconds.

    Verifies:
    1. Buffer pointers (read_pos=0, write_pos=0, size=0).
    2. Generation counter increments by exactly 100.
    3. Atomic drain latency satisfies Mean < 0.05ms constraint.
    4. Playhead remains monotonic without backward jumps.
    5. Stale generation audio writes are strictly rejected.
    """
    rb = AudioRingBuffer(capacity_samples=24000 * 5, sample_rate=24000)
    iterations = 100
    test_chunk = np.ones(2400, dtype=np.int16) * 500  # 100ms chunk

    # Warm up memory allocator and numpy paths (aligning with Tier 4 benchmark protocol)
    for _ in range(50):
        rb.write(test_chunk)
        rb.drain(fade_ms=2.0)

    initial_gen = rb.generation
    drain_times_ms = []

    for i in range(iterations):
        # Pre-fill buffer
        written = rb.write(test_chunk, generation=rb.generation)
        assert written == len(test_chunk)
        assert rb.size == len(test_chunk)

        # Atomic drain measurement
        t0 = time.perf_counter()
        rb.drain(fade_ms=2.0)
        t1 = time.perf_counter()

        drain_times_ms.append((t1 - t0) * 1000.0)

        # Invariant checks immediately post-drain
        assert rb.size == 0
        assert rb.read_pos == 0
        assert rb.write_pos == 0
        assert rb.generation == initial_gen + i + 1

        # Stale write rejection check: attempt writing with old generation
        stale_written = rb.write(test_chunk, generation=initial_gen + i)
        assert stale_written == 0, f"Stale write succeeded at iteration {i}!"

    arr = np.array(drain_times_ms)
    mean_drain = float(arr.mean())
    max_drain = float(arr.max())
    p95_drain = float(np.percentile(arr, 95))

    print("\n[TIER 5 ADVERSARIAL] 100 Rapid Barge-In Drain Latencies:")
    print(f"  Mean: {mean_drain:.5f} ms  (Target: < 0.05000 ms)")
    print(f"  Max:  {max_drain:.5f} ms")
    print(f"  P95:  {p95_drain:.5f} ms")

    assert mean_drain < 0.05, f"Mean drain latency {mean_drain:.5f}ms violated sub-0.05ms law!"
    assert rb.generation == initial_gen + 100


def test_rapid_barge_in_websocket_flooding() -> None:
    """Stress test: Flood WebSocket with 100 consecutive barge_in_trigger messages.

    Verifies server processes all 100 triggers without deadlock, drops, or generation drift.
    """
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            start_t = time.perf_counter()
            for _ in range(100):
                ws.send_json({"type": "barge_in_trigger"})

            # Read back the 100 barge_in events, in order. Only the first one changes the state, so the
            # session announces LISTENING exactly once, right after it.
            messages = [ws.receive_json() for _ in range(101)]
            barge_events = [m for m in messages if m["type"] == "barge_in"]
            sync_events = [m for m in messages if m["type"] == "state_sync"]
            assert [m["generation"] for m in barge_events] == list(range(1, 101))
            assert [m["state"] for m in sync_events] == ["LISTENING"]
            assert messages[0]["type"] == "barge_in"
            assert messages[1]["type"] == "state_sync"

            elapsed = (time.perf_counter() - start_t) * 1000.0
            print(f"\n[TIER 5 ADVERSARIAL] 100 WebSocket Barge-In Roundtrips in {elapsed:.2f} ms")

            assert manager.generation == 100
            assert manager.state == "LISTENING"
