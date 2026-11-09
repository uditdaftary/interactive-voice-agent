"""Tier 2: WebSocket Protocol & Handshake Integration Tests.

Validates:
1. WebSocket connection handshake at `/ws` via Starlette TestClient.
2. Immediate transmission of initial `state_sync` JSON handshake message.
3. Client action message dispatch:
   - `select_voice`
   - `set_mic_source`
   - `barge_in_trigger`
   - `set_cadence`
   - `playback_control`
4. Bi-directional 24kHz binary PCM audio frame streaming.
5. Server event broadcasting (VAD telemetry, cursor update, barge-in, prompt envelope).
6. Multi-client session tracking and graceful client disconnection without resource leaks.
7. Defensive handling of malformed client payloads without dropping connection.
"""

import pytest
from fastapi.testclient import TestClient

from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import (
    BargeInEventMessage,
    CursorUpdateMessage,
    PromptEnvelopeMessage,
    TelemetryMessage,
    VADEventMessage,
)


@pytest.fixture
def isolated_app():
    """Provides a fresh FastAPI app instance with an isolated ConnectionManager for each test."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)
    return app, manager


def test_websocket_handshake_and_initial_state_sync(isolated_app) -> None:
    """Test connecting to /ws yields 101 Switching Protocols and immediate state_sync message."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1. First message must be initial state_sync
            init_msg = ws.receive_json()
            assert init_msg["type"] == "state_sync"
            assert init_msg["state"] == "NARRATING"
            assert init_msg["active_voice"] == "david-martinez"
            assert init_msg["mic_source"] == "browser"
            assert init_msg["cadence"] == 1.0
            assert init_msg["sentence_idx"] == 0
            assert init_msg["word_offset"] == 0
            assert "latency_ms" in init_msg

            assert manager.connection_count == 1


def test_client_select_voice_action(isolated_app) -> None:
    """Test select_voice action updates session active_voice and broadcasts state_sync."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Consume initial state_sync

            # Send select_voice
            ws.send_json({"type": "select_voice", "voice_id": "lucy-kushinada"})

            # Receive broadcasted state_sync
            response = ws.receive_json()
            assert response["type"] == "state_sync"
            assert response["active_voice"] == "lucy-kushinada"
            assert manager.active_voice == "lucy-kushinada"


def test_client_set_mic_source_action(isolated_app) -> None:
    """Test set_mic_source action updates session mic_source and broadcasts state_sync."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Consume initial state_sync

            ws.send_json({"type": "set_mic_source", "source": "host"})

            response = ws.receive_json()
            assert response["type"] == "state_sync"
            assert response["mic_source"] == "host"
            assert manager.mic_source == "host"


def test_client_barge_in_trigger_action(isolated_app) -> None:
    """Test barge_in_trigger increments generation, sets LISTENING, and broadcasts events."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Initial state_sync

            ws.send_json({"type": "barge_in_trigger"})

            # 1. Receives BargeInEventMessage
            barge_in_event = ws.receive_json()
            assert barge_in_event["type"] == "barge_in"
            assert barge_in_event["generation"] == 1
            assert "latency_ms" in barge_in_event
            assert "flushed_samples" in barge_in_event

            # 2. Receives updated state_sync with LISTENING
            state_event = ws.receive_json()
            assert state_event["type"] == "state_sync"
            assert state_event["state"] == "LISTENING"
            assert manager.state == "LISTENING"
            assert manager.generation == 1


def test_client_set_cadence_action(isolated_app) -> None:
    """Test set_cadence action updates reading speed multiplier and broadcasts state_sync."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            ws.send_json({"type": "set_cadence", "cadence": 1.25})

            response = ws.receive_json()
            assert response["type"] == "state_sync"
            assert response["cadence"] == 1.25
            assert manager.cadence == 1.25


def test_client_playback_control_pause_and_resume(isolated_app) -> None:
    """Test playback_control toggles state between PAUSED and NARRATING."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            # Pause playback
            ws.send_json({"type": "playback_control", "action": "pause"})
            resp_pause = ws.receive_json()
            assert resp_pause["type"] == "state_sync"
            assert resp_pause["state"] == "PAUSED"
            assert manager.state == "PAUSED"

            # Resume playback
            ws.send_json({"type": "playback_control", "action": "resume"})
            resp_resume = ws.receive_json()
            assert resp_resume["type"] == "state_sync"
            assert resp_resume["state"] == "NARRATING"
            assert manager.state == "NARRATING"


def test_bidirectional_binary_pcm_audio_streaming(isolated_app) -> None:
    """Test client -> server raw PCM ingestion and server -> client PCM streaming."""
    app, manager = isolated_app

    received_frames = []

    def audio_sink(frame: bytes):
        received_frames.append(frame)

    manager.register_audio_sink(audio_sink)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            # Client -> Server: Stream 30ms 24kHz 16-bit PCM (720 samples * 2 bytes = 1440 bytes)
            pcm_chunk = b"\x01\x00" * 720
            ws.send_bytes(pcm_chunk)

            assert len(received_frames) == 1
            assert received_frames[0] == pcm_chunk
            assert len(received_frames[0]) == 1440

            # Server -> Client: Broadcast synthesized binary audio PCM
            synth_chunk = b"\x7f\x00" * 480
            # TestClient loop handles broadcast_bytes inside thread-safe context
            import asyncio
            asyncio.run(manager.broadcast_bytes(synth_chunk))

            client_received = ws.receive_bytes()
            assert client_received == synth_chunk
            assert len(client_received) == 960


def test_server_event_broadcasting(isolated_app) -> None:
    """Test server broadcasting VADEvent, CursorUpdate, Telemetry, and PromptEnvelope."""
    app, manager = isolated_app
    import asyncio

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            # 1. Broadcast VADEventMessage
            vad_msg = VADEventMessage(
                speech_active=True,
                dbfs=-18.4,
                noise_floor_dbfs=-45.2,
                threshold_dbfs=-33.0,
            )
            asyncio.run(manager.broadcast_json(vad_msg))
            r_vad = ws.receive_json()
            assert r_vad["type"] == "vad_event"
            assert r_vad["speech_active"] is True
            assert r_vad["dbfs"] == -18.4

            # 2. Broadcast CursorUpdateMessage
            cursor_msg = CursorUpdateMessage(
                sentence_idx=2,
                word_offset=5,
                char_offset=42,
                total_sentences=7,
                sentence_text="The city never sleeps under the acid neon lights.",
            )
            asyncio.run(manager.broadcast_json(cursor_msg))
            r_cursor = ws.receive_json()
            assert r_cursor["type"] == "cursor_update"
            assert r_cursor["sentence_idx"] == 2
            assert r_cursor["word_offset"] == 5

            # 3. Broadcast TelemetryMessage
            telemetry_msg = TelemetryMessage(
                buffer_samples=12000,
                buffer_capacity=240000,
                drain_latency_ms=0.032,
                rms_db=-12.5,
                fft_bins=[0.1, 0.2, 0.3, 0.4],
            )
            asyncio.run(manager.broadcast_json(telemetry_msg))
            r_telem = ws.receive_json()
            assert r_telem["type"] == "telemetry"
            assert r_telem["buffer_samples"] == 12000
            assert r_telem["drain_latency_ms"] == 0.032

            # 4. Broadcast PromptEnvelopeMessage
            prompt_msg = PromptEnvelopeMessage(
                state="LISTENING",
                context="Cyberpunk narrative context",
                query="Who are you?",
                reply="I am your offline assistant.",
                pre_roll_words=3,
            )
            asyncio.run(manager.broadcast_json(prompt_msg))
            r_prompt = ws.receive_json()
            assert r_prompt["type"] == "prompt_envelope"
            assert r_prompt["query"] == "Who are you?"


def test_multi_client_session_and_graceful_disconnect(isolated_app) -> None:
    """Test multiple simultaneous clients receive broadcasts and handle disconnections cleanly."""
    app, manager = isolated_app
    import asyncio

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws1:
            _ = ws1.receive_json()
            assert manager.connection_count == 1

            with client.websocket_connect("/ws") as ws2:
                _ = ws2.receive_json()
                assert manager.connection_count == 2

                # Broadcast to both clients
                asyncio.run(manager.broadcast_json({"type": "test_broadcast", "data": "hello"}))

                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["data"] == "hello"
                assert msg2["data"] == "hello"

            # ws2 has exited and disconnected
            assert manager.connection_count == 1

        # ws1 has exited and disconnected
        assert manager.connection_count == 0


def test_malformed_client_payload_resilience(isolated_app) -> None:
    """Test server does not drop connection on invalid JSON or unrecognized message types."""
    app, manager = isolated_app
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()

            # Send raw non-JSON text
            ws.send_text("MALFORMED NON-JSON STRING")

            # Send unrecognized JSON message
            ws.send_json({"type": "non_existent_op_code", "foo": "bar"})

            # Connection remains open and responsive to valid action
            ws.send_json({"type": "set_cadence", "cadence": 1.1})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 1.1
