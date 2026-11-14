"""Tests for Server-Side Neural STT WebSocket Streaming Protocol & Pipeline."""

from unittest.mock import AsyncMock, MagicMock
import numpy as np
from fastapi.testclient import TestClient

from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import (
    UserSpeechTranscriptMessage,
    VoiceEndMessage,
    parse_client_message,
)
from fakes import FakeStt
from voice_agent.stt.accumulator import SpeechTurnAccumulator


# ============================================================================
# Schema Tests
# ============================================================================


def test_user_speech_transcript_schema() -> None:
    """Test UserSpeechTranscriptMessage serialization, defaults, and deserialization."""
    msg = UserSpeechTranscriptMessage(
        transcript="what is arasaka",
        is_final=True,
        latency_ms=45.2,
    )
    assert msg.type == "user_speech_transcript"
    assert msg.transcript == "what is arasaka"
    assert msg.is_final is True
    assert msg.latency_ms == 45.2

    # Serialization check
    data = msg.model_dump()
    assert data["type"] == "user_speech_transcript"
    assert data["transcript"] == "what is arasaka"
    assert data["is_final"] is True
    assert data["latency_ms"] == 45.2

    # Defaults check
    msg_default = UserSpeechTranscriptMessage(transcript="hello night city")
    assert msg_default.is_final is True
    assert msg_default.latency_ms is None

    # Deserialization check
    parsed = UserSpeechTranscriptMessage.model_validate(data)
    assert parsed.transcript == "what is arasaka"
    assert parsed.is_final is True
    assert parsed.latency_ms == 45.2


def test_voice_end_message_parsing() -> None:
    """Test parse_client_message supports voice_end and its aliases."""
    msg = parse_client_message({"type": "voice_end"})
    assert isinstance(msg, VoiceEndMessage)
    assert msg.type == "voice_end"

    # JSON string payload
    msg_str = parse_client_message('{"type": "voice_end"}')
    assert isinstance(msg_str, VoiceEndMessage)

    # Alias speech_end
    msg_alias = parse_client_message({"type": "speech_end"})
    assert isinstance(msg_alias, VoiceEndMessage)


# ============================================================================
# WebSocket Bi-Directional STT Streaming Integration Tests
# ============================================================================


def test_websocket_voice_end_triggers_stt_transcription_and_interruption() -> None:
    """Test streaming PCM chunks during LISTENING and sending voice_end broadcasts transcript."""
    manager = ConnectionManager()
    mock_stt = FakeStt("what is arasaka")
    manager.stt_engine = mock_stt

    mock_engine = MagicMock()
    mock_engine.handle_user_interruption = AsyncMock()

    app = create_app(connection_manager=manager, narration_engine=mock_engine)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # 1. Receive initial state_sync
            init_msg = ws.receive_json()
            assert init_msg["type"] == "state_sync"

            # 2. Trigger barge-in -> state switches to LISTENING
            ws.send_json({"type": "barge_in_trigger"})
            _ = ws.receive_json()  # barge_in event
            state_msg = ws.receive_json()  # state_sync
            assert state_msg["state"] == "LISTENING"

            # 3. Stream 16kHz speech PCM chunk
            t = np.linspace(0, 0.5, 8000, endpoint=False)
            speech_chunk = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
            ws.send_bytes(speech_chunk)

            # 4. Client signals end of voice turn
            ws.send_json({"type": "voice_end"})

            # 5. Expect UserSpeechTranscriptMessage broadcast
            transcript_event = ws.receive_json()
            assert transcript_event["type"] == "user_speech_transcript"
            assert transcript_event["transcript"] == "what is arasaka"
            assert transcript_event["is_final"] is True
            assert "latency_ms" in transcript_event

            # 6. Verify narration engine handle_user_interruption called
            mock_engine.handle_user_interruption.assert_called_once_with("what is arasaka")


def test_websocket_turn_accumulator_automatic_speech_segmentation() -> None:
    """Test accumulator detecting conversational silence timeout and triggering STT automatically."""
    manager = ConnectionManager()
    manager.stt_engine = FakeStt("query via auto pause")
    # Low silence timeout for fast test execution
    manager.accumulator = SpeechTurnAccumulator(
        sample_rate=16000,
        energy_threshold=0.02,
        silence_timeout_ms=100,
        min_speech_duration_ms=100,
    )

    mock_engine = MagicMock()
    mock_engine.handle_user_interruption = AsyncMock()

    app = create_app(connection_manager=manager, narration_engine=mock_engine)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            # Trigger barge-in
            ws.send_json({"type": "barge_in_trigger"})
            _ = ws.receive_json()  # barge_in
            _ = ws.receive_json()  # state_sync

            # Send 200ms speech audio (3200 samples @ 16kHz)
            t = np.linspace(0, 0.2, 3200, endpoint=False)
            speech_chunk = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
            ws.send_bytes(speech_chunk)

            # Send 150ms silence chunk (2400 samples @ 16kHz) to exceed 100ms silence timeout
            silence_chunk = np.zeros(2400, dtype=np.int16).tobytes()
            ws.send_bytes(silence_chunk)

            # Receive automatically broadcasted transcript
            transcript_event = ws.receive_json()
            assert transcript_event["type"] == "user_speech_transcript"
            assert transcript_event["transcript"] == "query via auto pause"
            assert transcript_event["is_final"] is True

            mock_engine.handle_user_interruption.assert_called_once_with("query via auto pause")


def test_websocket_binary_pcm_ignored_when_narrating() -> None:
    """Test audio frames received when state is NARRATING do not trigger transcription."""
    manager = ConnectionManager()
    manager.stt_engine = FakeStt("should not transcribe")

    mock_engine = MagicMock()
    mock_engine.handle_user_interruption = AsyncMock()

    app = create_app(connection_manager=manager, narration_engine=mock_engine)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            assert manager.state == "NARRATING"

            # Send speech audio while NARRATING
            speech_chunk = (np.ones(3200, dtype=np.int16) * 10000).tobytes()
            ws.send_bytes(speech_chunk)

            # Send voice_end
            ws.send_json({"type": "voice_end"})

            # Send harmless action to verify connection is alive
            ws.send_json({"type": "set_cadence", "cadence": 1.2})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["cadence"] == 1.2

            # handle_user_interruption should NOT have been called
            mock_engine.handle_user_interruption.assert_not_called()
