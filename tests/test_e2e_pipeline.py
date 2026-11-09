"""Tier 3: End-to-End Pipeline, Streaming & Loopback Integration Tests.

Validates the full system integration across:
- `create_app()` and `ConnectionManager`
- `HybridMicrophoneBridge`
- `VirtualAudioDevice` (headless audio loopback)
- `AudioRingBuffer` (sub-0.05ms atomic drainage & generation fencing)
- `VoiceActivityDetector` (16kHz 30ms frames, adaptive noise floor, DC blocker)
- `CursorTracker` (word-level reading cursor with pre-roll resumption)

Coverage:
1. End-to-end 30ms VAD event telemetry broadcast over WebSocket.
2. Word cursor tracking and playhead synchronization over WebSocket.
3. Hybrid mic toggle switching ("browser" vs "host" vs "virtual") without stream corruption.
4. Automatic virtual loopback fallback in headless/device-less environments.
5. End-to-end barge-in interruption flow: speech frame -> VAD onset -> atomic buffer drain -> WebSocket broadcast.
"""

import asyncio
from typing import List
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_agent.audio.hybrid import HybridMicrophoneBridge
from voice_agent.audio.loopback import VirtualAudioDevice
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.cursor.tracker import CursorTracker
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import CursorUpdateMessage, VADEventMessage
from voice_agent.vad.energy_vad import VoiceActivityDetector


def safe_broadcast_json(manager: ConnectionManager, message) -> None:
    """Thread-safe and loop-safe broadcast helper for sync callbacks and tests."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(manager.broadcast_json(message))
    except RuntimeError:
        asyncio.run(manager.broadcast_json(message))



@pytest.fixture
def e2e_system():
    """Builds a complete, integrated system harness wiring audio DSP, VAD, cursor, and WebSocket."""
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)

    ring_buffer = AudioRingBuffer(capacity_samples=24000 * 5, sample_rate=24000)
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    virtual_device = VirtualAudioDevice(sample_rate=24000, chunk_size=720)

    bridge = HybridMicrophoneBridge(
        source="browser",
        input_sample_rate=24000,
        target_sample_rate=16000,
        frame_duration_ms=30,
        ring_buffer=ring_buffer,
        vad=vad,
        virtual_device=virtual_device,
        auto_fallback=True,
    )

    story_sentences = [
        "In the neon-soaked alleys of Night City, rain drummed against the pavement.",
        "David Martinez adjusted his cyberware jacket as sirens echoed in the distance.",
        "Lucy stood quietly on the rooftop watching the orbital shuttles rise into the sky.",
    ]
    cursor = CursorTracker(story_sentences)

    # Wire Web Audio raw PCM from WebSocket client into the bridge
    manager.register_audio_sink(bridge.feed_browser_pcm)

    # Wire mic source change from WebSocket into bridge
    manager.register_mic_source_change_callback(bridge.set_source)

    # Wire barge-in callback from bridge into ConnectionManager and ring buffer
    def on_bridge_barge_in(barge_in_event):
        manager.trigger_barge_in(
            flushed_samples=barge_in_event.flushed_samples,
            latency_ms=barge_in_event.drain_latency_ms,
        )

    bridge.add_barge_in_callback(on_bridge_barge_in)

    yield {
        "app": app,
        "manager": manager,
        "bridge": bridge,
        "ring_buffer": ring_buffer,
        "vad": vad,
        "virtual_device": virtual_device,
        "cursor": cursor,
    }

    bridge.close()


def test_e2e_30ms_vad_event_broadcast(e2e_system) -> None:
    """Test 30ms VAD events are calculated by bridge and broadcast to WebSocket client."""
    app = e2e_system["app"]
    manager = e2e_system["manager"]
    bridge: HybridMicrophoneBridge = e2e_system["bridge"]

    received_vad_events: List[VADEventMessage] = []

    def on_vad_telemetry(telem):
        msg = VADEventMessage(
            speech_active=telem.speech_active,
            dbfs=telem.dbfs,
            noise_floor_dbfs=telem.noise_floor_dbfs,
            threshold_dbfs=telem.threshold_dbfs,
        )
        received_vad_events.append(msg)
        safe_broadcast_json(manager, msg)

    bridge.add_vad_callback(on_vad_telemetry)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake state_sync

            # Stream 2 frames of 24kHz 16-bit PCM (720 samples each = 1440 bytes per frame)
            # Frame 1: Silence
            silence_frame = (np.zeros(720, dtype=np.int16)).tobytes()
            ws.send_bytes(silence_frame)

            # Receive VAD event
            vad_payload1 = ws.receive_json()
            assert vad_payload1["type"] == "vad_event"
            assert vad_payload1["speech_active"] is False
            assert vad_payload1["dbfs"] < -60.0

            # Frame 2: Loud 440Hz tone (speech signal)
            t = np.linspace(0, 0.03, 720, endpoint=False)
            sine_samples = (0.7 * 32767 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
            ws.send_bytes(sine_samples.tobytes())

            vad_payload2 = ws.receive_json()
            assert vad_payload2["type"] == "vad_event"
            assert vad_payload2["dbfs"] > -10.0


def test_e2e_word_cursor_tracking_and_resumption(e2e_system) -> None:
    """Test playhead progress drives cursor advance, WebSocket updates, and pre-roll cues."""
    app = e2e_system["app"]
    manager = e2e_system["manager"]
    cursor: CursorTracker = e2e_system["cursor"]
    ring_buffer: AudioRingBuffer = e2e_system["ring_buffer"]

    assert cursor.current_sentence().startswith("In the neon-soaked")

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            # Simulate writing 24,000 samples (1.0 second @ 24kHz) to ring buffer
            audio_samples = np.zeros(24000, dtype=np.int16)
            ring_buffer.write(audio_samples)

            # Simulate playback consumption of 1.0 second (24,000 samples)
            _ = ring_buffer.read(24000)

            assert ring_buffer.playhead_seconds >= 1.0

            # At 150 WPM (2.5 words/sec), 1.0s elapsed is word offset 2
            word_offset = int(ring_buffer.playhead_seconds * 2.5)
            cursor.record_progress(word_offset=word_offset)
            manager.update_cursor(
                sentence_idx=cursor.pos.sentence_idx,
                word_offset=cursor.pos.word_offset,
            )

            # Broadcast cursor update to WebSocket
            cursor_msg = CursorUpdateMessage(
                sentence_idx=cursor.pos.sentence_idx,
                word_offset=cursor.pos.word_offset,
                char_offset=0,
                total_sentences=len(cursor.sentences),
                sentence_text=cursor.current_sentence(),
            )
            safe_broadcast_json(manager, cursor_msg)

            ws_msg = ws.receive_json()
            assert ws_msg["type"] == "cursor_update"
            assert ws_msg["sentence_idx"] == 0
            assert ws_msg["word_offset"] == word_offset
            assert "neon-soaked" in ws_msg["sentence_text"]

            # Verify pre-roll resumption text
            resumption = cursor.get_resumption_text(pre_roll_words=2)
            assert len(resumption) > 0
            assert "neon-soaked" in resumption or "alleys" in resumption


def test_e2e_hybrid_mic_switching_without_corruption(e2e_system) -> None:
    """Test switching between browser, virtual, and host mic modes preserves stream integrity."""
    app = e2e_system["app"]
    manager = e2e_system["manager"]
    bridge: HybridMicrophoneBridge = e2e_system["bridge"]
    virtual_dev: VirtualAudioDevice = e2e_system["virtual_device"]

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            # 1. Start in browser mode
            assert bridge.get_source() == "browser"

            # 2. Switch to virtual mode via WebSocket client action
            ws.send_json({"type": "set_mic_source", "source": "virtual"})
            resp = ws.receive_json()
            assert resp["type"] == "state_sync"
            assert resp["mic_source"] == "virtual"
            assert bridge.get_source() == "virtual"

            # Virtual mode generates frames
            virtual_dev.set_signal_mode("sine", sine_freq=440.0)
            processed_virtual = bridge.step_virtual()
            assert processed_virtual >= 1

            # 3. Switch back to browser mode
            ws.send_json({"type": "set_mic_source", "source": "browser"})
            resp2 = ws.receive_json()
            assert resp2["type"] == "state_sync"
            assert resp2["mic_source"] == "browser"
            assert bridge.get_source() == "browser"

            # Feed browser PCM
            test_chunk = np.zeros(720, dtype=np.int16).tobytes()
            processed_browser = bridge.feed_browser_pcm(test_chunk)
            assert processed_browser >= 1


def test_e2e_automatic_virtual_loopback_fallback(e2e_system) -> None:
    """Test switching to 'host' in headless environment automatically falls back to 'virtual'."""
    bridge: HybridMicrophoneBridge = e2e_system["bridge"]

    # Even in test environment without physical audio card, set_source("host") must not crash
    effective_source = bridge.set_source("host")
    assert effective_source in ("host", "virtual")

    # If physical sound card is absent, fallback must engage
    if effective_source == "virtual":
        assert bridge.is_fallback_active is True

    # Stepping virtual audio continues to operate reliably
    frames = bridge.step_virtual()
    assert frames >= 1


def test_e2e_barge_in_interruption_pipeline(e2e_system) -> None:
    """Test speech onset triggers atomic ring buffer drain and WebSocket barge-in broadcast."""
    app = e2e_system["app"]
    manager = e2e_system["manager"]
    bridge: HybridMicrophoneBridge = e2e_system["bridge"]
    ring_buffer: AudioRingBuffer = e2e_system["ring_buffer"]

    # Fill ring buffer with active speech audio (e.g. 2 seconds = 48,000 samples)
    active_audio = (np.sin(np.linspace(0, 100, 48000)) * 16000).astype(np.int16)
    ring_buffer.write(active_audio)
    assert ring_buffer.size == 48000

    barge_in_broadcasts = []

    def on_barge_in_event(event):
        barge_in_broadcasts.append(event)
        # Broadcast barge-in event and state_sync to WebSocket client
        barge_msg = manager.trigger_barge_in(
            flushed_samples=event.flushed_samples,
            latency_ms=event.drain_latency_ms,
        )
        safe_broadcast_json(manager, barge_msg)
        safe_broadcast_json(manager, manager.get_state_sync_message())

    bridge.add_barge_in_callback(on_barge_in_event)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # Handshake

            # Generate speech burst (onset >= 2 frames)
            t = np.linspace(0, 0.03, 720, endpoint=False)
            loud_sine = (0.8 * 32767 * np.sin(2 * np.pi * 300.0 * t)).astype(np.int16).tobytes()

            # Send Frame 1
            ws.send_bytes(loud_sine)
            # Send Frame 2 (speech onset detected here)
            ws.send_bytes(loud_sine)

            # Client receives barge_in event
            msg_barge = ws.receive_json()
            assert msg_barge["type"] == "barge_in"
            assert msg_barge["flushed_samples"] == 48000
            assert msg_barge["latency_ms"] < 50.0  # Must be sub-50ms

            # Client receives state_sync with LISTENING
            msg_state = ws.receive_json()
            assert msg_state["type"] == "state_sync"
            assert msg_state["state"] == "LISTENING"

            # Verify ring buffer was drained atomically
            assert ring_buffer.size == 0
            assert ring_buffer.generation >= 1
