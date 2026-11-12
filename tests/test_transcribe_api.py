"""Tests for REST API Endpoint POST /api/transcribe.

Validates:
1. POST /api/transcribe with uploaded WAV audio files.
2. POST /api/transcribe with uploaded raw PCM audio files.
3. Response schema: {"transcript": str, "duration_sec": float, "status": "ok"}.
4. Error handling for empty files (returns 400).
5. Error handling for missing file parameter (returns 422).
6. Integration with the STT engine (FakeStt here).
"""

import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_agent.server.app import create_app
from fakes import FakeStt


def make_synthetic_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory WAV audio bytes with a 440Hz tone."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    audio_int16 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@pytest.fixture
def mock_client():
    """Provides a TestClient configured with FakeStt."""
    mock_stt = FakeStt(fixed_transcript="cyberpunk voice agent online")
    app = create_app(stt_engine=mock_stt)
    with TestClient(app) as client:
        yield client


def test_transcribe_wav_file(mock_client: TestClient) -> None:
    """Test POST /api/transcribe with standard WAV file upload."""
    wav_bytes = make_synthetic_wav(duration_sec=1.5, sample_rate=16000)
    files = {"file": ("test_speech.wav", wav_bytes, "audio/wav")}

    resp = mock_client.post("/api/transcribe", files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["transcript"] == "cyberpunk voice agent online"
    assert pytest.approx(data["duration_sec"], rel=1e-2) == 1.5


def test_transcribe_raw_pcm_file(mock_client: TestClient) -> None:
    """Test POST /api/transcribe with raw PCM binary upload."""
    # 1.0 second of 16kHz 16-bit mono = 32000 bytes
    pcm_bytes = (np.zeros(16000, dtype=np.int16) + 1000).tobytes()
    files = {"file": ("audio.pcm", pcm_bytes, "application/octet-stream")}

    resp = mock_client.post("/api/transcribe", files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["transcript"] == "cyberpunk voice agent online"
    assert pytest.approx(data["duration_sec"], rel=1e-2) == 1.0


def test_transcribe_empty_file_returns_400(mock_client: TestClient) -> None:
    """Test POST /api/transcribe with empty file returns HTTP 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}

    resp = mock_client.post("/api/transcribe", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_transcribe_missing_file_returns_422(mock_client: TestClient) -> None:
    """Test POST /api/transcribe without file part returns HTTP 422."""
    resp = mock_client.post("/api/transcribe")
    assert resp.status_code == 422


def test_transcribe_with_default_app(offline_stt: None) -> None:
    """Test POST /api/transcribe with default app configuration (STT built on first use, no model available)."""
    app = create_app()
    wav_bytes = make_synthetic_wav(duration_sec=0.5, sample_rate=16000)
    files = {"file": ("query.wav", wav_bytes, "audio/wav")}

    with TestClient(app) as client:
        resp = client.post("/api/transcribe", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # With no Whisper model and no recognizer, the real engine reports an empty transcript.
        assert data["transcript"] == ""
        assert pytest.approx(data["duration_sec"], rel=1e-2) == 0.5
