"""Tests for REST API Endpoint POST /api/transcribe.

Validates:
1. POST /api/transcribe with uploaded WAV audio files.
2. POST /api/transcribe with uploaded raw PCM audio files.
3. Response schema: {"transcript": str, "duration_sec": float, "status": "ok"}.
4. Error handling for empty files (returns 400).
5. Error handling for missing file parameter (returns 422).
6. Integration with STTEngine / MockSTTEngine.
"""

import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_agent.server.app import create_app
from voice_agent.stt.recognizer import MockSTTEngine


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
    """Provides a TestClient configured with MockSTTEngine."""
    mock_stt = MockSTTEngine(fixed_transcript="cyberpunk voice agent online")
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


# ---------------------------------------------------------------------------
# Audio validation and resampling (M9)
# ---------------------------------------------------------------------------


class RecordingStt:
    """Records every buffer it is asked to transcribe, as int16 samples."""

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def transcribe(self, pcm: bytes, *args: object, **kwargs: object) -> str:
        self.received.append(np.frombuffer(pcm, dtype=np.int16))
        return "heard"


def make_wav(samples: np.ndarray, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def recording() -> tuple[TestClient, RecordingStt]:
    stt = RecordingStt()
    return TestClient(create_app(stt_engine=stt)), stt  # no lifespan needed: nothing here starts background work


def post_audio(client: TestClient, name: str, content: bytes):
    return client.post("/api/transcribe", files={"file": (name, content, "application/octet-stream")})


def test_transcribe_44k_wav_resampled(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording
    tone = (np.sin(2 * np.pi * 440 * np.arange(44100) / 44100) * 12000).astype(np.int16)

    resp = post_audio(client, "tone.wav", make_wav(tone, sample_rate=44100))

    assert resp.status_code == 200
    assert resp.json()["duration_sec"] == 1.0
    assert [len(received) for received in stt.received] == [16000]
    # The 440 Hz tone survives the resample: 16 kHz output still has most of its energy.
    assert np.abs(stt.received[0]).max() > 9000


def test_transcribe_8khz_wav_resampled_up(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording

    resp = post_audio(client, "phone.wav", make_wav(np.zeros(8000, dtype=np.int16), sample_rate=8000))

    assert resp.status_code == 200
    assert [len(received) for received in stt.received] == [16000]


def test_transcribe_stereo_wav_is_downmixed(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording
    left, right = np.full(1600, 1000, dtype=np.int16), np.full(1600, 3000, dtype=np.int16)
    stereo = np.column_stack([left, right]).astype(np.int16)

    resp = post_audio(client, "stereo.wav", make_wav(stereo, sample_rate=16000, channels=2))

    assert resp.status_code == 200
    assert len(stt.received[0]) == 1600
    assert set(stt.received[0].tolist()) == {2000}


def test_transcribe_8bit_wav_415(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording

    resp = post_audio(client, "eight.wav", make_wav(np.full(1600, 128, dtype=np.uint8), sample_rate=16000, sample_width=1))

    assert resp.status_code == 415
    assert "16-bit" in resp.json()["detail"]
    assert stt.received == []


def test_transcribe_float_wav_415(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording
    wav = bytearray(make_wav(np.zeros(1600, dtype=np.int16), sample_rate=16000))
    wav[20:22] = (3).to_bytes(2, "little")  # format tag 3: IEEE float

    resp = post_audio(client, "float.wav", bytes(wav))

    assert resp.status_code == 415
    assert stt.received == []


@pytest.mark.parametrize("sample_rate", [100, 4_000_000_000])
def test_transcribe_absurd_sample_rate_415(recording: tuple[TestClient, RecordingStt], sample_rate: int) -> None:
    """A header rate must not pick the resampler's filter size (memory) for us."""
    client, stt = recording
    wav = bytearray(make_wav(np.zeros(1600, dtype=np.int16), sample_rate=16000))
    wav[24:28] = sample_rate.to_bytes(4, "little")

    resp = post_audio(client, "rate.wav", bytes(wav))

    assert resp.status_code == 415
    assert stt.received == []


def test_transcribe_truncated_wav_400(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording
    wav = make_wav(np.zeros(1600, dtype=np.int16), sample_rate=16000)

    resp = post_audio(client, "cut.wav", wav[:30])

    assert resp.status_code == 400
    assert stt.received == []


def test_transcribe_wav_without_samples_400(recording: tuple[TestClient, RecordingStt]) -> None:
    client, _ = recording

    assert post_audio(client, "silence.wav", make_wav(np.zeros(0, dtype=np.int16), sample_rate=16000)).status_code == 400


def test_transcribe_odd_raw_400(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording

    resp = post_audio(client, "odd.pcm", b"\x01\x00\x02")

    assert resp.status_code == 400
    assert "odd" in resp.json()["detail"]
    assert stt.received == []


def test_transcribe_raw_pcm_is_passed_through_unchanged(recording: tuple[TestClient, RecordingStt]) -> None:
    client, stt = recording
    raw = (np.arange(3200) % 200).astype(np.int16)

    resp = post_audio(client, "mic.raw", raw.tobytes())

    assert resp.status_code == 200
    assert resp.json()["duration_sec"] == 0.2
    assert stt.received[0].tolist() == raw.tolist()


@pytest.mark.parametrize("name", ["voice.mp3", "voice.ogg", "voice", "voice.wav.exe"])
def test_transcribe_unknown_extension_415(recording: tuple[TestClient, RecordingStt], name: str) -> None:
    client, stt = recording

    assert post_audio(client, name, b"\x00\x00" * 160).status_code == 415
    assert stt.received == []
