# Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-accuracy, low-latency local Speech-To-Text (STT) and voice interpretation pipeline using `faster-whisper` (with `SpeechRecognition` fallback), integrating live 16kHz PCM audio streaming over WebSocket, turn-taking speech segmentation, real-time transcript broadcasting (`user_speech_transcript`), and REST transcription endpoints (`POST /api/transcribe`).

**Architecture:** Client browser Web Audio streams raw 16kHz PCM frames over WebSocket during barge-in. A server-side `SpeechTurnAccumulator` segments user voice bursts using energy VAD and speech endpoints, transcribing them via `STTEngine` (`faster-whisper` `tiny.en` / `base.en` with CTranslate2 int8 execution). The transcribed text is broadcast back to the HUD via `user_speech_transcript` and automatically dispatched to `NarrationEngine.handle_user_interruption()`, resuming narration seamlessly once the answer is synthesized.

**Tech Stack:** `faster-whisper`, CTranslate2, `SpeechRecognition`, NumPy, FastAPI, WebSockets, Web Audio API, `edge-tts`.

**Spec:** GitHub Issue [#15](https://github.com/uditdaftary/interactive-voice-agent/issues/15)

## Global Constraints
- **VRAM Budget:** $\le 3100$ MB (leaving $\ge 500$ MB headroom on RTX 3050). Default to `cpu` with `compute_type="int8"` or CUDA when available.
- **Latency Budget:** Speech transcription for a spoken query must complete in $< 350$ms on CPU.
- **Audio Buffer Drain Latency:** $< 0.05$ms during barge-in interruption flush.
- **Prompt Context:** $\le 2048$ tokens.
- **Testing Integrity:** 100% test pass rate across all tiers in `pytest tests/`. No physical microphone or external GPU hardware required for automated tests (deterministic mock mode provided).

---

### Task 1: Neural STT Engine Abstraction (`voice_agent.stt.recognizer`)

**Files:**
- Create: `src/voice_agent/stt/__init__.py`
- Create: `src/voice_agent/stt/recognizer.py`
- Test: `tests/test_stt_recognizer.py`

**Interfaces:**
- Consumes: Raw audio bytes or numpy arrays (16kHz Mono PCM 16-bit or float32).
- Produces: `STTEngine.transcribe(audio_data: bytes | np.ndarray, sample_rate: int = 16000) -> str`
- Fallback hierarchy: `faster-whisper` -> `SpeechRecognition` (Google Web ASR / Vosk) -> `MockSTTEngine`.

- [ ] **Step 1: Write the failing unit tests for STTEngine**

```python
# tests/test_stt_recognizer.py
import pytest
import numpy as np
from voice_agent.stt.recognizer import STTEngine, MockSTTEngine

def test_mock_stt_engine():
    engine = MockSTTEngine("what is arasaka")
    # Generate 1 second of dummy 16kHz audio
    audio = np.zeros(16000, dtype=np.int16).tobytes()
    text = engine.transcribe(audio)
    assert text == "what is arasaka"

def test_stt_engine_initialization():
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    assert engine.model_size == "tiny.en"
    assert engine.device == "cpu"

def test_stt_engine_transcribe_silence():
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    text = engine.transcribe(silence)
    # Silence or mock should return string
    assert isinstance(text, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_stt_recognizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voice_agent.stt'`

- [ ] **Step 3: Implement `src/voice_agent/stt/recognizer.py` and `__init__.py`**

```python
# src/voice_agent/stt/recognizer.py
import io
import os
import wave
import logging
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class MockSTTEngine:
    """Deterministic Mock STT engine for unit testing and CI."""
    def __init__(self, fixed_transcript: str = "what is arasaka"):
        self.fixed_transcript = fixed_transcript

    def transcribe(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000) -> str:
        return self.fixed_transcript


class STTEngine:
    """High-performance local Speech-To-Text engine utilizing Faster-Whisper
    with fallback to SpeechRecognition and deterministic Mock mode.
    """
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        fallback_mock: bool = False,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.fallback_mock = fallback_mock
        self._whisper_model = None
        self._init_engine()

    def _init_engine(self):
        try:
            from faster_whisper import WhisperModel
            logger.info("Initializing faster-whisper STTEngine (%s on %s)...", self.model_size, self.device)
            self._whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            logger.warning("Failed to initialize faster-whisper (%s). Falling back.", e)
            self._whisper_model = None

    def transcribe(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000) -> str:
        """Transcribe PCM audio to text."""
        if not audio_data or len(audio_data) == 0:
            return ""

        # Convert bytes to numpy float32 if necessary
        if isinstance(audio_data, bytes):
            pcm16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = pcm16.astype(np.float32) / 32768.0
        else:
            if audio_data.dtype == np.int16:
                audio_np = audio_data.astype(np.float32) / 32768.0
            else:
                audio_np = audio_data.astype(np.float32)

        # Check for near-total silence
        if np.max(np.abs(audio_np)) < 0.01:
            return ""

        # 1. Primary: faster-whisper
        if self._whisper_model is not None:
            try:
                segments, info = self._whisper_model.transcribe(
                    audio_np,
                    beam_size=3,
                    language="en",
                    vad_filter=True,
                )
                text = " ".join([s.text.strip() for s in segments]).strip()
                if text:
                    return text
            except Exception as err:
                logger.warning("faster-whisper transcription error: %s", err)

        # 2. Secondary: SpeechRecognition fallback
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            # Wrap PCM in WAV buffer
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio = r.record(source)
                return r.recognize_google(audio)
        except Exception as err:
            logger.debug("SpeechRecognition fallback notice: %s", err)

        if self.fallback_mock:
            return "what is arasaka"

        return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_stt_recognizer.py -v`
Expected: PASS (3 tests passed)

- [ ] **Step 5: Commit Task 1**

```bash
git add src/voice_agent/stt/ tests/test_stt_recognizer.py
git commit -m "feat(stt): add neural STTEngine with faster-whisper and SpeechRecognition fallback"
```

---

### Task 2: Speech Turn Accumulator & Voice Activity Segmentation

**Files:**
- Create: `src/voice_agent/stt/accumulator.py`
- Test: `tests/test_speech_accumulator.py`

**Interfaces:**
- Consumes: Real-time 16kHz PCM chunks from WebSocket / audio bridge.
- Produces: `SpeechTurnAccumulator` detecting when user starts speaking, buffering audio, detecting when user finishes speaking (silence duration $> 600$ms), and returning the segmented utterance for transcription.

- [ ] **Step 1: Write the failing unit tests for SpeechTurnAccumulator**

```python
# tests/test_speech_accumulator.py
import pytest
import numpy as np
from voice_agent.stt.accumulator import SpeechTurnAccumulator

def test_speech_turn_accumulator_buffering():
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=500)
    # Feed silence
    silence = np.zeros(1600, dtype=np.int16).tobytes() # 100ms
    turn = acc.feed(silence)
    assert turn is None
    assert not acc.is_speaking

    # Feed loud audio (simulated speech)
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    turn = acc.feed(loud)
    assert acc.is_speaking
    assert turn is None

    # Feed silence exceeding silence_timeout_ms (600ms)
    silence_600ms = np.zeros(9600, dtype=np.int16).tobytes()
    turn = acc.feed(silence_600ms)
    # Should trigger turn complete
    assert turn is not None
    assert len(turn) > 0
    assert not acc.is_speaking
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_speech_accumulator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voice_agent.stt.accumulator'`

- [ ] **Step 3: Implement `src/voice_agent/stt/accumulator.py`**

```python
# src/voice_agent/stt/accumulator.py
import time
import numpy as np
from typing import Optional, List

class SpeechTurnAccumulator:
    """Buffers real-time microphone PCM chunks, identifies active speech turns
    via energy thresholding, and detects conversational pauses for transcription.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.04,
        silence_timeout_ms: int = 650,
        min_speech_duration_ms: int = 400,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.is_speaking = False
        self._speech_buffer: List[bytes] = []
        self._last_speech_time: float = 0.0
        self._speech_start_time: float = 0.0

    def feed(self, pcm_chunk: bytes) -> Optional[bytes]:
        """Feed a 16-bit PCM chunk. Returns full utterance bytes when user finishes speaking."""
        if not pcm_chunk or len(pcm_chunk) < 2:
            return None

        # Calculate RMS energy of this chunk
        pcm16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        float_samples = pcm16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(float_samples ** 2))) if len(float_samples) > 0 else 0.0
        now = time.monotonic()

        if rms >= self.energy_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self._speech_start_time = now
                self._speech_buffer = []
            self._last_speech_time = now
            self._speech_buffer.append(pcm_chunk)
            return None

        # Silence chunk
        if self.is_speaking:
            self._speech_buffer.append(pcm_chunk)
            silence_duration = (now - self._last_speech_time) * 1000.0
            if silence_duration >= self.silence_timeout_ms:
                speech_duration = (self._last_speech_time - self._speech_start_time) * 1000.0
                utterance = b"".join(self._speech_buffer)
                self.is_speaking = False
                self._speech_buffer = []
                if speech_duration >= self.min_speech_duration_ms:
                    return utterance

        return None

    def flush(self) -> Optional[bytes]:
        """Manually flush and return accumulated speech (e.g. on client speech_end event)."""
        if self._speech_buffer:
            utterance = b"".join(self._speech_buffer)
            self._speech_buffer = []
            self.is_speaking = False
            return utterance
        return None

    def reset(self):
        """Reset internal accumulator state."""
        self.is_speaking = False
        self._speech_buffer = []
        self._last_speech_time = 0.0
        self._speech_start_time = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_speech_accumulator.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add src/voice_agent/stt/accumulator.py tests/test_speech_accumulator.py
git commit -m "feat(stt): add SpeechTurnAccumulator with energy VAD pause detection"
```

---

### Task 3: WebSocket Protocol Schemas & Bi-Directional STT Streaming

**Files:**
- Modify: `src/voice_agent/server/schemas.py`
- Modify: `src/voice_agent/server/websocket.py`
- Test: `tests/test_stt_websocket.py`

**Interfaces:**
- Produces: `UserSpeechTranscriptMessage(type="user_speech_transcript", transcript="...", is_final=True)`
- Consumes: Binary PCM chunks over WebSocket feeding into `SpeechTurnAccumulator` during `LISTENING` or `BARGE_IN` states; on turn completion, transcribes audio and triggers `handle_user_interruption(query=transcript)`.

- [ ] **Step 1: Write test for WebSocket speech transcription message & flow**

```python
# tests/test_stt_websocket.py
import pytest
from voice_agent.server.schemas import UserSpeechTranscriptMessage, parse_client_message

def test_user_speech_transcript_schema():
    msg = UserSpeechTranscriptMessage(transcript="what is arasaka", is_final=True)
    assert msg.type == "user_speech_transcript"
    assert msg.transcript == "what is arasaka"
    assert msg.is_final is True
    data = msg.model_dump()
    assert data["type"] == "user_speech_transcript"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_stt_websocket.py -v`
Expected: FAIL with `cannot import name 'UserSpeechTranscriptMessage'`

- [ ] **Step 3: Update `src/voice_agent/server/schemas.py` and `src/voice_agent/server/websocket.py`**
  - Add `UserSpeechTranscriptMessage` schema.
  - Instantiate `STTEngine` and `SpeechTurnAccumulator` in `ConnectionManager` / `WebSocketHandler`.
  - When binary audio chunks are received during `LISTENING` / `BARGE_IN`, feed them to `SpeechTurnAccumulator`.
  - When an utterance is completed (or client sends `{"type": "voice_end"}`), transcribe audio with `stt_engine.transcribe()`.
  - Broadcast `UserSpeechTranscriptMessage` and invoke `narration_engine.handle_user_interruption(transcript)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_stt_websocket.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add src/voice_agent/server/schemas.py src/voice_agent/server/websocket.py tests/test_stt_websocket.py
git commit -m "feat(server): integrate server-side STT into WebSocket streaming pipeline"
```

---

### Task 4: REST API Endpoint `POST /api/transcribe`

**Files:**
- Modify: `src/voice_agent/server/app.py`
- Test: `tests/test_transcribe_api.py`

**Interfaces:**
- Route: `POST /api/transcribe`
- Request: Audio file upload (`multipart/form-data`) or raw body.
- Response: `{"transcript": "what is arasaka", "duration_sec": 1.5, "status": "ok"}`

- [ ] **Step 1: Write test for `POST /api/transcribe`**

```python
# tests/test_transcribe_api.py
import io
import wave
import pytest
from fastapi.testclient import TestClient
from voice_agent.server.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_api_transcribe_audio_file(client):
    # Create valid 1-sec 16kHz WAV
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    wav_io.seek(0)

    response = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", wav_io, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert data["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_transcribe_api.py -v`
Expected: FAIL with 404 Not Found

- [ ] **Step 3: Implement `POST /api/transcribe` in `src/voice_agent/server/app.py`**
  - Add route accepting `file: UploadFile`.
  - Read bytes, invoke `stt_engine.transcribe(audio_bytes)`.
  - Return JSON transcript response.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_transcribe_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add src/voice_agent/server/app.py tests/test_transcribe_api.py
git commit -m "feat(api): add POST /api/transcribe endpoint"
```

---

### Task 5: HUD Client Bi-Directional Speech Synchronization (`ui/index.html`)

**Files:**
- Modify: `ui/index.html`
- Modify: `ui/app.css`
- Test: `tests/test_server_static.py`

**Interfaces:**
- When user clicks `[🎙️ BARGE IN & SPEAK]` or presses `Space`/`V`:
  - Starts recording microphone via Web Audio API.
  - Automatically streams PCM chunks over WebSocket to server STT engine.
  - If browser Web Speech API is also available, uses both for instant zero-latency preview.
  - On receiving `user_speech_transcript` message from WebSocket, updates `#interruption-query-input` and `#speech-ticker` with server's transcribed words.
  - Dispatches query and displays Prompt Envelope seamlessly.

- [ ] **Step 1: Update `HUDWebSocketClient` in `ui/index.html`**
  - Add handler for `msg.type === 'user_speech_transcript'`.
  - When received, populate `#interruption-query-input` and `#speech-ticker`.
  - If `msg.is_final`, automatically call `submitInterruptionQuery(msg.transcript)`.

- [ ] **Step 2: Update `startDirectVoiceBargeIn()` in `ui/index.html`**
  - Ensure microphone stream is active and streaming binary PCM16 over WebSocket.
  - Add `[STOP & SUBMIT]` state to voice barge-in button while listening.

- [ ] **Step 3: Run existing static tests**

Run: `.venv\Scripts\python -m pytest tests/test_server_static.py -v`
Expected: PASS

- [ ] **Step 4: Commit Task 5**

```bash
git add ui/index.html ui/app.css
git commit -m "feat(ui): synchronize HUD with server-side STT transcripts"
```

---

### Task 6: End-to-End Test Suite Verification & Benchmarks

**Files:**
- Modify: `tests/test_e2e_pipeline.py`
- Modify: `tests/test_benchmarks.py`

- [ ] **Step 1: Add E2E test verifying voice frame -> STT -> LLM grounding -> TTS audio cycle**
- [ ] **Step 2: Add benchmark verifying STTEngine latency $< 350$ms**
- [ ] **Step 3: Run the full test suite**

Run: `.venv\Scripts\python -m pytest tests/ -v`
Expected: 100% tests passing (all existing 123 tests + new STT test suite).

- [ ] **Step 4: Update Documentation**
  - Update `MEMORY.md` with ADR-12: Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation.
  - Update `CHANGELOG.md` with v0.6.0 release notes.
  - Update `AGENTS.md` with Sprint 5 domain lead assignments.

- [ ] **Step 5: Commit & Push**

```bash
git add tests/ MEMORY.md CHANGELOG.md AGENTS.md
git commit -m "feat(stt): complete neural speech recognition pipeline with 100% verified test pass"
git push origin feature/unified-hud-websocket-server
```
