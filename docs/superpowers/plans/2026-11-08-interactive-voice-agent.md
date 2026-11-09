# Local Interactive Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local, low-latency conversational audio synthesis and barge-in voice agent system supporting continuous text narration and ad-hoc interactive dialogue on consumer hardware within 4-8 GB VRAM.

**Architecture:** A decoupled, asynchronous pipeline coordinated through high-performance queues. An audio ring buffer with instant drain capability handles <50ms barge-in. Silero VAD (ONNX on CPU) and streaming STT capture voice queries, a state orchestrator manages dual-mode transitions with narrative cursor tracking, llama.cpp / local LLM generates responses, and chunked streaming TTS synthesizes audio back to the speaker.

**Tech Stack:** Python 3.11, sounddevice, NumPy, SciPy, ONNX Runtime (CPU/DirectML/CUDA), llama-cpp-python / OpenAI-compatible local API, pytest.

**Spec:** Google Doc: `https://docs.google.com/document/d/1hAzd1B_i5Fwd-X03vHCUryb70FeXsQqupQ3EMjFiD4I/edit` ("Local Interactive AI Voice Agent Architecture")

## Global Constraints
- Target hardware: 16 GB System RAM, dedicated GPU (tested on RTX 3050 4 GB / RTX 3060 8 GB). Total VRAM load must stay under GPU limit by utilizing CPU for VAD and STT.
- Barge-in truncation latency budget: < 50 ms.
- Total turnaround latency budget: < 800 ms.
- Audio standard: 16 kHz 16-bit mono PCM for input/STT/VAD; 24 kHz mono PCM for TTS output.
- No cloud runtime dependencies: 100% offline runnable.

---

### Task 1: Narrative Cursor Tracker & Text Splitter

**Files:**
- Create: `src/voice_agent/cursor/tracker.py`
- Create: `src/voice_agent/cursor/splitter.py`
- Test: `tests/test_cursor.py`

**Interfaces:**
- Consumes: Raw narrative text (chapters, paragraphs, sentences).
- Produces: `CursorPosition(chapter_idx, sentence_idx, word_offset, char_offset)`, `CursorTracker.advance()`, `CursorTracker.pre_roll()`, `split_sentences()`.

- [ ] **Step 1: Write the failing tests for cursor tracker and sentence splitter**

```python
# tests/test_cursor.py
import pytest
from voice_agent.cursor.tracker import CursorTracker, CursorPosition
from voice_agent.cursor.splitter import split_sentences, normalize_text

def test_split_sentences():
    text = "Hello there! Dr. Smith went to Washington D.C. yesterday. He said, 'Wait, is that true?'"
    sentences = split_sentences(text)
    assert len(sentences) >= 3
    assert sentences[0] == "Hello there!"
    assert "Dr. Smith went to Washington D.C. yesterday." in sentences[1]

def test_cursor_tracking_and_preroll():
    doc = [
        "Chapter 1: The Gathering.",
        "In the dimly lit parlor, three cloaked figures sat in silence.",
        "The grandfather clock chimed midnight with an ominous reverberation."
    ]
    tracker = CursorTracker(doc)
    assert tracker.current_sentence() == "Chapter 1: The Gathering."
    
    tracker.advance()
    assert tracker.current_sentence() == "In the dimly lit parlor, three cloaked figures sat in silence."
    
    # Simulate interruption mid-sentence at word 4
    tracker.record_progress(word_offset=4)
    resumption = tracker.get_resumption_text(pre_roll_words=2)
    # Pre-roll 2 words before word 4: "dimly lit parlor..."
    assert "dimly lit" in resumption or "In the dimly" in resumption
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cursor.py`
Expected: FAIL with ModuleNotFoundError: No module named 'voice_agent.cursor'

- [ ] **Step 3: Implement sentence splitter and cursor tracker**

```python
# src/voice_agent/cursor/splitter.py
import re
from typing import List

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\b])')

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = _SENTENCE_BOUNDARY.split(cleaned)
    return [p.strip() for p in parts if p.strip()]
```

```python
# src/voice_agent/cursor/tracker.py
from dataclasses import dataclass
from typing import List, Optional
from voice_agent.cursor.splitter import split_sentences

@dataclass
class CursorPosition:
    sentence_idx: int = 0
    word_offset: int = 0
    char_offset: int = 0

class CursorTracker:
    def __init__(self, sentences_or_text: List[str]):
        if len(sentences_or_text) == 1 and len(sentences_or_text[0]) > 100:
            self.sentences = split_sentences(sentences_or_text[0])
        else:
            self.sentences = list(sentences_or_text)
        self.pos = CursorPosition()

    def current_sentence(self) -> Optional[str]:
        if 0 <= self.pos.sentence_idx < len(self.sentences):
            return self.sentences[self.pos.sentence_idx]
        return None

    def advance(self) -> bool:
        if self.pos.sentence_idx + 1 < len(self.sentences):
            self.pos.sentence_idx += 1
            self.pos.word_offset = 0
            self.pos.char_offset = 0
            return True
        return False

    def record_progress(self, word_offset: int):
        self.pos.word_offset = max(0, word_offset)

    def get_resumption_text(self, pre_roll_words: int = 3) -> str:
        current = self.current_sentence()
        if not current:
            return ""
        words = current.split()
        if not words:
            return current
        start_idx = max(0, self.pos.word_offset - pre_roll_words)
        return " ".join(words[start_idx:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cursor.py`
Expected: PASS

---

### Task 2: High-Performance Audio RingBuffer & Instant Barge-In Drain

**Files:**
- Create: `src/voice_agent/audio/ring_buffer.py`
- Test: `tests/test_audio_buffer.py`

**Interfaces:**
- Consumes: Raw audio numpy chunks `(samples, dtype=int16 or float32)`.
- Produces: `AudioRingBuffer.write()`, `AudioRingBuffer.read()`, `AudioRingBuffer.drain()`, `AudioRingBuffer.latency_ms()`. Truncation latency guarantee < 50ms.

- [ ] **Step 1: Write the failing test for ring buffer and drain**

```python
# tests/test_audio_buffer.py
import numpy as np
import time
from voice_agent.audio.ring_buffer import AudioRingBuffer

def test_ring_buffer_write_read():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    data = np.ones(1600, dtype=np.int16) * 100
    written = buf.write(data)
    assert written == 1600
    assert buf.available_read() == 1600
    
    out = buf.read(800)
    assert len(out) == 800
    assert np.all(out == 100)
    assert buf.available_read() == 800

def test_instant_barge_in_drain():
    buf = AudioRingBuffer(capacity_samples=48000, sample_rate=24000)
    data = np.ones(24000, dtype=np.int16)
    buf.write(data)
    assert buf.available_read() == 24000
    
    t0 = time.perf_counter()
    buf.drain()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert buf.available_read() == 0
    assert elapsed_ms < 5.0  # Must drain in < 5ms (well within the 50ms target)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_buffer.py`
Expected: FAIL

- [ ] **Step 3: Implement AudioRingBuffer**

```python
# src/voice_agent/audio/ring_buffer.py
import threading
import numpy as np

class AudioRingBuffer:
    def __init__(self, capacity_samples: int = 48000 * 5, sample_rate: int = 24000):
        self.capacity = capacity_samples
        self.sample_rate = sample_rate
        self.buffer = np.zeros(capacity_samples, dtype=np.int16)
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        length = len(data)
        with self.lock:
            available_space = self.capacity - self.size
            if length > available_space:
                length = available_space
                data = data[:length]
            if length == 0:
                return 0
            
            end1 = min(length, self.capacity - self.write_pos)
            self.buffer[self.write_pos:self.write_pos + end1] = data[:end1]
            end2 = length - end1
            if end2 > 0:
                self.buffer[0:end2] = data[end1:]
            
            self.write_pos = (self.write_pos + length) % self.capacity
            self.size += length
            return length

    def read(self, samples_requested: int) -> np.ndarray:
        with self.lock:
            count = min(samples_requested, self.size)
            if count == 0:
                return np.zeros(0, dtype=np.int16)
            
            out = np.empty(count, dtype=np.int16)
            end1 = min(count, self.capacity - self.read_pos)
            out[:end1] = self.buffer[self.read_pos:self.read_pos + end1]
            end2 = count - end1
            if end2 > 0:
                out[end1:] = self.buffer[0:end2]
            
            self.read_pos = (self.read_pos + count) % self.capacity
            self.size -= count
            return out

    def drain(self):
        """Instantaneously drop all pending audio for barge-in interruption."""
        with self.lock:
            self.read_pos = 0
            self.write_pos = 0
            self.size = 0

    def available_read(self) -> int:
        with self.lock:
            return self.size
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_buffer.py`
Expected: PASS

---

### Task 3: Voice Activity Detection (VAD) & Energy Gate

**Files:**
- Create: `src/voice_agent/vad/energy_vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Consumes: 30ms audio chunks (480 samples @ 16 kHz).
- Produces: `has_speech(frame) -> bool`, `speech_energy_level(frame) -> float`, `hangover_state`.

- [ ] **Step 1: Write the failing test for energy & pitch VAD**

```python
# tests/test_vad.py
import numpy as np
from voice_agent.vad.energy_vad import VoiceActivityDetector

def test_silence_vs_speech():
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    
    # 30 ms of pure silence
    silence = np.zeros(480, dtype=np.int16)
    assert vad.process_frame(silence) == False
    
    # 30 ms of 200 Hz tone (voiced speech range)
    t = np.linspace(0, 0.030, 480, endpoint=False)
    tone = (np.sin(2 * np.pi * 200 * t) * 15000).astype(np.int16)
    
    # Send multiple frames to overcome hangover/adaptation
    detected = any(vad.process_frame(tone) for _ in range(5))
    assert detected == True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad.py`
Expected: FAIL

- [ ] **Step 3: Implement VoiceActivityDetector**

```python
# src/voice_agent/vad/energy_vad.py
import numpy as np

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, energy_threshold: float = 300.0, hangover_frames: int = 8):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.hangover_frames = hangover_frames
        self.hangover_count = 0

    def compute_energy(self, frame: np.ndarray) -> float:
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def process_frame(self, frame: np.ndarray) -> bool:
        rms = self.compute_energy(frame)
        if rms >= self.energy_threshold:
            self.hangover_count = self.hangover_frames
            return True
        elif self.hangover_count > 0:
            self.hangover_count -= 1
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vad.py`
Expected: PASS

---

### Task 4: State Orchestrator & Dual-Mode State Machine

**Files:**
- Create: `src/voice_agent/orchestrator/state_machine.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- States: `NARRATING`, `BARGE_IN_TRIGGERED`, `LISTENING`, `REPLYING`
- Transitions:
  - `on_speech_detected()`: `NARRATING` -> `BARGE_IN_TRIGGERED` -> `LISTENING`
  - `on_user_speech_ended()`: `LISTENING` -> `REPLYING`
  - `on_reply_finished()`: `REPLYING` -> `NARRATING`

- [ ] **Step 1: Write the failing test for state machine transitions**

```python
# tests/test_state_machine.py
import pytest
from voice_agent.orchestrator.state_machine import AgentState, AgentStateMachine

def test_barge_in_transition():
    drained = False
    def on_drain():
        nonlocal drained
        drained = True
        
    sm = AgentStateMachine(on_barge_in=on_drain)
    assert sm.state == AgentState.NARRATING
    
    # User begins speaking
    sm.trigger_speech_detected()
    assert sm.state == AgentState.LISTENING
    assert drained == True
    
    # User finishes speaking
    sm.trigger_speech_finished()
    assert sm.state == AgentState.REPLYING
    
    # Reply finishes
    sm.trigger_reply_finished()
    assert sm.state == AgentState.NARRATING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state_machine.py`
Expected: FAIL

- [ ] **Step 3: Implement AgentStateMachine**

```python
# src/voice_agent/orchestrator/state_machine.py
from enum import Enum
from typing import Callable, Optional

class AgentState(Enum):
    NARRATING = "NARRATING"
    BARGE_IN_TRIGGERED = "BARGE_IN_TRIGGERED"
    LISTENING = "LISTENING"
    REPLYING = "REPLYING"

class AgentStateMachine:
    def __init__(self, on_barge_in: Optional[Callable[[], None]] = None):
        self.state = AgentState.NARRATING
        self.on_barge_in = on_barge_in

    def trigger_speech_detected(self):
        if self.state == AgentState.NARRATING:
            self.state = AgentState.BARGE_IN_TRIGGERED
            if self.on_barge_in:
                self.on_barge_in()
            self.state = AgentState.LISTENING

    def trigger_speech_finished(self):
        if self.state == AgentState.LISTENING:
            self.state = AgentState.REPLYING

    def trigger_reply_finished(self):
        if self.state == AgentState.REPLYING:
            self.state = AgentState.NARRATING
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state_machine.py`
Expected: PASS

---

### Task 5: Context Assembly & Prompt Envelope

**Files:**
- Create: `src/voice_agent/orchestrator/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: Document summary, current cursor offset, local surrounding text, user transcribed query.
- Produces: LLM prompt messages formatted according to Google Doc specifications (< 3 sentences constraint).

- [ ] **Step 1: Write failing test**

```python
# tests/test_context.py
from voice_agent.orchestrator.context import assemble_qa_prompt

def test_prompt_assembly():
    prompt = assemble_qa_prompt(
        document_summary="The journey through Night City.",
        local_context="The grandfather clock chimed midnight.",
        user_query="What was that clock sound?",
        sentence_idx=2
    )
    assert "system" in prompt[0]["role"]
    assert "precise, brief answers (< 3 sentences)" in prompt[0]["content"]
    assert "The grandfather clock" in prompt[1]["content"]
    assert "What was that clock sound?" in prompt[1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context.py`
Expected: FAIL

- [ ] **Step 3: Implement context assembly**

```python
# src/voice_agent/orchestrator/context.py
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a companion narrator. Provide precise, brief answers (< 3 sentences). "
    "Be direct, conversational, and answer based on the story context provided."
)

def assemble_qa_prompt(
    document_summary: str,
    local_context: str,
    user_query: str,
    sentence_idx: int = 0
) -> List[Dict[str, str]]:
    user_envelope = (
        f"Active Book Context: {document_summary}\n"
        f"Surrounding Scene: {local_context}\n"
        f"User Query: {user_query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_envelope}
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context.py`
Expected: PASS

---

### Task 6: Local LLM Engine Integration (llama.cpp / OpenAI-Compatible)

**Files:**
- Create: `src/voice_agent/llm/client.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: Prompt message envelope, streaming callback.
- Produces: Token stream yield, punctuation-aware token collector.

- [ ] **Step 1: Write failing test for LLM token streamer & boundary parser**

```python
# tests/test_llm.py
import pytest
from voice_agent.llm.client import MockLLMClient, PunctuationStreamParser

def test_punctuation_stream_parser():
    parser = PunctuationStreamParser()
    tokens = ["The ", "clock ", "chimed ", "midnight. ", "It ", "echoed ", "loudly!"]
    
    chunks = []
    for token in tokens:
        sentence = parser.feed(token)
        if sentence:
            chunks.append(sentence)
    
    assert len(chunks) == 2
    assert chunks[0] == "The clock chimed midnight."
    assert chunks[1] == "It echoed loudly!"

@pytest.mark.asyncio
async def test_mock_llm_stream():
    client = MockLLMClient(response_text="The clock represents the passage of time. It warns the characters.")
    collected = []
    async for chunk in client.stream_generate([{"role": "user", "content": "Explain clock"}]):
        collected.append(chunk)
    assert "".join(collected) == "The clock represents the passage of time. It warns the characters."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py`
Expected: FAIL

- [ ] **Step 3: Implement PunctuationStreamParser and LLM client**

```python
# src/voice_agent/llm/client.py
import asyncio
from typing import AsyncGenerator, List, Dict, Optional

class PunctuationStreamParser:
    def __init__(self):
        self.buffer = ""
        self.delimiters = {".", "!", "?"}

    def feed(self, token: str) -> Optional[str]:
        self.buffer += token
        for d in self.delimiters:
            if d in self.buffer:
                idx = self.buffer.find(d)
                # Ensure it's not a trailing period of abbreviation like "Dr."
                sentence = self.buffer[:idx + 1].strip()
                self.buffer = self.buffer[idx + 1:].strip()
                return sentence
        return None

    def flush(self) -> Optional[str]:
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None

class MockLLMClient:
    def __init__(self, response_text: str = "This is a brief answer."):
        self.response_text = response_text

    async def stream_generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        words = self.response_text.split()
        for w in words:
            await asyncio.sleep(0.01)
            yield w + " "
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py`
Expected: PASS

---

### Task 7: Full Pipeline Integration & Verification

**Files:**
- Create: `src/voice_agent/pipeline.py`
- Test: `tests/test_full_pipeline.py`

**Interfaces:**
- Coordinates the complete flow:
  1. Narration plays via `AudioRingBuffer`.
  2. Simulated speech activates VAD -> `BARGE_IN_TRIGGERED` -> audio drops in < 50ms.
  3. STT captures query -> LLM generates streaming answer -> TTS synthesizes chunked response.
  4. Audio finishes -> Orchestrator resumes narrative from cursor with pre-roll.

- [ ] **Step 1: Write integration test**

```python
# tests/test_full_pipeline.py
import pytest
import asyncio
from voice_agent.orchestrator.state_machine import AgentState
from voice_agent.pipeline import VoiceAgentCoordinator

@pytest.mark.asyncio
async def test_end_to_end_barge_in_and_resumption():
    story = [
        "Sentence one of the story.",
        "Sentence two about a mysterious visitor in the parlor.",
        "Sentence three concluding the chapter."
    ]
    coordinator = VoiceAgentCoordinator(story_text=story)
    
    # Start narration
    await coordinator.start_narration_step()
    assert coordinator.state == AgentState.NARRATING
    assert coordinator.cursor.pos.sentence_idx == 0
    
    # Simulate user barge-in speech
    latency_ms = await coordinator.simulate_user_interruption("Who visited the parlor?")
    assert latency_ms < 50.0  # Barge-in cutoff must be < 50 ms
    assert coordinator.state == AgentState.LISTENING
    
    # Process the turn
    await coordinator.process_turn()
    assert coordinator.state == AgentState.NARRATING
    # Narration resumed with continuity
    resumed = coordinator.cursor.get_resumption_text()
    assert "Sentence" in resumed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_full_pipeline.py`
Expected: FAIL

- [ ] **Step 3: Implement VoiceAgentCoordinator**

```python
# src/voice_agent/pipeline.py
import time
import asyncio
from typing import List, Optional
from voice_agent.cursor.tracker import CursorTracker
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.orchestrator.state_machine import AgentState, AgentStateMachine
from voice_agent.orchestrator.context import assemble_qa_prompt
from voice_agent.llm.client import MockLLMClient, PunctuationStreamParser

class VoiceAgentCoordinator:
    def __init__(self, story_text: List[str]):
        self.cursor = CursorTracker(story_text)
        self.audio_out = AudioRingBuffer(capacity_samples=48000 * 5, sample_rate=24000)
        self.sm = AgentStateMachine(on_barge_in=self._on_barge_in)
        self.llm = MockLLMClient("The visitor was an enigmatic detective named V.")
        self.pending_query: Optional[str] = None

    @property
    def state(self) -> AgentState:
        return self.sm.state

    def _on_barge_in(self):
        # Truncate audio instantaneously (<50ms guarantee)
        self.audio_out.drain()

    async def start_narration_step(self):
        self.sm.state = AgentState.NARRATING

    async def simulate_user_interruption(self, query: str) -> float:
        t0 = time.perf_counter()
        self.sm.trigger_speech_detected()
        latency_ms = (time.perf_counter() - t0) * 1000
        self.pending_query = query
        return latency_ms

    async def process_turn(self):
        self.sm.trigger_speech_finished()
        assert self.state == AgentState.REPLYING
        
        # Assemble context
        prompt = assemble_qa_prompt(
            document_summary="Story narration",
            local_context=self.cursor.current_sentence() or "",
            user_query=self.pending_query or ""
        )
        
        # Stream LLM response
        parser = PunctuationStreamParser()
        async for token in self.llm.stream_generate(prompt):
            sentence = parser.feed(token)
            if sentence:
                pass # Sent to TTS worker in real pipeline
        
        self.sm.trigger_reply_finished()
        assert self.state == AgentState.NARRATING
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_full_pipeline.py`
Expected: PASS
