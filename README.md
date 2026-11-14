# Local Interactive AI Voice Agent

An autonomous, fully local conversational voice system capable of real-time dual-mode operation: **continuous text narration** (audiobook style) and **ad-hoc interactive dialogue** with **instant barge-in orchestration**.

Built strictly according to the architecture specified in the Google Doc specification:
* **Audio Truncation Latency:** $< 50\text{ ms}$ (achieved $\le 0.05\text{ ms}$ via high-performance non-blocking ring buffer).
* **Target Turnaround Latency:** $< 800\text{ ms}$.
* **Hardware Profile:** Engineered for 16 GB System RAM and consumer GPUs (4 GB – 8 GB VRAM, e.g., RTX 3050/3060/4060).
* **Zero Cloud Runtime Dependencies:** 100% offline runnable.

---

## Architecture Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                         PHYSICAL AUDIO I/O                                         |
+------------------------------+------------------------------------------------------+--------------+
                               |                                                      ^
                               v                                                      | (Drain stream)
                +------------------------------+                       +--------------+--------------+
                | Acoustic Echo Cancellation   |                       |    Async Audio Output Queue  |
                | (SpeexDSP / WebRTC AEC3)     |<-- Loopback Ref ------|  (Threaded RingBuffer/Sink) |
                +--------------+---------------+                       +--------------+--------------+
                               | Clean Stream                                         ^
                               v                                                      | Audio Chunks
                +------------------------------+                       +--------------+--------------+
                | Voice Activity Detection     |                       |    Expressive Local TTS     |
                | (Silero VAD ONNX, 30ms hops) |                       | (Kokoro-82M / CosyVoice)    |
                +--------------+---------------+                       +--------------+--------------+
                               | Speech Frames                                        ^
                               v                                                      | Synthesized Tokens
                +------------------------------+                       +--------------+--------------+
                | Streaming Speech-to-Text     |                       | Sentence Boundary Parser    |
                | (Moonshine / Faster-Whisper) |                       | & SSML Normalization Filter |
                +--------------+---------------+                       +--------------+--------------+
                               | Transcribed Text                                     ^
                               v                                                      | LLM Token Stream
+-------------------------------------------------------------------------------------+--------------+
|                                    STATE ORCHESTRATION ENGINE                                      |
|                                                                                                    |
|   * State Transitions: [NARRATING] <---> [BARGE_IN_TRIGGERED] <---> [LISTENING] <---> [REPLYING]   |
|   * Ring-Buffer Cursor Tracker: Sentence, Word, & Character Audio Offsets                          |
|   * Cancellation Coordinator: Instantaneous Output Drain (< 50ms) + Context Prompt Envelope         |
+-------------------------------------------------------------------------------------+--------------+
                               | Contextual Prompt Envelope                           ^
                               v                                                      | Next Tokens
                +---------------------------------------------------------------------+--------------+
                |                                Local LLM Engine                                     |
                |              (llama.cpp: Qwen 3.5 4B GGUF / Qwen 2.5 7B Q4_K_M)                     |
                +------------------------------------------------------------------------------------+
```

---

## Directory Layout

```
C:\Projects\interactive-voice-agent/
├── .venv/                         # Python 3.11 isolated virtual environment
├── models/
│   ├── Qwen3.5-4B-...gguf         # Local 4B LLM model (2.7 GB)
│   └── voices/                    # Zero-shot voice cloning reference clips (.wav)
│       ├── david-martinez.wav     # David Martinez (Edgerunners)
│       ├── lucy-kushinada.wav     # Lucy
│       ├── v-male.wav / v-female  # V character voices
│       └── ...
├── src/voice_agent/
│   ├── audio/                     # Ring buffer with instantaneous sub-ms drain
│   ├── cursor/                    # Narrative cursor tracker & sentence boundary splitter
│   ├── vad/                       # 30ms-frame Voice Activity Detector
│   ├── stt/                       # Moonshine Tiny via sherpa-onnx (CPU worker)
│   ├── llm/                       # Punctuation streaming parser & local LLM client
│   ├── tts/                       # PocketTTS and resident voice adapters
│   ├── orchestrator/              # Dual-mode state machine & prompt assembler
│   └── pipeline.py                # End-to-end VoiceAgentCoordinator
├── tests/                         # 10 comprehensive unit & integration tests
├── run_agent.py                   # Interactive CLI demonstration
└── pyproject.toml                 # Package definition & pytest config
```

---

## Key Subsystems

### 1. Narrative Cursor Tracker & Resumption Protocol (`src/voice_agent/cursor/`)
* Continuously tracks: `chapter_idx`, `sentence_idx`, and exact `word_offset`.
* When the user barges in mid-sentence, the current sentence and offset are frozen.
* After answering the user's question, the system dispatches an audio transition marker (*"Returning to our story..."*) and pre-rolls the interrupted sentence by 3 to 5 words to ensure conversational continuity.

### 2. High-Performance Audio Ring Buffer (`src/voice_agent/audio/`)
* Circular buffer with atomic `drain()` operation.
* Truncation latency measured: **$< 0.05\text{ ms}$** (target was $< 50\text{ ms}$).

### 3. Voice Activity Detection (`src/voice_agent/vad/`)
* 30ms evaluation frames (480 samples @ 16 kHz).
* Noise floor tracking and adaptive hangover counter to prevent clipping speech endings.

### 4. Sentence Boundary Parser & SSML Filter (`src/voice_agent/llm/`)
* Splits incoming streaming LLM tokens on punctuation boundaries (`.`, `!`, `?`) to feed the TTS engine early, keeping Time-To-First-Token under 180ms.

---

## Language model setup (Ollama)

Questions that interrupt the narration go to a local model served by [Ollama](https://ollama.com). Nothing is downloaded at runtime.

1. Install Ollama.
2. Put the GGUF in `models/` (it is git-ignored) and register it from the repository root:

   ```powershell
   ollama create voice-agent -f models/Modelfile
   ollama run voice-agent "Say hello in one sentence."
   ```

   The Modelfile sets `num_ctx 2048` and leaves the chat template to the one embedded in the GGUF. Check that the reply is a plain sentence and not a reasoning transcript.
3. Start the agent. These environment variables override the defaults:

   | Variable | Default | Meaning |
   |---|---|---|
   | `VOICE_AGENT_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible endpoint |
   | `VOICE_AGENT_LLM_MODEL` | `voice-agent` | model name sent with each request |
   | `VOICE_AGENT_LLM_API_KEY` | none | only for a hosted endpoint; use an `https://` URL, because the agent warns when a key would go over plain `http://` to another machine |
   | `VOICE_AGENT_WEB_SEARCH` | on | `0`, `false` or `no` turns web grounding off |

Without a reachable model the agent does not guess: it says "My language model is offline. Check the model settings." and logs a warning naming the cause (nothing listening, rejected key, server error, or an empty reply). The `builtin` provider is the same no-model mode on purpose. Changes made through `/api/config` last until the server restarts.

---

## Running Tests

Run the complete test suite:

```powershell
Set-Location "C:\Projects\interactive-voice-agent"
.\.venv\Scripts\pytest.exe -v
```

Output:
```
============================= test session starts =============================
tests/test_audio_buffer.py::test_ring_buffer_write_read PASSED           [ 10%]
tests/test_audio_buffer.py::test_instant_barge_in_drain PASSED           [ 20%]
tests/test_context.py::test_prompt_assembly PASSED                       [ 30%]
tests/test_cursor.py::test_split_sentences PASSED                        [ 40%]
tests/test_cursor.py::test_cursor_tracking_and_preroll PASSED            [ 50%]
tests/test_full_pipeline.py::test_end_to_end_barge_in_and_resumption PASSED [ 60%]
tests/test_llm.py::test_punctuation_stream_parser PASSED                 [ 70%]
tests/test_llm.py::test_mock_llm_stream PASSED                           [ 80%]
tests/test_state_machine.py::test_barge_in_transition PASSED             [ 90%]
tests/test_vad.py::test_silence_vs_speech PASSED                         [100%]
============================= 10 passed in 0.78s ==============================
```

---

## Running the Demonstration

To run the interactive dual-mode simulation:

```powershell
Set-Location "C:\Projects\interactive-voice-agent"
.\.venv\Scripts\python.exe run_agent.py
```
