# Multi-Agent Topology & Operational Registry

*Documentation of all agent personas, subagent clusters, communications protocols, and operational workflows for the Interactive Voice Agent project.*

---

## 1. Multi-Agent System Overview

This project is orchestrated using a decentralized, multi-agent collective. To scale engineering across complex full-stack voice, DSP, UI, and protocol requirements without context degradation or mono-agent blind spots, work is divided across specialized Sub-Orchestrators with strict domain boundaries, formal review gates, and cross-agent coordination protocols.

```
                              +--------------------------+
                              |   Lead Coordinator Agent |
                              |   (Antigravity Assistant)|
                              +-------------+------------+
                                            |
        +------------------+----------------+------------------+------------------+
        |                  |                |                  |                  |
        v                  v                v                  v                  v
+---------------+  +---------------+  +---------------+  +---------------+  +---------------+
|  SUB-ORCH 1   |  |  SUB-ORCH 2   |  |  SUB-ORCH 3   |  |  SUB-ORCH 4   |  | DOCUMENTATION |
|  (Backend &   |  | (Audio DSP &  |  | (Catalog Sheet|  | (Automated    |  |   CUSTODIAN   |
|   Streaming)  |  |  Abstraction) |  | HUD Visuals)  |  | Test & Bench) |  | (Memory/Git)  |
+---------------+  +---------------+  +---------------+  +---------------+  +---------------+
```

---

## 2. Five-Domain Multi-Agent Topology (Sprint 2 / v0.3.0)

Per user directive on multi-agent scale, execution fans out into 5 specialized Sub-Orchestrators acting as domain leads:

### 2.1. Sub-Orchestrator 1: Backend & Real-Time Streaming (`sub_orch_backend_1`)
- **Domain Scope**: `src/voice_agent/server/app.py`, `websocket.py`, `schemas.py`
- **Responsibilities**:
  - FastAPI application factory hosting static single-page HUD assets (`ui/`) at `http://localhost:8000/` and bi-directional streaming at `ws://localhost:8000/ws`.
  - Typed Pydantic message schemas for client-server protocol (`state_sync`, `vad_event`, `cursor_update`, `barge_in`, `telemetry`, `prompt_envelope`).
  - Active client connection manager and session state tracking.
  - Graceful disconnection cleanup without dangling background tasks or memory leaks.
- **Verification Gates**: 100% test pass on static asset routing, WebSocket handshakes, and binary audio frame handling.

### 2.2. Sub-Orchestrator 2: Audio DSP & Hardware Abstraction (`sub_orch_audio_1`)
- **Domain Scope**: `src/voice_agent/audio/loopback.py`, `hybrid.py`, `ring_buffer.py`
- **Responsibilities**:
  - Headless `VirtualAudioDevice` loopback generator producing synthetic PCM frames for hardware-free CI execution.
  - `HybridMicrophoneBridge` dynamically switching between browser Web Audio API PCM streaming and host Python `sounddevice`.
  - 30ms VAD telemetry streaming pipeline integration with adaptive noise floor and DC blocker.
  - Sub-50ms barge-in interruption pipeline trigger and atomic buffer drain latency verification (< 0.05ms).
- **Verification Gates**: Atomic drain latency benchmark < 0.05ms, zero-discontinuity half-cosine fadeout, smooth mic mode switching.

### 2.3. Sub-Orchestrator 3: Single-Page Catalog Sheet HUD (`sub_orch_frontend_1`)
- **Domain Scope**: `ui/index.html`, `ui/app.css`, `ui/fonts/`
- **Responsibilities**:
  - Strict conformance to "The Catalog Sheet" design system tokens (44px tabbed controls, 0px corner geometry, Ground `#0b0b0b`, Ink `#efede6`, Signal Yellow `#e8b622`, Fault Red `#e03a2f`).
  - 60 FPS live canvas voice graph oscilloscope and dynamic 16-bar FFT frequency spectrum.
  - Continuous narrative reader displaying streaming sentences with active word highlighting in inverted Plot White and pre-roll cues in Signal Yellow.
  - Sub-50ms barge-in disruption canvas and dynamic Q&A prompt envelope telemetry.
  - Resident character voice selector table (David Martinez, Lucy, V Male, V Female, Johnny Silverhand) with cadence WPM and timbre parameters.
  - Seamless fallback to local interactive simulation mode when WebSocket server is disconnected.
- **Verification Gates**: Zero 404 font/asset errors, verified 60 FPS canvas rendering, responsive 44px control layout.

### 2.4. Sub-Orchestrator 4: Testing & Verification Benchmark Suite (`sub_orch_testing_1`)
- **Domain Scope**: `tests/test_server_static.py`, `test_websocket_protocol.py`, `test_e2e_pipeline.py`, `test_benchmarks.py`
- **Responsibilities**:
  - 5-Tier test architecture management:
    - Tier 1: 56 existing unit and DSP regression tests.
    - Tier 2: FastAPI static asset serving and WebSocket protocol handshake tests.
    - Tier 3: End-to-end audio streaming, hybrid mic switching, and virtual loopback tests.
    - Tier 4: Latency benchmarks verifying atomic drain < 0.05ms and end-to-end barge-in < 50ms.
    - Tier 5: Adversarial edge case, race condition, and stress testing.
  - Maintain 100% headless test stability without physical audio hardware dependencies.
- **Verification Gates**: 100% test pass rate across all tiers in `pytest -v`, zero flaky tests.

### 2.5. Documentation Custodian (`custodian_doc_1`)
- **Domain Scope**: `pyproject.toml`, `requirements.txt`, `MEMORY.md`, `CHANGELOG.md`, `AGENTS.md`
- **Responsibilities**:
  - Development environment baseline and dependency locking (`fastapi`, `uvicorn`, `websockets`, `httpx`).
  - Git branching discipline on `feature/unified-hud-websocket-server` with conventional commit syntax.
  - Architectural Decision Record (ADR) synchronization and WebSocket protocol schemas in `MEMORY.md`.
  - Comprehensive changelog updates adhering to Keep a Changelog in `CHANGELOG.md`.
  - Multi-agent cluster topology and operational registry maintenance in `AGENTS.md`.
- **Verification Gates**: Clean git working tree, verified 56 existing regression tests pass on baseline changes.

---

## 3. Sub-Orchestrator Delegation & Cluster Coordination Protocols

To scale development without context degradation, each Sub-Orchestrator coordinates dedicated workers or discrete modular execution cycles:

### 3.1. Workspace Isolation Discipline
- Every active agent operates within an exclusive directory under `.agents/teamwork/<agent_name>/`.
- `.agents/teamwork/` is reserved strictly for operational metadata (`DISPATCH.md`, `BRIEFING.md`, `progress.md`, `handoff.md`).
- Source code, tests, and documentation must never be written into `.agents/teamwork/`.
- Strict exclusive write boundaries: Agents may only modify files within their assigned domain ownership.

### 3.2. Communications & Handoff Standard
- Inter-agent coordination is conducted via `send_message` with structured headers (`Context`, `Content`, `Action`).
- Completed deliverables require a formal 5-component `handoff.md` report:
  1. **Observation**: Exact file paths, line numbers, tool output verbatim.
  2. **Logic Chain**: Direct causal steps from evidence to conclusion.
  3. **Caveats**: Scope boundaries, assumptions, uninvestigated areas.
  4. **Conclusion**: Concrete status and artifact locations.
  5. **Verification Method**: Exact terminal commands to independently verify claims.

### 3.3. Quality Review Gates (Verification Before Completion)
- No agent may declare a task complete without executing `pytest` on relevant targets and providing verbatim pass logs.
- Hardcoded test outputs, mock shortcuts, or facade implementations are strictly prohibited and flagged by the Forensic Auditor.

### 3.4. Dedicated Interactive UI/UX Design System Conformance
- High-fidelity visual assets and CSS tokens strictly adhere to `DESIGN.md` ("The Catalog Sheet"):
  - Palette: Ground `#0b0b0b`, Ink `#efede6`, Signal Yellow `#e8b622`, Fault Red `#e03a2f`, Sleeve `#121211`.
  - Typography: Local Archivo (125% width display) and Martian Mono (tabular numerals) in `ui/fonts/`.
  - Layout: 44px tabbed controls with right-edge action tabs and 0px border radius.


---

## 4. Operational Guidelines for Future Subagents

1. **No Shared State Without Locks**: Any cross-thread communication must be guarded by `threading.Lock` or `threading.RLock`.
2. **Deterministic Budgets**:
   - Audio buffer drain: $< 5\text{ms}$.
   - Prompt context: $\le 2048\text{ tokens}$.
   - VRAM usage: $\le 3100\text{ MB}$ (leaving $\ge 500\text{ MB}$ headroom on RTX 3050).
3. **Verification Before Completion**: Never declare a task complete without executing `pytest` on the relevant test suite and confirming 100% test pass evidence.


---

## 5. Sprint 2: Post-Sprint Adversarial Scrutiny & Competitive Resolution Protocol

*Executed immediately upon completion of the Unified HUD & WebSocket Server development sprint.*

### 5.1. Phase 1: Adversarial Scrutiny Pass (4 Specialized Auditors)
*Skills Applied: `/code-review-and-quality` (5 axes) and `/bugs-are-annoying` (adversarial bug hunting)*
1. **Auditor 1 (`WebSocket & Protocol Auditor`)**:
   - Focus: `src/voice_agent/server/`, WebSocket framing, connection lifecycle, state race conditions.
2. **Auditor 2 (`Audio Loopback & DSP Auditor`)**:
   - Focus: `src/voice_agent/audio/virtual_device.py`, 24kHz PCM downmix, sub-50ms atomic drain verification.
3. **Auditor 3 (`VAD & Hybrid Mic Auditor`)**:
   - Focus: Browser WebAudio vs sounddevice hybrid bridge, 30ms hop alignment, false-positive barge-in resistance.
4. **Auditor 4 (`Frontend HUD & Canvas Auditor`)**:
   - Focus: `ui/index.html`, `ui/app.css`, 60 FPS canvas performance, word cursor drift, memory leaks.

**Action**: All findings compiled into `bugs.md` and posted via GitHub CLI:
- Individual GitHub issues created (`gh issue create`).
- PR review comments posted on active feature branches.

### 5.2. Phase 2: Competing Solution Architecture (2 Cross-Communicating Teams)
All issues triaged into two independent architecture teams cross-communicating via `send_message`:
* **Team Alpha (Subagents A1 & A2)**: Lead DSP/Audio Engineer + Systems Architect -> `docs/TEAM_ALPHA_UNITED_PROPOSAL.md`.
* **Team Beta (Subagents B1 & B2)**: Concurrency/Network Lead + Frontend Performance Architect -> `docs/TEAM_BETA_UNITED_SOLUTION_PROPOSAL.md`.

### 5.3. Phase 3: Lead Evaluation & Fixer Execution (4 Fixers)
*Skills Applied: `/test-driven-development` and `/verification-before-completion`*
- The Lead Coordinator compares Alpha and Beta proposals, selects the superior technical moves, and dispatches 4 Fixer subagents (`Fixer 1: Protocol`, `Fixer 2: Audio/VAD`, `Fixer 3: HUD/Canvas`, `Fixer 4: Integration`).
- All fixes verified with 100% pytest pass evidence before issue closure.

### 5.4. Phase 4: Verification & Operational Resolution
- **Fixer 1 (WebSocket Protocol & Concurrency)**: Implemented non-blocking concurrent fan-out broadcast via `asyncio.gather` and atomic set-difference dead socket pruning under lock, resolving BUG-001. Added structured future exception routing with loop liveness guards, resolving BUG-004.
- **Fixer 2 (Audio DSP & Hardware Abstraction)**: Flushed FIFO accumulator and reset IIR filter states while preserving `noise_floor_dbfs` on mic switch, resolving BUG-002. Implemented bounded monotonic playhead tracking in `read_into()`, resolving BUG-005. Sliced odd bytes defensively, resolving BUG-006. Added empty array guards in resamplers, resolving BUG-008. Precomputed half-cosine micro-fadeout window in `AudioRingBuffer.__init__`, resolving BUG-011. Raised normalization threshold to 256.0, resolving BUG-013. Exposed configurable `onset_frames`, resolving BUG-014.
- **Fixer 3 (Frontend HUD & Visualizers)**: Centralized teleprompter timer management in `clearNarrativeTimer()` with 50ms timestamp debouncing, resolving BUG-007. Suspended canvas RAF rendering on `document.hidden`, added `visibilitychange` listener to resynchronize Web Audio clock, and clamped telemetry bins to 240, resolving BUG-009. Added metric-compatible system font fallbacks in `:root` and enforced `font-synthesis: none;`, resolving BUG-010.
- **Fixer 4 (Test Suite Modernization & Verification)**: Migrated all test suites to `from fastapi.testclient import TestClient`, resolving BUG-003. Expanded test coverage to 116 tests across 5 tiers. Full test suite executes with 100% pass rate (116/116 tests passing in 11.07s).

---

## 6. Sprint 3: Document Ingestion, Speech Synthesis & Web Grounding Topology

*Executed to operationalize real-time audiobook narration from user documents with live internet query grounding.*

### 6.1. Domain Lead Assignments
1. **Narration & Audio Synthesis Lead**:
   - Built `voice_agent.server.narration.NarrationEngine` and `voice_agent.tts.synthesizer`.
   - Microsoft `edge-tts` character voice mapping with Windows SAPI5 (`pyttsx3`) local fallback.
   - Word cursor teleprompter synchronization broadcasting `cursor_update` and binary audio chunks over WebSocket.
2. **Document Ingestion Lead**:
   - Built `voice_agent.document.parser.DocumentParser`.
   - Robust multi-format parsing for PDF (`pypdf`), EPUB (XHTML/HTML zip extraction), Markdown, and Plain Text.
   - REST endpoints `POST /api/upload-document` and `GET /api/document`.
3. **LLM Provider & Search Grounding Lead**:
   - Extended `voice_agent.llm.client.LLMClient` supporting Ollama (`http://localhost:11434`), OpenAI, Groq, Together, DeepSeek.
   - Live DuckDuckGo internet web search grounding (`search_web` / `ddgs`) dynamically querying external web citations on user interruption questions.
   - Local extractive fallback formulation preventing pipeline crashes when external models are offline.
4. **HUD Design & Audio Frontend Lead**:
   - Modernized Web Audio DAC in `ui/index.html` to hardware-decode MP3/WAV/PCM chunks via `audioContext.decodeAudioData()`.
   - Master volume slider (0-100%) and mute toggle.
   - Added Document Ingestion Modal (`#document-upload-modal`) with drag-and-drop file upload.
   - Added Model & Web Search Configuration Modal (`#model-config-modal`).
   - Added Interactive Question Interruption bar with live web search status display.
5. **Testing & Verification Lead**:
   - Created `tests/test_document_and_api.py`.
   - Validated complete test suite: 123/123 tests passing with 100% pass rate.

---

## 7. Sprint 4: Production Audio Studio, Direct Barge-In & Authentic Character Profiles

*Executed to transition UI from an ability/benchmark showcase to an actual production-grade software experience, implement direct microphone barge-in, and integrate authentic Cyberpunk character voices and personas.*

### 7.1. Domain Lead Assignments
1. **Character Voice & Audio Prosody Lead**:
   - Implemented authentic Cyberpunk neural voice profiles and calibrated prosody in `voice_agent.tts.synthesizer`:
     - Johnny Silverhand: `en-US-RogerNeural` (Keanu baritone, pitch `-14Hz`, rate `-5%`).
     - Lucy Kushinada: `en-US-AnaNeural` (intimate Netrunner whisper, pitch `-3Hz`, rate `-4%`).
     - David Martinez: `en-US-BrianNeural` (Santo Domingo edgerunner, pitch `+4Hz`, rate `+6%`).
     - V (Male): `en-US-GuyNeural` (mercenary solo raspy baritone, pitch `-6Hz`, rate `+3%`).
     - V (Female): `en-US-AriaNeural` (tactical solo dynamic sharp, pitch `-3Hz`, rate `+4%`).
   - Validated Edge-TTS pitch string formatting adhering to `^[+-]\d+Hz$`.
2. **Character Dialogue & Persona Lead**:
   - Authored character-specific system prompts and response templates in `voice_agent.llm.client`.
   - Formulates responses adhering to game lore, authentic street slang ("choom", "preem", "Sandevistan", "ICE", "corpo"), and distinctive character emotional tones.
3. **Direct Microphone Barge-In Lead**:
   - Integrated Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) for real-time speech-to-text input.
   - Built dual barge-in interaction:
     - Push-to-Talk / Click-to-Speak button with pulsing recording ring (`@keyframes pulseMic`) and keyboard shortcuts (`Space` / `V`).
     - Hands-Free Voice Detection: Web Audio RMS threshold monitoring automatically interrupting playback and capturing user questions when speaking aloud.
   - Fixed microphone audio feedback loop by isolating processor output with a zero-gain muted monitor.
4. **Production Studio UI/UX Lead**:
   - Modernized `ui/index.html` and `ui/app.css` from a benchmark lab demo to **Cybervox Audio Studio**.
   - Cleaned test labels ("VA 010", "VA 020", "SIMULATE BARGE-IN", "BUFFER SINK").
   - Implemented an illuminated reading teleprompter with active sentence highlight, player transport bar (Prev, Next, Play/Pause, Speed), and audio autoplay unlock alert.
5. **Testing & Verification Lead**:
   - Verified 123/123 tests passing with 100% success rate.

---

## 8. Sprint 4.1: Web Grounding Sanity, Typo Tolerance & Resumption Continuity

*Executed to eliminate web snippet truncation ('...'), support phonetic speech-to-text typos (e.g. 'arakasa' -> Arasaka), search complete document notes, and restore seamless post-barge-in narration resumption.*

### 8.1. Domain Lead Assignments
1. **Web Grounding & NLP Lead**:
   - Built `sanitize_web_snippet` stripping citation tags (`[1]`), date prefixes, and Unicode artifacts.
   - Normalized ellipses (`...`, `…`) to prevent false sentence termination.
   - Implemented 2-sentence substantive fact scoring prioritizing definition markers ("is a", "megacorporation").
   - Integrated ASCII normalization preventing Windows `cp1252` encoding crashes.
2. **Document Ingestion & Context Matching Lead**:
   - Built `search_document_passages` searching the full loaded document (`self.sentences`) with token matching.
   - Added phonetic typo tolerance for Cyberpunk lore terms (`arakas` / `arakasa` -> `arasaka`, `sande` -> `sandevistan`).
3. **State Machine & Streaming Resumption Lead**:
   - Fixed `triggerBargeIn()` removing rogue offline simulation timers that stomped over live WebSocket state.
   - Server dynamically waits for question answer playback duration before broadcasting pre-roll resumption cues (`PromptEnvelopeMessage(state="NARRATING", pre_roll_words=4)`).
   - Ensured atomic restart of `NarrationEngine._narration_loop()` post-interruption.
4. **Testing & Verification Lead**:
   - Re-verified full test suite: 123/123 tests passing with 100% success rate.

---

## 9. Sprint 5: Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation

*Executed to implement local offline Faster-Whisper speech-to-text recognition, speech turn accumulation with pause segmentation, WebSocket transcript streaming, REST transcription endpoint, and HUD client speech synchronization.*

### 9.1. Domain Lead Assignments
1. **Neural STT Recognition Lead**:
   - Implemented `STTEngine` (`voice_agent.stt.recognizer`) using `faster-whisper` (`tiny.en`, INT8 on CPU) delivering $< 350$ms latency.
   - Built resilient multi-tier fallback (Faster-Whisper -> SpeechRecognition -> `MockSTTEngine`).
   - Integrated zero-noise amplitude gating (< 0.01 peak) to prevent silence hallucinations.
2. **Speech Turn & Conversational Pause Lead**:
   - Developed `SpeechTurnAccumulator` (`voice_agent.stt.accumulator`) buffering real-time 16kHz PCM chunks.
   - Implemented energy thresholding VAD and automatic silence pause detection ($> 600$ms trailing silence after $\ge 300$ms speech) for hands-free utterance segmentation.
   - Handled explicit `voice_end` / `speech_end` client flush signals.
3. **WebSocket Streaming & Interruption Orchestration Lead**:
   - Defined typed Pydantic protocol messages: `UserSpeechTranscriptMessage` and `VoiceEndMessage`.
   - Wired WebSocket binary audio ingestion to `SpeechTurnAccumulator` during `LISTENING` and `BARGE_IN`.
   - Automatically piped recognized user transcripts into `NarrationEngine.handle_user_interruption(query)`.
   - Built REST `POST /api/transcribe` endpoint for multipart audio file transcription.
4. **Client Voice Synchronization & HUD UI Lead**:
   - Updated `ui/index.html` and `ui/app.css` to stream 16-bit PCM microphone chunks over WebSocket.
   - Rendered live speech recognition transcripts into query input and speech ticker.
   - Automated query submission and prompt envelope display upon receiving final server transcripts.
5. **Testing & Benchmark Verification Lead**:
   - Built end-to-end voice barge-in test in `tests/test_e2e_pipeline.py`.
   - Implemented STT transcription latency benchmark in `tests/test_benchmarks.py`.
   - Verified 100% test pass rate across the full test suite (148+ tests).

---

## 10. Sprint 5.1: Conversational Speech Sensitivity & Grounding Disambiguation

*Executed to eliminate premature voice barge-in timeouts on desktop microphones, preserve initial plosives via pre-roll audio buffering, upgrade web search grounding to `ddgs`, and disambiguate character queries.*

### 10.1. Domain Lead Assignments
1. **Audio DSP & VAD Sensitivity Lead**:
   - Lowered `SpeechTurnAccumulator.energy_threshold` to `0.015` and expanded conversational pause timeout to `1400ms`.
   - Implemented 4-chunk rolling pre-roll ring buffer in `SpeechTurnAccumulator` capturing initial consonants before speech onset detection.
   - Reduced silence amplitude gate in `STTEngine` to `0.004` peak to allow Faster-Whisper Silero VAD to process faint voices.
2. **Web Grounding & Disambiguation Lead**:
   - Upgraded search library to `ddgs` (v9.16.1).
   - Added character contextual anchoring: queries for character names automatically append "Cyberpunk" if not present.
   - Enhanced `sanitize_web_snippet` to eliminate leading dangling punctuation.
   - Enforced query term presence and hominid/fossil filtering in extractive fallback formulation.
3. **HUD Audio & Protocol Lead**:
   - Decoupled Web Speech API timeout in `ui/index.html` from stopping the active microphone session.
   - Lowered client-side RMS threshold to `0.015` and increased pause timeout to `1800ms`.
   - Added `fallback_query` to `VoiceEndMessage` for zero-loss speech preview fallback.
4. **Testing & Verification Lead**:
   - Verified 149/149 tests passing with 100% success rate.



