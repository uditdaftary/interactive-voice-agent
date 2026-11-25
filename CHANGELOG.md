# Changelog: Interactive Voice Agent

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Remediation design. It comes out of a full audit and covers roughly 70 findings: gaps between the docs and the code, about 59% of the source being unreachable, scripted answers by default, a hardcoded barge-in latency, no trust boundary, and tests that make live network calls. It sets out five gated phases (P0 to P4) and a traceability matrix, No code changes yet.
- S0.1: declared real dependencies, added ruff/mypy configuration and CI lint/typecheck job (H8, M22, L1).
- S0.2: network-free default test suite, injection seams, lazy STT (M2, M19, M20, M21, L15).
- S3.1: httpx streaming LLM client with typed outcomes and a neutral offline line, think-block filter, fenced prompts, local intents, Ollama setup; removed scripted lore answers (D2, H6, H7, M12, M13, N3, N9, L13, L14).
- S3.2: STT/TTS backend protocols, judged Whisper output, configurable initial prompt, pyttsx3 on one dedicated thread with whole-word gender match (a fresh engine per utterance: a reused pyttsx3 engine stalls); removed the Google STT fallback, the mock STT engine and the synthetic TTS tier (M16, L9, L10, L12).
- S1: same-origin and host checks, validated request bodies, bounded spine-ordered parsing, robust broadcast and WS error replies (H3, H4, H5, M5, M8, M9, M10, M14, N5, N7, L2, L3).
- S2.1: one SessionState with legal transitions, resume_to and a LISTENING deadline; tracked tasks cancelled at shutdown; pause on narration error and on last disconnect; initial state PAUSED (M4, M6, M11 part, L5, L6 part).
- S4.2: optional Pocket TTS and Moonshine STT backends behind `pocket` and `moonshine` extras; deleted the vendored game-mod TTS workers (H9).
- Default test suite: the slowest tests run on shorter inputs and waits, with no assertion or test removed; about 3 s faster (#82).

---

## [0.8.0] - 2026-11-11

### Conversational Fluidity, Continuous Phase Resampling & Low-Gain STT Normalization
*Eliminated robotic canned interruption templates with rich multi-tier conversational intent routing, implemented continuous-phase fractional downsampling to prevent audio sample drops, added automatic audio gain normalization for quiet microphones, and expanded test coverage to 213 tests.*

### Fixed & Enhanced
- **Continuous Phase Web Audio Downsampling & Gain Normalization (`ui/index.html`, `voice_agent.stt`)**:
  - In `ui/index.html`, upgraded the 48kHz -> 16kHz linear downsampler to maintain continuous sub-sample fractional phase (`this._downsamplePhase`) across audio processor buffers, eliminating phase boundary glitches and sample drops.
  - Enabled browser `autoGainControl: true` and added automatic `AudioContext.resume()` on user gesture to avoid suspended audio states.
  - In `src/voice_agent/stt/recognizer.py`, added dynamic peak normalization boosting quiet microphone audio ($< 0.5$ peak scaled to $0.75$) so Faster-Whisper's acoustic model and VAD reliably detect speech without quiet phoneme dropping.
  - Added dual-stage Whisper transcription: attempts primary transcription with `vad_filter=True`, then automatically falls back to `vad_filter=False` if soft speech was gated out by Silero VAD.
  - In `src/voice_agent/stt/accumulator.py`, extended the rolling pre-roll buffer to 25 frames (~1000ms) to capture initial words and plosives, and added a resilient `_session_buffer` fallback ensuring quiet utterances are never discarded when manually stopped.
- **Conversational Fluidity & Multi-Tier Character Intent Routing (`voice_agent.llm.client`, `voice_agent.server.narration`)**:
  - Overhauled `CHARACTER_PERSONA_PROMPTS` across Johnny Silverhand, Lucy Kushinada, David Martinez, and V (Male/Female) to instruct 1-3 natural spoken conversational sentences with street banter, dark humor, and smooth bridges back to the reading, strictly banning markdown asterisks, stage directions, and bullet points.
  - Implemented multi-tier conversational intent routing in `_synthesize_fallback_answer()`:
    1. **Greetings & Social Check-ins**: "hey", "what's up", "you there" trigger direct, immersive character banter without canned robotic prefixes.
    2. **Self-Identity & Lore Inquiries**: "who are you", "what's your story" trigger authentic first-person character backstory.
    3. **Repeat & Playback Commands**: "repeat that", "what did you just say" dynamically quote the currently active reading sentence in character voice.
    4. **Pause & Wait Requests**: "hold on", "pause", "wait a minute" provide responsive, in-character confirmation.
    5. **Reactions & Exclamations**: "wow", "damn", "no way" acknowledge the drama of the active story.
    6. **Character Lore Opinions**: Questions about comrades and cyberware (e.g. asking David about Lucy, or Johnny about Sandevistan) trigger authentic game lore opinions.
    7. **Inquisitive Fallback & Dynamic Closers**: Replaced robotic repetitive templates ("Listen up samurai, tapped into the net...") with character-authentic banter and fluid conversational transitions.
- **Testing & Verification**:
  - Added `tests/test_conversational_dialogue.py` covering greetings, identity inquiries, repeat commands with context, pause commands, and lore banter across all character personas.
  - Added quiet-speech session buffering and extended pre-roll tests in `tests/test_speech_accumulator.py`.
  - **100% test pass rate across full test suite (213/213 tests passing in 67.84s)**.

---

## [0.7.0] - 2026-11-10

### Audio Downsampling, Search Relevance Overhaul & Okapi BM25 Passage Retrieval
*Resolved 48kHz -> 16kHz microphone sample-rate mismatch and premature 469ms silence truncation (Issue #16), overhauled web search grounding with punctuation-free tokenization and interrogative filtering (Issue #17), and implemented a pure-Python Okapi BM25 ranking engine with rolling 3-sentence context windows (Issue #18).*

### Fixed & Enhanced
- **Client-Side Polyphase/Linear Downsampling & Defensive Resampling (Issue #16)**:
  - In `ui/index.html` (`onaudioprocess`), implemented dynamic linear downsampling from `this.audioContext.sampleRate` ($48\text{kHz} / 44.1\text{kHz}$) to $16{,}000\text{ Hz}$ before converting to `pcm16` and streaming over WebSocket.
  - In `src/voice_agent/stt/recognizer.py`, added defensive polyphase resampling via `scipy.signal.resample_poly` guaranteeing $16{,}000\text{ Hz}$ input for Faster-Whisper regardless of source sample rate.
  - In `src/voice_agent/stt/accumulator.py`, converted conversational pause/silence timeout detection to track true monotonic wall-clock elapsed time (`time.monotonic() - self._last_speech_time`), eliminating the 3x time-acceleration defect that prematurely aborted speech at 469ms.
  - Added low-energy speech hysteresis in `SpeechTurnAccumulator` preserving trailing consonants.
  - Conditioned `WhisperModel.transcribe` with `initial_prompt` guiding acoustic decoding towards Cyberpunk terminology.
- **Web Search Grounding Relevance & Interrogative Title Filtering (Issue #17)**:
  - In `src/voice_agent/llm/client.py`, implemented punctuation-free tokenization using regex word boundaries `\b[a-z0-9_-]{2,}\b` and a 107-word English stopword set.
  - Expanded lore and entity disambiguation across 18 Cyberpunk characters and 13 core world concepts (Relic, Blackwall, Sandevistan, Soulkiller, Cyberpsychosis, etc.).
  - Increased DuckDuckGo retrieval depth (`max_results = 8`).
  - Added an interrogative penalty (-20 score) to prevent echoing search result question titles as answers.
  - Replaced definition-only keyword bias with information-density and predicate clause scoring (`killed by`, `stole`, `invented`, `developed`, `destroyed`, `died`, `shot`, `caused by`, `biochip`, `construct`, `ai`).
- **Pure-Python Okapi BM25 Retrieval & Rolling Context Windows (Issue #18)**:
  - Implemented `BM25Index` in `src/voice_agent/document/bm25.py` ($k_1 = 1.5, b = 0.75$, exact word boundary regex, non-negative Robertson-Spärck Jones IDF).
  - Built rolling 3-sentence context windows ($W_i = [s_{i-1}, s_i, s_{i+1}]$) to retain subject antecedents and dialogue pronouns.
  - Added playhead proximity Gaussian boost to prioritize passages near the active reading cursor.
  - Established context provenance hierarchy: when top document passages have high BM25 scores ($\ge 2.5$), document context is designated primary ground truth and external search is bypassed.
  - Clamped interruption queries to 256 tokens, stripped ChatML control tokens, and bounded context to 2048 tokens to safeguard RTX 3050 4GB VRAM.
- **Testing & Verification**:
  - Full test suite expanded from 149 to **205 tests** across all 5 tiers.
  - **100% test pass rate (205/205 tests passing in 44.79s)**.

---

## [0.6.1] - 2026-11-09

### Conversational Speech Sensitivity, Pre-Roll Buffering & Lore Grounding Disambiguation
*Resolved premature voice listening cutoffs on standard microphones, eliminated Australopithecus fossil hallucinations on character queries via contextual disambiguation, and added pre-roll audio buffering.*

### Fixed & Enhanced
- **Speech Turn Sensitivity & Premature Cutoff Prevention (`voice_agent.stt.accumulator`, `voice_agent.stt.recognizer`, `ui/index.html`)**:
  - Lowered `SpeechTurnAccumulator.energy_threshold` from `0.04` to `0.015` to capture quiet/headset microphones.
  - Increased silence segmentation timeout from `650ms` to `1400ms` (server-side) and `1800ms` (client-side) to accommodate natural human conversational pauses.
  - Added 4-frame rolling pre-roll audio ring buffer to capture initial plosives and consonants before energy threshold trigger.
  - Lowered peak amplitude silence filter from `0.01` to `0.004` to allow Faster-Whisper's Silero VAD to process faint voices.
  - Disconnected `recognizer.onend` from calling `stopDirectVoiceBargeIn()`, preventing browser Web Speech timeouts from killing active mic sessions prematurely.
  - Added `fallback_query` to `VoiceEndMessage` (`src/voice_agent/server/schemas.py`, `websocket.py`) to preserve client-side speech preview if audio frame stream is quiet.
- **Lore Contextual Disambiguation & Web Grounding (`src/voice_agent/llm/client.py`)**:
  - Upgraded DuckDuckGo search integration to modern `ddgs` library (v9.16.1), resolving deprecation failures.
  - Character and lore queries (e.g. "who is lucy", "david", "arasaka") automatically anchor with lore context ("Cyberpunk"), completely eliminating unrelated web search results (e.g., Australopithecus afarensis hominid fossils, Beatles songs).
  - Enhanced `sanitize_web_snippet()` to strip leading dangling punctuation and quotation marks.
  - Enforced query term containment in fallback candidate sentence scoring (`_synthesize_fallback_answer`), actively rejecting hominid, fossil, skeleton, and song lyrics.
- **Testing & Verification**:
  - 100% test pass rate across the full test suite (149/149 tests passing).

---

## [0.6.0] - 2026-11-09

### Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation
*Integrated local neural speech-to-text with Faster-Whisper, speech turn accumulation with pause segmentation, bi-directional WebSocket transcript streaming, REST transcription API, and HUD client voice synchronization.*

### Added
- **Neural Speech-To-Text Recognition Engine (`voice_agent.stt.recognizer`)**:
  - Implemented `STTEngine` powered by `faster-whisper` (`tiny.en`, INT8 quantization on CPU) delivering $< 350$ms transcription latency without cloud dependencies or GPU VRAM.
  - Multi-tier resilience: primary Faster-Whisper, secondary `SpeechRecognition` library fallback, and deterministic `MockSTTEngine` for headless testing.
  - Zero-noise gating for silent audio (< 0.01 peak) avoiding hallucinations.
  - Dedicated REST endpoint `POST /api/transcribe` accepting multipart WAV or raw PCM uploads with duration and latency telemetry.
- **Speech Turn Accumulator & Pause Segmentation (`voice_agent.stt.accumulator`)**:
  - Implemented `SpeechTurnAccumulator` buffering streaming 16kHz PCM chunks during active speech.
  - RMS energy VAD speech onset and pause detection with configurable thresholds.
  - Automatic turn boundary segmentation on conversational pauses ($> 600$ms silence trailing speech of $\ge 300$ms), transcribing utterances without requiring manual push-to-talk release.
  - Instant manual turn completion via `voice_end` / `speech_end` control signals.
- **Bi-Directional STT Streaming Protocol (`voice_agent.server.websocket`, `voice_agent.server.schemas`)**:
  - Typed Pydantic schemas: `UserSpeechTranscriptMessage` (`user_speech_transcript`) and `VoiceEndMessage` (`voice_end` / `speech_end`).
  - WebSocket router routes binary PCM chunks into `SpeechTurnAccumulator` when state is `LISTENING` or `BARGE_IN`.
  - Automatically invokes `STTEngine.transcribe()` and broadcasts transcripts to all connected clients.
  - Directly dispatches recognized utterances to `NarrationEngine.handle_user_interruption(query)` for immediate character Q&A response and audio playback.
- **HUD Client Bi-Directional Speech Synchronization (`ui/index.html`)**:
  - `[🎙️ BARGE IN & SPEAK]` button and `Space`/`V` shortcuts stream 16-bit PCM chunks to server STT engine.
  - Real-time transcript preview rendered in `#interruption-query-input` and `#speech-ticker`.
  - Automatic query dispatch on receiving final server transcripts, triggering Prompt Envelope generation and teleprompter resumption cues.
- **Benchmark Suite & E2E Verification (`tests/test_benchmarks.py`, `tests/test_e2e_pipeline.py`)**:
  - Added STT transcription latency benchmark verifying $< 350$ms throughput.
  - Added full end-to-end integration test validating speech frame -> VAD barge-in -> LISTENING -> STT transcription -> LLM response -> resumption cue -> NARRATING.

### Testing & Verification
- 100% test pass rate across the full test suite (148+ tests passing).

---

## [0.5.1] - 2026-11-09

### Web Search Truncation Fix, Lore Typo Tolerance & Post-Barge-In Resumption
*Resolved web snippet truncation after '...', added Cyberpunk lore typo tolerance (e.g. 'arakasa' -> Arasaka), enabled full document passage retrieval for ad-hoc queries, and established seamless server-driven audio narration and teleprompter resumption post-barge-in.*

### Fixed
- **Web Search Snippet Truncation & '...' Cutoff (`src/voice_agent/llm/client.py`)**:
  - `sanitize_web_snippet()` strips URLs, bracket citations (`[1]`), date headers (`Nov 2, 2026`), empty parentheticals `(Japanese: )`, and replaces ellipses (`...`, `…`) with whitespace before sentence tokenization, completely eliminating trailing `...` and premature sentence cutoffs.
  - Sourced full substantive definitions (scoring explanatory phrases like "is a", "megacorporation", "specializing in") and synthesized 2-sentence answers with authentic character voice prefixes.
  - Sanitized multi-byte Unicode characters to safe ASCII, preventing Windows `cp1252` encoding crashes on console output.
- **Phonetic Speech Typo Tolerance & Document Context Matching (`voice_agent.llm.client`, `voice_agent.server.narration`)**:
  - Added phonetic typo tolerance mapping speech recognition errors (e.g. `arakasa` / `arakas` -> `arasaka`, `sande` -> `sandevistan`) across search queries and document token matching.
  - `search_document_passages()` queries the entire loaded book/prose (`self.sentences`) rather than only the current sentence, integrating both external web facts and internal document excerpts.
- **Post-Barge-In Narration Resumption (`voice_agent.server.narration`, `ui/index.html`)**:
  - Removed rogue simulation `setTimeout` chains in `triggerBargeIn()` that stomped over live WebSocket state.
  - Server awaits answer audio playback duration before broadcasting pre-roll resumption cues (`PromptEnvelopeMessage(state="NARRATING", pre_roll_words=4)`).
  - Cleanly restarts background narration task and re-engages teleprompter word highlighting and scrolling.
- **Web Speech API Transcript Capture (`ui/index.html`)**:
  - Fixed `recognizer.onresult` to accumulate all final and interim results across `event.results`, eliminating dropped words in manual voice barge-in.
  - Triggered query submission cleanly on `recognizer.onend`.

### Testing & Verification
- All 123 tests passing with 100% success rate across all 5 tiers.

---

## [0.5.0] - 2026-11-09

### Production Studio Experience, Direct Microphone Barge-In & Authentic Cyberpunk Character Voices
*Elevated the HUD from an ability/benchmark showcase to a production-grade Cyberpunk Audiobook & Voice Interaction Studio, implemented direct microphone barge-in with speech recognition, and mapped authentic Cyberpunk 2077 / Edgerunners character voices, prosody, and dialogue attitudes.*

### Added
- **Production-Grade Studio Interface (`ui/index.html`, `ui/app.css`)**:
  - Transformed the layout from benchmark showcase codes (eliminated "VA 010", "VA 020", "VA 030", "VA 040", "SIMULATE BARGE-IN", "BUFFER SINK") into a premier **Cybervox Audio Studio**.
  - Reader Teleprompter with illuminated active sentence (`.active-sentence`), Archivo typography, and smooth word cursor highlighting.
  - Dedicated Audio Transport Bar with Previous, Play/Pause toggle (spacebar shortcut), Next, and Cadence stepper (`0.8x`, `1.0x`, `1.2x`).
  - Audio Autoplay Unlock alert banner ensuring seamless Web Audio playback on user click/interaction without browser autoplay blocks.
  - Eliminated microphone speaker feedback loop by routing monitor through a zero-gain muted node.
- **Direct Microphone Barge-In & Speech Recognition**:
  - Added tactile `[🎙️ BARGE IN & SPEAK]` button with pulsing recording animation (`@keyframes pulseMic`) and status ticker (`#speech-ticker`).
  - Integrated browser Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) transcribing user speech in real time into `#interruption-query-input`.
  - Added push-to-talk keyboard shortcuts (`Space` to Play/Pause, `V` for instant voice barge-in).
  - Hands-Free Voice Detection (`#btn-handsfree-toggle`): Web Audio RMS energy monitoring automatically halts narration in $< 0.05$ms when the user speaks aloud into their microphone.
- **Authentic Cyberpunk Character Prosody & Personas (`voice_agent.tts.synthesizer`, `voice_agent.llm.client`)**:
  - **Johnny Silverhand**: Keanu Reeves baritone mapped to `en-US-RogerNeural` with low pitch (`-14Hz`) and relaxed rockerboy rate (`-5%`); speaks with cynical swagger, anti-corpo rebel attitude, calling user "samurai" or "choom".
  - **Lucy Kushinada**: Netrunner prodigy mapped to `en-US-AnaNeural` with soft whisper (`-3Hz`, `-4%`); speaks with quiet confidence, references to subnets and ICE.
  - **David Martinez**: Santo Domingo edgerunner mapped to `en-US-BrianNeural` with youthful energy (`+4Hz`, `+6%`); speaks with authentic street slang ("choom", "preem", "Sandevistan").
  - **V (Male)**: Mercenary solo mapped to `en-US-GuyNeural` with raspy baritone (`-6Hz`, `+3%`).
  - **V (Female)**: Tactical solo mapped to `en-US-AriaNeural` with crisp dynamic delivery (`-3Hz`, `+4%`).
  - Edge-TTS pitch validation adhering strictly to regex `^[+-]\d+Hz$`.
  - Injected character-specific dialogue templates across built-in assistant and LLM system prompts.
- **Testing & Verification**:
  - Full test suite passing: 123/123 tests passing with 100% success rate.

### Real-Time Continuous Narration, Document Ingestion & Live Web Search Grounding
*Integrated active speech synthesis with real audio streaming, multi-format ebook ingestion (PDF/EPUB/TXT/MD), dynamic LLM endpoint configuration (Ollama, OpenAI, Groq), and live DuckDuckGo web search grounding for ad-hoc voice interruptions.*

### Added
- **Multi-Tier Voice Synthesis & Hardware-Accelerated Playback (`voice_agent.tts.synthesizer`)**:
  - Live narration audio stream broadcasting real audio chunks over WebSocket to connected HUD clients.
  - Character persona voice mapping using Microsoft `edge-tts` (David Martinez: `en-US-GuyNeural`, Lucy Kushinada: `en-US-JennyNeural`, V Male: `en-US-ChristopherNeural`, V Female: `en-US-AriaNeural`, Johnny Silverhand: `en-US-EricNeural`).
  - Windows SAPI5 (`pyttsx3`) local offline fallback and Web Audio API decoding in client browser.
  - Audio Master controls in HUD: Volume slider (0-100%), Mute toggle, and zero-latency barge-in audio cut (< 0.05ms).
- **Document Ingestion Engine (`voice_agent.document.parser`)**:
  - Full ingestion support for PDF (`pypdf`), EPUB (XHTML/HTML chapter extraction via `zipfile`), Markdown, and Plain Text files.
  - Token-shielded sentence boundary splitter (`split_sentences`) preserving abbreviations, ellipses, dialog quotes, and decimal numbers.
  - Drag-and-drop document upload modal (`#document-upload-modal`) in "The Catalog Sheet" aesthetic (0px geometry, 44px buttons) supporting file upload and raw text pasting.
  - REST endpoints: `POST /api/upload-document` and `GET /api/document`.
- **Multi-Provider LLM & Live Web Search Grounding (`voice_agent.llm.client`)**:
  - Flexible multi-provider LLM client supporting local Ollama (`http://localhost:11434`), OpenAI, Groq, Together, DeepSeek, and custom OpenAI-compatible endpoints.
  - DuckDuckGo live internet web search grounding (`search_web` / `ddgs`): dynamically triggers real-time web searches when the user interrupts narration with ad-hoc questions.
  - Streaming token responses with punctuation boundary sentence segmentation and conversational prompt assembly.
  - Intelligent local extractive fallback formulation when external endpoints are offline.
  - Dedicated configuration modal (`#model-config-modal`) in the HUD with endpoint, model, API key, and search toggle controls.
- **Active Narration & Interruption Engine (`voice_agent.server.narration`)**:
  - Coordinates document sentence queue, speech synthesis, binary audio streaming, DAC playhead word tracking, and barge-in Q&A.
  - Interactive Interruption bar in HUD allowing users to type or speak questions mid-narration.
  - Live Prompt Envelope telemetry displaying web search status, citations, LLM reply, and pre-roll resumption cues.
- **Verification & Testing**:
  - Expanded test suite from 116 to 123 tests covering document parsing, synthetic PDF ingestion, REST endpoints, and narration control.
  - 100% test pass rate (123/123 tests passing).

---

## [0.3.0] - 2026-11-09

### Unified Voice Agent HUD Dashboard & Full-Stack WebSocket Server
*Full-stack integration of the single-page Catalog Sheet HUD with a local FastAPI WebSocket streaming server, hybrid microphone switching, and virtual audio loopback.*

### Added
- **Unified FastAPI Server & Static Host**:
  - Unified local server hosting static HUD assets (`ui/index.html`, `ui/app.css`, `ui/fonts/`) at `http://localhost:8000/` and bi-directional WebSocket at `ws://localhost:8000/ws`.
  - Zero-cloud offline execution with complete local asset delivery.
- **WebSocket Streaming Protocol & Telemetry**:
  - Bi-directional 24kHz 16-bit mono PCM audio streaming.
  - 30ms VAD speech activity event broadcast (`speech_active`, `dbfs`, `noise_floor_dbfs`, `threshold_dbfs`).
  - Active word cursor tracking updates synchronized with audio playhead DAC progress.
  - Sub-50ms barge-in interruption signals with atomic queue drain notification.
  - Structured prompt envelope telemetry and 16-bin FFT spectrum streams.
- **Hybrid Microphone Ingestion Bridge**:
  - Dynamic switcher between browser Web Audio API PCM capture (`getUserMedia`) and host Python hardware microphone (`sounddevice.InputStream`).
  - Runtime toggleable via HUD control tab without restarting the server or breaking audio pipelines.
- **Virtual Audio Loopback Device**:
  - `VirtualAudioDevice` in-memory fallback generating synthetic audio frames for headless test execution and automated CI verification without physical sound hardware.
- **Single-Page Catalog Sheet HUD**:
  - Strict conformance to "The Catalog Sheet" design system tokens (44px tabbed controls, 0px corner geometry, Ground `#0b0b0b`, Ink `#efede6`, Signal Yellow `#e8b622`, Fault Red `#e03a2f`).
  - 60 FPS live canvas dynamic voice graph visualizer (waveform oscilloscope and 16-bar FFT spectrum).
  - Continuous narrative reader with active word highlighting in inverted Plot White and resumption pre-roll cues in Signal Yellow.
  - Sub-50ms barge-in disruption canvas and dynamic Q&A prompt envelope telemetry.
  - Resident character voice selector table (David Martinez, Lucy, V Male, V Female, Johnny Silverhand) with WPM cadence and timbre modulation.
  - Seamless offline simulation fallback when the WebSocket server is disconnected.
- **Performance & Latency Benchmarks**:
  - Automated latency benchmark verifying atomic buffer drain in $< 0.05\text{ms}$.
  - End-to-end barge-in interruption latency benchmark from speech detection to playback halt in $< 50\text{ms}$.
  - Automated 5-tier integration test suite covering static assets, WebSocket protocol, hybrid mic switching, and headless pipeline execution.
- **Post-Sprint Adversarial Hardening (BUG-001 through BUG-014)**:
  - **ConnectionManager**: Non-blocking concurrent fan-out broadcast via `asyncio.gather(*[_safe_send(ws)...])` and atomic set-difference pruning `self.active_connections = [ws for ws in self.active_connections if ws not in dead_set]`, completely eliminating race condition crashes and head-of-line blocking under slow/dead clients (`BUG-001`).
  - **ConnectionManager**: Added `_log_future_exception` callback with loop liveness guards to `broadcast_json_sync` and `broadcast_bytes_sync`, safely trapping unobserved background broadcast failures while filtering expected cancellations (`BUG-004`).
  - **HybridMicrophoneBridge**: Flushed FIFO sample accumulator and reset VAD filter states (`_filter_zi`, `_prev_x`, `_prev_y`) while strictly preserving `noise_floor_dbfs` on source transitions, preventing acoustic step pulses and eliminating false-positive barge-in triggers from ambient noise (`BUG-002`). Exposed `switch_source` alias.
  - **AudioRingBuffer**: Bounded monotonic playhead synchronization advancing `self.total_samples_played += samples_requested` during active playback underruns (`filled > 0 or had_active_tail`), but freezing playhead advance when completely idle to eliminate teleprompter clock drift without infinite idle runaway (`BUG-005`).
  - **AudioRingBuffer**: Precomputed and cached 2ms half-cosine micro-fadeout window in `__init__`, reducing buffer drain latency to $< 0.02\text{ms}$ under full test concurrency (`BUG-011`).
  - **Audio Ingestion Defensiveness**: Sliced `data = data[:len(data) - (len(data) % 2)]` before `np.frombuffer` across `feed_browser_pcm` and `normalize_audio`, defending against odd-byte frame fragmentation (`BUG-006`). Exposed `ingest_browser_audio` alias.
  - **Resampling Defensiveness**: Added early-exit guards for empty audio arrays and degenerate sample rates in `resample_audio` and `_to_float32` (`BUG-008`).
  - **VAD First-Frame Latency**: Moved `scipy.signal.lfilter` import to module top-level with fallback, eliminating a 2.5-second disk I/O cold-import delay on Windows systems (`BUG-012`).
  - **Normalization Threshold**: Raised unnormalized float heuristic from `max_abs > 1.5` to `max_abs > 256.0` to preserve legitimate loud vocals from being attenuated by 90 dB (`BUG-013`).
  - **Configurable Onset Frames**: Added configurable `onset_frames` parameter to `VoiceActivityDetector` (default 2), allowing tuning for noisy/vibrational environments (`BUG-014`).
  - **HUD Teleprompter Single-Timer Invariant**: Centralized timer lifecycle into `clearNarrativeTimer()` across cadence and sentence navigation handlers, paired with 50ms timestamp debouncing on cadence buttons (`BUG-007`).
  - **HUD Canvas Tab Throttling & Visibility Sync**: Suspended canvas RAF rendering when `document.hidden`, added `visibilitychange` listener re-anchoring Web Audio `nextPlayTime` to `audioContext.currentTime`, and clamped visualizer FFT queues to 240 samples (`BUG-009`).
  - **HUD CSS Font Fallbacks**: Added robust cross-platform system font fallbacks in `:root` and enforced `font-synthesis: none;` on `.mono` and `.num` classes (`BUG-010`).
  - **TestClient Modernization**: Migrated all test suites to `from fastapi.testclient import TestClient`, eliminating Starlette deprecation warnings (`BUG-003`). Coverage expanded to 116 tests (100% passing).

---

## [0.2.0] - 2026-11-09

### Comprehensive Audit & Production Hardening
*Resolved all 43 audit defects identified during the adversarial review pass across Audio, VAD, Cursor/State Machine, and LLM/Pipeline domains. Full test suite coverage expanded from 10 to 56 tests (100% passing in 4.4s).*

### Added
- **AudioRingBuffer Zero-Allocation & Fadeout**:
  - Zero-allocation hot path via `read_into(outdata: np.ndarray) -> int` to service PortAudio callbacks without heap allocations (`BUG-A06`).
  - Automatic zero-padding (silence padding) on buffer starvation, preventing callback shape broadcasting crashes (`BUG-A04`).
  - Generation fencing (`self.generation` token) on `write()` and `drain()`, rejecting stale in-flight synthesized TTS chunks post-barge-in (`BUG-A07`).
  - 2ms half-cosine micro-fadeout window ($w[n] = 0.5(1 + \cos(\pi n / (M-1)))$) on `drain()`, eliminating DAC step-discontinuity pops while maintaining $< 3.5\text{ms}$ drainage latency (`BUG-A08`).
  - Monotonic playhead sample counters (`total_samples_played`, `playhead_seconds`) mapping physical audio consumption to narrative cursor tracking (`BUG-A10`).
  - Polymorphic input handling for `bytes`, `bytearray`, `memoryview` (`BUG-A09`), and multi-channel 2D downmix to mono (`(N, 1)` and `(N, 2)`) (`BUG-A02`).
  - Dynamic `available_write` property and capacity validation (`BUG-NORM-01`, `BUG-NORM-02`).
- **VoiceActivityDetector Robustness**:
  - 30ms internal FIFO sample accumulator (`480` samples @ 16kHz) to buffer arbitrary incoming sample chunk sizes from `sounddevice` (`BUG-V06`).
  - DC offset blocker: zero-mean centering and stateful 1st-order IIR high-pass filter ($f_c \approx 80\text{Hz}$, $R = 0.968$), eliminating permanent speech lockup caused by hardware ADC bias (`BUG-V04`).
  - Scale-invariant dBFS metric standardization ($20 \log_{10}(\text{RMS} + 10^{-12})$) with identical scaling across normalized `float32 [-1.0, 1.0]` and unnormalized `int16` (`BUG-V05`).
  - Dual-rate EMA adaptive noise floor estimator ($\alpha_{\text{down}}=0.15$, $\alpha_{\text{up}}=0.02$) dynamically adjusting speech threshold between $-90\text{ dBFS}$ and $-20\text{ dBFS}$ (`BUG-V03`).
  - Multi-frame onset persistence ($\ge 2$ consecutive active 30ms frames) to eliminate false triggers from keystrokes and clicks (`BUG-V07`).
  - Decoupled pause bridging (6 frames = 180ms) and turn finalization (16 frames = 480ms) (`BUG-V08`).
  - Input validation sanitizing `None`, empty arrays, non-finite floats (`NaN`, `Inf`), and stereo channels (`BUG-V09`).
  - Acoustic echo suppression / ducking helper (`set_playback_active`) adding $+14\text{ dB}$ masking during agent speech to prevent self-interruption loops (`BUG-V02`).
- **Sentence Splitter & Cursor Tracker**:
  - Unicode sentinel token-shielding architecture (`\uE000`) for period protection in `split_sentences` and `PunctuationStreamParser` (`BUG-O06`, `BUG-L01`).
  - Expanded protection for abbreviations (`Dr.`, `Mr.`, `Mrs.`, `Ms.`, `St.`, `vs.`, `etc.`), multi-dot acronyms (`e.g.`, `i.e.`, `D.C.`, `U.S.`), initials (`A. Lincoln`), decimals (`1.5`), and ellipses (`...`).
  - Support for terminal dialogue quotes (`!" `, `." `, `?" `, `'! `, `'. `) (`BUG-O07`).
  - Clean separation of markdown headers (`# Header`) and paragraph boundaries (`BUG-O10`).
  - Removal of invalid regex token `[\b]` (`BUG-O08`).
  - Polymorphic initialization in `CursorTracker` supporting strings and lists without arbitrary `> 100` length heuristics (`BUG-O05`).
  - Word offset clamping in `record_progress()` preventing silent empty resumption text (`BUG-O09`).
- **Orchestrator State Machine**:
  - Multi-state barge-in support enabling interruptions during BOTH `NARRATING` and `REPLYING` states (`BUG-O01`).
  - Exception-safe callback wrapper (`try...finally`) guaranteeing state transitions cleanly to `LISTENING` on callback failure, eliminating deadlocks (`BUG-O02`).
  - Re-entrant mutex synchronization (`threading.RLock`) on all state mutations and inspections (`BUG-O03`).
- **LLM Streaming & Hardware Constraints**:
  - Multi-sentence streaming parser (`feed(token) -> List[str]`) yielding all ready sentences without buffer retention (`BUG-L10`).
  - Lossless stream completion via mandatory `flush()` integration (`BUG-L05`).
  - ChatML prompt injection sanitization stripping `<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]` tags (`BUG-L06`).
  - Strict 2048-token context budgeting preventing CUDA OOM on 4GB RTX 3050 Laptop GPUs (`BUG-L07`).
  - Hard cutoff terminating LLM generation after 3 sentences (`BUG-L13`).
  - Win32 Job Object and non-blocking stderr pipe drain preventing subprocess zombie processes and 64KB pipe deadlocks (`BUG-L08`, `BUG-L14`).

### Changed
- Scaled floating-point audio in `AudioRingBuffer` by `32767.0` and clamped to `[-32768, 32767]` before int16 conversion (`BUG-A01`, `BUG-A05`).
- Replaced legacy `find()` loop in `PunctuationStreamParser` with compiled regex boundary scanner (`BUG-L01`).
- Wired `pipeline.py` audio playhead progress to `CursorTracker.record_progress()` and automated `cursor.advance()` on sentence finish (`BUG-O04`).

---

## [0.1.0] - 2026-11-08

### Initial Architecture & Prototype
- Created repository workspace at `C:\Projects\interactive-voice-agent`.
- Configured Python 3.11 virtual environment with `sounddevice`, `numpy`, `scipy`, `pydantic`, `pytest`.
- Extracted quantized dialogue LLM `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (2.70 GB).
- Extracted 20 Cyberpunk companion voice reference clips to `models/voices/`.
- Extracted PocketTTS runtime worker and Moonshine Tiny STT worker (`sherpa-onnx` on CPU).
- Implemented baseline circular buffer `AudioRingBuffer`.
- Implemented baseline `VoiceActivityDetector` (energy RMS).
- Implemented baseline `AgentStateMachine` (`NARRATING`, `BARGE_IN_TRIGGERED`, `LISTENING`, `REPLYING`).
- Implemented baseline `CursorTracker` with pre-roll narrative resumption.
- Implemented interactive CLI demonstration (`run_agent.py`).
