# Changelog: Interactive Voice Agent

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Comprehensive Multi-Agent Audit & Production Hardening
*Resolved all 43 audit defects identified during the multi-agent adversarial scrutiny pass across Audio, VAD, Cursor/State Machine, and LLM/Pipeline domains. Full test suite coverage expanded from 10 to 56 tests (100% passing in 4.4s).*

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
