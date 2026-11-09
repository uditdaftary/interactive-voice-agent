# Changelog: Interactive Voice Agent

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - Unreleased

### Unified Voice Agent HUD Dashboard & Full-Stack WebSocket Server
*Full-stack integration of the single-page Catalog Sheet HUD with a local FastAPI WebSocket streaming server.*

### Added
- **Unified FastAPI Server & Static Host**:
  - Unified local server hosting both the static HUD assets (`ui/`) and bi-directional WebSocket (`ws://localhost:8000/ws`).
  - Bi-directional 24kHz audio PCM streaming and 30ms VAD event telemetry.
- **Audio Virtual Device & Mock Loopback**:
  - `VirtualAudioDevice` enabling headless test execution and automated CI verification without physical sound hardware.
- **Single-Page Catalog Sheet HUD**:
  - 60 FPS live canvas dynamic voice graph visualizer and FFT spectrum.
  - Continuous narrative teleprompter with word-level cursor token tracking in inverted Plot White.
  - Sub-50ms barge-in disruption canvas with atomic queue drain latency benchmark.
  - Resident character voice selector table (David Martinez, Lucy, V Male/Female, Johnny Silverhand).
- **Post-Sprint Adversarial Quality Protocol**:
  - 4-auditor adversarial review pass (`/code-review-and-quality` and `/bugs-are-annoying`).
  - Competing solution architecture teams (Team Alpha and Team Beta).
  - 4-fixer TDD remediation team with GitHub issue tracking.

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
