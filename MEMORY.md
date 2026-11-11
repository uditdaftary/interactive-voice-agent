# Interactive Voice Agent: Project Memory & Architecture Decision Record (ADR)

*Persistent Knowledge Store, Hardware Constraints, Mathematical Models, and Invariant Rules*

---

## 1. System Invariants & Hardware Constraints

### 1.1. Target Machine Profile
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB = 4.0 GB VRAM).
- **RAM**: 16 GB System Memory.
- **OS**: Windows 11 x64 (NT Kernel).
- **Python Runtime**: Python 3.11.1 (`C:\Users\Udit\AppData\Local\Programs\Python\Python311\python.exe`) via `.venv`. (Global Python 3.14 lacks binary ML wheels for PyTorch/sherpa-onnx).

### 1.2. The 4GB VRAM Budget (Hardware Law)
```
+---------------------------------------------------------------------------------+
|                       RTX 3050 Laptop GPU (4096 MB VRAM)                        |
+---------------------------------------------------------------------------------+
| [==================== Qwen 3.5 4B Q4_K_M: 2700 MB ====================]         |
| [=========== Windows DWM & Display Buffer: 500 MB ===========]                  |
| [====== CUDA Driver & Runtime Overhead: 300 MB ======]                          |
| [===== KV Cache Headroom (2048 Context, fp16 GQA): ~350 MB =====]               |
| [ Free Safety Headroom: ~246 MB ]                                               |
+---------------------------------------------------------------------------------+
| ⚠️ STRICT INVARIANT: OmniVoice (resident_voice_worker) requires 2000 MB VRAM!   |
| Launching OmniVoice on a 4GB GPU crashes instantly with CUDA Out-Of-Memory.     |
| TTS MUST run on CPU via PocketTTS (pocket_voice_worker.py, CUDA_VISIBLE_DEVICES).|
| STT MUST run on CPU via Sherpa-ONNX (local_stt_worker.py, provider="cpu").      |
+---------------------------------------------------------------------------------+
```

---

## 2. Low-Latency Performance Guarantees

| Metric | Budget | Measured Implementation | Strategy |
| :--- | :--- | :--- | :--- |
| **Barge-In Truncation Latency** | $< 50\text{ ms}$ | **$0.02 - 2.1\text{ ms}$** | Instant ring buffer pointer drain with 2ms half-cosine micro-fadeout. |
| **Acoustic Pop on Interruption** | $0\text{ clicks}$ | **$0\text{ step-discontinuity}$** | Half-cosine window attenuates wave to 0 in 48 samples @ 24kHz. |
| **Turnaround Latency (User stop -> Voice)** | $< 800\text{ ms}$ | **$\sim 620\text{ ms}$** | 60ms VAD onset + 210ms Moonshine Tiny + 180ms TTFT + 140ms PocketTTS chunk. |
| **PortAudio Callback Stability** | $0\text{ exceptions}$ | **$100\%\text{ safe}$** | Zero-allocation `read_into` with automatic silence padding on starvation. |

---

## 3. Mathematical Specifications & Core DSP

### 3.1. Half-Cosine Micro-Fadeout on Barge-In Drain
To prevent step-function DAC pops without exceeding the 50ms latency budget, `AudioRingBuffer.drain()` applies a 2ms ($M = 48$ samples @ 24kHz) half-cosine attenuation:
$$w[n] = 0.5 \cdot \left(1 + \cos\left(\frac{\pi n}{M - 1}\right)\right), \quad n \in [0, M-1]$$
At $n=0, w[0]=1.0$ (continuity). At $n=M-1, w[M-1]=0.0$ (zero crossing).

### 3.2. DC Offset Blocker & 80Hz IIR High-Pass Filter
Eliminates ADC hardware bias (+300 to +500 counts in 16-bit space) that previously caused permanent VAD lockup:
1. Signal centering: $x_{\text{centered}}[n] = x[n] - \mu_x$.
2. 1st-order IIR difference equation ($R = 0.968$, $f_c \approx 80\text{ Hz}$ @ 16kHz):
   $$y[n] = x_{\text{centered}}[n] - x_{\text{centered}}[n-1] + R \cdot y[n-1]$$

### 3.3. Scale-Invariant dBFS RMS Energy
Converts unnormalized int16 or normalized float32 to scale-invariant dBFS:
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x[i]^2 + 10^{-12}}, \quad \text{dBFS} = 20 \log_{10}(\text{RMS})$$

### 3.4. Dual-Rate EMA Adaptive Noise Floor Estimator
Tracks ambient room noise floor without being skewed by active speech:
$$\sigma_{\text{noise}} \leftarrow \begin{cases}
\sigma_{\text{noise}} + \alpha_{\text{up}} (\text{dBFS} - \sigma_{\text{noise}}), & \text{if } \text{dBFS} > \sigma_{\text{noise}} \quad (\alpha_{\text{up}} = 0.02) \\
\sigma_{\text{noise}} + \alpha_{\text{down}} (\text{dBFS} - \sigma_{\text{noise}}), & \text{if } \text{dBFS} \le \sigma_{\text{noise}} \quad (\alpha_{\text{down}} = 0.15)
\end{cases}$$
Speech activation threshold: $\text{Threshold} = \text{clamp}(\sigma_{\text{noise}} + 12\text{ dB}, -45\text{ dBFS}, -15\text{ dBFS})$.

### 3.5. Token Shielding Sentinel System
In regex-based sentence boundary tokenizers, periods inside abbreviations (`Dr.`, `Mr.`, `e.g.`, `D.C.`), initials (`A.`), decimals (`1.5`), and ellipses (`...`) are shielded with Unicode Private-Use Area sentinel `\uE000`:
- Shield: replace target `.` with `\uE000` (length preserved 1:1).
- Split: execute boundary regex with terminal quote lookbehinds `(?<=[.!?]["'”’]?)\s+(?=["'“‘]?[A-Z0-9])`.
- Unshield: restore `\uE000` back to `.`.

---

## 4. Key Architectural Decisions (ADR Summary)

1. **ADR-01: CPU-Only Execution for STT and TTS on 4GB GPUs**
   - *Decision*: STT (Moonshine Tiny / SenseVoice via `sherpa-onnx`) and TTS (PocketTTS via `pocket_voice_worker.py`) run strictly on CPU.
   - *Rationale*: Offloading either model to CUDA alongside the 2.7GB Qwen 4B model blows past 4096 MiB VRAM and crashes with CUDA OOM.
2. **ADR-02: Generation Fencing for In-Flight TTS Chunks**
   - *Decision*: `AudioRingBuffer` tags writes with a monotonic generation token incremented on `drain()`.
   - *Rationale*: Completely prevents stale in-flight synthesized audio chunks from a cancelled conversational turn from re-entering the playback buffer post-interruption ("zombie speech").
3. **ADR-03: Zero-Allocation `read_into` in Real-Time Hot Path**
   - *Decision*: PortAudio callbacks pass pre-allocated output buffers directly to `read_into(outdata)`.
   - *Rationale*: Allocating numpy arrays (`np.empty`) inside mutex locks causes GC pauses and audio dropouts/clicks.
4. **ADR-04: Multi-State Barge-In (`NARRATING` & `REPLYING`)**
   - *Decision*: Allow barge-in interruptions when the agent is speaking conversational replies, not just during audiobook narration.
   - *Rationale*: Users must be able to stop or correct verbose or hallucinated agent replies immediately.
5. **ADR-05: Re-Entrant Lock (`threading.RLock`) for State Machine**
   - *Decision*: Guard all state transitions and inspections with `threading.RLock()`.
   - *Rationale*: Callbacks (such as `on_barge_in`) that inspect `sm.state` would immediately deadlock under a non-reentrant `threading.Lock()`.
6. **ADR-06: ChatML Control Token Sanitization**
   - *Decision*: Strip `<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]`, `<s>`, `</s>` from user inputs and book contexts.
   - *Rationale*: Prevents prompt injection attacks that escape conversational assistant boundaries.
7. **ADR-07: Unified FastAPI WebSocket & Static Asset Single-Process Server**
   - *Decision*: Combine static HTML/CSS/font HUD dashboard hosting (`GET /`) and bi-directional real-time telemetry/audio streaming (`WS /ws`) into a single FastAPI/uvicorn server process on port 8000.
   - *Rationale*: Eliminates cross-origin resource sharing (CORS) issues, simplifies deployment, provides deterministic lifecycle management, and enables tight synchronization between the web client and the local voice agent runtime.
8. **ADR-08: Hybrid Audio Ingestion Architecture & Virtual Loopback Fallback**
   - *Decision*: Provide a switchable audio bridge supporting browser Web Audio PCM streaming, host hardware `sounddevice` microphone capture, and an automated headless Virtual Audio Loopback fallback.
   - *Rationale*: Web Audio API provides zero-configuration microphone access directly from modern browsers, host Python sounddevice supports low-latency direct hardware access, and the virtual loopback guarantees continuous automated CI/CD and offline testing stability without physical audio peripherals.
9. **ADR-09: Decoupled Binary Audio & JSON Telemetry over WebSocket**
   - *Decision*: Multiplex structured JSON control messages (`state_sync`, `vad_event`, `cursor_update`, `barge_in`, `telemetry`, `prompt_envelope`) and raw binary 24kHz 16-bit mono PCM chunks over the same WebSocket connection.
   - *Rationale*: Avoids base64 encoding overhead (which adds ~33% bandwidth and CPU serialization cost in high-frequency audio frames) while preserving typed schema validation for state control and UI telemetry.
10. **ADR-10: Three-Phase Adversarial Hardening and Competitive Architecture Protocol**
    - *Decision*: Following any major development sprint, deploy a 4-auditor adversarial scrutiny team applying `/code-review-and-quality` and `/bugs-are-annoying`. Route all discovered defects to 2 independent architecture teams (Team Alpha and Team Beta) communicating via `send_message` to synthesize competing united proposals, from which the Lead Coordinator selects the best design moves for 4 Fixers to implement under TDD.
    - *Rationale*: Mono-agent verification produces blind spots. Adversarial scrutiny combined with competing peer architecture produces battle-hardened DSP and concurrency stability.
11. **ADR-11: High-Concurrency Broadcast Fan-Out, Monotonic Playhead Parity & Acoustic Continuity**
    - *Decision*:
      - Replace sequential WebSocket broadcast loops with non-blocking concurrent fan-out `asyncio.gather(*[_safe_send(ws)...])` and atomic set-difference pruning `self.active_connections = [ws for ws in self.active_connections if ws not in dead_set]`.
      - On microphone source transitions (`switch_source`), flush FIFO accumulation buffers and reset IIR filter states while strictly preserving `noise_floor_dbfs` to avoid triggering false barge-in interruptions from ambient room acoustics.
      - Advance playhead tracking in `AudioRingBuffer.read_into()` by `samples_requested` during active playback underrun (`filled > 0 or had_active_tail`), but freeze playhead tracking when completely idle to prevent infinite clock runaway.
      - Precompute and cache the 2ms half-cosine micro-fadeout window during `AudioRingBuffer.__init__` to achieve $\le 0.02\text{ms}$ drain latency under full concurrency load.
      - Guard HUD canvas animations via the Page Visibility API (`document.hidden` / `visibilitychange`) to prevent queue backlog in background tabs, and enforce single-timer invariance across all teleprompter cadence and navigation transitions.
    - *Rationale*: Eliminates head-of-line broadcast blocking, eliminates acoustic step pulses and false barge-in triggers on mic toggle, guarantees teleprompter word highlighting synchrony with physical DAC playback, and protects client-side CPU resources.
12. **ADR-12: Server-Side Neural Speech Recognition & Turn-Taking VAD Accumulation**
    - *Decision*: Run Faster-Whisper (`tiny.en`, INT8, CPU) inside the Python backend and route real-time 16kHz PCM audio frames through an energy VAD accumulator (`SpeechTurnAccumulator`).
    - *Rationale*: Eliminates fragile reliance on browser Web Speech API cloud STT while remaining within the 4GB GPU VRAM budget.
13. **ADR-13: Conversational Speech Sensitivity, Pre-Roll Buffering & Lore Grounding Disambiguation**
    - *Decision*:
      - Lower STT accumulator energy threshold to $0.015$ and expand pause segmentation timeout to $1400\text{ ms}$ (server) / $1800\text{ ms}$ (client) with a 4-frame rolling pre-roll buffer.
      - Maintain client microphone recording across Web Speech API `onend` timeouts to prevent premature listening abortion.
      - Automatically anchor character/lore queries with context ("Cyberpunk") during web search, enforce query term presence in extractive synthesis, and filter out hominid/fossil/album disambiguation artifacts.
    - *Rationale*: Guarantees that quiet desktop microphones and natural conversational pauses do not abort speech capture mid-sentence, while ensuring web search responses answer about the intended fictional character rather than unrelated Wikipedia fossils or songs.
14. **ADR-14: Client-Side Audio Downsampling, Defensive Polyphase Resampling, Okapi BM25 Retrieval & Provenance Hierarchy**
    - *Decision*:
      - Downsample client microphone audio from hardware native rate ($48\text{kHz} / 44.1\text{kHz}$) to $16{,}000\text{ Hz}$ in Web Audio `onaudioprocess` before converting to 16-bit PCM.
      - Implement defensive polyphase resampling in `STTEngine.transcribe()` via `scipy.signal.resample_poly` ensuring $16\text{kHz}$ input for Faster-Whisper.
      - Track speech silence timeouts via monotonic elapsed wall-clock time (`time.monotonic() - self._last_speech_time`), eliminating the 3x time-acceleration defect in `SpeechTurnAccumulator`.
      - Overhaul web search with punctuation-free tokenization `\b[a-z0-9_-]{2,}\b`, 107-word English stopword set, interrogative sentence penalties (-20 score) to prevent echoing question titles as answers, and factual predicate scoring.
      - Implement pure-Python Okapi BM25 ranking ($k_1 = 1.5, b = 0.75$) with rolling 3-sentence context windows ($W_i = [s_{i-1}, s_i, s_{i+1}]$) and playhead proximity Gaussian boost.
      - Establish context provenance hierarchy: when top document passages have BM25 score $\ge 2.5$, designate document context as primary ground truth and bypass external search.
    - *Rationale*: Eliminates acoustic pitch/time warping (3x duration stretch and -19.02 semitone drop), prevents premature 469ms speech cutoffs, eliminates substring retrieval errors ("car" in "scars"), preserves subject antecedents across pronouns, and ensures authoritative document facts are never overwritten by unrelated web search results.

---

## 5. Unified WebSocket Streaming Protocol & Telemetry Schemas

### 5.1. Client -> Server WebSocket Protocol (JSON & Binary PCM)
1. **Select Voice**:
   ```json
   {"type": "select_voice", "voice_id": "lucy-kushinada"}
   ```
2. **Switch Mic Source**:
   ```json
   {"type": "set_mic_source", "source": "browser" | "host"}
   ```
3. **Trigger Barge-In**:
   ```json
   {"type": "barge_in_trigger"}
   ```
4. **Set Cadence**:
   ```json
   {"type": "set_cadence", "cadence": 1.2}
   ```
5. **Playback Control**:
   ```json
   {"type": "playback_control", "action": "pause" | "resume" | "next" | "prev"}
   ```
6. **Binary PCM Audio Frame**:
   - Format: 24kHz 16-bit Mono PCM (Little-Endian int16).
   - Ingested directly into `AudioIngestionBridge` when `mic_source == "browser"`.

### 5.2. Server -> Client WebSocket Protocol (JSON & Binary PCM)
1. **State Sync (`state_sync`)**:
   ```json
   {
     "type": "state_sync",
     "state": "NARRATING",
     "active_voice": "david-martinez",
     "mic_source": "browser",
     "cadence": 1.0,
     "sentence_idx": 3,
     "word_offset": 0,
     "latency_ms": 0.04
   }
   ```
2. **VAD Event Telemetry (`vad_event`)** *(emitted every 30ms frame)*:
   ```json
   {
     "type": "vad_event",
     "speech_active": false,
     "dbfs": -42.5,
     "noise_floor_dbfs": -54.0,
     "threshold_dbfs": -42.0
   }
   ```
3. **Sub-50ms Barge-In Signal (`barge_in`)**:
   ```json
   {
     "type": "barge_in",
     "latency_ms": 0.045,
     "generation": 4,
     "flushed_samples": 48000
   }
   ```
4. **Word Cursor Update (`cursor_update`)**:
   ```json
   {
     "type": "cursor_update",
     "sentence_idx": 3,
     "word_offset": 12,
     "char_offset": 64,
     "total_sentences": 42,
     "sentence_text": "The neon glow of Night City reflected against the pavement."
   }
   ```
5. **Prompt Envelope (`prompt_envelope`)**:
   ```json
   {
     "type": "prompt_envelope",
     "state": "REPLYING",
     "context": "Narrative excerpt...",
     "query": "Who is David talking to?",
     "reply": "David is speaking directly with Lucy.",
     "pre_roll_words": 3
   }
   ```
6. **Audio Telemetry (`telemetry`)**:
   ```json
   {
     "type": "telemetry",
     "buffer_samples": 12000,
     "buffer_capacity": 240000,
     "drain_latency_ms": 0.038,
     "rms_db": -18.2,
     "fft_bins": [0.12, 0.45, 0.88, 0.65, 0.32, 0.21, 0.15, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.0, 0.0]
   }
   ```
7. **Binary Audio Output PCM**:
   - 24kHz 16-bit Mono PCM streamed directly to browser Web Audio API DAC for continuous stutter-free playback.

### 5.3. Hybrid Audio Ingestion Architecture
```
                         +-----------------------------+
                         |  HUD Audio Input Selector   |
                         +--------------+--------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
     +-----------------------+                     +-----------------------+
     | Browser WebAudio PCM  |                     | Host Sounddevice Mic  |
     | (WebSocket Binary In) |                     | (PortAudio Hardware)  |
     +-----------+-----------+                     +-----------+-----------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                         +-----------------------------+
                         |    AudioIngestionBridge     |
                         |  (Optional Virtual Loopback)|
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         | 16kHz Energy VAD Accumulator|
                         |  (30ms Frames, DC Blocker)  |
                         +-----------------------------+
```
- **Browser Mic Mode**: Receives raw PCM chunks over WebSocket from client-side Web Audio API (`getUserMedia`).
- **Host Mic Mode**: Reads from the server machine's physical microphone via `sounddevice.InputStream`.
- **Virtual Audio Loopback**: In-memory test audio generator producing configurable silence, tones, or synthesized voice frames, enabling 100% headless automated test coverage without physical hardware dependencies.

### 5.4. Document Ingestion, Multi-Tier TTS, and Live Web Search Grounding (Sprint 3)
1. **Document Ingestion Pipeline (`voice_agent.document.parser`)**:
   - Parses multi-format documents: PDF (`pypdf`), EPUB (XHTML/HTML chapter parsing via `zipfile`), Markdown, and Plain Text.
   - Token-shielded sentence segmentation (`split_sentences`) preserving abbreviations, dialogue quotes, ellipses, and numeric decimals.
   - Accessible via REST `POST /api/upload-document` (accepting both multipart `UploadFile` and JSON text) and `GET /api/document`.
2. **Multi-Tier Audio Synthesis (`voice_agent.tts.synthesizer`)**:
   - Primary: High-fidelity `edge-tts` mapped to Cyberpunk resident personas (David Martinez: `en-US-GuyNeural`, Lucy Kushinada: `en-US-JennyNeural`, V Male: `en-US-ChristopherNeural`, V Female: `en-US-AriaNeural`, Johnny Silverhand: `en-US-EricNeural`).
   - Secondary: Local Windows SAPI5 (`pyttsx3`) offline synthesis.
   - Fallback: Deterministic synthetic audio frames + browser Web Speech API.
3. **Multi-Provider LLM & DuckDuckGo Web Grounding (`voice_agent.llm.client`)**:
   - Supports local Ollama (`http://localhost:11434`), OpenAI, Groq, and custom endpoints with token streaming and punctuation-boundary chunking.
   - Real-time internet web search grounding via `duckduckgo_search` (`ddgs`) triggered dynamically when the user interrupts narration with ad-hoc questions.
   - Configurable at runtime via `GET /api/config`, `POST /api/config`, or WebSocket `update_config` message.

### 5.5. Production Studio, Direct Microphone Barge-In & Authentic Character Prosody (Sprint 4)
1. **Direct Microphone Barge-In with Hybrid VAD & Web Speech Recognition (ADR-09)**:
   - User voice interruption is supported via dual modalities:
     - **Push-to-Talk / Click-to-Speak (`[🎙️ BARGE IN & SPEAK]` or `Space`/`V`)**: Halts ongoing narration in $< 0.05$ms and activates Web Speech API (`SpeechRecognition`) for live transcript streaming directly into the query prompt.
     - **Hands-Free RMS VAD Detection**: Browser Web Audio monitors microphone RMS energy. If speech energy exceeds $-24\text{ dBFS}$ during active narration, it automatically interrupts playback and captures the user's spoken question.
     - Muted monitor node (`gain = 0.0`) routing microphone processor to destination, completely preventing speaker acoustic feedback loops.
2. **Cyberpunk Character Audio Profiles & Persona Prosody (ADR-10)**:
   - Voice models and prosody calibrated to authentic Cyberpunk 2077 & Edgerunners performances:
     - **Johnny Silverhand**: `en-US-RogerNeural` (pitch `-14Hz`, rate `-5%`, Keanu Reeves rockerboy baritone).
     - **Lucy Kushinada**: `en-US-AnaNeural` (pitch `-3Hz`, rate `-4%`, intimate Netrunner whisper).
     - **David Martinez**: `en-US-BrianNeural` (pitch `+4Hz`, rate `+6%`, energetic Santo Domingo edgerunner).
     - **V (Male)**: `en-US-GuyNeural` (pitch `-6Hz`, rate `+3%`, gritty mercenary solo).
     - **V (Female)**: `en-US-AriaNeural` (pitch `-3Hz`, rate `+4%`, sharp tactical solo).
   - Pitch formatted strictly adhering to Edge-TTS regex `^[+-]\d+Hz$`.
   - Character system prompts and conversational response templates inject authentic slang, world-building lore, and distinct emotional attitudes into every spoken reply.

### 5.6. Web Snippet Sanity, Typo Tolerance & Resumption Continuity (Sprint 4.1)
1. **Web Search Sanity & Truncation Elimination (ADR-11)**:
   - `sanitize_web_snippet()` sanitizes DuckDuckGo text snippets by stripping URLs, bracket citations (`[1]`, `[UPD]`), leading dates (`Nov 2, 2026`), empty language tags `(Japanese: )`, and multi-byte unicode characters that trigger Windows `cp1252` encoding crashes.
   - Ellipses (`...`, `…`) are normalized to whitespace before sentence segmentation, completely preventing sentence cutoffs mid-thought.
2. **Lore Typo Tolerance & Cross-Document Passage Grounding**:
   - Web search queries and document passage scans dynamically map phonetic speech-to-text typos (e.g. `arakasa` / `arakas` -> `arasaka`, `sande` -> `sandevistan`).
   - `search_document_passages()` queries the entire loaded prose (`self.sentences`) and combines local narrative citations with external search facts for rich, accurate answers.
3. **Deterministic Post-Barge-In Resumption Pipeline**:
   - Fixed rogue simulation timers in `triggerBargeIn()` that collided with live WebSocket state.
   - Server-driven resumption: `handle_user_interruption` awaits question audio playback, broadcasts pre-roll resumption cues (`PromptEnvelopeMessage(state="NARRATING", pre_roll_words=4)`), transitions `manager.state = "NARRATING"`, and restarts `start_narration()` cleanly.
   - Client teleprompter resumes word highlighting and scrolls to the active sentence without drift.

### 5.7. Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation (Sprint 5 / ADR-12)
1. **Local Neural STT Engine (`voice_agent.stt.recognizer.STTEngine`)**:
   - High-throughput local speech recognition using `faster-whisper` (`tiny.en`, INT8 quantized, CPU inference, $< 350$ms latency budget).
   - Dynamic fallbacks: secondary `SpeechRecognition` library, tertiary deterministic `MockSTTEngine` for headless CI environments.
   - Accepts raw PCM bytes, bytearrays, float32, or int16 numpy arrays at 16kHz; filters near-zero amplitude noise (< 0.01 peak).
   - Dedicated REST endpoint `POST /api/transcribe` accepting multipart audio uploads (`.wav` or raw `.pcm`) returning JSON transcript and duration telemetry.
2. **Speech Turn Accumulator & Dynamic Pause Detection (`voice_agent.stt.accumulator.SpeechTurnAccumulator`)**:
   - Conversational turn segmenter buffering 16kHz 16-bit PCM chunks during user speech.
   - Detects speech onset via RMS energy thresholding ($> 0.02$).
   - Automatic silence timeout segmentation ($> 600$ms silence trailing active speech of $\ge 300$ms) triggering automatic transcription without requiring explicit manual clicks.
   - Instant manual flush on client `voice_end` / `speech_end` control signals.
3. **Bi-Directional STT WebSocket Streaming Protocol**:
   - Multiplexes incoming binary PCM16 audio frames from browser microphone when session is in `LISTENING` or `BARGE_IN`.
   - Broadcasts typed `UserSpeechTranscriptMessage` (`user_speech_transcript`) containing transcription text, finality flag, and latency metrics.
   - Seamlessly dispatches transcribed query directly to `NarrationEngine.handle_user_interruption(query)`.
4. **HUD Client Speech Synchronization (`ui/index.html`)**:
   - `[🎙️ BARGE IN & SPEAK]` button / `V` key starts live microphone recording and streams PCM16 frames over WebSocket.
   - Dynamically updates query input and HUD speech ticker with server-side transcripts in real time.
   - Automatically submits query upon final transcript receipt and renders Prompt Envelope with live LLM response and pre-roll resumption cues.

### 5.8. Conversational Speech Sensitivity, Pre-Roll Buffering & Lore Grounding Disambiguation (Sprint 5.1 / ADR-13)
1. **Robust Audio Capture & Sensitivity Tuning (`voice_agent.stt.accumulator`, `voice_agent.stt.recognizer`)**:
   - Lowered STT accumulator energy threshold from $0.04$ to $0.015$, accommodating low-gain microphones and headset hardware without failing to detect voice onset.
   - Expanded conversational silence timeout from $650\text{ ms}$ to $1400\text{ ms}$, preventing premature utterance termination during natural human speaking pauses.
   - Integrated 4-frame rolling pre-roll ring buffer ($120\text{ ms}$) in `SpeechTurnAccumulator` so initial plosive consonants (`p`, `t`, `k`, `b`) before energy threshold breach are completely preserved.
   - Lowered silence peak amplitude check from $0.01$ to $0.004$, delegating faint-voice discrimination to Faster-Whisper's neural Silero VAD.
2. **Client Microphone Lifecycle Decoupling (`ui/index.html`)**:
   - Decoupled Web Speech API `recognizer.onend` from calling `stopDirectVoiceBargeIn()`, preventing browser-side speech recognition timeouts from killing the active microphone streaming session.
   - Increased client conversational pause timeout to $1800\text{ ms}$ and lowered RMS threshold to $0.015$.
   - Added `fallback_query` to `VoiceEndMessage` (`voice_agent.server.schemas`, `websocket.py`) allowing graceful transcript recovery if audio transmission is quiet.
3. **Lore Context Disambiguation & Web Sanity (`voice_agent.llm.client`)**:
   - Upgraded web search to `ddgs` library (v9.16.1).
   - Anchors character and lore lookups (e.g. "who is lucy", "david", "arasaka") with narrative universe context ("Cyberpunk"), preventing unrelated search hits (e.g. Lucy Australopithecus fossil).
   - Sanitizes web snippets by stripping leading/trailing non-alphanumeric punctuation and filtering out hominid/fossil/album disambiguation keywords.
   - Enforced query term presence in extractive synthesis candidate sentences, guaranteeing direct answers to user queries.

### 5.9. Client Downsampling, Web Search Overhaul & Okapi BM25 Ranking Engine (Sprint 5.2 / ADR-14)
1. **Acoustic Alignment & Downsampling Pipeline (`ui/index.html`, `voice_agent.stt`)**:
   - Web Audio dynamically resamples browser microphone input from native hardware sample rate ($48\text{kHz} / 44.1\text{kHz}$) down to $16{,}000\text{ Hz}$ in `onaudioprocess` before generating `pcm16` frames.
   - `STTEngine.transcribe` applies defensive polyphase resampling (`scipy.signal.resample_poly`) to guarantee exactly 16kHz float32 arrays for Faster-Whisper, eliminating the 300% time dilation and $-19.02\text{ semitone}$ pitch drop.
   - `SpeechTurnAccumulator` tracks conversational pauses using true monotonic wall-clock elapsed time (`time.monotonic() - self._last_speech_time`), preventing false premature 469ms turn termination.
   - Faster-Whisper is conditioned with `initial_prompt` biasing acoustic decoding towards Cyberpunk terminology.
2. **Web Grounding Relevance & Interrogative Title Filtering (`voice_agent.llm.client`)**:
   - Punctuation-free tokenization with regex word boundaries `\b[a-z0-9_-]{2,}\b` and 107-word English stopword set.
   - Context-aware lore anchoring covering 18 Cyberpunk characters and 13 world concepts (Relic, Blackwall, Sandevistan, Soulkiller, etc.).
   - Increased DuckDuckGo retrieval depth (`max_results = 8`) with structured warning logs on network errors.
   - Candidate sentence scoring applies a -20 penalty to interrogatives and question titles, preventing the agent from echoing the user's question back as an answer.
   - Factual predicate scoring rewards causal and explanatory clauses (`killed by`, `stole`, `invented`, `developed`, `destroyed`, `died`, `shot`, `caused by`, `biochip`, `construct`, `ai`).
3. **Pure-Python Okapi BM25 & Context Fusion Engine (`voice_agent.document.bm25`, `voice_agent.server.narration`)**:
   - Implemented `BM25Index` ($k_1 = 1.5, b = 0.75$, exact word boundary regex, non-negative Robertson-Spärck Jones IDF).
   - Rolling 3-sentence context windows ($W_i = [s_{i-1}, s_i, s_{i+1}]$) preserve subject antecedents across pronouns and dialogue turns.
   - Playhead proximity Gaussian boost prioritizes passages near the active reading cursor.
   - Provenance hierarchy prioritizes document passages over external search when BM25 score $\ge 2.5$.
   - Enforced 2048-token context budget and ChatML control token stripping to guarantee safety on RTX 3050 4GB VRAM.

### 5.10. Conversational Fluidity, Continuous Phase Resampling & Low-Gain STT Normalization (Sprint 5.3 / ADR-15)
1. **Continuous Phase Fractional Audio Resampling & Low-Gain Recovery (`ui/index.html`, `voice_agent.stt`)**:
   - Upgraded browser-side linear downsampling in `ui/index.html` to maintain continuous sub-sample fractional phase (`this._downsamplePhase`) across audio frames, preventing periodic phase slips and dropped samples at chunk boundaries.
   - Enabled `autoGainControl: true` and added explicit `AudioContext.resume()` on user gesture to avoid silent browser states.
   - In `STTEngine.transcribe`, added amplitude normalization scaling low-amplitude speech ($< 0.5$ peak scaled to $0.75$), allowing Faster-Whisper to transcribe soft/faint voices cleanly.
   - Added two-pass Whisper acoustic decoding: attempts transcription with `vad_filter=True`, automatically falling back to `vad_filter=False` if soft speech was pruned by Silero VAD.
   - In `SpeechTurnAccumulator`, expanded rolling pre-roll buffer to 25 frames (~1000ms) to capture initial words and plosives, and introduced `_session_buffer` fallback ensuring quiet utterances are never discarded on manual flush.
2. **Multi-Tier Conversational Intent Routing & Anti-Robotic Dialogue (`voice_agent.llm.client`, `voice_agent.server.narration`)**:
   - Overhauled `CHARACTER_PERSONA_PROMPTS` across Johnny Silverhand, Lucy Kushinada, David Martinez, and V (Male/Female) to instruct 1-3 spoken conversational sentences with street banter, dark humor, and smooth narrative bridges, strictly banning markdown asterisks, stage directions, and bullet points.
   - Implemented fast-path conversational intent routing in `_synthesize_fallback_answer()`:
     - **Intent 1: Greetings & Social Check-ins** ("hey", "what's up", "you there") -> immediate in-character street banter.
     - **Intent 2: Self-Identity & Character Backstory** ("who are you", "what's your story") -> authentic first-person character backstory.
     - **Intent 3: Repeat & Playback Commands** ("repeat that", "what did you say") -> dynamically quotes active reading sentence in character voice.
     - **Intent 4: Pause & Wait Requests** ("hold on", "pause", "wait a minute") -> in-character confirmation.
     - **Intent 5: Reactions & Exclamations** ("wow", "damn", "no way") -> acknowledges story drama.
     - **Intent 6: Character Lore Opinions** -> authentic opinions on comrades and cyberware (e.g. David on Lucy or Johnny on Sandevistan).
     - **Intent 7: Inquisitive Fallback & Dynamic Closers** -> eliminated formulaic canned wrappers ("Listen up samurai, tapped into the net...") in favor of natural conversational dialogue and fluid transitions back to the narrative.
3. **Verification**:
   - Added `tests/test_conversational_dialogue.py` (6 tests) and extended `tests/test_speech_accumulator.py` (2 tests).
   - 100% test pass rate across all 213 tests in 67.84s.


