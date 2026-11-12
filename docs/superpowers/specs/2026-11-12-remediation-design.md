# Remediation master spec: make the voice agent do what its docs say

Date: 2026-11-12. Branch: `feature/unified-hud-websocket-server` at `e0037dd`.
Status: revision 3, final for approval. Two Opus review rounds are logged in section 13, and every owner question is answered (sections 1 and 10).
Source: the three-slice audit of 2026-11-12 (server, audio/STT/TTS, LLM/docs/UI/tests), spot-verified in code.

## 1. Intent

The app works as a demo (217 tests pass), but there are serious problems:
- about 59% of the source never runs;
- out-of-the-box answers are scripted;
- the HUD shows fabricated latency and runs a fake "offline simulation" when the server is down;
- any web page can reconfigure or feed the server.

This spec fixes every finding by root cause. When it lands, four things will be true:
- Every claim in README, AGENTS.md and MEMORY.md is true and backed by a test or a gate.
- Every module under `src/` is either called by the running app or deliberately removed.
- The app is safe to run on a personal PC while you browse the web.
- Failures are visible (logged and shown in the HUD), never masked by canned dialogue or simulated numbers.

### Decisions the owner made (2026-11-12, verbatim answers)

| Question | Answer |
|---|---|
| Unreachable code (~6,200 lines) | **Wire them in** |
| Offline claim vs edge-tts / DuckDuckGo / Google STT | **Make docs honest** (keep edge-tts + web search, say they need internet, drop the Google STT fallback, keep pyttsx3 as offline TTS) |
| LLM default | **Ollama + visible errors** (keep a small canned set only for greetings and pause/repeat commands) |
| Clients | **Single user, this PC** (127.0.0.1, origin check, one mic owner at a time) |

### Goals
- G1. Close every finding in the traceability matrix (section 9).
- G2. Lint, typecheck and tests exist, run in CI, and pass.
- G3. The default test run never touches the network and finishes in under 30 s.
- G4. Every number the HUD shows (barge-in latency, flushed audio, word cursor) is measured where the audio actually plays.
- G5. The host-mic mode (button already in the UI) works.

### Non-goals
- Multi-user or LAN serving, accounts, auth tokens.
- Fully offline TTS as the default.
- Replacing faster-whisper as the default STT.
- Redesigning the HUD look.

### Anti-goals (would make this a failure even if tests pass)
- A canned in-character line standing in for a broken model, a failed STT, or an empty reply.
- Any HUD number that is a constant, a mock, or a server-side guess about client playback.
- Porting the vendored game-mod packaging (hash manifests, MO2/Vortex/USVFS, Node bridge).
- A test suite that is green because it asserts on mocks or canned copy.

## 2. Constraints

- **Platform.** Windows 11, Python 3.11 at `.venv`, RTX 3050 class GPU. VRAM budget (AGENTS.md): 3,100 MB; the LLM gets the GPU, STT and TTS stay on CPU.
- **No LLM server is installed.** Neither `ollama` nor `llama-server` is on PATH. The only model is the git-ignored `models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`. Until setup (5.7) is done, a fresh start shows "model offline".
- **Dependencies.**
  - Not installed: `torch`, `pocket_tts`, `safetensors`, `sherpa_onnx`, `soundfile`, `omnivoice`.
  - Installed but undeclared: `faster_whisper`, `ddgs`, `speech_recognition`.
  - Installed version to code against: edge-tts 7.3.1. Its `boundary` default is `"SentenceBoundary"` (edge_tts/communicate.py:335).
- **Owner's global rules.**
  - Typecheck + lint + tests gate every "done".
  - Every fallback logs.
  - One source of truth per constant.
  - Conventional commits; commit only when asked.
- **Voice clips.** `models/voices/*.wav` are recordings of real voice actors' characters. They stay local and are never published or bundled.

## 3. Root causes

| ID | Root cause | Evidence |
|---|---|---|
| RC1 | No single owner of session state. A free string is set from 15+ places; `AgentStateMachine` exists but is unused. | `connection_manager.py:36,101`, `narration.py:88,96,231,370,396`, `websocket.py:142,146,170`, `app.py:255` |
| RC2 | Fire-and-forget tasks and swallowed task errors. | `websocket.py:55,80,96,166,206`, `app.py:244` (dropped task refs); `narration.py:172-175` (loop error swallowed) |
| RC3 | Blocking work on the event loop. | `app.py:144` parse, `app.py:286` transcribe, `client.py:886` sync DDG in async path, `app.py:120` + `connection_manager.py:152` Whisper load at import under RLock |
| RC4 | Silent fallbacks: errors turned into canned text, empty strings, DEBUG logs or simulated UI. | `client.py:555`, `narration.py:339-349,415-422`, `parser.py:113,138,159,161`, `recognizer.py:164`, `app.py:51`, `connection_manager.py:294,330,359`, `energy_vad.py:183-201`, `synthesizer.py:164`, `hybrid.py:177-182,551`, `ui/index.html:1032-1060,1112` |
| RC5 | Unbounded input. | `accumulator.py:23,44`, `app.py:141,280`, `parser.py:126-160`, `app.py:218`, `ring_buffer.py:144-146` (silent truncation) |
| RC6 | No trust boundary. Server binds `0.0.0.0`, no Host/Origin check on any route, `request.json()` ignores Content-Type, multipart routes are open to cross-site form posts, untrusted text is mixed into the prompt. | `app.py:321,132,201,238,250,271`, `websocket.py:231`, `narration.py:285-316` |
| RC7 | Fake measurements. | `connection_manager.py:191-192` (constant 0.04 ms, 0 samples), `ui/index.html:1009-1010,1645` (UI fabricates `+0.04`), `narration.py:144` (2.5 words/s guess), `pipeline.py:95-101`, `test_benchmarks.py:187-193` (MagicMock STT) |
| RC8 | Doc drift and undeclared deps. | README (10 tests, Moonshine/Kokoro/llama.cpp, "100% offline", "<0.05 ms"), MEMORY.md:28,82, pyproject |
| RC9 | Unwired and vendored modules (section 4). | |
| RC10 | No gates and no test isolation. Measured on `test_chatml_sanitization_and_token_budget_in_interruption` (real sleeps neutralised): 54.3 s unpatched, 1.55 s with web search patched, 0.05 s with search + TTS patched. | `tests/test_bm25.py:205-224` |

## 4. Module dispositions ("wire them in", made concrete)

Rule: a module counts as "wired in" only if a named call site in the running app uses it and a behaviour changes. Every bug in a module is fixed in the same change that activates it.

| Module | Disposition | Call site after this spec |
|---|---|---|
| `orchestrator/state_machine.py` | **Wire in.** The single state owner (5.1). | `ConnectionManager` and `NarrationEngine` delegate all transitions to it |
| `cursor/tracker.py` | **Wire in.** Position tracking and pre-roll resumption. | `NarrationEngine.resume()` uses `get_resumption_text(4)` |
| `llm/client.py PunctuationStreamParser` | **Wire in.** Currently created but never fed (`narration.py:319`). | Reply sentences go to TTS as they complete |
| `vad/energy_vad.py` | **Wire in.** The only server-side speech detector. | `AudioHub` frame path: segmentation, hands-free barge-in, `vad_event` |
| `audio/hybrid.py` | **Wire in** as `AudioHub`, the single mic ingress for `browser` and `host` sources. | WS binary frames; `set_mic_source` |
| `audio/loopback.py` | **Wire in** as the `virtual` source for tests. Not a user-facing demo, since it only plays tones. | Test harness through `AudioHub` |
| `audio/ring_buffer.py` | **Wire in** as the host-speaker output buffer. A PortAudio `OutputStream` reads it through `read_into`, the path it was designed for. | `HostAudioOutput` in host I/O mode (5.3) |
| `stt/local_stt_worker.py` | **Wire in** as the optional `moonshine` STT backend: a subprocess using its existing JSON-lines protocol. Its pure speech-judge functions move to `stt/speech_judge.py` and are shared with the Whisper backend. | `stt/backends.py` |
| `pipeline.py` | **Retire.** It duplicates `NarrationEngine`. Its generation-fence idea lives on in `HostAudioOutput`. | none |
| `run_agent.py` | **Rewrite** as the launcher: `python run_agent.py [--host-io]` starts the server on 127.0.0.1. | entry point |
| `tts/pocket_voice_worker.py` + `pocket_voice_runtime.py` (2,428 lines) | **Replace** with a ~200-line in-process `PocketTTSBackend` on the public `pocket-tts` package (optional extra), cloning from `models/voices/*.wav`, and delete the vendored files. | `tts/backends.py` |
| `tts/resident_voice_worker.py` (1,293 lines) | **Delete.** It needs CUDA, `services.tts_backend` and `bridge/voice-cuda-graph.py`, none of which exist here. A GPU TTS would also break the VRAM budget. | none |

Why the vendored workers can't run as they are:
- `pocket_voice_worker.py:24` uses a bare `from pocket_voice_runtime import`.
- `pocket_voice_runtime.py` hash-checks 140 pinned upstream files and a hashed "student export" built by `packaging/build-release.mjs`.
- Both files assume MO2/Vortex/USVFS paths (`pocket_voice_runtime.py:317-352`).
- `resident_voice_worker.py:1183` imports `services.tts_backend`, which isn't in the repo.

## 5. Target architecture

There are two audio I/O modes. Both share one detector and one state machine.

```
BROWSER I/O (default)                                   HOST I/O (--host-io / mic_source=host)
browser mic --ws binary 16 kHz--> AudioHub              PortAudio InputStream --> AudioHub
browser plays TTS audio (MP3/PCM) + reports playhead    HostAudioOutput: ring_buffer --read_into--> PortAudio OutputStream
                         \                                   /
                          AudioHub (hybrid.py): callback thread only enqueues; processing on the event loop
                          -> VoiceActivityDetector (energy_vad.py) -> vad_event (<=10 Hz)
                          -> onset while NARRATING/REPLYING (hands-free) -> Session.barge_in()
                          -> SpeechTurnAccumulator (VAD-driven, bounded) -> STT backend -> speech judge
                                              |
                          AgentStateMachine (single owner, resume_to) <- typed questions, push-to-talk
                                              |
                          NarrationEngine: narration task | reply task (cancellable, TaskRegistry)
                          LLM (httpx stream) -> PunctuationStreamParser -> TTS per sentence
```

### 5.1 Session and state (RC1, RC2)

**States and transitions**
- States: `NARRATING`, `PAUSED`, `LISTENING`, `REPLYING`. The unused `BARGE_IN` state is removed; barge-in is an event, not a state.
- The machine stores `resume_to` (`NARRATING` or `PAUSED`), set when leaving either of those states.
- Legal transitions:
  - `NARRATING | PAUSED -> LISTENING` (barge-in or push-to-talk start).
  - `NARRATING | PAUSED | LISTENING | REPLYING -> REPLYING` (typed question via `/api/ask` or `query_interruption`, or a final transcript; a new reply cancels the old one).
  - `LISTENING -> resume_to` (deadline, empty or rejected transcript).
  - `REPLYING -> LISTENING` (barge-in, or the follow-up window after a reply).
  - `REPLYING -> resume_to` (done).
  - Any state `-> PAUSED` (user pause, narration error, last client disconnects). Entering `PAUSED` cancels the narration task, the reply task and any pending STT task.
  - `PAUSED -> NARRATING` (resume).
- An illegal transition logs a WARNING and returns False; callers act on the return value.
- `ConnectionManager` stops storing state. A single `on_change` hook broadcasts `state_sync`, replacing the scattered `broadcast_json(get_state_sync_message())` calls.
- The initial state is `PAUSED` (today it is `NARRATING` with no task running).

**Deadlines**
- `LISTENING` has a deadline: 8 s after a barge-in, 2.5 s for the follow-up window after a reply.
- The deadline is extended whenever speech is detected, so it never fires while the user is talking. Speech is detected by the server VAD from P4 on, and by the accumulator's `is_speaking` in P2-P3 (section 8, interim input path).
- The deadline is suspended while an utterance is with STT, and resolves on the transcript (go to `REPLYING`) or on a rejection (go to `resume_to`). This way a slow transcript can't restart narration and then cancel it again.
- A typed question cancels any pending STT task before starting its reply. A late transcript can't replace a typed question.
- Values live in `voice_agent/settings.py` (P0).

**Tasks and lifecycle**
- `TaskRegistry` (`server/tasks.py`):
  - `spawn(coro, name)` keeps a strong reference.
  - A done-callback logs exceptions with traceback.
  - `cancel_all()` is awaited in the lifespan shutdown.
  - ruff `RUF006` enforces stored task references.
- When the last client disconnects, the state goes to `PAUSED`.

### 5.2 Interruption handling (H1, H2, M6, M7, N1, N6)

**Barge-in and replies**
- `NarrationEngine.barge_in()`:
  - cancels the narration and reply tasks;
  - stops playback (5.3);
  - resets the accumulator pre-roll, so the agent's own reply echo never becomes part of the question;
  - transitions to `LISTENING`.
  - It is idempotent.
- `NarrationEngine.answer(query)` runs as the single *reply task*. Starting a new one cancels the previous one. Cancellation reaches the LLM stream because of httpx (5.7). The lock guards only the short transition sections, never I/O.
- The reply body uses `try/except/finally`. On any exception:
  - log with traceback;
  - broadcast `reply_error`;
  - speak one neutral line ("Something went wrong answering that.");
  - end in `resume_to`.
- An empty LLM reply gets the same treatment. This replaces the canned persona lines at `narration.py:339-349`.
- An unrecognised utterance gives the neutral line "I didn't catch that." plus `stt_status`, replacing the persona lines at `narration.py:415-422`.

**Speaking the reply**
- Reply sentences stream to TTS as `PunctuationStreamParser` completes them (N4).
- "Reply finished playing" is the client's `playback_done` ack (browser I/O) or the outbox reaching empty (host I/O). It is not `reply_words / 2.2`. A 30 s safety timeout logs a WARNING if no ack arrives.
- **Narration pacing.** The loop stays at most one sentence ahead of what the listener hears:
  - it prefetches (synthesizes) sentence N+1 while N plays;
  - it sends N+1 only after `playback_done` for N-1;
  - `current_sentence_idx` is the sentence that is playing, not the last one sent.
  - So a barge-in flushes at most about one sentence, and the book never queues up in the browser. This replaces the `duration_sec` sleep at `narration.py:144-167`.
- A follow-up utterance during the 2.5 s window starts a new reply directly; narration never restarts in between. Today the lock is held across `narration.py:367-383`, which causes a ~3.3 s stall and a narration flicker.

**Resuming and navigation**
- Resume narrates the rest of the interrupted sentence from 4 words before the stop point (`CursorTracker.get_resumption_text`). The word offset comes from `playback_stopped` / the pause ack (browser) or the outbox playhead (host).
- `next`, `prev`, `load_document` and `resume` go through engine methods. Each one:
  - takes the transition lock;
  - cancels the narration task;
  - sets the index;
  - restarts narration if it was narrating.
  - This fixes the skip-one-sentence bug at `narration.py:170`.
- `load_document` clears `_conversation_history`.
- A narration-loop exception logs with traceback, transitions to `PAUSED`, and broadcasts the error.

### 5.3 Audio out (RC7, G4, D3, L4)

**Browser I/O (default)**
- Keep browser playback of edge-tts MP3. No server-side decoding or pacing beyond the one-sentence lead (5.2).
- Audio and its messages go only to the owner connection (5.4). Acks are accepted only from it, so two tabs never play twice or double-ack.
- **P2, minimal ack:**
  - before each audio blob the server sends `audio_segment {segment_id, generation}`;
  - the browser sends `playback_done {segment_id}` from the last source node's `onended`.
  - This is all that 5.2's reply wait and pacing need.
- **P4, real measurements:**
  - `audio_segment` gains `word_times`. Edge-tts gets them from `WordBoundary` events (`Communicate(..., boundary="WordBoundary")` on 7.3.1); other backends get proportional-by-characters estimates.
  - The browser drives the word cursor from its own `AudioContext` playhead.
  - On `barge_in_event` or a pause, the browser sends `playback_stopped {generation, segment_id, word_idx, flushed_ms, stop_latency_ms}`, all measured in the browser:
    - `flushed_ms` is the queued audio it discarded;
    - `stop_latency_ms` runs from receiving the event to the sources being stopped;
    - `word_idx` is the resume position.
  - There is no separate cursor stream.
- The HUD shows those reported numbers plus the server's `detect_to_send_ms` (VAD onset frame to `barge_in_event` sent).
- Remove `latency_ms=0.04` / `flushed_samples=0` (`connection_manager.py:191-192`) in P2 and show "not measured" until P4. Remove the client's fabricated `+0.04` (`ui/index.html:1009-1010,1645`) in P3.

**Host I/O (owner: yes, Q2)**
- `HostAudioOutput` owns an `AudioRingBuffer(sample_rate=24000)`; a PortAudio `OutputStream` callback calls `read_into`.
- TTS audio is decoded to 24 kHz int16 PCM for this mode only (decoder is the builder's call: `miniaudio`, or `soundfile` if its libsndfile decodes MP3). pyttsx3 output is resampled with `resample_poly`.
- The writer applies back-pressure: it awaits free space instead of letting `write` truncate (`ring_buffer.py:144-146`).
- Barge-in calls `drain()` on this buffer only; it is the single drain. `HybridMicrophoneBridge` is built with `ring_buffer=None`, so its own drain at `hybrid.py:340-343` is a no-op. The returned flushed samples and measured drain time are the HUD numbers.
- The cursor comes from the ring-buffer playhead mapped through `word_times`.
- Echo handling follows the `host_output` setting (section 10, Q3). With the default `speakers`, hands-free barge-in is disabled while the outbox is playing and push-to-talk works. With `headset`, it stays on.
- Fix in the same change:
  - `drain()` drops an unconsumed fade tail on a second drain (`ring_buffer.py:187`);
  - drain allocates (`ring_buffer.py:172,184`) and the float32 `read_into` path allocates (`:297`). Preallocate, or remove the "zero-allocation" claims (`:11,195`). The int16 `read_into` path is already allocation-free.

### 5.4 Audio in: `AudioHub` (G5, M11, H3)

**Threading**
- One `HybridMicrophoneBridge` per app, created in the lifespan. Browser frames come in at 16 kHz.
- The PortAudio input callback only calls `loop.call_soon_threadsafe(queue.put_nowait, indata.copy())`. All resampling, VAD, accumulation and barge-in run on the event loop in one consumer task.
- This removes the callback-under-RLock chain (`hybrid.py:538 -> 246 -> 283-359`), the stop-under-lock deadlock (`hybrid.py:166,569`) and the unlocked reads (`:341-343`).

**Sources**
- `auto_fallback=False`. A failed host open reports `mic_error` to the HUD at WARNING and stays on `browser`. It never falls back to the 440 Hz sine (`hybrid.py:177-182`, `loopback.py:29-30`), which would fake speech and trigger barge-in.
- Owner connection (built in P2, ahead of the rest of AudioHub):
  - The first connection to send audio or `set_mic_source` owns the mic and the playback stream until it disconnects or another connection explicitly sends `set_mic_source`.
  - Other connections get `mic_owner: false` in `state_sync`. They receive state and text but no audio. Their binary frames and acks are dropped, with one INFO log per connection.
- `mic_source` is an enum: `browser | host`. `virtual` is test-only, set through injection, not the protocol.

**VAD**
- `energy_vad.py` is the only detector.
- In `_process_calibrated_frame`, run the VAD step before the frame callbacks (`hybrid.py:287-295`), so consumers get this frame's decision, and reuse the step's dBFS instead of re-running the stateful filter (`hybrid.py:298`, `energy_vad.py:210`).
- Fix the filter fallback: un-nest the pure-numpy path from `if lfilter is not None` (`energy_vad.py:183-201`), keep one filter state, and log once.
- VAD parameters carried over from the current tuning (Sprint 5.1/5.2), so behaviour does not regress:

  | Parameter | Value | Note |
  |---|---|---|
  | Turn-end silence | 1,400 ms | `turn_completion_frames=47` at 30 ms, not the default 16 |
  | Onset | 2 frames | |
  | Pre-roll | 1,000 ms | |
  | Ducking during browser-mode playback | 0 dB | The browser's `echoCancellation` handles it |
  | Ducking during host-mode playback | not applicable | Hands-free barge-in is disabled (5.3) |

  The energy threshold is calibrated against the values in the code today:
  - server accumulator RMS 0.010 (`accumulator.py:14`), with half that while speech is ongoing;
  - client hands-free RMS 0.013 (`ui/index.html:1841`).
  - The AGENTS.md figure of 0.015 is not what the code uses.
  - A test pins a 0.010 RMS frame as speech at the default noise floor.

**Segmentation**
- One segmenter per input style:
  - **Hands-free:** the server VAD decides onset (barge-in) and turn end (utterance), and the accumulator returns the utterance.
  - **Push-to-talk:**
    - the client sends `voice_start {mode: "ptt"}` on press, which acts as a barge-in and switches off VAD turn-end for that turn;
    - the turn ends only on `voice_end`;
    - the accumulator returns everything since press.
- The client's own silence auto-stop (`ui/index.html:1846-1866`) and RMS barge-in (`ui/index.html:1835`) are deleted **in P4, in the same change that adds server VAD barge-in**, so hands-free barge-in never goes missing between phases and two segmenters never produce self-cancelling double replies.

**Accumulator** (`feed(frame, is_speech)`)
- Bounded: pre-roll ring of 1,000 ms; utterance cap of 30 s (cut and transcribe).
- `_session_buffer` and `_silence_duration_ms` are deleted.
- The pre-roll is reset on barge-in and on entering the follow-up window.

`vad_event` is broadcast at most 10 Hz (the UI already handles it at `ui/index.html:1582`).

### 5.5 STT (H8, M1, M2, M3, M9, M16, L12)

- **Backends.** `stt/backends.py`:
  - `SttBackend.transcribe(pcm16: bytes, sample_rate: int) -> Transcript(text, rejected_reason | None)`.
  - `WhisperBackend` is the default (current `STTEngine` logic).
  - `MoonshineBackend` spawns `local_stt_worker.py --model models/stt/moonshine-tiny-en-int8` and uses its JSON-lines protocol. Optional extra `moonshine = ["sherpa-onnx"]`; the model download is documented, not automated.
- **Speech judge.** Both backends pass their output through `stt/speech_judge.py`: `evidence`, `acoustic`, `judge` and `HALLUCINATIONS`, moved from the worker, which then imports them.
- **Removals.** Delete the Google `recognize_google` fallback (`recognizer.py:150-163`), the `speech_recognition` dependency, and `MockSTTEngine` from production code.
- **Whisper loading.** Whisper loads lazily on first use in a thread, behind an `asyncio.Lock`. It never loads under the manager RLock or at import (M2, moved to P0).
- **Off the receive loop.** Transcription is a single-flight task in the TaskRegistry, never awaited inside the WS receive loop (`websocket.py:260`).
- **`fallback_query` (owner: opt-in, Q4).** The browser's `fallback_query` comes from the Web Speech API (`ui/index.html:835,978`), which sends mic audio to Google in Chrome. That conflicts with owner answer 2. Owner decision (Q4):
  - the Web Speech preview becomes an opt-in setting, off by default, listed under "needs internet";
  - when it is enabled, `parse_client_message` keeps `fallback_query` (`schemas.py:374`) and it is used only when the server transcript is empty;
  - the triplicated transcript-and-answer block (`websocket.py:48-56,74-81,199-206`) becomes one function.
- **`/api/transcribe`.**
  - parse the WAV and resample to 16 kHz;
  - unsupported sample widths or channel layouts return 415 instead of the RIFF header being treated as PCM (`app.py:50-55`);
  - odd-length raw PCM returns 400;
  - it runs in a thread.
- **`initial_prompt`.** It becomes a setting, empty by default, instead of hardcoded Cyberpunk names (`recognizer.py:111`).

### 5.6 TTS (H9, L9, L10)

- **Backends.** `tts/backends.py`: `TtsBackend.synthesize(text, voice_id, cadence) -> SpeechAudio(data, mime, word_times)`.
  - The order is the configured primary (default `edge`), then `pyttsx3`.
  - A fallback logs a WARNING and sets `tts_status`.
  - The synthetic PCM tier is deleted. If every backend fails, the cursor still advances and the HUD shows "TTS unavailable".
- **`PocketTTSBackend` (owner: yes, Q1).**
  - Optional extra; loads once on CPU in a thread; caches one voice state per `models/voices/<voice_id>.wav`; synthesis is cancellable between chunks.
  - The builder verifies the public API and whether its voice-cloning weights require an accepted Hugging Face licence.
- **pyttsx3 fixes.**
  - Use one engine on a dedicated thread instead of calling `init()` per call (`synthesizer.py:132`).
  - Temp-file deletion failures log a WARNING (`:164`).
  - Match the voice gender on whole words, so "female" no longer matches "male" (`:138,145`).

### 5.7 LLM (H6, H7, D2, M12, M13, L13, N3)

**Defaults and setup**
- One `voice_agent/settings.py` (created in P0) reads these env vars:
  - `VOICE_AGENT_LLM_BASE_URL` (default `http://127.0.0.1:11434/v1`)
  - `VOICE_AGENT_LLM_MODEL` (default `voice-agent`)
  - `VOICE_AGENT_LLM_API_KEY`
  - `VOICE_AGENT_WEB_SEARCH`
- `/api/config` changes are in-memory only.
- Setup:
  - ship `models/Modelfile` (`FROM ./Qwen3.5-4B-...gguf`, `num_ctx 2048`);
  - add README steps: install Ollama, then `ollama create voice-agent -f models/Modelfile`;
  - the builder verifies at setup that the installed Ollama loads the architecture and picks the chat template.
- Thinking output:
  - Turn thinking off at the source: the Modelfile template, or the request option the installed Ollama supports (the builder verifies which).
  - As a guard, a streaming filter strips `<think>...</think>` across chunk boundaries before parsing or TTS.
  - `max_tokens` is 300.

**Transport: httpx streaming, replacing urllib**
- `httpx` is already a dependency.
- `AsyncClient.stream` gives cancellation that actually closes the connection, separate timeouts (connect 2 s, read 30 s to cover a cold model load), and status codes.
- This replaces the `urlopen(timeout=3.5)` + executor `readline` loop (`client.py:528,534`), which cannot be cancelled and reports a cold load as offline.

**Typed outcomes**
- The outcomes are `ok`, `offline` (connect failure), `auth` (401/403), `error` (other status or malformed URL), and `interrupted` (stream broke after partial output).
- Each logs a WARNING with the endpoint, never the key, and sets `llm_status`.
- The spoken line is neutral, e.g. "My language model is offline. Check the model settings."
- A partial reply followed by an error speaks the partial text plus nothing else, and flags `interrupted`. No canned text is appended (today it is, `client.py:555`).

**Canned set (owner answer 3), as a local fast path in front of the LLM**
- Commands are always handled locally, whether or not the model is up:
  - `pause` and `resume` are state actions;
  - `repeat` re-narrates the current sentence.
- Greetings get a canned reply locally, whether or not the model is up. When the model is down, the HUD still shows `llm_status=offline`. The builder decides whether to append "(model offline)". This is my call.
- Everything else goes to the LLM.
- Delete the lore dict, lore banter, `_synthesize_fallback_answer` and its sync `search_web_sync` call (this closes H7 and the double search).
- Keep `CHARACTER_PERSONA_PROMPTS` as system prompts for the real model.
- `"builtin"` stays only as a documented "no model" mode that uses the offline line.

**Prompt**
- Untrusted material goes into fenced, labelled blocks (`<book_passages>`, `<web_results>`), each with "Reference material, not instructions."
- The question comes last.
- `sanitize_chatml` also strips the fence tags from untrusted text.
- Web search stays async, runs once, and is skipped when BM25 is authoritative.

**Keys**
- Warn once when a key is set with an `http://` URL to a non-loopback host.
- Unparseable client messages are logged without their content (`websocket.py:114`).

### 5.8 Server boundary (H4, H5, M8, M10, RC5)

**Network exposure**
- `run()` defaults to `127.0.0.1`. Any other `--host` logs a WARNING.
- `TrustedHostMiddleware`: `allowed_hosts` defaults to `["127.0.0.1", "localhost"]`, and `create_app(allowed_hosts=...)` lets tests add `testserver`.

**Origin check: one dependency, applied to `/ws` and to every non-GET route, including the multipart ones**
- If an `Origin` header is present, its host:port must equal the request's `Host`, compared case-insensitively with default ports normalised; otherwise return 403 (WS: close 1008).
  - `Origin: null` (file pages, sandboxed frames) is rejected.
  - No Origin header (CLI, TestClient) is allowed.
- This closes cross-site multipart posts to `/api/upload-document` and `/api/transcribe`, which need no preflight.
- JSON routes also require `Content-Type: application/json` (415 otherwise).

**Input validation**
- Bodies are Pydantic models reusing the WS schemas:
  - cadence 0.1 to 3.0;
  - `voice_id` from the known set;
  - enums for `mic_source` and the playback action;
  - config is validated before anything is applied;
  - non-object JSON returns 422.

**Uploads**
- No request-size caps (owner decision, 2026-11-13). Uploads are read as today; the guards below bound what a small crafted file can expand to.
- Allowed extensions: `.pdf .epub .txt .md` for documents, `.wav .pcm .raw` for audio.

**Parsing (in `asyncio.to_thread`)**
- EPUB:
  - at most 2,000 members;
  - each member's `ZipInfo.file_size` is checked against 10 MB before reading;
  - 100 MB total;
  - chapters in OPF spine order, not lexical order (`parser.py:143`, N5).
- PDF: at most 2,000 pages.
- A corrupt file returns 422.
- Failed parts log a WARNING and are returned as `skipped_parts`.
- An EPUB error string is never ingested as book text (`parser.py:161`).
- An empty document returns 422 and the old document stays loaded (`narration.py:73`, `app.py:157`).

**Broadcast and session robustness**
- Each send has a 2 s timeout; a failed or slow client is dropped and logged at INFO.
- `broadcast_json_sync` with no loop logs a WARNING.
- Import `concurrent.futures` (`connection_manager.py:347`).
- The WS catch-all (`websocket.py:267`) logs with traceback. A bad message gets an `error` reply and the session stays open.

### 5.9 UI

- Delete the offline simulation (`ui/index.html:1032-1060,1112`). A disconnected HUD shows "Disconnected, retrying" and does nothing else.
- Delete the fabricated latency (`:1009-1010,1645`).
- WS URL: the base already comes from `location`; delete the hardcoded port-8080 and `file:` branch (`:1364-1372,484-486`).
- Reconnect with exponential backoff, 1 s up to 30 s (`:1348,1486`).
- Show `llm_status`, `tts_status`, `stt_status`, `mic_owner`, `mic_error`, and `reply_error` in the existing status area, as squared chips in sentence case.
- 5.3 browser reporting:
  - `audio_segment` handling and `playback_done` in P2;
  - `word_times` cursor and `playback_stopped` in P4.
- P4: remove the client segmenters and add `voice_start {mode:"ptt"}` (5.4), in the same change as server VAD barge-in. Keep push-to-talk and the barge-in button.
- P3: add the Web Speech preview opt-in toggle (5.5).
- innerHTML: replace the two data-carrying uses with `textContent`/DOM building (`:582` VOICES, `:1075` server state). Add a test that fails on new non-static `innerHTML`.
- The RAF loops already skip drawing while hidden (`:1186-1189,1310-1312`). No change is needed; AGENTS.md is corrected instead.

## 6. Gates, tests and tooling (P0 unless stated)

**Dependencies (`pyproject.toml`)**
- Add `faster-whisper` and `ddgs`.
- Remove `duckduckgo_search` and `onnxruntime`, plus `pyyaml`/`colorama` if unused.
- Extras: `pocket`, `moonshine`, `hostaudio` (MP3 decoder for host I/O), `dev` (`pytest`, `pytest-asyncio`, `ruff`, `mypy`).
- `requirements.txt` is deleted (one source of truth).

**Lint and typecheck**
- ruff rules: defaults + `F`, `E`, `B`, `ASYNC`, `BLE`, `S110`, `RUF006`.
- mypy on `src/voice_agent`, `ignore_missing_imports`; strict for new modules.
- The vendored TTS workers are excluded from ruff and mypy in P0 and deleted in P4.

**CI**
- GitHub Actions on `ubuntu-latest` (owner decision, 2026-11-13: private repo, Windows minutes bill 2x), Python 3.11, with `libportaudio2` installed: ruff, mypy, `pytest -m "not network and not slow"`. Windows-only paths (pyttsx3 SAPI) carry a `windows` marker and are skipped on Linux; the orchestrator runs the full suite locally on Windows before each merge.

**Test isolation**
- `tests/conftest.py`:
  - sets `HF_HUB_OFFLINE=1`;
  - autouse guard blocks outbound non-loopback sockets;
  - `network` and `slow` markers, skipped by default.
- `tests/fakes.py`: `FakeLLM`, `FakeTts`, `FakeStt`, `FakeSearch`, and a fake browser client that acks playback.
- Short windows come from `settings` overrides. No patching of the global `asyncio.sleep` (`test_e2e_pipeline.py:346`).
- **Seams (built in P0):**
  - `create_app(settings=, llm=, tts=, stt=, search=, allowed_hosts=)`;
  - `NarrationEngine` takes the same;
  - Whisper is never loaded at import (M2).

**Benchmarks**
- Delete the MagicMock STT benchmark (`test_benchmarks.py:187-223`) and add a `slow` real-Whisper benchmark on a bundled 2 s WAV.
- Timing assertions use medians over N runs with generous bounds.

**Test rewrites**
- `test_conversational_dialogue.py` is deleted with the canned engine; keep tests for the greeting and command intents.
- `test_full_pipeline.py` is rewritten against `NarrationEngine` + `AudioHub` with the virtual source and fakes: speech burst during narration -> barge-in -> answer -> pre-roll resume.

**Other checks**
- **Import-graph check:** every module under `src/voice_agent` is reachable from `create_app` or `run_agent`. `stt/local_stt_worker.py` is allowlisted because it is spawned rather than imported.
- **TDD:** each fix starts with a failing test, named in section 9.

## 7. Documentation (P3)

- README rewritten to the real system:
  - architecture as in section 5 (both I/O modes);
  - setup (`pip install -e .[dev]`, Ollama + Modelfile, optional extras);
  - **What needs internet:** edge-tts, web search, the Web Speech preview if enabled, the first-run Whisper download, Pocket weights;
  - **What works offline:** pyttsx3, cached Whisper, Moonshine, Ollama;
  - measured numbers only, each with how it was measured.
- AGENTS.md: the current module map and rules, under 150 lines; sprint history moves to CHANGELOG.md. Correct the false claims:
  - "100% headless";
  - "<0.05 ms barge-in";
  - PocketTTS-on-CPU as the current TTS;
  - the BUG-009 note (RAF already pauses; the Web Audio resync claim must be re-checked).
- MEMORY.md: mark superseded ADRs (MEMORY.md:28,82).
- `TEST_READY.md`, `bugs.md`, `docs/AUDIT_FINDINGS_CONSOLIDATED.md` and `docs/TEAM_*` move to `docs/archive/`.
- CHANGELOG.md gets one entry per phase.

## 8. Phases (each ships alone and gates the next)

| Phase | Content | Exit gate |
|---|---|---|
| P0 Gates and seams | Section 6 tooling and deps; `settings.py`; injection seams; Whisper lazy load (M2); conftest guard and `HF_HUB_OFFLINE`; fakes; markers; vendored workers excluded from lint. No user-visible behaviour change. | ruff + mypy + default pytest green in CI; under 30 s; zero network |
| P1 Safety | 5.8 in full; parse and transcribe off the loop; accumulator bounded per the interim input path below | cross-origin WS and multipart POST rejected, `text/plain` POST gets 415, bad Host gets 400, zip-bomb EPUB gets 422 in bounded memory |
| P2 State and concurrency | 5.1, 5.2 including narration pacing; the minimal `audio_segment` / `playback_done` ack (5.3, client and server); audio to the owner connection only; remove the server's constant 0.04 | the 5.1/5.2 rows of section 9; the browser never holds more than one queued sentence |
| P3 Honesty | 5.5 (except Moonshine), 5.6 (except Pocket), 5.7, 5.9 P3 items, the RC4 logging sweep, section 7 docs | `llm_status=offline` with no Ollama; 401 gives `auth`; no canned persona lines remain; the HUD shows no number that wasn't reported |
| P4 Wire-ins | 5.3 `word_times` and `playback_stopped`, host I/O with the `host_output` setting, 5.4 AudioHub with server VAD barge-in **and** removal of the client segmenters plus `voice_start`, Moonshine, Pocket backend, retire pipeline.py, rewrite run_agent.py, delete vendored workers | e2e virtual-source test; host mic opens or reports `mic_error`; manual check in a real browser and with host I/O in both `host_output` settings |

**Interim input path (P1-P3).** Server VAD, the push-to-talk since-press buffer and `voice_start` arrive in P4, so until then the current input path stays, with three changes:
- **P1:** `_session_buffer` stays, so quiet-mic push-to-talk keeps working. It is capped at 30 s, filled only in `LISTENING`, and cleared on leaving it. It is deleted in P4, when the since-press buffer replaces it.
- **P2:** the `LISTENING` deadline is extended while the accumulator reports `is_speaking`.
- **P2, turn dedupe:** each entry into `LISTENING` gets a turn id. Only the first finished utterance or `voice_end` for that turn is processed; later ones are dropped with a DEBUG log. The client's silence auto-stop still runs until P4, and this stops a stray second turn-end from cancelling the real reply or triggering "I didn't catch that" mid-reply (N20).

Each phase gets its own branch off `feature/unified-hud-websocket-server`, with conventional commits, one logical change per commit.

## 9. Traceability matrix

| ID | Finding | Fix | Phase | Test / gate |
|---|---|---|---|---|
| D1 | ~59% of the source is unreachable | sec. 4 | P4 | import-graph check |
| D2 | Default answers are scripted | 5.7 | P3 | default targets Ollama; offline sets `llm_status=offline` |
| D3 | "<0.05 ms" is a microbenchmark; HUD shows a constant | 5.3 | P4 | HUD values come from `playback_stopped` / host drain |
| D4 | "100% offline" is false | sec. 7 | P3 | doc review |
| D5 | README is stale | sec. 7 | P3 | doc review |
| D6 | AGENTS/MEMORY drift | sec. 7 | P3 | doc review |
| D7 | Pre-roll resume is claimed but narration restarts at word 0 | 5.2 | P2 | resume at word 10 starts at word 6 |
| D8 | "Zero-allocation" claim is false for drain and the float path | 5.3 | P4 | claim fixed or allocation test |
| H1 | Reply handler wedges in REPLYING | 5.2 | P2 | FakeLLM raises: state goes to `resume_to`, `reply_error` is sent |
| H2 | Lock held across playback and the listen window | 5.2 | P2 | follow-up in the window gives one reply and no narration restart |
| H3 | Accumulator buffers are unbounded | 5.4, sec. 8 interim path | P1 | an hour of frames stays bounded; a 31 s utterance is cut at 30 s; quiet-mic push-to-talk still returns the whole turn |
| H4 | No Host/Origin check, 0.0.0.0, `text/plain` accepted | 5.8 | P1 | security tests |
| H5 | Unbounded uploads, sync parse, corrupt PDF gives 500 | 5.8 | P1 | 422 on corrupt or crafted files; loop latency under 50 ms during a parse |
| H6 | LLM errors are swallowed | 5.7 | P3 | fake server: refused, 401, 500, cut mid-stream |
| H7 | Sync DDG on the event loop; double search | 5.7 | P3 | one `FakeSearch` call per question |
| H8 | Undeclared and unused deps | sec. 6 | P0 | clean `pip install -e .[dev]` then the suite passes |
| H9 | Vendored workers can't be imported | sec. 4 | P4 | removed or replaced |
| M1 | `voice_end` drops `fallback_query` | 5.5 | P3 | kept when the opt-in is on; used only on an empty transcript |
| M2 | Whisper loads at import under the RLock | 5.5 | P0 | app import under 1 s, no Hub access |
| M3 | STT awaited in the receive loop | 5.5 | P2 | a control message is handled during a slow FakeStt |
| M4 | Untracked tasks | 5.1 | P2 | logged exception; no pending tasks after shutdown |
| M5 | Broadcast has no timeout; silent failures | 5.8 | P1 | stalled socket dropped after 2 s with a log |
| M6 | Narration error leaves a silent NARRATING; initial NARRATING with no task | 5.1, 5.2 | P2 | TTS raises: PAUSED + error; initial state is PAUSED |
| M7 | resume/next/prev unlocked; sentence skip; history kept on load | 5.2 | P2 | next from 0 narrates 1; load clears history |
| M8 | REST input unvalidated; partial updates | 5.8 | P1 | cadence 1e-9 gives 422 and applies nothing |
| M9 | WAV header read as PCM; sample rate ignored; odd PCM crashes | 5.5 | P1 | 44.1 kHz resampled; 8-bit gives 415; odd PCM gives 400 |
| M10 | Empty document reported as ok | 5.8 | P1 | 422; old document kept |
| M11 | Shared state across clients; narration plays for no one; LISTENING forever | 5.1, 5.4 | P2/P4 | second client is not mic owner; zero clients goes to PAUSED; deadline fires |
| M12 | Untrusted prompt text not delimited | 5.7 | P3 | prompt snapshot test |
| M13 | Key sent over http; no env var | 5.7 | P3 | warning once; env var read |
| M14 | Parser swallows errors; EPUB error text narrated | 5.8 | P1 | `skipped_parts`; no error text in sentences |
| M15 | `pipeline.py` timing is meaningless | sec. 4 | P4 | test replaced |
| M16 | Google STT fallback; DEBUG-only log | 5.5 | P3 | code removed; WARNING + `stt_status` |
| M17 | Hybrid stops stream under lock; callback holds the RLock | 5.4 | P4 | source switch during callback feed finishes within 1 s |
| M18 | Drain and float path allocate | 5.3 | P4 | as D8 |
| M19 | Tests hit the network (54 s measured) | sec. 6 | P0 | socket guard; under 30 s |
| M20 | Benchmarks measure mocks | sec. 6 | P0 | removed; real one marked `slow` |
| M21 | Flaky thresholds | sec. 6 | P0 | medians |
| M22 | No CI, lint or typecheck | sec. 6 | P0 | CI green |
| L1 | `concurrent` not imported | 5.8 | P1 | callback handles cancelled futures |
| L2 | `broadcast_json_sync` drops silently | 5.8 | P1 | WARNING |
| L3 | WS catch-all ends the session | 5.8 | P1 | session survives a malformed message |
| L4 | Heuristic cursor drifts | 5.3 | P4 | cursor follows the client playhead |
| L5 | Cancellation swallowed | 5.2 | P2 | caller cancellation propagates |
| L6 | Dead server code and ~25 aliases | 5.1, 5.4, 5.5 | P2-P4 | ruff `F`; aliases cut to what the UI sends |
| L7 | Second drain drops the fade tail | 5.3 | P4 | unit test |
| L8 | VAD filter fallback nested and desynced | 5.4 | P4 | runs without scipy; one filter state |
| L9 | pyttsx3 temp files and init per call | 5.6 | P3 | WARNING; one engine |
| L10 | "male" matches "female" | 5.6 | P3 | unit test |
| L11 | Hybrid unlocked reads; DEBUG-only host failure | 5.4 | P4 | consumer task owns state; `mic_error` |
| L12 | Vendored MO2 assumptions; Cyberpunk `initial_prompt` | sec. 4, 5.5 | P3/P4 | setting test |
| L13 | Debug log may contain the API key | 5.7 | P3 | log capture test |
| L14 | Tests assert canned strings | sec. 6 | P3 | removed |
| L15 | Test patches the global `asyncio.sleep` | sec. 6 | P0 | settings injection |
| L16 | UI hardcodes port 8080 / `file:` branch | 5.9 | P3 | branch deleted |
| L17 | Reconnect fixed at 3 s | 5.9 | P3 | backoff |
| L18 | *Withdrawn*: the RAF loops already pause; the fix is the AGENTS.md wording | sec. 7 | P3 | doc review |
| N1 | Canned persona lines for empty reply and unrecognised speech (`narration.py:339-349,415-422`) | 5.2 | P3 | neutral line + status |
| N2 | HUD offline simulation and fabricated latency (`ui/index.html:1009,1032-1060,1112,1645`) | 5.9 | P3 | disconnected HUD shows the state only; no client-made numbers |
| N3 | Partial LLM reply + error gets canned text appended (`client.py:555`) | 5.7 | P3 | stream cut mid-way gives `interrupted` with no appended text |
| N4 | Parser never fed, so TTS waits for the full reply (`narration.py:319`) | 5.2 | P3 | first sentence reaches TTS before the stream ends |
| N5 | EPUB chapters sorted lexically, not by spine (`parser.py:143`) | 5.8 | P1 | spine-ordered fixture |
| N6 | Follow-up pre-roll holds the reply's echo | 5.4 | P2 | pre-roll empty on entering the follow-up window |
| N7 | Cross-site multipart POST (no preflight) to upload/transcribe | 5.8 | P1 | cross-origin multipart gives 403 |
| N8 | Web Speech `fallback_query` sends audio to Google (conflicts with answer 2) | 5.5 | P3 | opt-in, off by default |
| N9 | Cancelled reply leaves the LLM thread and Ollama generating; 3.5 s timeout reports a cold load as offline | 5.7 | P3 | cancel closes the connection; slow first token is not `offline` |
| N10 | Two segmenters (client silence auto-stop + server) | 5.4 | P4 (same change as server VAD barge-in) | one `voice_end` per turn; no double reply |
| N11 | Host-open failure falls back to a 440 Hz sine | 5.4 | P4 | `mic_error`; stays on `browser` |
| N12 | Two drains with separate buffers | 5.3 | P4 | single drain |
| N13 | VAD frame callbacks fire before the VAD step; dBFS recomputed | 5.4 | P4 | consumer sees this frame's decision |
| N14 | `ring_buffer.write` silently truncates | 5.3 | P4 | back-pressure test |
| N15 | innerHTML with server data (`ui/index.html:582,1075`) | 5.9 | P3 | innerHTML test |
| N16 | Without a 2.5 words/s sleep, narration would queue the whole book in the browser | 5.2 pacing | P2 | at most one sentence ahead; a barge-in flushes at most one sentence |
| N17 | Deadline fires while STT is pending, so narration restarts and is then cancelled | 5.1 | P2 | slow FakeStt in the follow-up window gives one reply and no narration start |
| N18 | Push-to-talk turn ended by both VAD and `voice_end`; typed question replaced by a late transcript | 5.1, 5.4 | P2/P4 | one reply per PTT turn; typed question wins over a pending STT |
| N19 | Audio broadcast to every tab (double playback, double acks) | 5.3, 5.4 | P2 | a second tab receives no audio; its acks are ignored |
| N20 | In P2-P3, a second turn-end (client auto-stop + accumulator) would cancel the real reply | sec. 8 interim path | P2 | two turn-ends for one LISTENING entry give one reply |

## 10. Owner decisions, round 2 (2026-11-12, verbatim answers)

| Question | Answer | Effect on this spec |
|---|---|---|
| Q1. Pocket TTS and OmniVoice | **Pocket adapter, drop Omni** | ~200-line `PocketTTSBackend` on the public `pocket-tts` package (5.6); delete the three vendored worker files |
| Q2. Host I/O mode with the ring buffer driving the speakers | **Yes, add host I/O** | 5.3 host I/O is in scope for P4; server-side paced streaming to the browser stays rejected |
| Q3. Headset or speakers in host mode | **Both / not sure** | setting `host_output` (see below) |
| Q4. Web Speech preview (sends audio to Google in Chrome) | **Opt-in, off by default** | 5.5 as written |

`host_output` setting (Q3), values `speakers | headset`, default `speakers`:
- `speakers`: hands-free barge-in is disabled while host playback is active; push-to-talk works.
- `headset`: hands-free barge-in stays on during host playback.
- The setting is shown in the HUD next to the mic source. It lives in `settings.py`, can be changed at runtime through the validated config schema (5.8), and has no effect in browser mode.
- Real acoustic echo cancellation is out of scope.

Decided by me, reversible on request:
  - Moonshine STT becomes an optional backend, with Whisper staying the default;
  - greetings and pause/resume/repeat are a local fast path in front of the LLM.

## 11. Left to the builder

- The MP3 decoder for host I/O.
- Names beyond those fixed here.
- HUD status wording (sentence case, no hype).
- Whether to append "(model offline)" to the canned greeting.
- Whether to archive or delete stale docs.
- The JS test harness, if any.
- The exact energy-threshold calibration method (pinned by a test, 5.4).

## 12. What breaks first

- **Host I/O mode:** PortAudio device selection and echo on real hardware. That is why P4 ends with a manual check on real hardware, in both `host_output` settings, and why the default is the speakers-safe behaviour.
- **Browser mode:** if the client stops sending `playback_done` (tab suspended), replies wait up to the 30 s safety timeout.

## 13. Review log (Opus 5.5 review, 2026-11-12)

All points were accepted unless marked.

| Review item | Resolution |
|---|---|
| A1 RAF already pauses | L18 withdrawn; AGENTS.md fix (sec. 7) |
| A2 WS URL already uses `location` | L16 narrowed to the 8080/`file:` branch |
| A3 innerHTML with server data exists | N15; gate allowlist replaced by fixes |
| A4 VAD fallback uses `_prev_x/_prev_y`, nested | 5.4, L8 restated |
| A5 int16 `read_into` already allocation-free | 5.3, D8/M18 retargeted |
| A6 `test_search_relevance.py:166` is not a fake measurement | removed from RC7 |
| A7 `narration.py:90` stores its task | RC2 re-cited to 172-175 |
| B1 multipart CSRF | Origin = Host on all non-GET routes; N7 |
| B2 Web Speech sends audio to Google | Q4, N8 |
| B3 no return target; missing typed-question edges; deadline during speech | `resume_to`, edges, deadline extension (5.1) |
| B4 LLM not cancellable; 3.5 s timeout | httpx streaming (5.7), N9 |
| B5 two segmenters | one per input style (5.4), N10 |
| B6 VAD defaults would regress tuning | parameter table (5.4) |
| B7 callback runs everything under the RLock | enqueue-only callback (5.4) |
| B8 sine fallback; virtual source needs a pump | `auto_fallback=False`, N11; virtual is test-only |
| B9 two drains | single drain (5.3), N12 |
| B10 server send time is not playback | client reports playhead (5.3) |
| B11 host mic has no AEC | Q3, default disables hands-free during host playback |
| B12 phase ordering | M2, settings and seams moved to P0; `HF_HUB_OFFLINE`; vendored files excluded from lint |
| B13 TrustedHost breaks TestClient | injectable `allowed_hosts` |
| B14 ownership flapping | first owner keeps the mic (5.4) |
| B15 silent `write` truncation | back-pressure, N14 |
| B16 callbacks before the VAD step | reorder and reuse dBFS, N13 |
| B17 import graph flags the worker | allowlisted |
| C1-C6 | N1, N2, N3, N4, N5, N6 |
| D1 headset vs speakers | Q3 |
| D2 thinking model / Ollama architecture | always strip `<think>`, `max_tokens` 300, builder verifies the template at setup |
| D3 greetings with the model up | my call: local fast path always (5.7) |
| D4 Web Speech | Q4 |
| D5 P4 blocked on Q1-Q3 | yes; P0-P3 do not depend on them |
| E1 drop server-side decoding and pacing | adopted (5.3, Q2) |
| E2 `RUF006` instead of a custom grep | adopted |
| E3 httpx | adopted |
| E4 compare Origin to Host | adopted |
| E5 virtual-mic demo plays a tone | virtual is test-only |
| E6 drop the L18 work | adopted |

Round 2 (verification of revision 2):

| Review item | Resolution |
|---|---|
| 1.1 client segmenters removed in P3 but server barge-in only in P4 | both move to P4, in one change (5.4, 5.9, sec. 8) |
| 1.2 / 2a ack scheduled in three places; P2 depends on P4 | minimal ack in P2; `word_times` and `playback_stopped` in P4 (5.3) |
| 1.3 no narration pacing once the sleep is gone | one-sentence lead (5.2), N16 |
| 1.4 RMS 0.015 is the doc value, not the code value | thresholds pinned to 0.010 / 0.013 (5.4) |
| 1.5 stripping `<think>` after generation wastes tokens | thinking off at the source; streaming strip as a guard (5.7) |
| 1.6 PAUSED does not say what it cancels | cancels narration, reply and pending STT (5.1) |
| 2b deadline vs pending STT | deadline suspended while STT is pending (5.1), N17 |
| 2b2 PTT double turn-end; typed question vs late transcript | `voice_start {mode:"ptt"}`; a typed question cancels pending STT, N18 |
| 2b3 two tabs play twice | audio and acks for the owner connection only, N19 |
| 2c Origin check details | case-insensitive host:port compare, default ports normalised, `null` rejected |
| 3.1 the 10 Hz cursor stream is unnecessary | dropped; position travels in `playback_stopped` / the pause ack |
| 3.2 host I/O exists only for the ring buffer | stated in Q2: if the answer is no, `ring_buffer.py` is deleted |
| 4 builder questions | none beyond Q1-Q4 |
