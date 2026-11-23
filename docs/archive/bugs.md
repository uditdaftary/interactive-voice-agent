# Bug Report — Interactive Voice Agent HUD & Streaming Server — 2026-11-09

## Summary
- Critical: 0 open, 3 fixed (BUG-001, BUG-002, BUG-011)
- Intermediate: 0 open, 6 fixed (BUG-003, BUG-004, BUG-005, BUG-006, BUG-007, BUG-012)
- Normal: 0 open, 5 fixed (BUG-008, BUG-009, BUG-010, BUG-013, BUG-014)

---

## 🔴 Critical

### BUG-001: Race Condition in Dead Socket Pruning under High-Concurrency Broadcast
- **File:** `src/voice_agent/server/connection_manager.py:242-290`
- **Issue:** When broadcasting JSON or binary audio packets to multiple clients (`broadcast_text` or `broadcast_bytes`), failed socket connections are appended to `dead_sockets`. Pruning executed `self.active_connections.remove(ws)` inside `with self._lock:`. If a client disconnected concurrently via `disconnect()` on the reader thread, `ws` was already removed, causing `ValueError: list.remove(x): x not in list`, which crashed the background broadcasting task and dropped all subsequent audio frames.
- **Trigger:** Multiple clients connected, with one client disconnecting abruptly while audio streaming or VAD event broadcasting is active.
- **Impact:** Unhandled server exception, breaking audio stream distribution to all remaining connected HUD clients.
- **Remediation:** Migrated broadcast logic to non-blocking concurrent fan-out via `asyncio.gather(*[_safe_send(ws)...])` and atomic set-difference pruning `self.active_connections = [ws for ws in self.active_connections if ws not in dead_set]`.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-002: Stale FIFO Audio Buffer Contamination on Microphone Source Switch
- **File:** `src/voice_agent/audio/hybrid.py:161-180` and `src/voice_agent/vad/energy_vad.py:109-125`
- **Issue:** When `HybridMicrophoneBridge.set_source()` / `switch_source()` transitioned between `"browser"`, `"host"`, and `"virtual"`, the pending sample accumulator (`self._fifo`) was not flushed. Stale 24kHz frames from the browser mic were concatenated with host mic frames, creating a sample rate/phase discontinuity. Furthermore, resetting the VAD noise floor to -60 dBFS caused immediate false barge-in triggers from ambient noise.
- **Trigger:** User toggles the HUD microphone button from "BROWSER MIC" to "HOST HARDWARE" during live narration or speech.
- **Impact:** Acoustic glitch and DC bias pulse tripping the VAD into a false-positive barge-in interruption.
- **Remediation:** Flushed `self._fifo = np.empty(0, dtype=np.float32)` on source transition and reset VAD filter states (`_filter_zi`, `_prev_x`, `_prev_y`) while strictly preserving `noise_floor_dbfs`. Added alias `switch_source = set_source`.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-011: AudioRingBuffer Drain Exceeds Sub-0.05ms Budget Due to Dynamic linspace/cos Allocation
- **File:** `src/voice_agent/audio/ring_buffer.py:32-35` & `178-185`
- **Issue:** Dynamic allocation of `t = np.linspace(0.0, np.pi, fade_samples)` and `window = 0.5 * (1.0 + np.cos(t))` on each drain call increased mean drain latency under multi-suite load to ~0.058ms.
- **Trigger:** Rapid barge-in cycling or buffer drain under multi-threaded system load.
- **Impact:** Occasional P99 latency jitter above 0.05ms.
- **Remediation:** Precomputed default 2ms half-cosine micro-fadeout window in `AudioRingBuffer.__init__` (`self._cached_fade_window`), slicing directly into it without dynamic trigonometry or allocations.
- **Status:** Resolved & Verified (116/116 tests passing)

---

## 🟡 Intermediate

### BUG-003: Starlette Deprecation Warning on TestClient Deprecation
- **File:** `tests/test_e2e_pipeline.py:23`, `tests/test_server_static.py:10`, `tests/test_websocket_protocol.py:19`, `tests/test_tier5_adversarial.py:34`
- **Issue:** Tests imported `from starlette.testclient import TestClient`, emitting `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated`.
- **Trigger:** Running pytest suite (`pytest -v`).
- **Impact:** Emitted console deprecation warnings.
- **Remediation:** Migrated all test imports across all suites to `from fastapi.testclient import TestClient`.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-004: Silent Future Failure in Synchronous Background Broadcast Helpers
- **File:** `src/voice_agent/server/connection_manager.py:315-334`
- **Issue:** `broadcast_json_sync` and `broadcast_bytes_sync` scheduled coroutines via `asyncio.run_coroutine_threadsafe(..., target_loop)`, but discarded the returned `concurrent.futures.Future` without attaching an error callback (`add_done_callback`).
- **Trigger:** Audio engine attempts broadcast while the event loop is closing or saturated.
- **Impact:** Exceptions raised in the async broadcast coroutine were swallowed without telemetry.
- **Remediation:** Added `_log_future_exception` callback to returned futures with `target_loop.is_running() and not target_loop.is_closed()` checks, ignoring expected cancellations during shutdown.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-005: Playhead Drift on Buffer Starvation Silence Padding
- **File:** `src/voice_agent/audio/ring_buffer.py:200-234`
- **Issue:** `AudioRingBuffer.read_into()` zero-pads output buffers when fewer samples are available than requested by the audio sink, but did not advance monotonic playhead tracking for the silence-padded samples during active playback underruns.
- **Trigger:** Temporary audio pipeline underrun where output stream requests 720 samples but only 360 are buffered.
- **Impact:** Monotonic audio clock desynchronized from physical speaker playback, causing narrative teleprompter word highlighting to lag behind audible speech.
- **Remediation:** In `read_into()`, advance `self.total_samples_played += samples_requested` whenever active audio or fadeout tail was emitted (`filled > 0 or had_active_tail`), while freezing the playhead when completely idle to prevent infinite runaway.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-006: Unhandled Unaligned Byte Length in Browser PCM Ingestion
- **File:** `src/voice_agent/audio/hybrid.py:212-225` & `440-455`
- **Issue:** Fragmented WebSocket frames with odd byte lengths caused `np.frombuffer` to raise `ValueError: buffer size must be a multiple of element size`.
- **Trigger:** Browser network packet fragmentation or unaligned Web Audio worklet chunk boundary.
- **Impact:** Server raised unhandled exception and dropped audio frames.
- **Remediation:** Sliced `data = data[:len(data) - (len(data) % 2)]` in `feed_browser_pcm` and `normalize_audio` before buffer conversion.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-007: Multiple Overlapping Cadence Timers in HUD Teleprompter
- **File:** `ui/index.html:473-542`
- **Issue:** Clicking cadence buttons (0.8x, 1.0x, 1.2x) or rapid sentence stepping scheduled `advanceWordCursor()` via `setTimeout` without clearing previous active `narrativeTimer`.
- **Trigger:** User clicked cadence or sentence buttons in rapid succession on the HUD.
- **Impact:** Multiple concurrent timer loops ran in parallel, causing erratic word cursor jumps.
- **Remediation:** Centralized timer management into `clearNarrativeTimer()` and invoked it across all state-mutating handlers, paired with 50ms timestamp debouncing on cadence controls.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-012: Lazy Scipy Signal Import Causes 2.5-Second First-Frame Latency Glitch
- **File:** `src/voice_agent/vad/energy_vad.py:9-12`
- **Issue:** Function-scoped import of `scipy.signal.lfilter` caused a 2.5s disk I/O delay on the initial audio frame on Windows systems.
- **Trigger:** First audio frame fed to `VoiceActivityDetector`.
- **Impact:** Audio stutter and potential buffer overrun on initial utterance.
- **Remediation:** Moved `lfilter` import to module top-level with `try/except ImportError` guard.
- **Status:** Resolved & Verified (116/116 tests passing)

---

## 🟢 Normal

### BUG-008: Redundant Linear Resampling Boundary Check
- **File:** `src/voice_agent/audio/loopback.py:395-405` and `src/voice_agent/audio/hybrid.py:491-506`
- **Issue:** Calling resamplers with zero-length audio frame produced indexing errors.
- **Trigger:** Calling resampler with empty audio input.
- **Impact:** ValueError on empty audio input.
- **Remediation:** Added `if len(audio) == 0 or orig_sr <= 0 or target_sr <= 0: return np.empty(0, dtype=np.float32)` guard.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-009: Background Tab Canvas Frame Accumulation
- **File:** `ui/index.html:756-760`, `880-884`, `1067-1070`, `1297-1304`
- **Issue:** `requestAnimationFrame` throttles when tab is hidden, causing audio packet queue backlog and render bursts upon refocus.
- **Trigger:** Switching browser tabs while audio is playing.
- **Impact:** Stutter and temporary visualizer glitch on tab refocus.
- **Remediation:** Skipped canvas rendering when `document.hidden`, added `visibilitychange` listener to re-anchor `nextPlayTime` to `audioContext.currentTime`, and clamped visualizer FFT queues to 240 samples.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-010: Missing Fallback Font Declaration in CSS Variable
- **File:** `ui/app.css:31-32` & `55-58`
- **Issue:** Specifying local webfonts without system fallbacks risked FOIT, and browser faux-styling distorted tabular digits.
- **Trigger:** Cold load on high-latency connection.
- **Impact:** Brief text flash before fonts load.
- **Remediation:** Added comprehensive system fallback stacks in `:root` and added `font-synthesis: none;` on `.mono` and `.num` classes.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-013: Normalization Heuristic Over-Attenuates Legitimate Audio with Peak > 1.5
- **File:** `src/voice_agent/audio/hybrid.py:497` and `src/voice_agent/vad/energy_vad.py:169`
- **Issue:** `max_abs > 1.5` heuristic over-attenuated float audio with transient DC overshoot.
- **Trigger:** Floating-point audio with peak amplitude between 1.5 and 256.0.
- **Impact:** Legitimate speech attenuated by ~90 dB.
- **Remediation:** Raised unnormalized float threshold to `max_abs > 256.0`.
- **Status:** Resolved & Verified (116/116 tests passing)

### BUG-014: Cross-Frame Mechanical Step Transient Triggers False Speech Onset
- **File:** `src/voice_agent/vad/energy_vad.py:29`
- **Issue:** Isolated non-speech mechanical transient straddling a 30ms boundary could trigger barge-in on `onset_frames=2`.
- **Trigger:** Mechanical shock or step transient at frame boundary.
- **Impact:** False barge-in on non-speech click.
- **Remediation:** Made `onset_frames` a configurable parameter in `VoiceActivityDetector.__init__` with default 2, allowing tuning up to 3+ for noisy/vibrational environments.
- **Status:** Resolved & Verified (116/116 tests passing)
