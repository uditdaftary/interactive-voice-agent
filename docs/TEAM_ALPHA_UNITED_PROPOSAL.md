# Team Alpha: United Architectural Proposal (Sprint 2 Hardening)

> **Architects:** Senior DSP & Audio Concurrency Engineer (A1) & Systems Protocol Architect (A2)  
> **Target Scope:** Comprehensive resolution of `BUG-001` through `BUG-010` in `bugs.md`  
> **Status:** APPROVED & READY FOR FIXER EXECUTION  

---

## 1. Executive Summary & Design Philosophy

Team Alpha approaches the Sprint 2 defects with a focus on **deterministic DSP boundaries, zero-allocation real-time safety, and deadlock-free socket concurrency**. Our solution guarantees:
1. Zero thread race crashes during high-concurrency client drops.
2. Complete acoustic isolation when toggling between browser Web Audio and host physical microphones.
3. Microsecond-accurate monotonic playhead synchronization during ring buffer starvation.
4. Clean deprecation-free test harnesses and guarded linear resampling.

---

## 2. Technical Remediation Specifications

### 2.1. BUG-001 (Critical): Atomic Set-Comprehension Socket Pruning
**Target:** `src/voice_agent/server/connection_manager.py`  
**Problem:** `self.active_connections.remove(ws)` inside mutex raises `ValueError` if a socket was deregistered concurrently.  
**Team Alpha Solution:**
Replace element-wise removal with atomic set-difference list filtering:
```python
if dead_sockets:
    with self._lock:
        dead_set = set(dead_sockets)
        self.active_connections = [ws for ws in self.active_connections if ws not in dead_set]
```
This guarantees $O(N)$ safe pruning without exceptions, preserving all active connections.

### 2.2. BUG-002 (Critical): FIFO Purge & Filter State Reset on Mic Source Switch
**Target:** `src/voice_agent/audio/hybrid.py`  
**Problem:** Stale 24kHz browser audio frames in FIFO buffer concatenate with new host mic frames, tripping false-positive VAD barge-ins.  
**Team Alpha Solution:**
In `HybridMicrophoneBridge.switch_source(new_source)`:
```python
with self._lock:
    self.source = new_source
    self._unprocessed_samples = np.empty(0, dtype=np.float32)
    if self._vad is not None:
        self._vad.reset()
```
Flushes all stale samples and resets the DC-blocker IIR filter and EMA noise floor estimator to prevent filter ringing.

### 2.3. BUG-003: Starlette TestClient Migration
**Target:** `tests/test_e2e_pipeline.py`  
**Problem:** Deprecated Starlette import.  
**Team Alpha Solution:**
Replace `from starlette.testclient import TestClient` with `from fastapi.testclient import TestClient`.

### 2.4. BUG-004: Future Exception Logger Callback
**Target:** `src/voice_agent/server/connection_manager.py`  
**Problem:** Unobserved background broadcast exceptions.  
**Team Alpha Solution:**
Add static future callback:
```python
def _handle_future_done(fut: concurrent.futures.Future) -> None:
    try:
        exc = fut.exception()
        if exc is not None:
            logger.warning(f"Background WebSocket broadcast failed: {exc}")
    except (asyncio.CancelledError, concurrent.futures.CancelledError):
        pass
```
Attach to both `broadcast_json_sync` and `broadcast_bytes_sync`.

### 2.5. BUG-005: Playhead Advancement on Silence Padding
**Target:** `src/voice_agent/audio/ring_buffer.py`  
**Problem:** Playhead desync during buffer underrun.  
**Team Alpha Solution:**
In `AudioRingBuffer.read_into(outdata)`:
```python
# Advance playhead by total physical audio output duration (samples_to_read)
self._total_samples_played += samples_to_read
```
Ensures narrative teleprompter word highlighting never lags behind DAC physical output.

### 2.6. BUG-006: Byte-Alignment Guard for Browser PCM Ingestion
**Target:** `src/voice_agent/audio/hybrid.py`  
**Problem:** Odd-length byte arrays cause `np.frombuffer` crash.  
**Team Alpha Solution:**
```python
if isinstance(data, (bytes, bytearray, memoryview)):
    aligned_len = len(data) - (len(data) % 2)
    data = data[:aligned_len]
    pcm16 = np.frombuffer(data, dtype=np.int16)
```

### 2.7. BUG-007: HUD Teleprompter Timer Isolation
**Target:** `ui/index.html`  
**Problem:** Rapid cadence switching creates runaway overlapping timers.  
**Team Alpha Solution:**
Call `clearTimeout(narrativeTimer)` inside `setCadence()` and at the entry of `advanceWordCursor()`.

### 2.8. BUG-008: Linear Resampler Empty Array Guard
**Target:** `src/voice_agent/audio/loopback.py`  
**Problem:** Empty input array causes indexing error.  
**Team Alpha Solution:**
```python
if len(signal) == 0:
    return np.empty(0, dtype=signal.dtype)
```

### 2.9. BUG-009: Background Tab Canvas Frame Clamping
**Target:** `ui/index.html`  
**Problem:** Hidden tab pauses `requestAnimationFrame` while audio queue balloons.  
**Team Alpha Solution:**
Clamp visualizer history buffer to `Math.min(buffer.length, 240)` samples on every frame.

### 2.10. BUG-010: CSS Font Fallback Robustness
**Target:** `ui/app.css`  
**Problem:** FOIT on high-latency networks.  
**Team Alpha Solution:**
Ensure system fallbacks are explicitly bound in root variables:
`--sans: "Archivo", ui-sans-serif, system-ui, sans-serif;`
`--mono: "Martian Mono", ui-monospace, "Cascadia Mono", monospace;`
