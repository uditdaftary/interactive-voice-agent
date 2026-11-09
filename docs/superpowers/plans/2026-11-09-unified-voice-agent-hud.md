# Unified Voice Agent HUD & WebSocket Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and integrate a unified local FastAPI WebSocket server and single-page HUD dashboard for the interactive voice agent, enabling real-time 24kHz audio streaming, sub-50ms barge-in orchestration, live telemetry, and zero-cloud offline capability.

**Architecture:** A FastAPI application serves both static HUD assets (`ui/`) and a duplex WebSocket (`ws://localhost:8000/ws`). The backend bridges incoming PCM microphone frames directly to the Silero VAD engine and `AudioRingBuffer`, broadcasting state machine transitions, word-level cursor tracking, and instantaneous drain signals to the single-page Catalog Sheet HUD.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, WebSockets, NumPy, PyTorch / Silero VAD, vanilla JS / HTML5 Canvas / CSS (0px square Catalog Sheet design).

**Spec:** `c:\Projects\interactive-voice-agent\DESIGN.md` & `c:\Projects\interactive-voice-agent\MEMORY.md`

## Global Constraints

- **Hardware Law:** Max 4.0 GB VRAM (RTX 3050 Laptop). TTS and STT run strictly on CPU; LLM strictly within 2048 context tokens.
- **Latency Budget:** Ring buffer drain must execute in $< 50\text{ ms}$ (target $\le 0.05\text{ ms}$). Turnaround latency $< 800\text{ ms}$.
- **Design System:** Strictly follow "The Catalog Sheet" (`DESIGN.md`): Ground `#0b0b0b`, Ink `#efede6`, Signal Yellow `#e8b622`, Fault Red `#e03a2f`, 0px square corners, 44px right-tab buttons, Archivo and Martian Mono typography.
- **Git & Issue Discipline:** Work strictly on branch `feature/unified-hud-websocket-server`. Commit atomically per task. Raise GitHub issues with `gh issue create` if blockers arise.
- **Documentation Discipline:** Maintain `MEMORY.md`, `CHANGELOG.md`, and `AGENTS.md` on every major architectural addition.

---

### Task 1: Feature Branch & Documentation Setup

**Files:**
- Create: `git branch feature/unified-hud-websocket-server`
- Modify: `MEMORY.md:80-110`
- Modify: `CHANGELOG.md:1-25`
- Modify: `AGENTS.md:1-50`

**Interfaces:**
- Consumes: Existing documentation ADRs and multi-agent topology.
- Produces: Updated `MEMORY.md` (ADR-07 WebSocket Protocol), `CHANGELOG.md` ([0.3.0] unreleased entry), and `AGENTS.md` (Server and WebSocket roles).

- [ ] **Step 1: Check out feature branch**
  ```powershell
  git checkout -b feature/unified-hud-websocket-server
  ```
- [ ] **Step 2: Update MEMORY.md with ADR-07 and WebSocket architecture**
- [ ] **Step 3: Update CHANGELOG.md with [0.3.0] Unreleased section**
- [ ] **Step 4: Update AGENTS.md with the server and integration subagent roles**
- [ ] **Step 5: Commit documentation setup**
  ```powershell
  git add MEMORY.md CHANGELOG.md AGENTS.md
  git commit -m "docs: establish ADR-07 and feature branch for unified hud websocket server"
  ```

---

### Task 2: WebSocket Streaming Protocol & Payload Schemas

**Files:**
- Create: `src/voice_agent/server/protocol.py`
- Create: `src/voice_agent/server/__init__.py`
- Test: `tests/test_server_protocol.py`

**Interfaces:**
- Consumes: Standard library `dataclasses`, `json`, `typing`.
- Produces: `ClientMessage` and `ServerMessage` dataclasses/enums with `.to_json()` and `.from_json()` serialization.

- [ ] **Step 1: Write the failing test for protocol serialization and message validation**
  ```python
  # tests/test_server_protocol.py
  from voice_agent.server.protocol import (
      ClientMessageType, ServerMessageType,
      ClientMessage, ServerMessage, parse_client_message
  )
  def test_protocol_roundtrip():
      msg = ClientMessage(type=ClientMessageType.BARGE_IN, payload={"timestamp": 12345.0})
      raw = msg.to_json()
      parsed = parse_client_message(raw)
      assert parsed.type == ClientMessageType.BARGE_IN
      assert parsed.payload["timestamp"] == 12345.0
  ```
- [ ] **Step 2: Run test to confirm it fails**
  ```powershell
  pytest tests/test_server_protocol.py
  ```
- [ ] **Step 3: Implement `src/voice_agent/server/protocol.py`**
  Implement `ClientMessageType`, `ServerMessageType`, `ClientMessage`, `ServerMessage`, and JSON parsers with strict type validation.
- [ ] **Step 4: Run test to verify it passes**
  ```powershell
  pytest tests/test_server_protocol.py
  ```
- [ ] **Step 5: Commit protocol implementation**
  ```powershell
  git add src/voice_agent/server/ tests/test_server_protocol.py
  git commit -m "feat(server): implement websocket protocol message schemas and validators"
  ```

---

### Task 3: Virtual / Mock Audio Loopback & Hardware Fallback

**Files:**
- Create: `src/voice_agent/audio/virtual_device.py`
- Test: `tests/test_virtual_audio.py`

**Interfaces:**
- Consumes: `voice_agent.audio.ring_buffer.AudioRingBuffer`
- Produces: `VirtualAudioDevice` class mimicking sounddevice/PyAudio stream callbacks with synthetic tones or silence buffers, ensuring headless tests pass 100%.

- [ ] **Step 1: Write the failing test for VirtualAudioDevice**
  ```python
  # tests/test_virtual_audio.py
  from voice_agent.audio.virtual_device import VirtualAudioDevice
  def test_virtual_audio_loopback():
      dev = VirtualAudioDevice(sample_rate=24000, channels=1)
      dev.start()
      assert dev.is_active
      data = dev.generate_chunk(samples=480)
      assert len(data) == 480
      dev.stop()
      assert not dev.is_active
  ```
- [ ] **Step 2: Run test to confirm it fails**
  ```powershell
  pytest tests/test_virtual_audio.py
  ```
- [ ] **Step 3: Implement `VirtualAudioDevice` in `src/voice_agent/audio/virtual_device.py`**
- [ ] **Step 4: Run test to verify it passes**
  ```powershell
  pytest tests/test_virtual_audio.py
  ```
- [ ] **Step 5: Commit virtual audio fallback**
  ```powershell
  git add src/voice_agent/audio/virtual_device.py tests/test_virtual_audio.py
  git commit -m "feat(audio): add virtual audio device loopback for headless execution"
  ```

---

### Task 4: FastAPI WebSocket & Static Server Implementation

**Files:**
- Create: `src/voice_agent/server/app.py`
- Test: `tests/test_server_websocket.py`

**Interfaces:**
- Consumes: `voice_agent.server.protocol`, `voice_agent.pipeline.VoiceAgentCoordinator`, `voice_agent.audio.ring_buffer.AudioRingBuffer`.
- Produces: `create_app()` returning FastAPI instance with:
  - `GET /` mounting `ui/` static files
  - `WebSocket /ws` handling duplex streaming, VAD broadcast, and sub-50ms barge-in drain.

- [ ] **Step 1: Write the failing test for FastAPI WebSocket handshake and drain**
  ```python
  # tests/test_server_websocket.py
  from fastapi.testclient import TestClient
  from voice_agent.server.app import create_app
  def test_websocket_connection_and_bargein():
      app = create_app(mock_mode=True)
      client = TestClient(app)
      with client.websocket_connect("/ws") as ws:
          # Verify initial connection state
          data = ws.receive_json()
          assert data["type"] == "STATE_CHANGE"
          # Send barge-in trigger
          ws.send_json({"type": "BARGE_IN", "payload": {}})
          resp = ws.receive_json()
          assert resp["type"] in ["DRAIN_ALERT", "STATE_CHANGE"]
  ```
- [ ] **Step 2: Run test to confirm it fails**
  ```powershell
  pytest tests/test_server_websocket.py
  ```
- [ ] **Step 3: Implement `src/voice_agent/server/app.py`**
  Implement FastAPI application mounting `ui/` static files and handling `/ws` endpoint with connection management, asynchronous audio queue, and instantaneous atomic drain.
- [ ] **Step 4: Run test to verify it passes**
  ```powershell
  pytest tests/test_server_websocket.py
  ```
- [ ] **Step 5: Commit server implementation**
  ```powershell
  git add src/voice_agent/server/app.py tests/test_server_websocket.py
  git commit -m "feat(server): implement fastapi unified websocket and static hud server"
  ```

---

### Task 5: Frontend WebSocket Client & Single-Page HUD Integration

**Files:**
- Modify: `ui/index.html`
- Modify: `ui/app.css`

**Interfaces:**
- Consumes: `ws://localhost:8000/ws` protocol messages.
- Produces: Live UI reactivity:
  - Dynamic Voice Graph renders real-time audio amplitude and FFT frequency spectrum
  - Word cursor highlights streamed tokens from backend `CursorTracker`
  - Barge-In button triggers sub-50ms queue drain over WebSocket and displays measured roundtrip latency
  - Character Voice Selector updates active backend voice profile
  - Graceful fallback to simulation mode when disconnected.

- [ ] **Step 1: Add WebSocket connection manager to `ui/index.html`**
  Add `connectWebSocket()` with automatic reconnect, state sync, and connection status indicator.
- [ ] **Step 2: Wire incoming audio chunks and FFT data to `renderDynamicVoiceGraph()`**
- [ ] **Step 3: Wire word cursor tracking to backend `CURSOR_UPDATE` events**
- [ ] **Step 4: Wire barge-in button to dispatch `BARGE_IN` packet and benchmark drain latency**
- [ ] **Step 5: Wire voice selector to dispatch `SELECT_VOICE` payload to server**
- [ ] **Step 6: Test in browser at `http://localhost:8000/` and verify clean console with zero errors**
- [ ] **Step 7: Commit frontend integration**
  ```powershell
  git add ui/index.html ui/app.css
  git commit -m "feat(ui): integrate live websocket client into unified single-page hud"
  ```

---

### Task 6: End-to-End Latency Verification & Automated Integration Test Suite

**Files:**
- Create: `tests/test_e2e_websocket_hud.py`
- Modify: `CHANGELOG.md`
- Modify: `MEMORY.md`

**Interfaces:**
- Consumes: Full stack: `app.py`, `pipeline.py`, `protocol.py`.
- Produces: Comprehensive integration test suite verifying:
  - WebSocket connection and 24kHz audio streaming
  - 30ms VAD event broadcast
  - Barge-in drain execution $< 50\text{ ms}$ (benchmark $\le 0.05\text{ ms}$)
  - Sentence resumption pre-roll protocol.

- [ ] **Step 1: Write comprehensive end-to-end integration test in `tests/test_e2e_websocket_hud.py`**
- [ ] **Step 2: Run full test suite across all 8 test modules**
  ```powershell
  pytest tests/ -v
  ```
- [ ] **Step 3: Update `CHANGELOG.md` with final [0.3.0] release notes and `MEMORY.md` benchmarks**
- [ ] **Step 4: Commit e2e test suite and documentation updates**
  ```powershell
  git add tests/test_e2e_websocket_hud.py CHANGELOG.md MEMORY.md
  git commit -m "test(e2e): add websocket integration test suite and update documentation"
  ```
