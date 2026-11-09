Implement and integrate a full-stack interactive local voice agent application with a unified single-page HUD dashboard as a capability showcase. Deliver an offline, zero-cloud-dependency voice assistant featuring real-time bi-directional streaming, sub-50ms barge-in interruption, and live audio telemetry.

Working directory: c:\Projects\interactive-voice-agent
Integrity mode: development

## Requirements

### R1. Unified FastAPI WebSocket Server & Static Dashboard Hosting
Host a unified local FastAPI server on port 8000 that serves both the static single-page HUD dashboard (`ui/index.html`, `ui/app.css`, `ui/fonts/`) at `http://localhost:8000/` and provides a bi-directional WebSocket endpoint at `ws://localhost:8000/ws`.
- Stream bi-directional 24kHz audio PCM chunks, 30ms VAD speech activity events, word cursor offset positions, and instantaneous barge-in interruption signals over WebSocket.
- Support hybrid microphone input: default to browser Web Audio API (`navigator.mediaDevices.getUserMedia`) with a HUD toggle to switch to host Python hardware microphone (`sounddevice`).
- Provide an automatic virtual/mock audio loopback fallback so the server and automated test suites operate reliably in environments without active physical audio hardware.
- Maintain complete offline execution with zero external cloud or network dependencies.

### R2. Unified Single-Page HUD Dashboard Integration
Integrate the live WebSocket connection into the unified single-page HUD dashboard (`ui/index.html`, `ui/app.css`):
- Dynamic Voice Graph & Audio Visualizer: Render real-time voice waveforms, decibel level meters, and FFT frequency spectrum at 60 FPS on HTML5 canvas.
- Continuous Narrative Reader: Display streaming narrative sentences with active word cursor tracking in inverted Plot White, resumption pre-roll cues in Signal Yellow, and cadence playback controls.
- Sub-50ms Barge-In Disruption Pipeline: Provide instantaneous visual feedback upon user speech disruption, render a disruption wave canvas, and display dynamic Q&A prompt envelope telemetry.
- Resident Character Voice Selector Table: Display and enable live selection of character voices (David Martinez, Lucy, V-Male, V-Female, Johnny Silverhand) with corresponding WPM cadences and timbre modulation.
- Implement seamless fallback to local simulated preview mode when the WebSocket server is disconnected or unreachable.

### R3. Visual Design System Conformance
Strictly adhere to the design system tokens and layout of "The Catalog Sheet" documented in `c:\Projects\interactive-voice-agent\DESIGN.md`:
- Palette: Ground `#0b0b0b`, Ink `#efede6`, Signal Yellow `#e8b622`, Fault Red `#e03a2f`, Sleeve `#121211`, Hairlines `rgba(239, 237, 230, 0.15)`.
- Typography: Local Archivo (display 125% width) and Martian Mono (tabular numerals, metrics) located in `ui/fonts/`.
- Geometry: Square 0px corners throughout, 44px tabbed controls with right-edge action tabs.

### R4. Development Workflow, Documentation & Git Discipline
- Feature Branching & Commits: Create and work on a dedicated feature branch (`feature/unified-hud-websocket-server`). Make atomic, descriptive git commits following conventional commit syntax for each discrete task or component.
- Issue Tracking: File GitHub issues via `gh issue create` for any discovered bugs, architectural bottlenecks, or blockers.
- Strict Documentation Maintenance:
  - `MEMORY.md`: Maintain and update architectural decisions, WebSocket protocol specifications, hardware limits, and DSP/math models.
  - `CHANGELOG.md`: Maintain and update following Keep a Changelog format and Semantic Versioning.
  - `AGENTS.md`: Maintain and document multi-agent execution topology, operational roles, subagent responsibilities, and review protocols.
- Execution Discipline: Formulate an implementation plan with bite-sized tasks, apply test-driven development (TDD), and execute per-task review gates before integrating.

## Verification Resources
- Existing unit and pipeline tests in `tests/` (`test_audio_buffer.py`, `test_vad.py`, `test_cursor.py`, `test_state_machine.py`, `test_llm.py`, `test_context.py`, `test_full_pipeline.py`).
- CLI interactive runner `run_agent.py`.
- Automated pytest integration test suite covering WebSocket handshake, bi-directional 24kHz PCM audio streaming, VAD 30ms event broadcast, cursor position tracking, and atomic buffer drain latency benchmark (< 0.05ms).

## Acceptance Criteria

### Server & Streaming Integration
- [ ] FastAPI server starts locally on `http://localhost:8000` and serves the HUD dashboard static assets and fonts with zero 404s.
- [ ] WebSocket endpoint `ws://localhost:8000/ws` accepts client connections, exchanges bi-directional 24kHz PCM audio frames, and handles client disconnections gracefully without crashing.
- [ ] VAD emits speech activity events for 30ms audio frames over the WebSocket connection.
- [ ] Word cursor offset positions are transmitted over WebSocket in sync with audio output.
- [ ] Switching between browser Web Audio mic input and host Python sounddevice mic functions via the WebSocket protocol without stream corruption.
- [ ] Server and tests automatically use virtual/mock audio loopback when physical audio devices are absent.
- [ ] All components run strictly offline with no external network requests.

### Performance & Barge-In Latency
- [ ] Ring buffer drains atomically in < 0.05ms upon barge-in signal detection.
- [ ] End-to-end barge-in interruption latency from user speech detection to playback halt is under 50ms.
- [ ] Automated pytest integration test verifies WebSocket streaming, VAD events, and queue drain benchmark with 100% pass rate.

### HUD Dashboard Interface
- [ ] Opening `http://localhost:8000/` connects to `ws://localhost:8000/ws` and transitions the HUD connection status indicator to connected.
- [ ] Voice graph canvas renders dynamic 60 FPS waveform and FFT spectrum reflecting live audio telemetry.
- [ ] Narrative reader streams incoming text and highlights active words in inverted Plot White with Signal Yellow pre-roll cues.
- [ ] Barge-in disruption canvas and HUD state indicator switch instantaneously to disruption state upon barge-in.
- [ ] Selecting character personas (David, Lucy, V-Male, V-Female, Johnny Silverhand) updates active voice settings and parameters over the WebSocket.
- [ ] When the WebSocket connection is closed or offline, the HUD falls back seamlessly to interactive simulation mode.

### Development Workflow & Documentation
- [ ] All development work is contained on branch `feature/unified-hud-websocket-server` with conventional commits for each discrete unit of work.
- [ ] Any architectural blockers or bugs encountered during implementation are logged via `gh issue create`.
- [ ] `MEMORY.md` documents WebSocket protocols, state sync schemas, and hardware bounds.
- [ ] `CHANGELOG.md` records all additions and changes per Keep a Changelog specifications.
- [ ] `AGENTS.md` is updated with multi-agent roles, subagent responsibilities, and verification gates.
