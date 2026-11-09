Implement and integrate the full-stack interactive local voice agent application with the unified single-page HUD dashboard.

Working directory: c:\Projects\interactive-voice-agent
Integrity mode: [OPEN: Which shortcuts the team should NOT be allowed to take]

## Requirements

### R1. Full-Stack WebSocket Server & Bi-Directional Streaming
Implement a FastAPI WebSocket server (e.g. `src/voice_agent/server.py`) exposing `ws://localhost:8000/ws` for offline local execution with zero cloud dependencies.
- Stream bi-directional 24kHz audio PCM chunks, 30ms VAD activity events, word cursor offset positions, and instantaneous barge-in interruption signals.
- Support hybrid microphone input: Browser Web Audio API (`navigator.mediaDevices.getUserMedia`) by default, with HUD toggle to switch to host Python hardware mic (`sounddevice`).
- Serve static HUD assets from the server or support unified local access: [OPEN: Server hosting strategy for static HUD files].

### R2. Unified Single-Page HUD Dashboard
Integrate the live WebSocket connection into the single-page HUD dashboard (`ui/index.html`, `ui/app.css`):
- Dynamic Voice Graph & Audio Visualizer: Live 60 FPS canvas rendering real-time voice waveforms, decibel meters, and FFT spectrum.
- Continuous Narrative Reader: Sentence streaming, word cursor tracking in inverted Plot White, resumption pre-roll cues in Signal Yellow, cadence controls.
- Sub-50ms Barge-In Disruption Pipeline: Instant queue drain benchmark (< 0.05ms), disruption wave canvas, dynamic Q&A prompt envelope display.
- Resident Character Voice Selector Table: David Martinez, Lucy, V-Male, V-Female, Johnny Silverhand with live selection, WPM cadences, and timbre modulation.
- Connect `ui/index.html` to the live WebSocket server with seamless fallback/simulation when disconnected.

### R3. Visual Design System Conformance
Strictly follow the visuals and tokens of "The Catalog Sheet" documented in `c:\Projects\interactive-voice-agent\DESIGN.md`:
- Ground: #0b0b0b, Ink: #efede6, Signal Yellow: #e8b622, Fault Red: #e03a2f, Sleeve: #121211, Hairlines: rgba(239, 237, 230, 0.15).
- Typography: Local Archivo (display 125% width) and Martian Mono (tabular numerals, metrics) located in `ui/fonts/`.
- Geometry: Square 0px corners everywhere, 44px tabbed controls with right-edge action tabs.

## Verification Resources
- Existing unit and pipeline tests in `tests/` (`test_audio_buffer.py`, `test_vad.py`, `test_cursor.py`, `test_state_machine.py`, `test_llm.py`, `test_context.py`, `test_full_pipeline.py`).
- CLI demonstration runner `run_agent.py`.
- [OPEN: Verification resources and automated test requirements for WebSocket server and HUD].

## Acceptance Criteria

### WebSocket & Audio Streaming
- [ ] FastAPI WebSocket endpoint `ws://localhost:8000/ws` accepts client connections and handles bidirectional 24kHz PCM audio without buffer overflows or audio artifacts.
- [ ] Audio ring buffer drains within < 0.05ms upon barge-in signal.
- [ ] VAD 30ms frames emit real-time speech activity events over WebSocket.
- [ ] Word cursor offset positions are broadcast synchronously with audio playback.
- [ ] Hybrid mic toggle switches between browser Web Audio and host sounddevice cleanly.
- [ ] System operates completely locally with zero external network or cloud dependencies.

### HUD Dashboard Interface
- [ ] Real-time 60 FPS canvas visualizer renders live waveform and FFT spectrum when connected to WebSocket.
- [ ] Narrative reader renders streaming words with highlighted active cursor and pre-roll cues.
- [ ] Disruption canvas triggers immediately (< 50ms) upon user speech detection during agent speech.
- [ ] Character voice selector dynamically changes active voice and timbre parameters.
- [ ] UI gracefully switches to simulation mode if WebSocket is disconnected.
