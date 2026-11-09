# SDD ledger — plan: docs/superpowers/plans/2026-11-09-neural-stt-and-voice-interpretation.md

Base Commit: ea99850cc70bb4cfb0f07791a9c3e123e81b6944
Branch: feature/unified-hud-websocket-server

## Plan Pre-Flight Scan
| Task Pair / Task | Interfaces Checked | Finding | Ruling |
| --- | --- | --- | --- |
| Task 1 (STTEngine) -> Task 2 (Accumulator) | `audio_data: bytes -> STTEngine.transcribe` | Clean match | Proceed |
| Task 2 (Accumulator) -> Task 3 (WebSocket) | `SpeechTurnAccumulator.feed` -> `flush` -> `STTEngine.transcribe` | Clean match | Proceed |
| Task 3 (WebSocket) -> Task 5 (HUD) | `user_speech_transcript` schema & handling | Clean match | Proceed |
| Task 4 (REST API) -> Task 1 (STTEngine) | `/api/transcribe` -> `STTEngine.transcribe` | Clean match | Proceed |
| Task 6 (E2E & Benchmarks) -> All | Latency $< 350$ms & 100% test pass rate | Clean match | Proceed |

## Execution Progress
- [x] Task 1: Neural STT Engine Abstraction (`voice_agent.stt.recognizer`)
- [ ] Task 2: Speech Turn Accumulator & Voice Activity Segmentation (`voice_agent.stt.accumulator`)
- [ ] Task 3: WebSocket Protocol Schemas & Bi-Directional STT Streaming (`voice_agent.server.schemas`, `voice_agent.server.websocket`)
- [ ] Task 4: REST API Endpoint `POST /api/transcribe` (`voice_agent.server.app`)
- [ ] Task 5: HUD Client Bi-Directional Speech Synchronization (`ui/index.html`)
- [ ] Task 6: End-to-End Test Suite Verification & Benchmarks (`tests/test_e2e_pipeline.py`, `tests/test_benchmarks.py`)
