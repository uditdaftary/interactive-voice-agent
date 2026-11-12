# Task 3 Brief: WebSocket Protocol Schemas & Bi-Directional STT Streaming

## Files to Create/Modify
- Modify: `src/voice_agent/server/schemas.py`
- Modify: `src/voice_agent/server/websocket.py`
- Create/Test: `tests/test_stt_websocket.py`

## Instructions
1. Write failing unit test in `tests/test_stt_websocket.py`:
   - Test `UserSpeechTranscriptMessage` schema serialization and deserialization (`type="user_speech_transcript"`, `transcript: str`, `is_final: bool = True`, `latency_ms: Optional[float] = None`).
   - Test `parse_client_message` with `voice_end` message type.
2. Run test: `.venv\Scripts\python -m pytest tests/test_stt_websocket.py -v` (verify it fails).
3. Implement updates:
   - In `schemas.py`: Add `UserSpeechTranscriptMessage` to exports and `ServerMessage` union.
   - In `websocket.py`:
     - Integrate `SpeechTurnAccumulator` and `STTEngine` into `WebSocketHandler` / `ConnectionManager`.
     - When receiving raw PCM binary chunks while in `LISTENING` / `BARGE_IN` mode, feed into accumulator.
     - On utterance detected or `voice_end` message received:
       - Run `stt_engine.transcribe(utterance)`.
       - If transcript is non-empty:
         - Broadcast `UserSpeechTranscriptMessage(transcript=transcript, is_final=True)`.
         - Call `narration_engine.handle_user_interruption(transcript)`.
4. Run tests: `.venv\Scripts\python -m pytest tests/test_stt_websocket.py -v` (verify all pass).
5. Run full test suite: `.venv\Scripts\python -m pytest tests/` (verify all pass).
6. Commit:
   ```bash
   git add src/voice_agent/server/schemas.py src/voice_agent/server/websocket.py tests/test_stt_websocket.py
   git commit -m "feat(server): integrate server-side STT into WebSocket streaming pipeline"
   ```
7. Report back with exact test output and git commit hash.
