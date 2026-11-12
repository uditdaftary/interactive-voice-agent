# Task 4 Brief: REST API Endpoint `POST /api/transcribe` (`voice_agent.server.app`)

## Files to Create/Modify
- Modify: `src/voice_agent/server/app.py`
- Create/Test: `tests/test_transcribe_api.py`

## Instructions
1. Write failing unit test in `tests/test_transcribe_api.py`:
   - Test `POST /api/transcribe` with an uploaded audio file (`multipart/form-data`, WAV or PCM audio bytes).
   - Test response format: `{"transcript": str, "duration_sec": float, "status": "ok"}`.
   - Test error handling when empty/invalid file provided (returns 400 or empty transcript).
2. Run test: `.venv\Scripts\python -m pytest tests/test_transcribe_api.py -v` (verify it fails).
3. Implement `POST /api/transcribe` in `src/voice_agent/server/app.py`:
   - Instantiate or access `STTEngine` (with lazy singleton or app state).
   - Read file content from `UploadFile`.
   - Invoke `stt_engine.transcribe(audio_bytes)`.
   - Calculate audio duration if WAV header present, or from sample count.
   - Return JSON with transcript, duration, and status.
4. Run tests: `.venv\Scripts\python -m pytest tests/test_transcribe_api.py -v` (verify all pass).
5. Run full test suite: `.venv\Scripts\python -m pytest tests/` (verify all pass).
6. Commit:
   ```bash
   git add src/voice_agent/server/app.py tests/test_transcribe_api.py
   git commit -m "feat(api): add POST /api/transcribe endpoint"
   ```
7. Report back with exact test output and git commit hash.
