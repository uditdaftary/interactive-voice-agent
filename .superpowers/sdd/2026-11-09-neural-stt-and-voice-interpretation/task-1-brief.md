# Task 1 Brief: Neural STT Engine Abstraction (`voice_agent.stt.recognizer`)

## Files to Create/Modify
- Create: `src/voice_agent/stt/__init__.py`
- Create: `src/voice_agent/stt/recognizer.py`
- Create/Test: `tests/test_stt_recognizer.py`

## Instructions
1. Write failing unit test in `tests/test_stt_recognizer.py`:
   - Test `MockSTTEngine` returns fixed transcript.
   - Test `STTEngine` initialization (default model_size="tiny.en", device="cpu", compute_type="int8").
   - Test `STTEngine.transcribe()` with silence returns empty string or mock without crashing.
   - Test `STTEngine.transcribe()` with real audio bytes (or simulated 16kHz PCM).
2. Run test: `.venv\Scripts\python -m pytest tests/test_stt_recognizer.py -v` (verify it fails).
3. Implement `src/voice_agent/stt/recognizer.py` and `src/voice_agent/stt/__init__.py`:
   - Import `faster_whisper.WhisperModel`.
   - Wrap transcription with error handling and fallback to `SpeechRecognition` (`speech_recognition.Recognizer`).
   - Add `MockSTTEngine` for deterministic testing.
   - Guard against silence: if peak amplitude < 0.01, return empty string.
4. Run tests: `.venv\Scripts\python -m pytest tests/test_stt_recognizer.py -v` (verify all pass).
5. Run full test suite: `.venv\Scripts\python -m pytest tests/` (verify 123+ tests pass).
6. Commit:
   ```bash
   git add src/voice_agent/stt/ tests/test_stt_recognizer.py
   git commit -m "feat(stt): add neural STTEngine with faster-whisper and SpeechRecognition fallback"
   ```
7. Report back with exact test output and git commit hash.
