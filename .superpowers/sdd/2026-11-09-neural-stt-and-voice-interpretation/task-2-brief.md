# Task 2 Brief: Speech Turn Accumulator & Voice Activity Segmentation (`voice_agent.stt.accumulator`)

## Files to Create/Modify
- Create: `src/voice_agent/stt/accumulator.py`
- Create/Test: `tests/test_speech_accumulator.py`

## Instructions
1. Write failing unit test in `tests/test_speech_accumulator.py`:
   - Test `SpeechTurnAccumulator` buffers incoming PCM chunks.
   - Test that silence chunks under energy threshold don't trigger speech.
   - Test loud audio triggers `is_speaking = True`.
   - Test silence exceeding `silence_timeout_ms` triggers turn completion and returns the utterance bytes.
   - Test `flush()` returns accumulated speech and resets state.
2. Run test: `.venv\Scripts\python -m pytest tests/test_speech_accumulator.py -v` (verify it fails).
3. Implement `src/voice_agent/stt/accumulator.py`:
   - Class `SpeechTurnAccumulator(sample_rate=16000, energy_threshold=0.04, silence_timeout_ms=650, min_speech_duration_ms=400)`.
   - `feed(pcm_chunk: bytes) -> Optional[bytes]`: Computes RMS, tracks speech start/end timestamps, returns completed utterance bytes when pause detected.
   - `flush() -> Optional[bytes]`: Flushes active buffer.
   - `reset()`: Resets all state.
4. Run tests: `.venv\Scripts\python -m pytest tests/test_speech_accumulator.py -v` (verify all pass).
5. Run full test suite: `.venv\Scripts\python -m pytest tests/` (verify all pass).
6. Commit:
   ```bash
   git add src/voice_agent/stt/accumulator.py tests/test_speech_accumulator.py
   git commit -m "feat(stt): add SpeechTurnAccumulator with energy VAD pause detection"
   ```
7. Report back with exact test output and git commit hash.
