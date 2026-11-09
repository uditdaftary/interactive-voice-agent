# Task 6 Brief: End-to-End Test Suite Verification & Benchmarks

## Files to Create/Modify
- Modify: `tests/test_e2e_pipeline.py`
- Modify: `tests/test_benchmarks.py`
- Modify: `MEMORY.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

## Instructions
1. In `tests/test_e2e_pipeline.py`:
   - Add `test_e2e_voice_barge_in_stt_transcription()`:
     - Simulate sending 16kHz PCM audio bytes of spoken speech ("what is arasaka") over WebSocket while `NARRATING`.
     - Verify state changes from `NARRATING` -> `BARGE_IN` -> `LISTENING`.
     - Verify accumulator detects speech, transcribes utterance via `STTEngine`.
     - Verify client receives `user_speech_transcript` message containing `"arasaka"`.
     - Verify client receives `prompt_envelope` with reply.
     - Verify client receives resumption cue and returns to `NARRATING`.
2. In `tests/test_benchmarks.py`:
   - Add `test_benchmark_stt_transcription_latency()`:
     - Verify `STTEngine.transcribe()` finishes within $< 350$ms budget on CPU.
3. Run complete test suite: `.venv\Scripts\python -m pytest tests/ -v`.
   - Ensure 100% test pass rate across all tests.
4. Update `MEMORY.md`:
   - Add Section 5.7: ADR-12: Server-Side Neural Speech Recognition & Turn-Taking Voice Interpretation.
5. Update `CHANGELOG.md`:
   - Add `## [0.6.0] - 2026-11-09` documenting Faster-Whisper neural STT, turn accumulator, WebSocket transcript streaming, and HUD bi-directional speech sync.
6. Update `AGENTS.md`:
   - Add Sprint 5 domain lead assignments.
7. Commit:
   ```bash
   git add tests/ MEMORY.md CHANGELOG.md AGENTS.md
   git commit -m "feat(stt): complete neural speech recognition pipeline with 100% verified test pass"
   ```
8. Report back with exact test output and git commit hash.
