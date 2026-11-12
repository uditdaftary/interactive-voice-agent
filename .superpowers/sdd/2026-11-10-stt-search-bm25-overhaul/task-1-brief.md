# Task 1 Brief: Audio Downsampling, STT Monotonic Silence & Faster-Whisper Resampling (Issue #16)

## Objective
Resolve Issue #16: Eliminate the 48kHz -> 16kHz sample-rate mismatch, prevent the 469ms premature silence cut-off in `SpeechTurnAccumulator`, and add defensive resampling + terminology prompt conditioning in `STTEngine`.

## Assigned Files
- `ui/index.html` (L1800-1845)
- `src/voice_agent/stt/accumulator.py`
- `src/voice_agent/stt/recognizer.py`
- `src/voice_agent/server/websocket.py`
- `tests/test_speech_accumulator.py`
- `tests/test_stt_recognizer.py`

## Requirements
1. **Client-Side Polyphase/Linear Downsampler in `ui/index.html`**:
   - In `this.micProcessor.onaudioprocess`:
     - Determine ratio: `const ratio = this.audioContext.sampleRate / 16000;`
     - Downsample `input` (float32 array) to 16,000 Hz using linear interpolation.
     - Convert the resampled 16kHz float32 array to `Int16Array` and transmit `pcm16.buffer` over WebSocket.
     - Also calculate RMS on the downsampled buffer for hands-free and silence tracking.
2. **Monotonic Elapsed-Time Silence Tracking in `SpeechTurnAccumulator`**:
   - In `src/voice_agent/stt/accumulator.py`:
     - Base silence duration strictly on true monotonic wall-clock elapsed time: `now = time.monotonic()`, `elapsed_silence_ms = (now - self._last_speech_time) * 1000.0`.
     - When `elapsed_silence_ms >= self.silence_timeout_ms`, emit utterance.
     - Keep `energy_threshold = 0.015` and add tolerance for low-energy speech.
3. **Defensive Resampling & Initial Prompt in `STTEngine`**:
   - In `src/voice_agent/stt/recognizer.py`:
     - In `STTEngine.transcribe(audio_data, sample_rate=16000)`:
     - If `sample_rate != 16000` and `audio_np.size > 0`:
       - Resample using linear interpolation or `scipy.signal.resample_poly` to guarantee 16,000 Hz.
     - In `_whisper_model.transcribe`:
       - Add `initial_prompt="Cyberpunk 2077, Night City, Arasaka, Sandevistan, Johnny Silverhand, Lucy, David Martinez."` to bias phoneme decoding towards Cyberpunk terminology.
4. **WebSocket Handling & Error Propagation**:
   - In `src/voice_agent/server/websocket.py`:
     - When Faster-Whisper returns `""`, log debug info. If `fallback_query` is present, use it.
5. **Testing**:
   - Write tests in `tests/test_speech_accumulator.py` and `tests/test_stt_recognizer.py` validating 48kHz audio handling, resampling, and monotonic silence timeout.
   - Run `pytest tests/test_speech_accumulator.py tests/test_stt_recognizer.py`.
