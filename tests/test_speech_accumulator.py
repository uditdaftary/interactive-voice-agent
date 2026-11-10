import pytest
import numpy as np
from voice_agent.stt.accumulator import SpeechTurnAccumulator


def test_speech_turn_accumulator_buffering():
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=500)
    # Feed silence (100ms: 1600 samples = 3200 bytes)
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    turn = acc.feed(silence)
    assert turn is None
    assert not acc.is_speaking

    # Feed loud audio (simulated speech: 500ms, 8000 samples)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    turn = acc.feed(loud)
    assert acc.is_speaking
    assert turn is None

    # Feed silence exceeding silence_timeout_ms (600ms: 9600 samples)
    silence_600ms = np.zeros(9600, dtype=np.int16).tobytes()
    turn = acc.feed(silence_600ms)
    # Should trigger turn complete
    assert turn is not None
    assert len(turn) > 0
    assert not acc.is_speaking


def test_speech_turn_accumulator_silence_below_threshold():
    acc = SpeechTurnAccumulator(sample_rate=16000, energy_threshold=0.04)
    # Low energy noise below threshold (peak ~ 200, RMS ~ 0.004)
    noise = np.random.randint(-200, 200, 1600, dtype=np.int16).tobytes()
    turn = acc.feed(noise)
    assert turn is None
    assert not acc.is_speaking


def test_speech_turn_accumulator_flush():
    acc = SpeechTurnAccumulator(sample_rate=16000)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    acc.feed(loud)
    assert acc.is_speaking

    flushed = acc.flush()
    assert flushed is not None
    assert len(flushed) == len(loud)
    assert not acc.is_speaking

    # Subsequent flush returns None
    assert acc.flush() is None


def test_speech_turn_accumulator_reset():
    acc = SpeechTurnAccumulator(sample_rate=16000)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    acc.feed(loud)
    assert acc.is_speaking

    acc.reset()
    assert not acc.is_speaking
    assert acc.flush() is None


def test_speech_turn_accumulator_ignores_brief_noise():
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=500, min_speech_duration_ms=400)
    # Feed brief click/noise of 100ms (< 400ms)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    noise_burst = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    turn = acc.feed(noise_burst)
    assert turn is None
    assert acc.is_speaking

    # Feed silence exceeding timeout
    silence_600ms = np.zeros(9600, dtype=np.int16).tobytes()
    turn = acc.feed(silence_600ms)
    # Discarded because speech duration (100ms) < min_speech_duration_ms (400ms)
    assert turn is None
    assert not acc.is_speaking


def test_speech_turn_accumulator_inter_word_pause():
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=500, min_speech_duration_ms=300)
    # Word 1: 200ms
    t1 = np.linspace(0, 0.2, 3200, endpoint=False)
    loud1 = (np.sin(2 * np.pi * 440 * t1) * 16000).astype(np.int16).tobytes()
    acc.feed(loud1)
    assert acc.is_speaking

    # Short pause: 150ms (< 500ms timeout)
    short_silence = np.zeros(2400, dtype=np.int16).tobytes()
    turn = acc.feed(short_silence)
    assert turn is None
    assert acc.is_speaking

    # Word 2: 200ms
    t2 = np.linspace(0, 0.2, 3200, endpoint=False)
    loud2 = (np.sin(2 * np.pi * 440 * t2) * 16000).astype(np.int16).tobytes()
    turn = acc.feed(loud2)
    assert turn is None
    assert acc.is_speaking

    # Final silence exceeding timeout (600ms > 500ms)
    final_silence = np.zeros(9600, dtype=np.int16).tobytes()
    turn = acc.feed(final_silence)
    assert turn is not None
    assert not acc.is_speaking
    # Contains all chunks: loud1 + short_silence + loud2 + final_silence
    expected_len = len(loud1) + len(short_silence) + len(loud2) + len(final_silence)
    assert len(turn) == expected_len


def test_speech_turn_accumulator_invalid_and_empty_chunks():
    acc = SpeechTurnAccumulator(sample_rate=16000)
    assert acc.feed(b"") is None
    assert acc.feed(b"\x00") is None
    assert not acc.is_speaking


def test_speech_turn_accumulator_export():
    from voice_agent.stt import SpeechTurnAccumulator as ExportedAccumulator
    assert ExportedAccumulator is SpeechTurnAccumulator


def test_speech_turn_accumulator_monotonic_silence_timeout():
    import time
    # Low silence timeout for fast test execution (60ms)
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=60, min_speech_duration_ms=50)
    # Feed 100ms loud speech
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    turn = acc.feed(loud)
    assert turn is None
    assert acc.is_speaking

    # Sleep for 100ms to let monotonic clock elapse beyond 60ms timeout (robust against Windows timer quantum)
    time.sleep(0.10)

    # Feed a very short silence chunk (10ms: 160 samples), which alone is < 60ms audio duration
    short_silence = np.zeros(160, dtype=np.int16).tobytes()
    turn = acc.feed(short_silence)
    # Monotonic elapsed time should trigger utterance completion
    assert turn is not None
    assert not acc.is_speaking


def test_speech_turn_accumulator_handles_48khz_without_premature_cutoff():
    import time
    # Default silence timeout is 1400ms
    acc = SpeechTurnAccumulator(sample_rate=16000, silence_timeout_ms=1400, min_speech_duration_ms=100)

    # Feed 200ms of loud speech at 48kHz (9600 samples)
    t = np.linspace(0, 0.2, 9600, endpoint=False)
    loud_48k = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    acc.feed(loud_48k)
    assert acc.is_speaking

    # Simulate 469ms of silence arriving in 48kHz chunks (2048 samples = 4096 bytes per chunk)
    # 22512 samples at 16kHz would be 1407ms, which caused premature cut-off in Issue #16.
    # But wall-clock time has NOT reached 1400ms.
    chunk_48k_silence = np.zeros(2048, dtype=np.int16).tobytes()
    for _ in range(11):
        turn = acc.feed(chunk_48k_silence)
        # Must NOT cut off prematurely at 469ms
        assert turn is None
        assert acc.is_speaking


def test_speech_turn_accumulator_low_energy_speech_tolerance():
    acc = SpeechTurnAccumulator(sample_rate=16000, energy_threshold=0.015)
    # Start speech turn with loud chunk (RMS ~ 0.35)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    loud = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()
    acc.feed(loud)
    assert acc.is_speaking

    # Feed lower energy speech (e.g. quiet syllable / phoneme: RMS ~ 0.012, below 0.015 but above low-energy tolerance)
    # Amplitude ~ 400 corresponds to RMS ~ 400 / (sqrt(2)*32768) ~ 0.0086
    # Let amplitude be 520 -> RMS ~ 0.0112
    quiet = (np.sin(2 * np.pi * 440 * t) * 520).astype(np.int16).tobytes()
    turn = acc.feed(quiet)
    assert turn is None
    # Must remain speaking due to hysteresis tolerance for active speech turns
    assert acc.is_speaking


