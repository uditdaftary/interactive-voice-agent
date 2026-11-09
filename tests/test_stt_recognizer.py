import pytest
import numpy as np
from voice_agent.stt.recognizer import STTEngine, MockSTTEngine

def test_mock_stt_engine():
    engine = MockSTTEngine("what is arasaka")
    # Generate 1 second of dummy 16kHz audio
    audio = np.zeros(16000, dtype=np.int16).tobytes()
    text = engine.transcribe(audio)
    assert text == "what is arasaka"

def test_stt_engine_initialization():
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    assert engine.model_size == "tiny.en"
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"
    assert engine.fallback_mock is True

def test_stt_engine_transcribe_silence():
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    text = engine.transcribe(silence)
    # Near-total silence (<0.01 peak) should return empty string even if fallback_mock is True
    assert text == ""

def test_stt_engine_transcribe_empty():
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    assert engine.transcribe(b"") == ""
    assert engine.transcribe(np.array([], dtype=np.int16)) == ""

def test_stt_engine_transcribe_with_audio_bytes_or_array():
    # Non-silent audio signal (e.g. 440 Hz sine wave)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio_bytes = sine_wave.tobytes()

    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    text_from_bytes = engine.transcribe(audio_bytes)
    assert isinstance(text_from_bytes, str)

    text_from_np_int16 = engine.transcribe(sine_wave)
    assert isinstance(text_from_np_int16, str)

    sine_float = sine_wave.astype(np.float32) / 32768.0
    text_from_np_float = engine.transcribe(sine_float)
    assert isinstance(text_from_np_float, str)

def test_stt_engine_fallback_mock():
    # Force whisper model to None and test fallback_mock
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8", fallback_mock=True)
    engine._whisper_model = None

    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # When Google speech recognition fails/is offline, fallback_mock returns default
    result = engine.transcribe(sine_wave)
    assert result == "what is arasaka"
