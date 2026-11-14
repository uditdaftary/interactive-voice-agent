"""The STTEngine adapter: text in, text out, over the Whisper backend. No model loads (see fake_whisper)."""

import numpy as np
import pytest

from fakes import FakeWhisper
from speech_audio import after_silence, speech_like_pcm, voiced_tone
from voice_agent.stt.recognizer import STTEngine

SAMPLE_RATE = 16000


def test_adapter_returns_backend_text(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "what is arasaka"
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8")

    assert engine.transcribe(speech_like_pcm()) == "what is arasaka"


def test_adapter_accepts_int16_and_float_arrays(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "what is arasaka"
    engine = STTEngine()
    speech = after_silence(voiced_tone(8000))

    assert engine.transcribe(speech) == "what is arasaka"
    assert engine.transcribe(speech.astype(np.float32) / 32768.0) == "what is arasaka"


def test_silence_and_empty_input_give_empty_text(fake_whisper: FakeWhisper) -> None:
    engine = STTEngine()

    assert engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()) == ""
    assert engine.transcribe(b"") == ""
    assert engine.transcribe(np.array([], dtype=np.int16)) == ""
    assert fake_whisper.constructions == []


def test_48khz_input_is_resampled_to_16khz(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "test transcription"
    t = np.arange(48000) / 48000.0
    tone_48k = np.clip(8000 * np.sin(2 * np.pi * 150 * t), -32768, 32767).astype(np.int16)
    audio_48k = np.concatenate([np.zeros(24000, dtype=np.int16), tone_48k])

    text = STTEngine().transcribe(audio_48k.tobytes(), sample_rate=48000)

    assert text == "test transcription"
    # 0.5 s of quiet plus 1 s of tone, at 16 kHz.
    assert fake_whisper.inputs[0].size == 24000


def test_initial_prompt_reaches_the_model(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "sandevistan arasaka"

    STTEngine(initial_prompt="Arasaka, Sandevistan").transcribe(speech_like_pcm())

    assert fake_whisper.transcribe_kwargs[0]["initial_prompt"] == "Arasaka, Sandevistan"


def test_unsupported_input_type_fails_loud() -> None:
    with pytest.raises(TypeError):
        STTEngine().transcribe(["not audio"])  # type: ignore[arg-type]  # deliberately the wrong type
