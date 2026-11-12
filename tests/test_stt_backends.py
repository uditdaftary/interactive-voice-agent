"""The Whisper backend: its words are judged against the audio that produced them, and the model loads lazily."""

import inspect
import logging
import sys

import pytest

from fakes import FakeWhisper
from speech_audio import after_silence, speech_like_pcm, voiced_tone
from voice_agent.stt import backends
from voice_agent.stt.backends import STT_UNAVAILABLE, Transcript, WhisperBackend

SAMPLE_RATE = 16000


def test_empty_audio_is_quiet(fake_whisper: FakeWhisper) -> None:
    assert WhisperBackend().transcribe(b"", SAMPLE_RATE) == Transcript("", "quiet")
    assert fake_whisper.constructions == []


def test_initial_prompt_defaults_empty(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "hello there"
    WhisperBackend().transcribe(speech_like_pcm(), SAMPLE_RATE)

    assert fake_whisper.constructions == [{"model_size": "tiny.en", "device": "cpu", "compute_type": "int8"}]
    assert fake_whisper.transcribe_kwargs[0]["initial_prompt"] is None


def test_initial_prompt_from_setting(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "hello there"
    WhisperBackend(initial_prompt="Arasaka").transcribe(speech_like_pcm(), SAMPLE_RATE)

    assert fake_whisper.transcribe_kwargs[0]["initial_prompt"] == "Arasaka"


def test_judge_rejects_decode_of_noise(fake_whisper: FakeWhisper) -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    noise = np.clip(rng.standard_normal(SAMPLE_RATE) * 3000, -32768, 32767).astype(np.int16)
    fake_whisper.text = "thank you"

    result = WhisperBackend().transcribe(after_silence(noise).tobytes(), SAMPLE_RATE)

    assert result.text == ""
    assert result.rejected_reason is not None


def test_judge_rejects_stock_phrase_after_decode(fake_whisper: FakeWhisper) -> None:
    # Weak voiced audio passes the acoustic checks, so the decoded words are judged after the decode.
    fake_whisper.text = "thank you"

    result = WhisperBackend().transcribe(speech_like_pcm(amplitude=300), SAMPLE_RATE)

    assert result == Transcript("", "stock-phrase")
    assert len(fake_whisper.inputs) == 1


def test_judge_runs_before_gain_normalisation(fake_whisper: FakeWhisper) -> None:
    # A 150 Hz tone at amplitude 140 has frame RMS about -50.4 dBFS, under QUIET_PEAK_DB. Normalising first would lift its peak
    # to 0.75 and let it through, so the judge must see the int16 audio.
    fake_whisper.text = "hello there"
    quiet_tone = after_silence(voiced_tone(140, noise=0.0)).tobytes()

    result = WhisperBackend().transcribe(quiet_tone, SAMPLE_RATE)

    assert result.rejected_reason == "quiet"
    assert fake_whisper.inputs == []


def test_voiced_audio_returns_text(fake_whisper: FakeWhisper) -> None:
    fake_whisper.text = "hello there"

    assert WhisperBackend().transcribe(speech_like_pcm(), SAMPLE_RATE) == Transcript("hello there", None)


def test_no_google_fallback(fake_whisper: FakeWhisper, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "speech_recognition", raising=False)
    fake_whisper.text = ""

    WhisperBackend().transcribe(speech_like_pcm(), SAMPLE_RATE)

    assert "speech_recognition" not in sys.modules
    assert "recognize_google" not in inspect.getsource(backends)


def test_whisper_unavailable_reason(fake_whisper: FakeWhisper, caplog: pytest.LogCaptureFixture) -> None:
    fake_whisper.fail_loads = 1
    caplog.set_level(logging.WARNING, logger="voice_agent.stt.backends")

    result = WhisperBackend().transcribe(speech_like_pcm(), SAMPLE_RATE)

    assert result == Transcript("", STT_UNAVAILABLE)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_initial_prompt_env() -> None:
    from voice_agent.settings import load_settings

    assert load_settings({"VOICE_AGENT_STT_INITIAL_PROMPT": "Arasaka"}).stt_initial_prompt == "Arasaka"
    assert load_settings({}).stt_initial_prompt == ""
