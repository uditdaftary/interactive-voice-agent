"""The speech judge decides whether decoded words may stand, from the audio that produced them."""

import numpy as np

from voice_agent.stt import local_stt_worker, speech_judge
from voice_agent.stt.speech_judge import acoustic, evidence, judge, looping

SAMPLE_RATE = 16000


def voiced_tone(amplitude: float, seconds: float = 1.0, hz: float = 150.0) -> np.ndarray:
    """A pitched tone with a little noise, the shape of a sustained voiced vowel."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rng = np.random.default_rng(7)
    wave = amplitude * np.sin(2 * np.pi * hz * t) + 200.0 * rng.standard_normal(t.size)
    return np.clip(wave, -32768, 32767).astype(np.int16)


def after_silence(signal: np.ndarray) -> np.ndarray:
    """Sound after a half second of quiet. The gate sits above the recording's own noise floor, so a recording that is
    all sound has no floor to rise above, and the worker only ever sees speech set in quiet."""
    return np.concatenate([np.zeros(SAMPLE_RATE // 2, dtype=np.int16), signal])


def test_silence_is_quiet() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    assert judge("hello there", evidence(silence)) == "quiet"


def test_voiced_tone_passes() -> None:
    audio = after_silence(voiced_tone(8000))
    assert acoustic(evidence(audio)) is None
    assert judge("hello there", evidence(audio)) is None


def test_white_noise_not_voiced() -> None:
    rng = np.random.default_rng(3)
    noise = np.clip(rng.standard_normal(SAMPLE_RATE) * 3000, -32768, 32767).astype(np.int16)
    assert acoustic(evidence(after_silence(noise))) == "not-voiced"


def test_stock_phrase_rejected_on_weak_audio() -> None:
    weak = after_silence(voiced_tone(300))
    assert judge("thank you", evidence(weak)) == "stock-phrase"


def test_looping_detected() -> None:
    assert looping("tooth tooth tooth tooth on tooth".split())
    assert not looping("no no no".split())


def test_worker_reexports_judge() -> None:
    assert local_stt_worker.judge is speech_judge.judge
