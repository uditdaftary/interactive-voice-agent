"""The speech judge decides whether decoded words may stand, from the audio that produced them."""

import numpy as np

from speech_audio import SAMPLE_RATE, after_silence, voiced_tone
from voice_agent.stt import local_stt_worker, speech_judge
from voice_agent.stt.speech_judge import acoustic, evidence, judge, looping


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
