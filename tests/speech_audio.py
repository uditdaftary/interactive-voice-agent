"""Audio for the speech judge and Whisper backend tests: sound set in quiet, the way a real utterance sits in a recording."""

import numpy as np

SAMPLE_RATE = 16000


def voiced_tone(amplitude: float, seconds: float = 1.0, hz: float = 150.0, noise: float = 200.0) -> np.ndarray:
    """A pitched tone with a little noise, the shape of a sustained voiced vowel."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rng = np.random.default_rng(7)
    wave = amplitude * np.sin(2 * np.pi * hz * t) + noise * rng.standard_normal(t.size)
    return np.clip(wave, -32768, 32767).astype(np.int16)


def after_silence(signal: np.ndarray, lead_seconds: float = 0.5) -> np.ndarray:
    """Sound after quiet. The gate sits above the recording's own noise floor, so a recording that is all sound has no floor
    to rise above; the worker only ever sees speech set in quiet."""
    return np.concatenate([np.zeros(int(lead_seconds * SAMPLE_RATE), dtype=np.int16), signal])


def speech_like_pcm(amplitude: float = 8000) -> bytes:
    """A voiced tone in quiet, as PCM bytes: passes the acoustic judge."""
    return after_silence(voiced_tone(amplitude)).tobytes()
