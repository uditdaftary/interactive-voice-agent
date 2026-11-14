"""Decides whether decoded words may stand, from the audio that produced them.

Speech models write words for breaths, key presses, fans and dead inputs. The judge reads the audio (loudness, duration,
voicing) and the words (repetition, pace, stock phrases) and returns the reason to drop a decode, or None to keep it.
Shared by every STT backend: the Moonshine worker imports these, and the Whisper backend judges its own output.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME = 480               # 30 ms
GATE_DB = -52.0           # a recording whose loudest 30 ms frames never reach this holds no speech
SPEECH_FRAMES = 5         # 150 ms above the gate
QUIET_PEAK_DB = -50.0     # no sound reached this: a dead or wrong input (or a muted one), not a person speaking
MIN_SOUND_MS = 150        # fewer 30 ms frames of sound than this (5) are a click or a breath
MIN_VOICED_MS = 60        # and at least 2 of them must carry a voice pitch
VOICED_LAG_MIN, VOICED_LAG_MAX = 40, 230   # 400 Hz down to 70 Hz at 16 kHz
VOICED_LEVEL = 0.4
LONG_SOUND_MS = 3000      # a piece of sound this long is speech only if a good share of it is voiced
MIN_VOICED_SHARE = 0.35   # speech measured 0.51 and up (clean, -30 dB, loud room, 150-level noise); a hands-free headset's hiss and a fan far less
MAX_WORDS_PER_SECOND = 7.5
# What a speech model says on sound that holds no speech, whatever the model (heard from Whisper, Moonshine and the like).
HALLUCINATIONS = frozenset(["you", "thank you", "thanks for watching", "thank you for watching", "thank you very much", "bye",
                            "bye bye", "so", "the", "i", "uh", "um", "hmm", "mm", "mhm", "oh", "ah", "huh"])
FILLER = frozenset(["the", "a", "an", "and", "to", "of", "i", "you", "it", "is", "in", "that", "this", "we", "my", "me", "so", "but", "or", "on"])


def frame_levels(samples: np.ndarray) -> np.ndarray:
    """dBFS of every 30 ms frame."""
    count = len(samples) // FRAME
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: count * FRAME].reshape(count, FRAME).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return 20.0 * np.log10(rms)


def speech_gate(levels: np.ndarray) -> float:
    return max(GATE_DB, float(np.percentile(levels, 10)) + 9.0)


def has_speech(samples: np.ndarray) -> bool:
    levels = frame_levels(samples)
    if levels.size < SPEECH_FRAMES:
        return False
    return int(np.count_nonzero(levels > speech_gate(levels))) >= SPEECH_FRAMES


def evidence(samples: np.ndarray) -> dict[str, int | float]:
    """What the audio itself says about whether a person spoke: how loud the loudest sound was, how long the sound lasted,
    and how much of it was voiced (a pitch between 70 and 400 Hz repeating inside the 30 ms frame: vowels have one, a breath,
    a click, a fan and a key press do not). The words are judged against this (judge()), so a decode of noise is never shown."""
    levels = frame_levels(samples)
    result: dict[str, int | float] = {"peakDb": -120.0, "soundMs": 0, "spanMs": 0, "voicedMs": 0}
    if levels.size == 0:
        return result
    above = levels > speech_gate(levels)
    result["peakDb"] = round(float(levels.max()), 1)
    if not above.any():
        return result
    indexes = np.nonzero(above)[0]
    result["soundMs"] = int(indexes.size * FRAME * 1000 // SAMPLE_RATE)
    result["spanMs"] = int((indexes[-1] - indexes[0] + 1) * FRAME * 1000 // SAMPLE_RATE)
    result["peakDb"] = round(float(levels[above].max()), 1)
    frames = samples[: levels.size * FRAME].reshape(levels.size, FRAME)[above].astype(np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames, n=1024, axis=1)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=1024, axis=1)
    energy = autocorrelation[:, 0] + 1e-9
    best = (autocorrelation[:, VOICED_LAG_MIN:VOICED_LAG_MAX] / energy[:, None]).max(axis=1)
    result["voicedMs"] = int(np.count_nonzero(best >= VOICED_LEVEL) * FRAME * 1000 // SAMPLE_RATE)
    return result


def acoustic(proof: dict[str, int | float]) -> str | None:
    """None when the sound could be a person speaking, else why it could not."""
    if proof["peakDb"] < QUIET_PEAK_DB:
        return "quiet"
    if proof["soundMs"] < MIN_SOUND_MS:
        return "short"
    if proof["voicedMs"] < MIN_VOICED_MS:
        return "not-voiced"
    if proof["soundMs"] >= LONG_SOUND_MS and proof["voicedMs"] < proof["soundMs"] * MIN_VOICED_SHARE:
        return "not-voiced"
    return None


def looping(words: list[str]) -> bool:
    """A decoder stuck in a loop: one word three times in a row among others, a 2-3 word chunk three times in a row, or one word
    making up much of a longer text ("tooth tooth on tooth tooth"). A person saying "no no no" or "very very good" is not caught."""
    count = len(words)
    for size in (1, 2, 3):
        for start in range(0, count - size * 3 + 1):
            chunk = words[start: start + size]
            if not all(words[start + size * i: start + size * (i + 1)] == chunk for i in range(3)):
                continue
            if size > 1 or count >= 5:
                return True
    if count >= 6:
        best = max((words.count(word) for word in set(words) - FILLER), default=0)
        if best >= 4 and best * 3 >= count:
            return True
    return False


def judge(text: str, proof: dict[str, int | float]) -> str | None:
    """None when the words may stand, else why they may not. Moonshine can put words on a breath, a key press, a fan or a dead input;
    those never reach the screen, the voice or the model."""
    words = [word for word in "".join(c.lower() if c.isalnum() or c == "'" else " " for c in text).split() if word]
    if not words:
        return None
    why = acoustic(proof)
    if why:
        return why
    if proof["spanMs"] and len(words) * 1000 / proof["spanMs"] > MAX_WORDS_PER_SECOND and len(words) > 3:
        return "too-many-words"
    if looping(words):
        return "repeating"
    if " ".join(words) in HALLUCINATIONS and (proof["voicedMs"] < 240 or proof["peakDb"] < -36.0):
        return "stock-phrase"
    return None


def tail_is_quiet(samples: np.ndarray, start: int) -> bool:
    """True when the audio after `start` holds no speech (judged against the whole recording's noise floor)."""
    levels = frame_levels(samples)
    if levels.size < SPEECH_FRAMES:
        return False
    first = start // FRAME
    return int(np.count_nonzero(levels[first:] > speech_gate(levels))) < 3
