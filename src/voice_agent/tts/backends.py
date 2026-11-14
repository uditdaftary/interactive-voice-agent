"""Text-to-speech backends behind one protocol, and the chain that tries them in order.

A backend turns one sentence into a `SpeechAudio`, or raises `TtsError`. The chain logs every failed backend and returns the
first audio that succeeds; when none does, the caller gets None and the narration moves on without sound for that sentence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, TypedDict

logger = logging.getLogger(__name__)

# Anything shorter is a header-only reply from the engine, not speech.
MIN_AUDIO_BYTES = 100


class CharacterAudioProfile(TypedDict):
    voice: str
    pitch_hz: int
    rate_offset: int
    name: str
    persona: str
    quote: str


# Character audio profiles with authentic Cyberpunk prosody and Edge-TTS neural models
CHARACTER_AUDIO_PROFILES: dict[str, CharacterAudioProfile] = {
    "johnny-silverhand": {
        "voice": "en-US-RogerNeural",
        "pitch_hz": -14,
        "rate_offset": -5,
        "name": "Johnny Silverhand",
        "persona": "Rebel Rockerboy / Keanu Baritone",
        "quote": "Wake up, samurai. We got a city to burn.",
    },
    "lucy-kushinada": {
        "voice": "en-US-AnaNeural",
        "pitch_hz": -3,
        "rate_offset": -4,
        "name": "Lucy Kushinada",
        "persona": "Netrunner Prodigy / Intimate Whisper",
        "quote": "I don't look back anymore. Keep your head down, David.",
    },
    "david-martinez": {
        "voice": "en-US-BrianNeural",
        "pitch_hz": 4,
        "rate_offset": 6,
        "name": "David Martinez",
        "persona": "Santo Domingo Solo / Streetkid Energetic",
        "quote": "I'm built different, choomba. Fast as light.",
    },
    "v-male": {
        "voice": "en-US-GuyNeural",
        "pitch_hz": -6,
        "rate_offset": 3,
        "name": "V (Male)",
        "persona": "Mercenary Solo / Gritty Raspy Baritone",
        "quote": "Night City always collects its debt. Let's make this quick.",
    },
    "v-female": {
        "voice": "en-US-AriaNeural",
        "pitch_hz": -3,
        "rate_offset": 4,
        "name": "V (Female)",
        "persona": "Tactical Solo / Crisp Dynamic Sharp",
        "quote": "Eyes sharp, weapons hot. Let's see what they got.",
    },
}

# Used for a voice id with no character profile: the neutral voice with no prosody offset.
DEFAULT_EDGE_VOICE = "en-US-BrianNeural"


@dataclass(frozen=True)
class SpeechAudio:
    """One sentence of speech. `word_times` are word start times in seconds, when the engine reports them."""

    data: bytes
    mime: str  # "audio/mpeg" or "audio/wav"
    word_times: list[float] | None = None


class TtsError(RuntimeError):
    """A backend could not produce audio for this request."""


class TtsBackend(Protocol):
    name: str

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        """Raises TtsError when no audio comes back."""
        ...


class EdgeTtsBackend:
    """Microsoft's neural voices through edge-tts. Needs the network; the chain falls back to pyttsx3 when it is unreachable."""

    name = "edge"

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        import edge_tts

        clean_text = text.strip()
        if not clean_text:
            raise TtsError("edge: nothing to say")

        # Unknown voice ids get the default voice with neutral prosody.
        profile = CHARACTER_AUDIO_PROFILES.get(voice_id)
        edge_voice = profile["voice"] if profile is not None else DEFAULT_EDGE_VOICE
        base_rate_offset = profile["rate_offset"] if profile is not None else 0
        cadence_offset = int(round((cadence - 1.0) * 100))
        rate = f"{base_rate_offset + cadence_offset:+d}%"
        # Edge-TTS pitch must match ^[+-]\d+Hz$
        pitch = f"{profile['pitch_hz'] if profile is not None else 0:+d}Hz"

        communicate = edge_tts.Communicate(clean_text, edge_voice, rate=rate, pitch=pitch, boundary="WordBoundary")
        audio_chunks: list[bytes] = []
        word_times: list[float] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts reports offsets in 100-nanosecond ticks.
                word_times.append(chunk["offset"] / 1e7)
        return _require_audio("edge", SpeechAudio(b"".join(audio_chunks), "audio/mpeg", word_times))


class SapiVoice(Protocol):
    id: str
    name: str


# Whole words only: "female" contains "male", so a substring test would give a male request the female voice.
_FEMALE_VOICE_ID = re.compile(r"\bfemale\b|lucy")
_FEMALE_VOICE_NAME = re.compile(r"\bfemale\b|\bzira\b|\bhazel\b")
_MALE_VOICE_NAME = re.compile(r"\bmale\b|\bdavid\b")


def is_female_voice_id(voice_id: str) -> bool:
    return _FEMALE_VOICE_ID.search(voice_id) is not None


def pick_voice(voices: Sequence[SapiVoice], female: bool) -> str | None:
    """The id of the first installed SAPI voice of the wanted gender, or None."""
    pattern = _FEMALE_VOICE_NAME if female else _MALE_VOICE_NAME
    for voice in voices:
        if pattern.search(voice.name.lower()) is not None:
            return voice.id
    return None


class Pyttsx3Backend:
    """Windows SAPI5 through pyttsx3, offline. Speaks to a WAV file in a temp directory and returns its bytes."""

    name = "pyttsx3"

    def __init__(self) -> None:
        # pyttsx3 and SAPI are not thread-safe: creating the engine and every call on it run on this one thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyttsx3")

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        clean_text = text.strip()
        if not clean_text:
            raise TtsError("pyttsx3: nothing to say")
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(self._executor, self._speak_to_wav, clean_text, voice_id, cadence)
        return _require_audio("pyttsx3", SpeechAudio(data, "audio/wav"))

    def _speak_to_wav(self, text: str, voice_id: str, cadence: float) -> bytes:
        import pyttsx3

        # pyttsx3.init hands back a live engine rather than a new one, and a reused engine stalls on its second runAndWait
        # (pyttsx3 2.99, SAPI5). The engine is local to this call, so it is released before the next utterance.
        engine = pyttsx3.init()
        engine.setProperty("rate", int(175 * cadence))
        voice = pick_voice(engine.getProperty("voices"), is_female_voice_id(voice_id))
        if voice is not None:
            engine.setProperty("voice", voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                with open(temp_path, "rb") as handle:
                    return handle.read()
            raise TtsError("pyttsx3: the engine wrote no audio")
        finally:
            _remove_temp_file(temp_path)


def _remove_temp_file(path: str) -> None:
    """A temp WAV that cannot be deleted is left behind; the warning names it so it can be cleared by hand."""
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError as error:
        logger.warning("could not delete temp audio file %s: %s", path, error)


def _require_audio(name: str, audio: SpeechAudio) -> SpeechAudio:
    if len(audio.data) <= MIN_AUDIO_BYTES:
        raise TtsError(f"{name}: no audio returned")
    return audio


class ChainedTts:
    """Tries each backend in order. `status` says how the last call ended: "ok", "fallback:<name>" or "unavailable"."""

    def __init__(self, backends: Sequence[TtsBackend]) -> None:
        if not backends:
            raise ValueError("ChainedTts needs at least one backend")
        self.backends = list(backends)
        self.status = "unavailable"

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio | None:
        if not text.strip():
            self.status = "unavailable"
            return None
        for index, backend in enumerate(self.backends):
            try:
                audio = await backend.synthesize(text, voice_id, cadence)
            except Exception as error:  # noqa: BLE001  # a failed backend is logged, then the next one is tried
                logger.warning("TTS backend %s failed (%s): %s", backend.name, type(error).__name__, error)
                continue
            self.status = "ok" if index == 0 else f"fallback:{backend.name}"
            return audio
        self.status = "unavailable"
        logger.warning("every TTS backend failed; this sentence is spoken without audio")
        return None


_BACKEND_FACTORIES: dict[str, Callable[[], TtsBackend]] = {
    "edge": EdgeTtsBackend,
    "pyttsx3": Pyttsx3Backend,
}
# The names a setting may choose from; settings validates against this, so the two cannot drift apart.
TTS_BACKEND_NAMES: tuple[str, ...] = tuple(_BACKEND_FACTORIES)


def build_tts(primary: str = "edge") -> ChainedTts:
    """The primary backend first, then the other one."""
    if primary not in _BACKEND_FACTORIES:
        raise ValueError(f"unknown TTS backend {primary!r}; expected one of {sorted(_BACKEND_FACTORIES)}")
    order = [primary, *(name for name in _BACKEND_FACTORIES if name != primary)]
    return ChainedTts([_BACKEND_FACTORIES[name]() for name in order])
