"""Text-to-speech backends behind one protocol, and the chain that tries them in order.

A backend turns one sentence into a `SpeechAudio`, or raises `TtsError`. The chain logs every failed backend and returns the
first audio that succeeds; when none does, the caller gets None and the narration moves on without sound for that sentence.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np

from voice_agent.retry import RETRY_COOLDOWN_S

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


POCKET_LICENCE_MESSAGE = "pocket weights need the Hugging Face licence; see README"
# A voice id names a file in the voices folder, so it must be a plain name that cannot step out of that folder.
_SAFE_VOICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_INT16_SCALE = 32767.0


class PocketTtsBackend:
    """Kyutai's Pocket TTS on the CPU, in process (the `pocket` extra). Each voice is cloned from `voices_dir/<voice_id>.wav`.

    The model loads on the first sentence and stays loaded. A refused licence (only the open weights could be loaded, so voices
    cannot be cloned) is kept for the session: a retry would reload those weights for every sentence. Any other load failure
    is retried after RETRY_COOLDOWN_S. pocket-tts is not thread-safe, so every call into it runs on one thread, which also
    means one generation at a time. A voice without a usable file speaks in `default_voice`, with one WARNING per voice id.
    `cadence` is ignored: pocket-tts 4.0.1 has no speed control. Cancelling the awaiting task stops generation after the chunk
    in progress, and the next sentence waits until that generation has ended.
    """

    name = "pocket"

    def __init__(
        self,
        voices_dir: Path = Path("models/voices"),
        default_voice: str = "v-male",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._voices_dir = voices_dir
        self._default_voice = default_voice
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pocket")
        self._model: Any = None
        self._licence_refused = False
        self._failed_at: float | None = None
        self._voice_states: dict[str, Any] = {}
        self._warned_voices: set[str] = set()

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        clean_text = text.strip()
        if not clean_text:
            raise TtsError("pocket: nothing to say")
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(self._executor, self._speak_to_wav, clean_text, voice_id, stop)
        except asyncio.CancelledError:
            # A thread cannot be cancelled; the flag ends its generation after the chunk in progress.
            stop.set()
            raise
        except TtsError:
            raise
        except Exception as error:  # noqa: BLE001  # re-raised as TtsError so the chain logs one clear reason
            raise TtsError(f"pocket: {type(error).__name__}: {error}") from error
        return _require_audio("pocket", SpeechAudio(data, "audio/wav"))

    def _speak_to_wav(self, text: str, voice_id: str, stop: threading.Event) -> bytes:
        model = self._loaded_model()
        state = self._voice_state(model, voice_id)
        chunks: list[np.ndarray] = []
        # The stream is read to its end even after stop is set: pocket-tts joins its generation and decoder threads only when the
        # stream runs out, and a stopped stream ends within one frame. Breaking out early would let the next generation overlap it.
        for chunk in model.generate_audio_stream(state, text, stop=stop):
            if not stop.is_set():
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return _wav_bytes(samples, int(model.sample_rate))

    def _loaded_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._licence_refused:
            raise TtsError(POCKET_LICENCE_MESSAGE)
        if self._failed_at is not None and self._clock() - self._failed_at < RETRY_COOLDOWN_S:
            raise TtsError(f"pocket: the model did not load; the next attempt is after {RETRY_COOLDOWN_S:g} s")
        try:
            self._model = self._load()
        except TtsError:
            if not self._licence_refused:
                self._failed_at = self._clock()
            raise
        self._failed_at = None
        return self._model

    def _load(self) -> Any:
        try:
            from pocket_tts import TTSModel
        except Exception as error:  # noqa: BLE001  # a broken install can fail with any error (a torch DLL raises OSError); it is one backend failing
            logger.warning("pocket-tts did not import (%s: %s)", type(error).__name__, error)
            raise TtsError(f"pocket: pocket-tts did not import ({type(error).__name__}: {error}); install the pocket extra") from error
        try:
            model = TTSModel.load_model()
        except Exception as error:  # noqa: BLE001  # logged, then raised as TtsError so the chain falls back and the load is retried later
            logger.warning("pocket model did not load (%s: %s)", type(error).__name__, error)
            raise TtsError(f"pocket: the model did not load ({type(error).__name__}: {error})") from error
        if not model.has_voice_cloning:
            # pocket-tts falls back to the open weights when the gated cloning weights are refused, and cloning then fails for every voice.
            logger.warning("pocket loaded without voice cloning: the gated cloning weights are not available")
            self._licence_refused = True
            raise TtsError(POCKET_LICENCE_MESSAGE)
        return model

    def _voice_state(self, model: Any, voice_id: str) -> Any:
        cached = self._voice_states.get(voice_id)
        if cached is None:
            path = self._voice_path(voice_id)
            try:
                cached = model.get_state_for_audio_prompt(path)
            except Exception as error:  # noqa: BLE001  # an unreadable voice file is one backend failing; the chain moves on
                raise TtsError(f"pocket: could not clone {path.name}: {type(error).__name__}: {error}") from error
            self._voice_states[voice_id] = cached
        return cached

    def _voice_path(self, voice_id: str) -> Path:
        if _SAFE_VOICE_ID.fullmatch(voice_id):
            path = self._voices_dir / f"{voice_id}.wav"
            if path.is_file():
                return path
        fallback = self._voices_dir / f"{self._default_voice}.wav"
        if voice_id not in self._warned_voices:
            self._warned_voices.add(voice_id)
            logger.warning("no usable Pocket voice file for %r; speaking it in the %s voice", voice_id, self._default_voice)
        if not fallback.is_file():
            raise TtsError(f"pocket: no default voice file at {fallback}")
        return fallback


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Mono 16-bit WAV from float samples in [-1, 1]."""
    pcm = np.clip(np.round(samples * _INT16_SCALE), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


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
    "pocket": PocketTtsBackend,
}
# The names a setting may choose from; settings validates against this, so the two cannot drift apart.
TTS_BACKEND_NAMES: tuple[str, ...] = tuple(_BACKEND_FACTORIES)
# The default fallbacks. Pocket is only ever the primary: a fallback would load its model for every sentence edge-tts drops.
_FALLBACK_ORDER: tuple[str, ...] = ("edge", "pyttsx3")


def build_tts(primary: str = "edge") -> ChainedTts:
    """The primary backend first, then the default fallbacks in their own order."""
    if primary not in _BACKEND_FACTORIES:
        raise ValueError(f"unknown TTS backend {primary!r}; expected one of {sorted(_BACKEND_FACTORIES)}")
    order = [primary, *(name for name in _FALLBACK_ORDER if name != primary)]
    return ChainedTts([_BACKEND_FACTORIES[name]() for name in order])
