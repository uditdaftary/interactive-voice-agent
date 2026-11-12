"""Speech-to-text backends behind one protocol.

A backend turns 16-bit mono PCM into a `Transcript`. Its words are judged against the audio that produced them
(`speech_judge`), so a model's guess at a breath, a fan or a dead input never reaches the screen or the model.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from voice_agent.stt.speech_judge import SAMPLE_RATE, acoustic, evidence, judge

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

STT_UNAVAILABLE = "stt-unavailable"
QUIET = "quiet"


@dataclass(frozen=True)
class Transcript:
    """The words heard, or "" with the reason they were withheld (`speech_judge` reason, or `QUIET`, `STT_UNAVAILABLE`)."""

    text: str
    rejected_reason: str | None = None


class SttBackend(Protocol):
    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        """Synchronous: callers run it in a worker thread."""
        ...


class WhisperBackend:
    """faster-whisper on the CPU. The model loads on the first utterance that reaches it, and a failed load is retried on the next."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str = "",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self._model: WhisperModel | None = None
        # Guards the load only: transcribe runs on worker threads, and two first utterances must not load the model twice.
        self._load_lock = threading.Lock()

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        if not pcm16:
            return Transcript("", QUIET)
        audio = np.frombuffer(pcm16, dtype=np.int16)
        if sample_rate != SAMPLE_RATE:
            audio = _resample_to_16k(audio, sample_rate)

        # The judge reads the int16 audio before gain normalisation: lifting a dead input to 0.75 peak would hide that it was dead.
        proof = evidence(audio)
        reason = acoustic(proof)
        if reason is not None:
            logger.info("Speech rejected before decoding: %s", reason)
            return Transcript("", reason)

        model = self._load_model()
        if model is None:
            return Transcript("", STT_UNAVAILABLE)

        text = self._decode(model, _normalised(audio))
        reason = judge(text, proof)
        if reason is not None:
            logger.info("Whisper output rejected: %s", reason)
            return Transcript("", reason)
        return Transcript(text, None)

    def _load_model(self) -> WhisperModel | None:
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel

                    logger.info("Loading faster-whisper %s on %s", self.model_size, self.device)
                    self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
                except Exception as error:  # noqa: BLE001  # a missing model must not end the session; the next utterance retries
                    logger.warning("faster-whisper %s did not load (%s); transcripts are unavailable until it loads", self.model_size, error)
            return self._model

    def _decode(self, model: WhisperModel, samples: np.ndarray) -> str:
        prompt = self.initial_prompt or None
        # Silero VAD can prune a short utterance to nothing, so clips under 1.5 s skip it, and a pass that heard nothing retries without it.
        passes = (True, False) if samples.size / float(SAMPLE_RATE) >= 1.5 else (False,)
        for vad_filter in passes:
            try:
                segments, _info = model.transcribe(
                    samples,
                    beam_size=3,
                    language="en",
                    vad_filter=vad_filter,
                    initial_prompt=prompt,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except Exception as error:  # noqa: BLE001  # one failed pass is logged and the other is still tried
                logger.warning("faster-whisper transcription error (vad_filter=%s): %s", vad_filter, error)
                continue
            if text:
                return text
        return ""


def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 at `sample_rate` to int16 at 16 kHz. Polyphase resampling; linear interpolation only if scipy is unusable."""
    try:
        from scipy import signal

        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        resampled = signal.resample_poly(audio.astype(np.float64), SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    except Exception as error:  # noqa: BLE001  # logged; the interpolation below is the explicit fallback
        logger.warning("scipy resample_poly failed (%s); using linear interpolation", error)
        target_len = int(round(audio.size * (SAMPLE_RATE / sample_rate)))
        resampled = np.interp(np.linspace(0, audio.size - 1, num=target_len), np.arange(audio.size), audio.astype(np.float64))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def _normalised(audio: np.ndarray) -> np.ndarray:
    """Float samples in [-1, 1], lifted to a 0.75 peak (at most 12x) when quiet, so a soft voice still decodes."""
    samples = audio.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if 0.0 < peak < 0.6:
        samples = samples * min(12.0, 0.75 / peak)
    return samples
