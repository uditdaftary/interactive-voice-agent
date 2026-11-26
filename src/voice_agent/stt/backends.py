"""Speech-to-text backends behind one protocol.

A backend turns 16-bit mono PCM into a `Transcript`. Its words are judged against the audio that produced them
(`speech_judge`), so a model's guess at a breath, a fan or a dead input never reaches the screen or the model.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import math
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from voice_agent.retry import RETRY_COOLDOWN_S
from voice_agent.stt.speech_judge import SAMPLE_RATE, acoustic, evidence, judge

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from voice_agent.settings import Settings

logger = logging.getLogger(__name__)

STT_UNAVAILABLE = "stt-unavailable"
STT_ERROR = "stt-error"
QUIET = "quiet"
# The names a setting may choose from; settings validates against this, so the two cannot drift apart.
STT_BACKEND_NAMES: tuple[str, ...] = ("whisper", "moonshine")
WORKER_PATH = Path(__file__).resolve().parent / "local_stt_worker.py"
# A reply must arrive within this many seconds; the same bound caps the writes, so a hung worker cannot hold the lock forever.
READ_TIMEOUT_S = 30.0
# Loading the model and its warm-up decode can take far longer than one exchange on a slow disk, so the ready wait has its own bound.
READY_TIMEOUT_S = 120.0
# One second of 16 kHz int16 mono per audio line keeps each JSON line small.
CHUNK_BYTES = SAMPLE_RATE * 2


@dataclass(frozen=True)
class Transcript:
    """The words heard, or "" with the reason they were withheld (`speech_judge` reason, or `QUIET`, `STT_UNAVAILABLE`)."""

    text: str
    rejected_reason: str | None = None


def stt_status(transcript: Transcript) -> str:
    """What a transcript says about the recognizer: "ok", "unavailable" (it could not run) or "rejected" (the audio was not speech)."""
    if transcript.text:
        return "ok"
    return "unavailable" if transcript.rejected_reason in (STT_UNAVAILABLE, STT_ERROR) else "rejected"


class SttBackend(Protocol):
    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        """Synchronous: callers run it in a worker thread."""
        ...


class WhisperBackend:
    """faster-whisper on the CPU. The model is loaded by `load()` (the STT build does this), or by the first transcribe if that
    never ran. A failed load is not retried for LOAD_RETRY_COOLDOWN_S seconds, so a missing model does not stall every turn."""

    LOAD_RETRY_COOLDOWN_S = RETRY_COOLDOWN_S

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self._clock = clock
        self._model: WhisperModel | None = None
        self._failed_at: float | None = None
        # Guards the load only: transcribe runs on worker threads, and two first utterances must not load the model twice.
        self._load_lock = threading.Lock()

    def load(self) -> bool:
        """Loads the model now, if it is not loaded and the retry cooldown allows. Returns whether it is loaded."""
        return self._load_model() is not None

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
            if self._model is not None:
                return self._model
            if self._failed_at is not None and self._clock() - self._failed_at < self.LOAD_RETRY_COOLDOWN_S:
                return None  # inside the cooldown: no attempt, no log
            try:
                from faster_whisper import WhisperModel

                logger.info("Loading faster-whisper %s on %s", self.model_size, self.device)
                self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            except Exception as error:  # noqa: BLE001  # a missing model must not end the session; the load is retried after the cooldown
                self._failed_at = self._clock()
                logger.warning(
                    "faster-whisper %s did not load (%s); transcripts are unavailable, retrying in %d s",
                    self.model_size,
                    error,
                    int(self.LOAD_RETRY_COOLDOWN_S),
                )
                return None
            self._failed_at = None
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


class _WorkerProcess:
    """One running worker. A reader thread turns each stdout line into a queue item and puts None when stdout closes, so a reply can
    be waited for with a timeout (a blocking pipe read cannot). stderr is drained into the log, so a full pipe never stalls the worker."""

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lines: queue.Queue[dict[str, Any] | None] = queue.Queue()
        threading.Thread(target=self._read_replies, name="moonshine-replies", daemon=True).start()
        threading.Thread(target=self._drain_log, name="moonshine-log", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def send(self, message: dict[str, Any]) -> None:
        """Raises OSError once the worker has closed its input: it died, or the watchdog killed it."""
        stdin = self.process.stdin
        if stdin is None:
            raise OSError("the worker has no input pipe")
        stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        stdin.flush()

    def kill(self) -> None:
        """Safe to call again, and from the watchdog thread while a request is being written."""
        self.process.kill()
        self.process.wait()

    def _read_replies(self) -> None:
        stdout = self.process.stdout
        if stdout is not None:
            for raw in stdout:
                try:
                    message = json.loads(raw.decode("utf-8"))
                except ValueError:  # includes a UnicodeDecodeError
                    logger.warning("moonshine worker sent a line that is not JSON: %r", raw[:120])
                    continue
                if isinstance(message, dict):
                    self.lines.put(message)
                else:
                    logger.warning("moonshine worker sent a line that is not an object: %r", raw[:120])
        self.lines.put(None)

    def _drain_log(self) -> None:
        stderr = self.process.stderr
        if stderr is not None:
            for raw in stderr:
                logger.info("moonshine worker: %s", raw.decode("utf-8", "replace").rstrip())


class MoonshineBackend:
    """Moonshine Tiny (sherpa-onnx, the `moonshine` extra), run by local_stt_worker.py in a child process on the CPU.

    The worker starts on the first utterance and is reused. A failure never raises: the worker is killed, the attempt is logged,
    and the next attempt waits RETRY_COOLDOWN_S. Requests are serialised, since the worker holds one recording at a time. The
    worker judges its own words (`speech_judge`), so its final maps straight onto a Transcript.
    """

    def __init__(
        self,
        model_dir: str = "models/stt/moonshine-tiny-en-int8",
        worker_path: Path = WORKER_PATH,
        python: str = sys.executable,
        timeout_s: float = READ_TIMEOUT_S,
        ready_timeout_s: float = READY_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_dir = model_dir
        self._worker_path = worker_path
        self._python = python
        self._timeout_s = timeout_s
        self._ready_timeout_s = ready_timeout_s
        self._clock = clock
        self._worker: _WorkerProcess | None = None
        self._failed_at: float | None = None
        self._request_number = 0
        self._lock = threading.Lock()
        atexit.register(self.close)

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.alive

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        pcm = pcm16[: len(pcm16) - len(pcm16) % 2]  # whole int16 samples only; a stray byte is not a sample
        if not pcm:
            return Transcript("", QUIET)
        audio = np.frombuffer(pcm, dtype="<i2")
        if sample_rate != SAMPLE_RATE:
            audio = _resample_to_16k(audio, sample_rate)
        with self._lock:
            return self._exchange(audio.astype("<i2").tobytes())

    def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.kill()

    def _exchange(self, pcm: bytes) -> Transcript:
        worker = self._ready_worker()
        if worker is None:
            return Transcript("", STT_UNAVAILABLE)
        self._request_number += 1
        request_id = f"m{self._request_number}"
        # The watchdog bounds the writes as well as the reads: a worker that stops reading would otherwise block the write forever,
        # and with it the lock. It records that it fired, so the kill is logged by its cause rather than as a plain exit.
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            worker.kill()

        watchdog = threading.Timer(self._timeout_s, expire)
        watchdog.start()
        try:
            try:
                worker.send({"op": "begin", "id": request_id})
                for start in range(0, len(pcm), CHUNK_BYTES):
                    chunk = pcm[start : start + CHUNK_BYTES]
                    worker.send({"op": "audio", "id": request_id, "pcm": base64.b64encode(chunk).decode("ascii")})
                worker.send({"op": "finish", "id": request_id})
            except OSError as error:
                reason = self._expired_reason() if expired.is_set() else f"the worker stopped reading ({error})"
                self._give_up(worker, reason)
                return Transcript("", STT_UNAVAILABLE)
            return self._await_final(worker, request_id, expired)
        finally:
            watchdog.cancel()

    def _expired_reason(self) -> str:
        return f"the exchange timeout of {self._timeout_s:g} s expired and the watchdog killed the worker"

    def _await_final(self, worker: _WorkerProcess, request_id: str, expired: threading.Event) -> Transcript:
        while True:
            try:
                message = worker.lines.get(timeout=self._timeout_s)
            except queue.Empty:
                # The same bound as the watchdog, which may fire a moment later on the timer thread: either way it is this timeout.
                self._give_up(worker, f"the exchange timeout of {self._timeout_s:g} s expired with no reply")
                return Transcript("", STT_UNAVAILABLE)
            if message is None:
                self._give_up(worker, self._expired_reason() if expired.is_set() else "the worker exited")
                return Transcript("", STT_UNAVAILABLE)
            if message.get("id") != request_id:
                continue  # the ready line, or a reply to an earlier request
            if message.get("type") == "error":
                logger.warning("moonshine worker error: %s", message.get("message"))
                return Transcript("", STT_ERROR)
            if message.get("type") == "final":
                return _worker_transcript(message)

    def _ready_worker(self) -> _WorkerProcess | None:
        worker = self._worker
        if worker is not None:
            if worker.alive:
                return worker
            self._give_up(worker, "the worker exited between utterances")
            return None
        if self._failed_at is not None and self._clock() - self._failed_at < RETRY_COOLDOWN_S:
            return None  # inside the cooldown: no attempt, no log
        return self._start()

    def _start(self) -> _WorkerProcess | None:
        command = [self._python, "-I", str(self._worker_path), "--model", self.model_dir, "--kind", "moonshine"]
        try:
            worker = _WorkerProcess(command)
        except OSError as error:
            self._give_up(None, f"could not start the worker ({error})")
            return None
        try:
            message = worker.lines.get(timeout=self._ready_timeout_s)
        except queue.Empty:
            self._give_up(worker, f"the ready timeout of {self._ready_timeout_s:g} s expired and the worker was killed")
            return None
        if message is not None and message.get("type") == "ready":
            logger.info("moonshine worker ready (%s, loaded in %s ms)", message.get("engine"), message.get("loadMs"))
            self._worker = worker
            return worker
        detail = "it exited before reporting ready" if message is None else str(message.get("message", message))
        self._give_up(worker, f"the worker did not start: {detail}")
        return None

    def _give_up(self, worker: _WorkerProcess | None, reason: str) -> None:
        """Kills the worker, starts the restart cooldown and logs why. The next attempt is RETRY_COOLDOWN_S later."""
        if worker is not None:
            worker.kill()
        self._worker = None
        self._failed_at = self._clock()
        logger.warning("Moonshine unavailable (%s); the next attempt is in %d s", reason, int(RETRY_COOLDOWN_S))


def _worker_transcript(message: dict[str, Any]) -> Transcript:
    """The worker's final as a Transcript. Words come only with speech and judged text; otherwise the worker's reason, or "not-speech"."""
    text = str(message.get("text", "")).strip()
    if text and message.get("speech") is True:
        return Transcript(text, None)
    rejected = message.get("rejected")
    reason = str(rejected) if rejected else "not-speech"
    logger.info("Moonshine output rejected: %s", reason)
    return Transcript("", reason)


def build_stt_backend(settings: Settings) -> SttBackend:
    """The backend the settings choose. Whisper is the default; Moonshine needs the `moonshine` extra and its model files."""
    if settings.stt_backend == "moonshine":
        return MoonshineBackend(model_dir=settings.moonshine_model_dir)
    if settings.stt_backend == "whisper":
        return WhisperBackend(initial_prompt=settings.stt_initial_prompt)
    raise ValueError(f"unknown STT backend {settings.stt_backend!r}; expected one of {STT_BACKEND_NAMES}")
