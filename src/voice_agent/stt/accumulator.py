from collections import deque
import time
from typing import Deque, List, Optional
import numpy as np

# A single turn is cut here even if the user is still talking, and the quiet-mic session buffer
# never holds more than this much audio, so a stuck-open mic cannot grow memory without bound.
MAX_UTTERANCE_S = 30.0
# ~1000 ms of audio kept ahead of speech onset so initial plosives are not clipped.
PRE_ROLL_CHUNKS = 25


class SpeechTurnAccumulator:
    """Buffers real-time microphone PCM chunks, identifies active speech turns
    via energy thresholding, and detects conversational pauses for transcription.

    While disarmed (the agent is narrating or replying, not listening) it only keeps the rolling
    pre-roll, so nothing else accumulates. `arm()` starts a listening turn. A new instance is armed,
    so standalone use needs no setup.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.010,
        silence_timeout_ms: int = 1400,
        min_speech_duration_ms: int = 150,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.is_speaking: bool = False
        self._armed: bool = True
        self._max_utterance_bytes = int(MAX_UTTERANCE_S * sample_rate) * 2
        self._speech_buffer: List[bytes] = []
        self._speech_bytes: int = 0
        self._pre_roll_buffer: List[bytes] = []
        self._session_buffer: Deque[bytes] = deque()
        self._session_bytes: int = 0
        self._speech_duration_ms: float = 0.0
        self._last_speech_time: float = 0.0
        self._speech_start_time: float = 0.0

    def arm(self) -> None:
        """Start capturing a listening turn. Anything already buffered is kept."""
        self._armed = True

    def disarm(self) -> None:
        """Stop capturing: drop the half-built turn and the session audio, keep the pre-roll."""
        if self._armed:
            self._armed = False
            self.reset(preserve_pre_roll=True)

    def _keep_pre_roll(self, pcm_chunk: bytes) -> None:
        self._pre_roll_buffer.append(pcm_chunk)
        if len(self._pre_roll_buffer) > PRE_ROLL_CHUNKS:
            self._pre_roll_buffer.pop(0)

    def _append_to_session(self, pcm_chunk: bytes) -> None:
        self._session_buffer.append(pcm_chunk)
        self._session_bytes += len(pcm_chunk)
        while self._session_bytes > self._max_utterance_bytes and len(self._session_buffer) > 1:
            self._session_bytes -= len(self._session_buffer.popleft())

    def _append_to_speech(self, pcm_chunk: bytes) -> None:
        self._speech_buffer.append(pcm_chunk)
        self._speech_bytes += len(pcm_chunk)

    def _take_utterance(self) -> bytes:
        utterance = b"".join(self._speech_buffer)
        self.reset()
        return utterance

    def feed(self, pcm_chunk: bytes, *, end_on_silence: bool = True) -> Optional[bytes]:
        """Feed a 16-bit PCM chunk. Returns full utterance bytes when user finishes speaking.

        With `end_on_silence=False` a pause never ends the turn: a client that records until it says it is done
        (push-to-talk) owns the end, so a breath in the middle of a question must not cut it in half. The utterance
        cap still applies.
        """
        if not pcm_chunk or len(pcm_chunk) < 2:
            return None

        if not self._armed:
            self._keep_pre_roll(pcm_chunk)
            return None

        # Truncate any partial sample (if odd number of bytes)
        valid_len = len(pcm_chunk) - (len(pcm_chunk) % 2)
        pcm16 = np.frombuffer(pcm_chunk[:valid_len], dtype=np.int16)
        if len(pcm16) == 0:
            return None

        # Track all incoming audio chunks during active listening session for zero-loss manual flush
        self._append_to_session(pcm_chunk)

        float_samples = pcm16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(float_samples ** 2)))
        chunk_duration_ms = (len(pcm16) / self.sample_rate) * 1000.0
        now = time.monotonic()

        # Adaptive threshold with hysteresis: lower threshold for ongoing speech to preserve quiet phonemes
        active_threshold = self.energy_threshold * 0.5 if self.is_speaking else self.energy_threshold

        if rms >= active_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self._speech_buffer = list(self._pre_roll_buffer)
                self._speech_bytes = sum(len(chunk) for chunk in self._speech_buffer)
                self._pre_roll_buffer = []
                self._speech_duration_ms = 0.0
                self._speech_start_time = now

            self._last_speech_time = now
            self._append_to_speech(pcm_chunk)
            self._speech_duration_ms += chunk_duration_ms
            if self._speech_bytes >= self._max_utterance_bytes:
                return self._take_utterance()
            return None

        # Silence chunk
        if self.is_speaking:
            self._append_to_speech(pcm_chunk)
            elapsed_silence_ms = (now - self._last_speech_time) * 1000.0 if self._last_speech_time > 0 else 0.0

            # Monotonic wall-clock elapsed silence governs active voice turns to prevent
            # sample-rate mismatches (e.g. 48kHz vs 16kHz) from triggering premature cut-offs.
            is_silence_timeout = end_on_silence and (
                elapsed_silence_ms >= self.silence_timeout_ms
                or chunk_duration_ms >= self.silence_timeout_ms
            )

            if is_silence_timeout:
                speech_duration = max(
                    self._speech_duration_ms,
                    (self._last_speech_time - self._speech_start_time) * 1000.0,
                )
                utterance = self._take_utterance()
                if speech_duration >= self.min_speech_duration_ms:
                    return utterance
            elif self._speech_bytes >= self._max_utterance_bytes:
                return self._take_utterance()
        else:
            self._keep_pre_roll(pcm_chunk)

        return None

    def flush(self) -> Optional[bytes]:
        """Manually flush and return accumulated speech (e.g. on client speech_end event)."""
        if self._speech_buffer:
            utterance = b"".join(self._speech_buffer)
            self.reset()
            return utterance
        elif self._session_buffer:
            # Resilient fallback: user spoke into mic but RMS was below threshold (e.g. quiet mic)
            utterance = b"".join(self._session_buffer)
            self.reset()
            return utterance
        elif self._pre_roll_buffer:
            utterance = b"".join(self._pre_roll_buffer)
            self.reset()
            return utterance
        return None

    def reset(self, preserve_pre_roll: bool = False):
        """Reset internal accumulator state."""
        self.is_speaking = False
        self._speech_buffer = []
        self._speech_bytes = 0
        if not preserve_pre_roll:
            self._pre_roll_buffer = []
        self._session_buffer = deque()
        self._session_bytes = 0
        self._speech_duration_ms = 0.0
        self._last_speech_time = 0.0
        self._speech_start_time = 0.0
