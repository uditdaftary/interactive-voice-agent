import time
from typing import List, Optional
import numpy as np


class SpeechTurnAccumulator:
    """Buffers real-time microphone PCM chunks, identifies active speech turns
    via energy thresholding, and detects conversational pauses for transcription.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.012,
        silence_timeout_ms: int = 1400,
        min_speech_duration_ms: int = 150,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.is_speaking: bool = False
        self._speech_buffer: List[bytes] = []
        self._pre_roll_buffer: List[bytes] = []
        self._session_buffer: List[bytes] = []
        self._speech_duration_ms: float = 0.0
        self._silence_duration_ms: float = 0.0
        self._last_speech_time: float = 0.0
        self._speech_start_time: float = 0.0

    def feed(self, pcm_chunk: bytes) -> Optional[bytes]:
        """Feed a 16-bit PCM chunk. Returns full utterance bytes when user finishes speaking."""
        if not pcm_chunk or len(pcm_chunk) < 2:
            return None

        # Truncate any partial sample (if odd number of bytes)
        valid_len = len(pcm_chunk) - (len(pcm_chunk) % 2)
        pcm16 = np.frombuffer(pcm_chunk[:valid_len], dtype=np.int16)
        if len(pcm16) == 0:
            return None

        # Track all incoming audio chunks during active listening session for zero-loss manual flush
        self._session_buffer.append(pcm_chunk)

        float_samples = pcm16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(float_samples ** 2)))
        chunk_duration_ms = (len(pcm16) / self.sample_rate) * 1000.0
        now = time.monotonic()

        # Adaptive threshold with hysteresis: lower threshold for ongoing speech to preserve quiet phonemes
        active_threshold = self.energy_threshold * 0.6 if self.is_speaking else self.energy_threshold

        if rms >= active_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self._speech_buffer = list(self._pre_roll_buffer)
                self._pre_roll_buffer = []
                self._speech_duration_ms = 0.0
                self._silence_duration_ms = 0.0
                self._speech_start_time = now

            self._last_speech_time = now
            self._speech_buffer.append(pcm_chunk)
            self._speech_duration_ms += chunk_duration_ms
            self._silence_duration_ms = 0.0
            return None

        # Silence chunk
        if self.is_speaking:
            self._speech_buffer.append(pcm_chunk)
            self._silence_duration_ms += chunk_duration_ms
            elapsed_silence_ms = (now - self._last_speech_time) * 1000.0 if self._last_speech_time > 0 else 0.0

            # Monotonic wall-clock elapsed silence governs active voice turns to prevent
            # sample-rate mismatches (e.g. 48kHz vs 16kHz) from triggering premature cut-offs.
            # Offline / synthetic test feeds that pass an entire silence segment in a single chunk
            # are also supported via chunk_duration_ms.
            is_silence_timeout = (
                elapsed_silence_ms >= self.silence_timeout_ms
                or chunk_duration_ms >= self.silence_timeout_ms
            )

            if is_silence_timeout:
                speech_duration = max(
                    self._speech_duration_ms,
                    (self._last_speech_time - self._speech_start_time) * 1000.0,
                )
                utterance = b"".join(self._speech_buffer)
                self.reset()
                if speech_duration >= self.min_speech_duration_ms:
                    return utterance
        else:
            # Maintain rolling pre-roll buffer of up to 25 chunks (~1000ms) to capture initial plosives/words
            self._pre_roll_buffer.append(pcm_chunk)
            if len(self._pre_roll_buffer) > 25:
                self._pre_roll_buffer.pop(0)

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

    def reset(self):
        """Reset internal accumulator state."""
        self.is_speaking = False
        self._speech_buffer = []
        self._pre_roll_buffer = []
        self._session_buffer = []
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._last_speech_time = 0.0
        self._speech_start_time = 0.0
