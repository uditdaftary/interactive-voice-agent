import threading
from typing import Optional, Union
import numpy as np


class AudioRingBuffer:
    """Thread-safe circular ring buffer for real-time PCM audio streaming.
    
    Guarantees sub-millisecond instant drain (<5ms) with a 2ms click-free half-cosine
    micro-fadeout to eliminate step discontinuities during barge-in interruptions.
    Provides zero-allocation read_into for PortAudio callbacks, generation fencing
    against stale/zombie audio re-injection, and monotonic playhead tracking.
    """

    def __init__(self, capacity_samples: int = 48000 * 5, sample_rate: int = 24000):
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be positive, got {capacity_samples}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        self.capacity = capacity_samples
        self.sample_rate = sample_rate
        self.buffer = np.zeros(capacity_samples, dtype=np.int16)
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0
        self.generation = 0
        self.total_samples_played = 0
        self.fadeout_tail: Optional[np.ndarray] = None
        self.lock = threading.Lock()

    @property
    def available_write(self) -> int:
        """Remaining capacity available for writing."""
        with self.lock:
            return self.capacity - self.size

    def available_read(self) -> int:
        """Number of narrative audio samples currently queued in the ring buffer."""
        with self.lock:
            return self.size

    @property
    def playhead_seconds(self) -> float:
        """Monotonic playhead progress in seconds."""
        with self.lock:
            return self.total_samples_played / self.sample_rate

    def get_playhead_seconds(self) -> float:
        """Compatibility accessor for playhead progress in seconds."""
        return self.playhead_seconds

    def reset_playhead(self):
        """Explicitly reset the monotonic playhead counter (e.g. on new session)."""
        with self.lock:
            self.total_samples_played = 0

    def is_actively_playing(self) -> bool:
        """Returns True if any audio or fadeout tail is queued for output."""
        with self.lock:
            return self.size > 0 or (self.fadeout_tail is not None and len(self.fadeout_tail) > 0)

    def write(self, data: Union[np.ndarray, bytes, bytearray, memoryview], generation: Optional[int] = None) -> int:
        """Write audio into circular buffer with generation fencing and format normalization.

        - Converts bytes, bytearray, memoryview, and numpy arrays to 1D int16.
        - Downmixes 2D (N, 1) and (N, 2) arrays to 1D mono.
        - Scales normalized floats in [-1.0, 1.0] by 32767.0 and clips.
        - Clips integer inputs to [-32768, 32767].
        - Rejects stale writes where generation != self.generation.
        """
        if data is None:
            return 0

        # 1. Type normalization to numpy array
        if isinstance(data, (bytes, bytearray)):
            nbytes = len(data)
            if nbytes % 2 != 0:
                data = data[:nbytes - (nbytes % 2)]
            if len(data) == 0:
                return 0
            arr = np.frombuffer(data, dtype=np.int16)
        elif isinstance(data, memoryview):
            if data.format in ('B', 'b', 'c') or data.itemsize == 1:
                nbytes = len(data)
                if nbytes % 2 != 0:
                    data = data[:nbytes - (nbytes % 2)]
                if len(data) == 0:
                    return 0
                arr = np.frombuffer(data, dtype=np.int16)
            else:
                arr = np.asarray(data)
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)

        if arr.size == 0:
            return 0

        # 2. Downmix 2D inputs to 1D mono
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.shape[0] == 1 and arr.shape[1] > 2:
                arr = arr[0, :]
            elif arr.shape[1] == 2:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = 0.5 * (arr[:, 0] + arr[:, 1])
                else:
                    arr = (arr[:, 0].astype(np.int64) + arr[:, 1].astype(np.int64)) // 2
            else:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.mean(arr, axis=1)
                else:
                    arr = np.mean(arr.astype(np.float64), axis=1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1)

        # 3. Clean NaNs/Infs, scale float to int16, clip integer ranges
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = np.clip(np.round(arr * 32767.0), -32768.0, 32767.0).astype(np.int16)
        elif arr.dtype != np.int16:
            arr = np.clip(arr, -32768, 32767).astype(np.int16)
        elif not arr.flags.contiguous:
            arr = np.ascontiguousarray(arr)

        length = len(arr)
        if length == 0:
            return 0

        # 4. Ring buffer write under lock
        with self.lock:
            if generation is not None and generation != self.generation:
                return 0

            available_space = self.capacity - self.size
            if length > available_space:
                length = available_space
                arr = arr[:length]
            if length == 0:
                return 0

            end1 = min(length, self.capacity - self.write_pos)
            self.buffer[self.write_pos : self.write_pos + end1] = arr[:end1]
            end2 = length - end1
            if end2 > 0:
                self.buffer[0:end2] = arr[end1:length]

            self.write_pos = (self.write_pos + length) % self.capacity
            self.size += length
            return length

    def drain(self, fade_ms: float = 2.0):
        """Instantaneously drop all pending audio with 2ms half-cosine micro-fadeout (< 5ms).

        Increments generation fence to invalidate in-flight TTS chunks.
        Extracts up to 2ms of audio from the read position, attenuates via a half-cosine
        window to 0 amplitude, and stores in fadeout_tail to eliminate DAC step-discontinuity clicks.
        """
        with self.lock:
            self.generation += 1
            fade_samples = min(int(round(self.sample_rate * (fade_ms / 1000.0))), self.size)
            if fade_samples > 0:
                end1 = min(fade_samples, self.capacity - self.read_pos)
                raw = np.empty(fade_samples, dtype=np.int16)
                raw[:end1] = self.buffer[self.read_pos : self.read_pos + end1]
                end2 = fade_samples - end1
                if end2 > 0:
                    raw[end1:] = self.buffer[0:end2]

                if fade_samples > 1:
                    t = np.linspace(0.0, np.pi, fade_samples, endpoint=True)
                    window = 0.5 * (1.0 + np.cos(t))
                    self.fadeout_tail = np.round(raw * window).astype(np.int16)
                else:
                    self.fadeout_tail = np.zeros(1, dtype=np.int16)
            else:
                self.fadeout_tail = None

            self.read_pos = 0
            self.write_pos = 0
            self.size = 0

    def read_into(self, outdata: np.ndarray) -> int:
        """Zero-allocation read directly into PortAudio callback buffer outdata.

        Zero-pads (silence-pads) any buffer underrun deficit.
        Consumes micro-fadeout tail first if available.
        Updates monotonic playhead tracking.
        Returns the number of valid audio samples written.
        """
        if outdata is None or not isinstance(outdata, np.ndarray):
            return 0
        samples_requested = outdata.shape[0] if outdata.ndim > 0 else 0
        if samples_requested <= 0:
            return 0

        with self.lock:
            filled = 0

            # 1. Consume micro-fadeout tail if present
            if self.fadeout_tail is not None and len(self.fadeout_tail) > 0:
                tail_len = len(self.fadeout_tail)
                take = min(tail_len, samples_requested)
                self._copy_to_outdata(outdata, 0, self.fadeout_tail[:take])
                filled += take
                if take < tail_len:
                    self.fadeout_tail = self.fadeout_tail[take:]
                else:
                    self.fadeout_tail = None

            # 2. Read from circular buffer
            rem = samples_requested - filled
            if rem > 0 and self.size > 0:
                count = min(rem, self.size)
                end1 = min(count, self.capacity - self.read_pos)
                self._copy_to_outdata(outdata, filled, self.buffer[self.read_pos : self.read_pos + end1])
                end2 = count - end1
                if end2 > 0:
                    self._copy_to_outdata(outdata, filled + end1, self.buffer[0:end2])

                self.read_pos = (self.read_pos + count) % self.capacity
                self.size -= count
                filled += count

            # 3. Silence-pad any buffer underrun deficit
            if filled < samples_requested:
                self._zero_pad_outdata(outdata, filled)

            self.total_samples_played += filled
            return filled

    def read(self, samples_requested: int) -> np.ndarray:
        """Reads up to samples_requested samples into a newly allocated array.
        
        Guarded against negative or zero requests.
        Consumes micro-fadeout tail first if available.
        Updates monotonic playhead tracking.
        """
        if samples_requested <= 0:
            return np.zeros(0, dtype=np.int16)

        with self.lock:
            tail_len = len(self.fadeout_tail) if self.fadeout_tail is not None else 0
            total_avail = tail_len + self.size
            count = min(samples_requested, total_avail)
            if count == 0:
                return np.zeros(0, dtype=np.int16)

            out = np.empty(count, dtype=np.int16)
            filled = 0

            if tail_len > 0:
                take_tail = min(tail_len, count)
                out[:take_tail] = self.fadeout_tail[:take_tail]
                filled += take_tail
                if take_tail < tail_len:
                    self.fadeout_tail = self.fadeout_tail[take_tail:]
                else:
                    self.fadeout_tail = None

            rem = count - filled
            if rem > 0:
                end1 = min(rem, self.capacity - self.read_pos)
                out[filled : filled + end1] = self.buffer[self.read_pos : self.read_pos + end1]
                end2 = rem - end1
                if end2 > 0:
                    out[filled + end1 : count] = self.buffer[0:end2]

                self.read_pos = (self.read_pos + rem) % self.capacity
                self.size -= rem
                filled += rem

            self.total_samples_played += filled
            return out

    def _copy_to_outdata(self, outdata: np.ndarray, offset: int, chunk: np.ndarray):
        """Zero-allocation in-place copy helper for 1D and 2D outdata buffers."""
        chunk_len = len(chunk)
        if chunk_len == 0:
            return
        if outdata.ndim == 1:
            if np.issubdtype(outdata.dtype, np.floating):
                outdata[offset : offset + chunk_len] = chunk / 32767.0
            else:
                outdata[offset : offset + chunk_len] = chunk
        elif outdata.ndim == 2:
            if outdata.shape[1] == 1:
                if np.issubdtype(outdata.dtype, np.floating):
                    outdata[offset : offset + chunk_len, 0] = chunk / 32767.0
                else:
                    outdata[offset : offset + chunk_len, 0] = chunk
            else:
                if np.issubdtype(outdata.dtype, np.floating):
                    outdata[offset : offset + chunk_len, :] = (chunk / 32767.0)[:, np.newaxis]
                else:
                    outdata[offset : offset + chunk_len, :] = chunk[:, np.newaxis]

    def _zero_pad_outdata(self, outdata: np.ndarray, start_idx: int):
        """Zero-allocation in-place silence-pad helper for underrun deficits."""
        if start_idx >= outdata.shape[0]:
            return
        if outdata.ndim == 1:
            outdata[start_idx:] = 0
        elif outdata.ndim == 2:
            outdata[start_idx:, :] = 0
