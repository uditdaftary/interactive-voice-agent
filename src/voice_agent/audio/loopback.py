"""Virtual Audio Device and Loopback Generator for Headless CI and Automated Testing.

Provides deterministic synthetic PCM audio generation (24kHz and 16kHz) and
bidirectional loopback capabilities without requiring physical sound cards or
PortAudio drivers.
"""

import threading
import time
from typing import Any, Callable, Generator, List, Optional, Union
import numpy as np


class VirtualAudioDevice:
    """Headless audio loopback generator and mock audio hardware device.

    Generates deterministic synthetic PCM frames (24kHz, 16kHz, etc.) including
    silence, sinusoids, speech-like bursts, and loopback streams without requiring
    physical sound cards or PortAudio / sounddevice initialization.
    Provides mock microphone streaming and mock speaker sinks that integrate
    cleanly with AudioRingBuffer and VoiceActivityDetector.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        chunk_size: int = 720,
        signal_mode: str = "sine",
        sine_freq: float = 440.0,
        amplitude: float = 0.5,
    ):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.signal_mode = signal_mode.lower()
        self.sine_freq = float(sine_freq)
        self.amplitude = float(np.clip(amplitude, 0.0, 1.0))
        self.is_active = False

        # Phase tracking for click-free continuous sinusoids across chunks
        self._phase = 0.0

        # Speech burst simulation parameters
        self._burst_active_ms = 300.0
        self._burst_silence_ms = 500.0
        self._burst_elapsed_ms = 0.0

        # Speaker sink and loopback storage
        self._captured_audio: List[np.ndarray] = []
        self._loopback_fifo: np.ndarray = np.empty(0, dtype=np.int16)

        # Custom signal source
        self._custom_buffer: Optional[np.ndarray] = None
        self._custom_pos: int = 0
        self._custom_loop: bool = True
        self._custom_generator: Optional[Callable[[int], np.ndarray]] = None

        self._lock = threading.Lock()

    def start(self) -> None:
        """Activate the virtual audio device."""
        with self._lock:
            self.is_active = True

    def stop(self) -> None:
        """Deactivate the virtual audio device."""
        with self._lock:
            self.is_active = False

    def reset(self) -> None:
        """Reset internal phase, burst counters, captured audio, and loopback FIFO."""
        with self._lock:
            self._phase = 0.0
            self._burst_elapsed_ms = 0.0
            self._captured_audio.clear()
            self._loopback_fifo = np.empty(0, dtype=np.int16)
            self._custom_pos = 0

    def __enter__(self) -> "VirtualAudioDevice":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def set_signal_mode(self, mode: str, **kwargs) -> None:
        """Configure synthetic signal generation mode and parameters.

        Supported modes:
        - "silence": Generates all-zero PCM frames.
        - "sine": Generates continuous sine wave without phase discontinuities.
        - "speech_burst": Alternates between active harmonic voiced speech and silence.
        - "loopback": Streams audio previously written to the speaker sink.
        - "custom": Streams from a pre-loaded array or callable generator.
        """
        valid_modes = {"silence", "sine", "speech_burst", "burst", "loopback", "custom"}
        m = mode.lower()
        if m == "burst":
            m = "speech_burst"
        if m not in valid_modes:
            raise ValueError(f"Unknown signal mode '{mode}'. Choose from: {valid_modes}")

        with self._lock:
            self.signal_mode = m
            if "freq_hz" in kwargs:
                self.sine_freq = float(kwargs["freq_hz"])
            if "sine_freq" in kwargs:
                self.sine_freq = float(kwargs["sine_freq"])
            if "amplitude" in kwargs:
                self.amplitude = float(np.clip(kwargs["amplitude"], 0.0, 1.0))
            if "burst_active_ms" in kwargs:
                self._burst_active_ms = float(kwargs["burst_active_ms"])
            if "burst_silence_ms" in kwargs:
                self._burst_silence_ms = float(kwargs["burst_silence_ms"])
            if "custom_buffer" in kwargs:
                buf = kwargs["custom_buffer"]
                if isinstance(buf, np.ndarray):
                    self._custom_buffer = buf
                else:
                    self._custom_buffer = np.asarray(buf)
                self._custom_pos = 0
            if "custom_loop" in kwargs:
                self._custom_loop = bool(kwargs["custom_loop"])
            if "custom_generator" in kwargs:
                self._custom_generator = kwargs["custom_generator"]

    def generate_chunk(self, samples: Optional[int] = None, dtype: Any = np.int16) -> np.ndarray:
        """Generate synthetic audio chunk based on current signal mode.

        Maintains phase continuity for sinusoids and speech bursts.
        """
        n_samples = samples if samples is not None else self.chunk_size
        if n_samples <= 0:
            return np.zeros(0, dtype=dtype)

        with self._lock:
            mode = self.signal_mode
            sr = self.sample_rate

            if mode == "silence":
                raw_float = np.zeros(n_samples, dtype=np.float32)

            elif mode == "sine":
                t = np.arange(n_samples, dtype=np.float64) / sr
                phase_inc = 2.0 * np.pi * self.sine_freq * t + self._phase
                raw_float = (self.amplitude * np.sin(phase_inc)).astype(np.float32)
                # Advance phase monotonically and keep in [0, 2pi)
                delta_phase = 2.0 * np.pi * self.sine_freq * (n_samples / sr)
                self._phase = float((self._phase + delta_phase) % (2.0 * np.pi))

            elif mode == "speech_burst":
                # Speech burst: multi-harmonic voiced signal (fundamental + formants + subtle breath)
                chunk_dur_ms = (n_samples / sr) * 1000.0
                period_ms = self._burst_active_ms + self._burst_silence_ms
                cycle_pos_ms = self._burst_elapsed_ms % period_ms

                is_active = cycle_pos_ms < self._burst_active_ms
                self._burst_elapsed_ms += chunk_dur_ms

                if is_active:
                    t = np.arange(n_samples, dtype=np.float64) / sr
                    # Formants at 150Hz (F0), 750Hz (F1), 1800Hz (F2), 2800Hz (F3)
                    f0 = 150.0
                    f1 = 750.0
                    f2 = 1800.0
                    f3 = 2800.0
                    h0 = 0.45 * np.sin(2.0 * np.pi * f0 * t + self._phase)
                    h1 = 0.25 * np.sin(2.0 * np.pi * f1 * t + (self._phase * 1.5))
                    h2 = 0.15 * np.sin(2.0 * np.pi * f2 * t + (self._phase * 2.1))
                    h3 = 0.10 * np.sin(2.0 * np.pi * f3 * t + (self._phase * 3.3))
                    # Deterministic pseudo-random breath noise
                    noise = 0.05 * np.sin(2.0 * np.pi * 3731.0 * t)
                    sig = (h0 + h1 + h2 + h3 + noise) * self.amplitude
                    raw_float = sig.astype(np.float32)
                    self._phase = float((self._phase + 2.0 * np.pi * f0 * (n_samples / sr)) % (2.0 * np.pi))
                else:
                    raw_float = np.zeros(n_samples, dtype=np.float32)

            elif mode == "loopback":
                avail = len(self._loopback_fifo)
                if avail >= n_samples:
                    raw_int = self._loopback_fifo[:n_samples]
                    self._loopback_fifo = self._loopback_fifo[n_samples:]
                    raw_float = raw_int.astype(np.float32) / 32768.0
                else:
                    raw_float = np.zeros(n_samples, dtype=np.float32)
                    if avail > 0:
                        raw_float[:avail] = self._loopback_fifo.astype(np.float32) / 32768.0
                        self._loopback_fifo = np.empty(0, dtype=np.int16)

            elif mode == "custom":
                if self._custom_generator is not None:
                    gen_data = self._custom_generator(n_samples)
                    raw_float = self._to_float32(gen_data, n_samples)
                elif self._custom_buffer is not None and len(self._custom_buffer) > 0:
                    buf_len = len(self._custom_buffer)
                    rem = n_samples
                    parts = []
                    while rem > 0:
                        take = min(rem, buf_len - self._custom_pos)
                        parts.append(self._custom_buffer[self._custom_pos : self._custom_pos + take])
                        self._custom_pos += take
                        rem -= take
                        if self._custom_pos >= buf_len:
                            if self._custom_loop:
                                self._custom_pos = 0
                            else:
                                if rem > 0:
                                    parts.append(np.zeros(rem, dtype=self._custom_buffer.dtype))
                                break
                    raw_float = self._to_float32(np.concatenate(parts), n_samples)
                else:
                    raw_float = np.zeros(n_samples, dtype=np.float32)

            else:
                raw_float = np.zeros(n_samples, dtype=np.float32)

        # Output conversion to requested dtype and channels
        return self._format_output(raw_float, dtype, self.channels)

    def read_mic(self, samples: Optional[int] = None, dtype: Any = np.int16) -> np.ndarray:
        """Mock microphone read interface returning synthetic PCM audio."""
        if not self.is_active:
            self.start()
        return self.generate_chunk(samples=samples, dtype=dtype)

    def stream_mic(
        self, chunk_size: Optional[int] = None, max_chunks: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """Generator yielding simulated microphone PCM audio frames."""
        c_size = chunk_size if chunk_size is not None else self.chunk_size
        produced = 0
        while max_chunks is None or produced < max_chunks:
            yield self.read_mic(samples=c_size)
            produced += 1

    def write_speaker(
        self, data: Union[np.ndarray, bytes, bytearray, memoryview], generation: Optional[int] = None
    ) -> int:
        """Mock speaker sink capturing output audio and populating loopback FIFO.

        Returns the number of audio samples written.
        """
        if data is None:
            return 0

        # Normalize incoming data to 1D int16 array
        if isinstance(data, (bytes, bytearray)):
            nbytes = len(data)
            if nbytes % 2 != 0:
                data = data[: nbytes - 1]
            if len(data) == 0:
                return 0
            arr = np.frombuffer(data, dtype=np.int16)
        elif isinstance(data, memoryview):
            arr = np.frombuffer(data, dtype=np.int16)
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)

        if arr.size == 0:
            return 0

        # Multi-channel downmix
        if arr.ndim > 1:
            if arr.shape[0] < arr.shape[1]:
                arr = np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=-1)
        arr = arr.flatten()

        # Scale float to int16 or clip int
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = np.clip(np.round(arr * 32767.0), -32768.0, 32767.0).astype(np.int16)
        elif arr.dtype != np.int16:
            arr = np.clip(arr, -32768, 32767).astype(np.int16)
        elif not arr.flags.contiguous:
            arr = np.ascontiguousarray(arr)

        with self._lock:
            self._captured_audio.append(arr.copy())
            self._loopback_fifo = np.concatenate([self._loopback_fifo, arr])

        return len(arr)

    def get_captured_audio(self) -> np.ndarray:
        """Retrieve all audio captured by the speaker sink as a single 1D int16 array."""
        with self._lock:
            if not self._captured_audio:
                return np.zeros(0, dtype=np.int16)
            return np.concatenate(self._captured_audio)

    def clear_captured_audio(self) -> None:
        """Clear all captured speaker audio and reset the loopback FIFO."""
        with self._lock:
            self._captured_audio.clear()
            self._loopback_fifo = np.empty(0, dtype=np.int16)

    @property
    def captured_samples(self) -> int:
        """Total number of samples captured in speaker sink history."""
        with self._lock:
            return sum(len(c) for c in self._captured_audio)

    # -------------------------------------------------------------------------
    # Deterministic Static Signal Generators
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_silence(duration_ms: float, sample_rate: int = 24000, dtype: Any = np.int16) -> np.ndarray:
        """Generate a deterministic frame of pure silence."""
        n_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        return np.zeros(n_samples, dtype=dtype)

    @staticmethod
    def generate_tone(
        freq_hz: float = 440.0,
        duration_ms: float = 100.0,
        sample_rate: int = 24000,
        amplitude: float = 0.5,
        dtype: Any = np.int16,
    ) -> np.ndarray:
        """Generate a deterministic pure sinusoid of specified frequency and duration."""
        n_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        raw = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        if dtype == np.int16:
            return np.clip(np.round(raw * 32767.0), -32768.0, 32767.0).astype(np.int16)
        return raw.astype(np.float32)

    @staticmethod
    def generate_sine(
        duration_ms: float,
        freq_hz: float = 440.0,
        sample_rate: int = 24000,
        amplitude: float = 0.5,
        dtype: Any = np.int16,
    ) -> np.ndarray:
        """Alias for generate_tone with duration_ms as first parameter."""
        return VirtualAudioDevice.generate_tone(
            freq_hz=freq_hz,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            amplitude=amplitude,
            dtype=dtype,
        )

    @staticmethod
    def generate_speech_burst(
        duration_ms: float = 200.0,
        sample_rate: int = 24000,
        amplitude: float = 0.5,
        dtype: Any = np.int16,
    ) -> np.ndarray:
        """Generate a deterministic multi-harmonic voiced burst simulating human speech acoustics."""
        n_samples = int(round(sample_rate * (duration_ms / 1000.0)))
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        h0 = 0.45 * np.sin(2.0 * np.pi * 150.0 * t)
        h1 = 0.25 * np.sin(2.0 * np.pi * 750.0 * t)
        h2 = 0.15 * np.sin(2.0 * np.pi * 1800.0 * t)
        h3 = 0.10 * np.sin(2.0 * np.pi * 2800.0 * t)
        noise = 0.05 * np.sin(2.0 * np.pi * 3731.0 * t)
        raw = (h0 + h1 + h2 + h3 + noise) * amplitude
        if dtype == np.int16:
            return np.clip(np.round(raw * 32767.0), -32768.0, 32767.0).astype(np.int16)
        return raw.astype(np.float32)

    @staticmethod
    def generate_pcm_frame(
        num_samples: int,
        mode: str = "sine",
        sample_rate: int = 24000,
        dtype: Any = np.int16,
        **kwargs,
    ) -> np.ndarray:
        """Generate a one-shot deterministic PCM frame."""
        dev = VirtualAudioDevice(sample_rate=sample_rate, chunk_size=num_samples, signal_mode=mode, **kwargs)
        dev.start()
        return dev.generate_chunk(samples=num_samples, dtype=dtype)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_float32(data: Any, target_len: int) -> np.ndarray:
        """Convert input data to 1D float32 array in [-1.0, 1.0]."""
        arr = np.asarray(data)
        if arr.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=-1)
        arr = arr.flatten()
        if np.issubdtype(arr.dtype, np.floating):
            out = np.clip(arr.astype(np.float32), -1.0, 1.0)
        else:
            out = np.clip(arr.astype(np.float32) / 32768.0, -1.0, 1.0)
        if len(out) < target_len:
            pad = np.zeros(target_len - len(out), dtype=np.float32)
            out = np.concatenate([out, pad])
        elif len(out) > target_len:
            out = out[:target_len]
        return out

    @staticmethod
    def _format_output(raw_float: np.ndarray, dtype: Any, channels: int) -> np.ndarray:
        """Format 1D float32 audio to requested dtype and channel dimensions."""
        if dtype == np.int16:
            out = np.clip(np.round(raw_float * 32767.0), -32768.0, 32767.0).astype(np.int16)
        elif dtype == np.float32:
            out = raw_float.astype(np.float32)
        else:
            out = raw_float.astype(dtype)

        if channels > 1:
            out = np.column_stack([out] * channels)
        return out
