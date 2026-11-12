"""Hybrid Microphone Input Bridge and Streaming Audio Coordinator.

Supports seamless, thread-safe switching between Browser Web Audio API PCM frames
(fed over WebSocket) and Host Python sounddevice hardware microphone input, with
automatic fallback to VirtualAudioDevice in headless/CI environments.
Integrates 16kHz 30ms VAD frame normalization, sub-50ms barge-in interruption,
and playback playhead tracking.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from voice_agent.audio.loopback import VirtualAudioDevice
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.vad.energy_vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


@dataclass
class VADTelemetry:
    """Telemetry data emitted for each 30ms audio frame."""

    speech_active: bool
    dbfs: float
    noise_floor_dbfs: float
    threshold_dbfs: float
    is_turn_complete: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "vad_event",
            "speech_active": self.speech_active,
            "dbfs": round(float(self.dbfs), 2),
            "noise_floor_dbfs": round(float(self.noise_floor_dbfs), 2),
            "threshold_dbfs": round(float(self.threshold_dbfs), 2),
            "is_turn_complete": self.is_turn_complete,
            "timestamp": self.timestamp,
        }


@dataclass
class BargeInEvent:
    """Event dispatched instantaneously upon speech detection or manual barge-in."""

    drain_latency_ms: float
    generation: int
    flushed_samples: int
    playhead_seconds: float
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "barge_in",
            "latency_ms": round(float(self.drain_latency_ms), 4),
            "generation": int(self.generation),
            "flushed_samples": int(self.flushed_samples),
            "playhead_seconds": round(float(self.playhead_seconds), 3),
            "timestamp": self.timestamp,
        }


class HybridMicrophoneBridge:
    """Thread-safe hybrid microphone bridge coordinating multi-source audio input.

    - Supports sources:
      - "browser": Web Audio API PCM frames received from client via WebSocket.
      - "host": Native Python sounddevice.InputStream microphone capture.
      - "virtual" (or "mock"): Headless synthetic audio loopback via VirtualAudioDevice.
    - Automatic hardware fallback: if sounddevice is missing or fails to open a host stream,
      automatically falls back to VirtualAudioDevice without crashing.
    - Normalizes multi-channel and varied bit-depth audio to 16kHz 30ms frames (480 samples).
    - Evaluates VoiceActivityDetector per frame and emits VADTelemetry.
    - Triggers sub-50ms barge-in interruption (< 0.05ms atomic buffer drain) on speech onset.
    - Tracks monotonic audio playhead progress to drive word-level cursor offsets.
    """

    VALID_SOURCES = {"browser", "host", "virtual", "mock"}

    def __init__(
        self,
        source: str = "browser",
        input_sample_rate: int = 24000,
        target_sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        ring_buffer: Optional[AudioRingBuffer] = None,
        vad: Optional[VoiceActivityDetector] = None,
        virtual_device: Optional[VirtualAudioDevice] = None,
        auto_fallback: bool = True,
    ):
        norm_source = source.lower().strip()
        if norm_source not in self.VALID_SOURCES:
            raise ValueError(f"Invalid source '{source}'. Must be one of: {self.VALID_SOURCES}")
        if norm_source == "mock":
            norm_source = "virtual"

        self.source = norm_source
        self.input_sample_rate = int(input_sample_rate)
        self.target_sample_rate = int(target_sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_size = int(self.target_sample_rate * (self.frame_duration_ms / 1000.0))
        self.chunk_size = int(self.input_sample_rate * (self.frame_duration_ms / 1000.0))

        self.ring_buffer = ring_buffer
        self.vad = vad or VoiceActivityDetector(
            sample_rate=self.target_sample_rate,
            frame_duration_ms=self.frame_duration_ms,
        )
        self.virtual_device = virtual_device or VirtualAudioDevice(
            sample_rate=self.input_sample_rate,
            chunk_size=self.chunk_size,
        )
        self.auto_fallback = bool(auto_fallback)
        self.is_fallback_active = False

        # Host sounddevice stream
        self._host_stream: Any = None

        # FIFO accumulator for calibrated 16kHz VAD frames
        self._fifo = np.empty(0, dtype=np.float32)

        # Callbacks
        self._barge_in_callbacks: List[Callable[[BargeInEvent], None]] = []
        self._vad_callbacks: List[Callable[[VADTelemetry], None]] = []
        self._frame_callbacks: List[Callable[[np.ndarray], None]] = []

        # Speech state tracking for onset detection
        self._speech_in_progress = False

        # Monotonic playhead tracking baseline when no ring buffer is attached
        self._local_samples_played = 0

        self._lock = threading.RLock()

        # If initialized in host mode, attempt to open the host stream
        if self.source == "host":
            self._start_host_stream()

    # -------------------------------------------------------------------------
    # Source Configuration & Thread-Safe Switching
    # -------------------------------------------------------------------------

    def set_source(self, source: str) -> str:
        """Switch audio input source ("browser", "host", or "virtual").

        Thread-safe. Gracefully opens/closes host hardware streams and falls back
        to VirtualAudioDevice if host hardware is absent or unavailable.
        Returns the effective active source name.
        """
        norm_source = source.lower().strip()
        if norm_source not in self.VALID_SOURCES:
            raise ValueError(f"Invalid source '{source}'. Must be one of: {self.VALID_SOURCES}")
        if norm_source == "mock":
            norm_source = "virtual"

        with self._lock:
            if self.source == norm_source and not self.is_fallback_active:
                return self.source

            # Clean up existing host stream when leaving "host" mode
            self._stop_host_stream()

            # Flush FIFO accumulator and reset VAD filter state to prevent sample discontinuity / false barge-in
            self._fifo = np.empty(0, dtype=np.float32)
            self._speech_in_progress = False
            if hasattr(self, "vad") and self.vad is not None:
                self.vad.reset(preserve_noise_floor=True)

            if norm_source == "host":
                success = self._start_host_stream()
                if not success:
                    if self.auto_fallback:
                        logger.warning("Host sounddevice unavailable; falling back to VirtualAudioDevice")
                        self.source = "virtual"
                        self.is_fallback_active = True
                        self.virtual_device.start()
                        return "virtual"
                    else:
                        raise RuntimeError("Failed to initialize host sounddevice input stream")
                else:
                    self.source = "host"
                    self.is_fallback_active = False
                    return "host"

            elif norm_source == "virtual":
                self.source = "virtual"
                self.is_fallback_active = False
                self.virtual_device.start()
                return "virtual"

            elif norm_source == "browser":
                self.source = "browser"
                self.is_fallback_active = False
                return "browser"

            return self.source

    def get_source(self) -> str:
        """Return the current active input source."""
        with self._lock:
            return self.source

    # -------------------------------------------------------------------------
    # Audio Ingestion & Normalization Paths
    # -------------------------------------------------------------------------

    def feed_browser_pcm(
        self,
        data: Union[np.ndarray, bytes, bytearray, memoryview],
        sample_rate: Optional[int] = None,
    ) -> int:
        """Ingest raw PCM audio chunks received from browser Web Audio API.

        Normalizes, resamples to 16kHz, evaluates VAD in 30ms frames, and triggers
        barge-in if speech is detected.
        Returns the number of 30ms frames processed.
        """
        if isinstance(data, (bytes, bytearray)):
            if len(data) % 2 != 0:
                data = data[: len(data) - 1]
        with self._lock:
            if self.source != "browser":
                return 0

            sr = sample_rate if sample_rate is not None else self.input_sample_rate
            return self._ingest_audio_data(data, sr)

    # Architectural aliases for cross-subagent compatibility
    switch_source = set_source
    ingest_browser_audio = feed_browser_pcm

    def feed_audio(
        self,
        data: Union[np.ndarray, bytes, bytearray, memoryview],
        input_sample_rate: Optional[int] = None,
    ) -> int:
        """Universal audio ingestion method (used by host callbacks and unit tests).

        Returns the number of 30ms frames processed.
        """
        with self._lock:
            sr = input_sample_rate if input_sample_rate is not None else self.input_sample_rate
            return self._ingest_audio_data(data, sr)

    def step_virtual(self, num_samples: Optional[int] = None) -> int:
        """Generate one frame from the virtual audio device and process it."""
        with self._lock:
            samples = num_samples if num_samples is not None else self.chunk_size
            chunk = self.virtual_device.read_mic(samples=samples, dtype=np.float32)
            return self.feed_audio(chunk, input_sample_rate=self.virtual_device.sample_rate)

    def _ingest_audio_data(self, data: Any, source_sr: int) -> int:
        """Internal audio pipeline: normalize -> resample to 16kHz -> buffer in FIFO -> process 30ms frames."""
        normalized = self.normalize_audio(data)
        if len(normalized) == 0:
            return 0

        # Resample to 16kHz target sample rate
        resampled = self.resample_audio(normalized, orig_sr=source_sr, target_sr=self.target_sample_rate)
        if len(resampled) == 0:
            return 0

        self._fifo = np.concatenate([self._fifo, resampled])

        frames_processed = 0
        while len(self._fifo) >= self.frame_size:
            frame = self._fifo[: self.frame_size]
            self._fifo = self._fifo[self.frame_size :]
            self._process_calibrated_frame(frame)
            frames_processed += 1

        return frames_processed

    # -------------------------------------------------------------------------
    # Frame Processing, VAD Telemetry & Sub-50ms Barge-In Trigger
    # -------------------------------------------------------------------------

    def _process_calibrated_frame(self, frame: np.ndarray) -> VADTelemetry:
        """Process a single calibrated 30ms frame (480 samples @ 16kHz)."""
        now = time.time()

        # Notify raw frame observers
        for cb in self._frame_callbacks:
            try:
                cb(frame)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error in frame callback: {e}")

        # Evaluate VAD step
        self.vad._step_frame(frame)

        is_speech = self.vad.is_speech
        dbfs = self.vad.compute_dbfs(frame)
        noise_floor = self.vad.noise_floor_dbfs
        threshold = self.vad.threshold_dbfs
        is_turn_complete = self.vad.is_turn_complete

        telemetry = VADTelemetry(
            speech_active=is_speech,
            dbfs=dbfs,
            noise_floor_dbfs=noise_floor,
            threshold_dbfs=threshold,
            is_turn_complete=is_turn_complete,
            timestamp=now,
        )

        # Notify VAD observers
        for vad_cb in self._vad_callbacks:
            try:
                vad_cb(telemetry)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error in VAD callback: {e}")

        # Check for speech onset to trigger sub-50ms barge-in interruption
        consecutive = getattr(self.vad, "_consecutive_speech_frames", 0)
        onset_required = getattr(self.vad, "onset_frames", 2)

        if is_speech and consecutive >= onset_required:
            if not self._speech_in_progress:
                self._speech_in_progress = True
                self._trigger_barge_in(timestamp=now)
        elif not is_speech:
            self._speech_in_progress = False

        return telemetry

    def _trigger_barge_in(self, timestamp: Optional[float] = None) -> BargeInEvent:
        """Execute instantaneous atomic buffer drain (< 0.05ms) and dispatch barge-in event."""
        ts = timestamp if timestamp is not None else time.time()
        t0 = time.perf_counter()

        flushed_samples = 0
        generation = 0

        if self.ring_buffer is not None:
            flushed_samples = self.ring_buffer.size
            self.ring_buffer.drain(fade_ms=2.0)
            generation = self.ring_buffer.generation

        drain_latency_ms = (time.perf_counter() - t0) * 1000.0

        event = BargeInEvent(
            drain_latency_ms=drain_latency_ms,
            generation=generation,
            flushed_samples=flushed_samples,
            playhead_seconds=self.playhead_seconds,
            timestamp=ts,
        )

        for cb in self._barge_in_callbacks:
            try:
                cb(event)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error in barge-in callback: {e}")

        return event

    def trigger_manual_barge_in(self) -> BargeInEvent:
        """Explicitly trigger barge-in interruption (e.g. from UI button or test harness)."""
        with self._lock:
            self._speech_in_progress = True
            return self._trigger_barge_in()

    # -------------------------------------------------------------------------
    # Playhead Progress & Word Cursor Offsets
    # -------------------------------------------------------------------------

    @property
    def playhead_seconds(self) -> float:
        """Monotonic DAC playback progress in seconds."""
        if self.ring_buffer is not None:
            return self.ring_buffer.playhead_seconds
        return self._local_samples_played / float(self.input_sample_rate)

    def get_playhead_seconds(self) -> float:
        """Accessor for monotonic playhead progress in seconds."""
        return self.playhead_seconds

    def reset_playhead(self) -> None:
        """Reset playhead progress counter."""
        with self._lock:
            if self.ring_buffer is not None:
                self.ring_buffer.reset_playhead()
            self._local_samples_played = 0

    def compute_word_offset(self, words_per_minute: float = 150.0, total_words: Optional[int] = None) -> int:
        """Compute current spoken word offset from playback playhead progress."""
        sec = self.playhead_seconds
        words_per_second = max(0.1, words_per_minute / 60.0)
        offset = int(sec * words_per_second)
        if total_words is not None and total_words > 0:
            offset = min(offset, total_words - 1)
        return max(0, offset)

    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------

    def add_barge_in_callback(self, callback: Callable[[BargeInEvent], None]) -> None:
        """Register callback invoked instantaneously on barge-in speech onset."""
        with self._lock:
            if callback not in self._barge_in_callbacks:
                self._barge_in_callbacks.append(callback)

    def remove_barge_in_callback(self, callback: Callable[[BargeInEvent], None]) -> None:
        """Remove previously registered barge-in callback."""
        with self._lock:
            if callback in self._barge_in_callbacks:
                self._barge_in_callbacks.remove(callback)

    def add_vad_callback(self, callback: Callable[[VADTelemetry], None]) -> None:
        """Register callback invoked every 30ms with VAD metrics."""
        with self._lock:
            if callback not in self._vad_callbacks:
                self._vad_callbacks.append(callback)

    def remove_vad_callback(self, callback: Callable[[VADTelemetry], None]) -> None:
        """Remove previously registered VAD callback."""
        with self._lock:
            if callback in self._vad_callbacks:
                self._vad_callbacks.remove(callback)

    def add_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register callback receiving calibrated 30ms 16kHz audio frames."""
        with self._lock:
            if callback not in self._frame_callbacks:
                self._frame_callbacks.append(callback)

    # -------------------------------------------------------------------------
    # Audio Normalization & Resampling Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_audio(data: Any) -> np.ndarray:
        """Normalize arbitrary audio input to 1D float32 array in [-1.0, 1.0].

        - Handles bytes, bytearray, memoryview, and numpy arrays.
        - Truncates odd byte lengths.
        - Downmixes multi-channel stereo / 2D arrays to 1D mono.
        - Converts int16 to float32 [-1.0, 1.0].
        - Replaces NaNs and Infs with 0.0.
        """
        if data is None:
            return np.empty(0, dtype=np.float32)

        # 1. Type normalization
        if isinstance(data, (bytes, bytearray)):
            nbytes = len(data)
            if nbytes % 2 != 0:
                data = data[: nbytes - 1]
            if len(data) == 0:
                return np.empty(0, dtype=np.float32)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(data, memoryview):
            if data.format in ("B", "b", "c") or data.itemsize == 1:
                nbytes = len(data)
                if nbytes % 2 != 0:
                    data = data[: nbytes - 1]
                if len(data) == 0:
                    return np.empty(0, dtype=np.float32)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                arr = np.asarray(data)
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            try:
                arr = np.asarray(data)
            except Exception:  # noqa: BLE001  # S3.3 RC4
                return np.empty(0, dtype=np.float32)

        if arr.size == 0:
            return np.empty(0, dtype=np.float32)

        # 2. Multi-channel downmix to mono
        if arr.ndim > 1:
            if arr.shape[0] < arr.shape[1]:
                arr = np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=-1)
        arr = arr.flatten()

        # 3. Sanitize NaNs and Infs
        arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        # 4. Scale parity
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = arr.astype(np.float32)
            max_abs = float(np.max(np.abs(arr))) if len(arr) > 0 else 0.0
            if max_abs > 256.0:
                arr = arr / 32768.0
            arr = np.clip(arr, -1.0, 1.0)

        return arr

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample 1D float32 audio from orig_sr to target_sr using fast linear interpolation."""
        if len(audio) == 0 or orig_sr <= 0 or target_sr <= 0:
            return np.empty(0, dtype=np.float32)
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        n_in = len(audio)
        n_out = int(round(n_in * (float(target_sr) / float(orig_sr))))
        if n_out == 0:
            return np.empty(0, dtype=np.float32)

        orig_indices = np.arange(n_in, dtype=np.float64)
        target_indices = np.linspace(0, n_in - 1, n_out, endpoint=True, dtype=np.float64)
        resampled = np.interp(target_indices, orig_indices, audio)
        return resampled.astype(np.float32)

    # -------------------------------------------------------------------------
    # Native sounddevice Hardware Integration
    # -------------------------------------------------------------------------

    def _start_host_stream(self) -> bool:
        """Attempt to open and start a native sounddevice.InputStream."""
        try:
            import sounddevice as sd

            # Check if input devices exist
            devices = sd.query_devices()
            if not devices:
                return False

            def _sd_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"sounddevice stream status: {status}")
                self.feed_audio(indata.copy(), input_sample_rate=self.input_sample_rate)

            stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=_sd_callback,
            )
            stream.start()
            self._host_stream = stream
            return True

        except (ImportError, Exception) as e:  # noqa: BLE001  # S3.3 RC4
            logger.debug(f"Failed to initialize host sounddevice: {e}")
            self._host_stream = None
            return False

    def _stop_host_stream(self) -> None:
        """Stop and close any active sounddevice input stream."""
        if self._host_stream is not None:
            try:
                self._host_stream.stop()
                self._host_stream.close()
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.debug(f"Error closing sounddevice stream: {e}")
            finally:
                self._host_stream = None

    def close(self) -> None:
        """Release all hardware resources and stop virtual generators."""
        with self._lock:
            self._stop_host_stream()
            self.virtual_device.stop()
            self._fifo = np.empty(0, dtype=np.float32)

    def __enter__(self) -> "HybridMicrophoneBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Drain Latency Benchmarking
    # -------------------------------------------------------------------------

    def benchmark_drain_latency(self, iterations: int = 1000, fade_ms: float = 2.0) -> Dict[str, float]:
        """Execute atomic buffer drain benchmark across iterations and return latency stats.

        Verifies that mean execution time satisfies the sub-0.05ms (< 50 microseconds) law.
        """
        rb = self.ring_buffer or AudioRingBuffer(capacity_samples=48000 * 5, sample_rate=self.input_sample_rate)
        times_ms = []

        # Warm up JIT/caches
        dummy = np.ones(48000, dtype=np.int16)
        for _ in range(20):
            rb.write(dummy)
            rb.drain(fade_ms=fade_ms)

        for _ in range(iterations):
            rb.write(dummy)
            t0 = time.perf_counter()
            rb.drain(fade_ms=fade_ms)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms)
        stats = {
            "iterations": float(iterations),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "target_met": bool(arr.mean() < 0.05),
        }
        return stats
