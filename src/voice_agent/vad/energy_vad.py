"""Voice Activity Detection (VAD) module with adaptive noise floor, DC blocking,

and onset persistence for conversational voice agents.
"""

from typing import Any, Generator, List, Optional, Union
import numpy as np

try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None


class VoiceActivityDetector:
    """Voice Activity Detector evaluating strictly 30ms frames with adaptive noise floor,

    DC blocking, multi-frame onset persistence, decoupled hangover bridging, and acoustic ducking.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: Optional[float] = 300.0,
        snr_margin_db: float = 12.0,
        hangover_frames: int = 6,
        turn_completion_frames: int = 16,
        onset_frames: int = 2,
        min_noise_floor_dbfs: float = -90.0,
        max_noise_floor_dbfs: float = -20.0,
        initial_noise_floor_dbfs: float = -60.0,
        ducking_attenuation_db: float = 14.0,
        high_pass_cutoff_hz: float = 80.0,
        high_pass_r: float = 0.968,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.snr_margin_db = float(snr_margin_db)
        self.hangover_frames = int(hangover_frames)
        self.turn_completion_frames = int(turn_completion_frames)
        self.onset_frames = int(onset_frames)
        self.min_noise_floor_dbfs = float(min_noise_floor_dbfs)
        self.max_noise_floor_dbfs = float(max_noise_floor_dbfs)
        self.initial_noise_floor_dbfs = float(initial_noise_floor_dbfs)
        self.ducking_attenuation_db = float(ducking_attenuation_db)
        self.high_pass_cutoff_hz = float(high_pass_cutoff_hz)
        self.high_pass_r = float(high_pass_r)

        # Baseline fixed threshold converted to dBFS if specified
        self._fixed_threshold_dbfs: Optional[float] = None
        if energy_threshold is not None:
            eth = float(energy_threshold)
            if eth > 1.0:
                # Scaled to int16 maximum
                self._fixed_threshold_dbfs = float(20.0 * np.log10(eth / 32768.0))
            elif eth <= 0.0:
                self._fixed_threshold_dbfs = eth
            else:
                self._fixed_threshold_dbfs = float(20.0 * np.log10(eth))

        # Filter state for stateful 1st-order IIR high-pass filter: y[n] = x[n] - x[n-1] + R * y[n-1]
        self._b = np.array([1.0, -1.0], dtype=np.float32)
        self._a = np.array([1.0, -self.high_pass_r], dtype=np.float32)
        self._filter_zi = np.zeros(1, dtype=np.float32)
        self._prev_x = 0.0
        self._prev_y = 0.0

        # Sample accumulator FIFO
        self._fifo = np.empty(0, dtype=np.float32)

        # Adaptive state
        self.noise_floor_dbfs = float(initial_noise_floor_dbfs)
        self.threshold_dbfs = self.noise_floor_dbfs + self.snr_margin_db
        self.is_speech = False
        self.is_turn_complete = False
        self.hangover_count = 0
        self._consecutive_speech_frames = 0
        self._silence_frames = 0
        self._has_speech_occurred = False

        # Acoustic Echo Suppression (AES) / Ducking state
        self._is_playback_active = False
        self._reference_ducking_threshold: Optional[float] = None

    @property
    def is_playback_active(self) -> bool:
        """Indicates whether speaker playback is currently active."""
        return self._is_playback_active

    @property
    def playback_active(self) -> bool:
        """Alias for is_playback_active."""
        return self._is_playback_active

    def set_playback_active(
        self, active: bool, reference_signal: Optional[np.ndarray] = None
    ) -> None:
        """Set acoustic echo suppression state.

        Elevates threshold to prevent microphone self-triggering during agent speech.
        """
        self._is_playback_active = bool(active)
        if active and reference_signal is not None and len(reference_signal) > 0:
            ref_dbfs = self.compute_dbfs(reference_signal)
            # Estimated acoustic leakage with -12 dB coupling loss
            leakage_est = ref_dbfs - 12.0
            self._reference_ducking_threshold = leakage_est + self.ducking_attenuation_db
        else:
            self._reference_ducking_threshold = None

    def reset(self, preserve_noise_floor: bool = False) -> None:
        """Reset internal FIFO, DSP filter state, and speech detection counters.

        If preserve_noise_floor is True, retains the calibrated background noise floor
        and threshold to prevent false-positive barge-in interruptions on mic source switches.
        """
        self._fifo = np.empty(0, dtype=np.float32)
        self._filter_zi = np.zeros(1, dtype=np.float32)
        self._prev_x = 0.0
        self._prev_y = 0.0
        if not preserve_noise_floor:
            self.noise_floor_dbfs = float(self.initial_noise_floor_dbfs)
            self.threshold_dbfs = self.noise_floor_dbfs + self.snr_margin_db
        self.is_speech = False
        self.is_turn_complete = False
        self.hangover_count = 0
        self._consecutive_speech_frames = 0
        self._silence_frames = 0
        self._has_speech_occurred = False
        self._is_playback_active = False
        self._reference_ducking_threshold = None

    def _sanitize_audio(self, chunk: Any) -> np.ndarray:
        """Validate, sanitize, downmix to mono, and normalize input audio to float32 [-1.0, 1.0]."""
        if chunk is None:
            return np.empty(0, dtype=np.float32)

        if not isinstance(chunk, np.ndarray):
            try:
                chunk = np.asarray(chunk)
            except Exception:
                return np.empty(0, dtype=np.float32)

        if chunk.size == 0:
            return np.empty(0, dtype=np.float32)

        # Multi-channel downmix to mono
        if chunk.ndim > 1:
            if chunk.shape[0] < chunk.shape[1]:  # shape: (channels, samples)
                chunk = np.mean(chunk, axis=0)
            else:  # shape: (samples, channels)
                chunk = np.mean(chunk, axis=-1)

        chunk = chunk.flatten()

        # Sanitize non-finite values (NaN, +Inf, -Inf)
        chunk = np.nan_to_num(chunk, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        # Scale parity: handle int16 vs float32
        if np.issubdtype(chunk.dtype, np.integer):
            chunk = chunk.astype(np.float32) / 32768.0
        else:
            chunk = chunk.astype(np.float32)
            # Check for unnormalized float values
            max_abs = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
            if max_abs > 256.0:
                chunk = chunk / 32768.0
            chunk = np.clip(chunk, -1.0, 1.0)

        return chunk

    def _apply_filter(self, frame: np.ndarray) -> np.ndarray:  # type: ignore[return]  # tracked L8, fixed in S4.1
        """Apply zero-mean centering and stateful 1st-order IIR high-pass filter (80 Hz, R=0.968)."""
        if len(frame) == 0:
            return frame

        # Zero-mean centering
        centered = frame - np.mean(frame)

        if lfilter is not None:
            try:
                y, self._filter_zi = lfilter(self._b, self._a, centered, zi=self._filter_zi)
                return y.astype(np.float32)
            except Exception:
                pass
            # Fallback pure-numpy/python loop
            y = np.empty_like(centered)
            px, py = self._prev_x, self._prev_y
            r = self.high_pass_r
            for i in range(len(centered)):
                cx = centered[i]
                cy = cx - px + r * py
                y[i] = cy
                px = cx
                py = cy
            self._prev_x = float(px)
            self._prev_y = float(py)
            return y

    def compute_energy(self, frame: np.ndarray) -> float:
        """Computes scale-invariant RMS energy normalized in [0.0, 1.0]."""
        sanitized = self._sanitize_audio(frame)
        if len(sanitized) == 0:
            return 0.0
        centered = sanitized - np.mean(sanitized)
        rms_centered = float(np.sqrt(np.mean(centered**2)))
        filtered = self._apply_filter(sanitized)
        rms_filt = float(np.sqrt(np.mean(filtered**2)))
        return min(rms_filt, rms_centered)

    def compute_dbfs(self, audio_or_rms: Union[float, np.ndarray]) -> float:
        """Computes scale-invariant dBFS: 20 * log10(rms + 1e-12)."""
        if isinstance(audio_or_rms, (int, float, np.floating, np.integer)):
            rms = float(audio_or_rms)
        else:
            sanitized = self._sanitize_audio(audio_or_rms)
            if len(sanitized) == 0:
                rms = 0.0
            else:
                rms = self.compute_energy(sanitized)
        return float(20.0 * np.log10(rms + 1e-12))

    def _step_frame(self, frame: np.ndarray) -> bool:
        """Process a single calibrated 30ms frame (480 samples @ 16 kHz)."""
        centered = frame - np.mean(frame)
        rms_centered = float(np.sqrt(np.mean(centered**2)))

        filtered = self._apply_filter(frame)
        rms_filt = float(np.sqrt(np.mean(filtered**2)))

        # Effective RMS: filter attenuates low frequencies, but cannot exceed centered input energy
        rms = min(rms_filt, rms_centered)
        if rms_centered == 0.0:
            self._filter_zi = np.zeros(1, dtype=np.float32)
            self._prev_x = 0.0
            self._prev_y = 0.0

        frame_dbfs = float(20.0 * np.log10(rms + 1e-12))

        # Asymmetric dual-rate EMA noise floor tracking
        if frame_dbfs < self.noise_floor_dbfs:
            alpha_down = 0.15
            self.noise_floor_dbfs = (1.0 - alpha_down) * self.noise_floor_dbfs + alpha_down * frame_dbfs
        else:
            # Only adapt upwards during confirmed silence
            if not self.is_speech and self._consecutive_speech_frames == 0:
                alpha_up = 0.02
                self.noise_floor_dbfs = (1.0 - alpha_up) * self.noise_floor_dbfs + alpha_up * frame_dbfs

        self.noise_floor_dbfs = float(
            np.clip(self.noise_floor_dbfs, self.min_noise_floor_dbfs, self.max_noise_floor_dbfs)
        )

        # Calculate dynamic threshold
        adaptive_thresh = self.noise_floor_dbfs + self.snr_margin_db
        if self._fixed_threshold_dbfs is not None:
            base_thresh = max(adaptive_thresh, self._fixed_threshold_dbfs)
        else:
            base_thresh = adaptive_thresh

        # Clamp base speech threshold
        base_thresh = float(np.clip(base_thresh, -60.0, -15.0))

        # Apply acoustic echo suppression / ducking if playback is active
        if self._is_playback_active:
            if self._reference_ducking_threshold is not None:
                effective_thresh = max(base_thresh, self._reference_ducking_threshold)
            else:
                effective_thresh = base_thresh + self.ducking_attenuation_db
        else:
            effective_thresh = base_thresh

        self.threshold_dbfs = effective_thresh

        is_frame_speech = frame_dbfs >= effective_thresh

        if is_frame_speech:
            self._consecutive_speech_frames += 1
            if self.is_speech:
                # Refresh pause bridging while in active speech
                self.hangover_count = self.hangover_frames
                self._silence_frames = 0
            elif self._consecutive_speech_frames >= self.onset_frames:
                # Onset persistence met
                self.is_speech = True
                self._has_speech_occurred = True
                self.hangover_count = self.hangover_frames
                self._silence_frames = 0
                self.is_turn_complete = False
        else:
            self._consecutive_speech_frames = 0
            if self._has_speech_occurred:
                self._silence_frames += 1
                if self._silence_frames >= self.turn_completion_frames:
                    self.is_turn_complete = True

            if self.is_speech:
                if self.hangover_count > 0:
                    self.hangover_count -= 1
                    self.is_speech = True  # Pause bridging active
                else:
                    self.is_speech = False  # Pause bridging expired

        return self.is_speech

    def process_frame(self, chunk: Optional[np.ndarray]) -> bool:
        """Feed audio samples into 30ms accumulator and process complete frames.

        Returns True if speech is actively detected, False otherwise.
        """
        sanitized = self._sanitize_audio(chunk)
        if len(sanitized) > 0:
            self._fifo = np.concatenate([self._fifo, sanitized])

        while len(self._fifo) >= self.frame_size:
            frame = self._fifo[: self.frame_size]
            self._fifo = self._fifo[self.frame_size :]
            self._step_frame(frame)

        return self.is_speech

    def feed_audio(self, chunk: Optional[np.ndarray]) -> List[bool]:
        """Feed audio samples into accumulator and return per-frame speech decisions."""
        sanitized = self._sanitize_audio(chunk)
        if len(sanitized) > 0:
            self._fifo = np.concatenate([self._fifo, sanitized])

        results = []
        while len(self._fifo) >= self.frame_size:
            frame = self._fifo[: self.frame_size]
            self._fifo = self._fifo[self.frame_size :]
            results.append(self._step_frame(frame))
        return results

    def drain_frames(self) -> Generator[np.ndarray, None, None]:
        """Yield strictly calibrated 30ms frames from the FIFO accumulator."""
        while len(self._fifo) >= self.frame_size:
            frame = self._fifo[: self.frame_size]
            self._fifo = self._fifo[self.frame_size :]
            yield frame
