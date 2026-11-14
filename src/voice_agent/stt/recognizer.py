"""Text-in, text-out adapter over WhisperBackend, for the server call sites that predate SttBackend.

# removed in S3.3 when narration takes a SttBackend
"""

import numpy as np

from voice_agent.stt.backends import WhisperBackend


class STTEngine:
    """Keeps the call signature the server uses. `transcribe` returns the backend's words, and "" when they were withheld."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str = "",
    ) -> None:
        self.backend = WhisperBackend(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            initial_prompt=initial_prompt,
        )

    def load(self) -> bool:
        """Loads the Whisper model now (the STT build calls this off the event loop). Returns whether it is loaded."""
        return self.backend.load()

    def transcribe(self, audio_data: bytes | bytearray | memoryview | np.ndarray | None, sample_rate: int = 16000) -> str:
        if audio_data is None:
            return ""
        return self.backend.transcribe(_pcm16_bytes(audio_data), sample_rate).text


def _pcm16_bytes(audio_data: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Mono 16-bit PCM bytes. A float array is in [-1, 1]; a multi-channel array is averaged to mono."""
    if isinstance(audio_data, (bytes, bytearray, memoryview)):
        return bytes(audio_data)
    if isinstance(audio_data, np.ndarray):
        if audio_data.dtype == np.int16:
            samples = audio_data
        else:
            samples = np.clip(audio_data.astype(np.float32) * 32768.0, -32768, 32767).astype(np.int16)
        if samples.ndim > 1:
            samples = np.mean(samples.astype(np.float32), axis=-1).astype(np.int16)
        return samples.tobytes()
    raise TypeError(f"STTEngine.transcribe takes bytes or a numpy array, not {type(audio_data).__name__}")
