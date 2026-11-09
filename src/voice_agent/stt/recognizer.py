import io
import os
import wave
import logging
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class MockSTTEngine:
    """Deterministic Mock STT engine for unit testing and CI."""

    def __init__(self, fixed_transcript: str = "what is arasaka"):
        self.fixed_transcript = fixed_transcript

    def transcribe(self, audio_data: Union[bytes, np.ndarray], sample_rate: int = 16000) -> str:
        return self.fixed_transcript


class STTEngine:
    """High-performance local Speech-To-Text engine utilizing Faster-Whisper
    with fallback to SpeechRecognition and deterministic Mock mode.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        fallback_mock: bool = False,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.fallback_mock = fallback_mock
        self._whisper_model = None
        self._init_engine()

    def _init_engine(self):
        try:
            from faster_whisper import WhisperModel

            logger.info("Initializing faster-whisper STTEngine (%s on %s)...", self.model_size, self.device)
            self._whisper_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            logger.warning("Failed to initialize faster-whisper (%s). Falling back.", e)
            self._whisper_model = None

    def transcribe(self, audio_data: Union[bytes, bytearray, np.ndarray], sample_rate: int = 16000) -> str:
        """Transcribe PCM audio to text."""
        if audio_data is None:
            return ""

        # Convert bytes to numpy float32 if necessary
        if isinstance(audio_data, (bytes, bytearray, memoryview)):
            if len(audio_data) == 0:
                return ""
            pcm16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = pcm16.astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            if audio_data.size == 0:
                return ""
            if audio_data.dtype == np.int16:
                audio_np = audio_data.astype(np.float32) / 32768.0
            else:
                audio_np = audio_data.astype(np.float32)
        else:
            return ""

        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)

        # Check for near-total silence
        if audio_np.size == 0 or np.max(np.abs(audio_np)) < 0.004:
            return ""

        # 1. Primary: faster-whisper
        if self._whisper_model is not None:
            try:
                segments, info = self._whisper_model.transcribe(
                    audio_np,
                    beam_size=3,
                    language="en",
                    vad_filter=True,
                )
                text = " ".join([s.text.strip() for s in segments]).strip()
                if text:
                    return text
            except Exception as err:
                logger.warning("faster-whisper transcription error: %s", err)

        # 2. Secondary: SpeechRecognition fallback
        try:
            import speech_recognition as sr

            r = sr.Recognizer()
            # Wrap PCM in WAV buffer
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio = r.record(source)
                return r.recognize_google(audio)
        except Exception as err:
            logger.debug("SpeechRecognition fallback notice: %s", err)

        if self.fallback_mock:
            return "what is arasaka"

        return ""
