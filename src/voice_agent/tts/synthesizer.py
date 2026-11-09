"""Unified Speech Synthesis Engine for Interactive Voice Agent.

Provides multi-tier TTS:
1. Primary: edge-tts (ultra-high-quality neural Cyberpunk voices, zero GPU VRAM).
2. Secondary: pyttsx3 (100% offline local Windows SAPI5 desktop speech).
3. Tertiary: In-memory PCM synthetic audio generator.
"""

import asyncio
import io
import logging
import os
import re
import tempfile
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Character voice mappings to Edge TTS neural voices
VOICE_MAP_EDGE = {
    "david-martinez": "en-US-ChristopherNeural",
    "lucy-kushinada": "en-US-JennyNeural",
    "v-male": "en-US-GuyNeural",
    "v-female": "en-US-AriaNeural",
    "johnny-silverhand": "en-US-EricNeural",
}

# Fallback voice map
DEFAULT_EDGE_VOICE = "en-US-ChristopherNeural"


async def synthesize_speech_edge(
    text: str,
    voice_id: str = "david-martinez",
    cadence: float = 1.0,
) -> Optional[bytes]:
    """Synthesize speech audio using edge-tts. Returns MP3/audio bytes."""
    clean_text = text.strip()
    if not clean_text:
        return None

    edge_voice = VOICE_MAP_EDGE.get(voice_id, DEFAULT_EDGE_VOICE)

    # Cadence rate modifier in edge-tts: e.g. "+10%", "-15%"
    rate_pct = int(round((cadence - 1.0) * 100))
    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, edge_voice, rate=rate_str)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        if audio_chunks:
            return b"".join(audio_chunks)
    except Exception as exc:
        logger.warning(f"edge-tts synthesis failed, falling back: {exc}")

    return None


def synthesize_speech_pyttsx3(
    text: str,
    voice_id: str = "david-martinez",
    cadence: float = 1.0,
) -> Optional[bytes]:
    """Synthesize speech audio using local Windows SAPI5 pyttsx3. Returns WAV bytes."""
    clean_text = text.strip()
    if not clean_text:
        return None

    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Adjust rate
        base_rate = 175
        engine.setProperty("rate", int(base_rate * cadence))

        # Select female vs male voice if available
        is_female = "female" in voice_id or "lucy" in voice_id
        voices = engine.getProperty("voices")
        for v in voices:
            name_lower = v.name.lower()
            if is_female and ("zira" in name_lower or "hazel" in name_lower or "female" in name_lower):
                engine.setProperty("voice", v.id)
                break
            elif not is_female and ("david" in name_lower or "male" in name_lower):
                engine.setProperty("voice", v.id)
                break

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_path = tf.name

        try:
            engine.save_to_file(clean_text, temp_path)
            engine.runAndWait()
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                with open(temp_path, "rb") as f:
                    wav_bytes = f.read()
                return wav_bytes
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    except Exception as exc:
        logger.warning(f"pyttsx3 synthesis failed: {exc}")

    return None


def generate_synthetic_pcm_fallback(
    text: str,
    cadence: float = 1.0,
    sample_rate: int = 24000,
) -> bytes:
    """Deterministic in-memory synthetic voice audio frame generator fallback."""
    words = text.split()
    duration_sec = max(0.4, (len(words) / (2.5 * cadence)))
    n_samples = int(sample_rate * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    # Voiced multi-harmonic tone simulating speech formant acoustics
    f0 = 150.0
    signal = (
        0.45 * np.sin(2.0 * np.pi * f0 * t) +
        0.25 * np.sin(2.0 * np.pi * 750.0 * t) +
        0.15 * np.sin(2.0 * np.pi * 1800.0 * t) +
        0.05 * np.sin(2.0 * np.pi * 3700.0 * t)
    ) * 0.35
    # Smooth start and end with half-cosine envelopes
    fade = min(int(sample_rate * 0.02), n_samples // 4)
    if fade > 1:
        win_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade)))
        signal[:fade] *= win_in
        signal[-fade:] *= win_in[::-1]

    int16_pcm = np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return int16_pcm.tobytes()


async def synthesize_sentence(
    text: str,
    voice_id: str = "david-martinez",
    cadence: float = 1.0,
) -> Tuple[bytes, str]:
    """Universal synthesizer function: tries edge-tts -> pyttsx3 -> synthetic fallback.
    
    Returns (audio_bytes, format_type) where format_type is 'mp3', 'wav', or 'pcm_s16le'.
    """
    # 1. Try edge-tts
    edge_audio = await synthesize_speech_edge(text, voice_id=voice_id, cadence=cadence)
    if edge_audio is not None and len(edge_audio) > 100:
        return edge_audio, "mp3"

    # 2. Try pyttsx3
    loop = asyncio.get_running_loop()
    wav_audio = await loop.run_in_executor(None, synthesize_speech_pyttsx3, text, voice_id, cadence)
    if wav_audio is not None and len(wav_audio) > 100:
        return wav_audio, "wav"

    # 3. Synthetic fallback
    pcm_audio = generate_synthetic_pcm_fallback(text, cadence=cadence)
    return pcm_audio, "pcm_s16le"
