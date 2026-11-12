"""Unified Speech Synthesis Engine for Interactive Voice Agent.

Provides multi-tier TTS:
1. Primary: edge-tts (ultra-high-quality neural Cyberpunk voices, zero GPU VRAM).
2. Secondary: pyttsx3 (100% offline local Windows SAPI5 desktop speech).
3. Tertiary: In-memory PCM synthetic audio generator.
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple, TypedDict
import numpy as np

logger = logging.getLogger(__name__)

class CharacterAudioProfile(TypedDict):
    voice: str
    pitch_hz: int
    rate_offset: int
    name: str
    persona: str
    quote: str


# Character audio profiles with authentic Cyberpunk prosody and Edge-TTS neural models
CHARACTER_AUDIO_PROFILES: Dict[str, CharacterAudioProfile] = {
    "johnny-silverhand": {
        "voice": "en-US-RogerNeural",
        "pitch_hz": -14,
        "rate_offset": -5,
        "name": "Johnny Silverhand",
        "persona": "Rebel Rockerboy / Keanu Baritone",
        "quote": "Wake up, samurai. We got a city to burn.",
    },
    "lucy-kushinada": {
        "voice": "en-US-AnaNeural",
        "pitch_hz": -3,
        "rate_offset": -4,
        "name": "Lucy Kushinada",
        "persona": "Netrunner Prodigy / Intimate Whisper",
        "quote": "I don't look back anymore. Keep your head down, David.",
    },
    "david-martinez": {
        "voice": "en-US-BrianNeural",
        "pitch_hz": 4,
        "rate_offset": 6,
        "name": "David Martinez",
        "persona": "Santo Domingo Solo / Streetkid Energetic",
        "quote": "I'm built different, choomba. Fast as light.",
    },
    "v-male": {
        "voice": "en-US-GuyNeural",
        "pitch_hz": -6,
        "rate_offset": 3,
        "name": "V (Male)",
        "persona": "Mercenary Solo / Gritty Raspy Baritone",
        "quote": "Night City always collects its debt. Let's make this quick.",
    },
    "v-female": {
        "voice": "en-US-AriaNeural",
        "pitch_hz": -3,
        "rate_offset": 4,
        "name": "V (Female)",
        "persona": "Tactical Solo / Crisp Dynamic Sharp",
        "quote": "Eyes sharp, weapons hot. Let's see what they got.",
    },
}

# Compatibility voice map
VOICE_MAP_EDGE = {
    k: v["voice"] for k, v in CHARACTER_AUDIO_PROFILES.items()
}

# Fallback voice map
DEFAULT_EDGE_VOICE = "en-US-BrianNeural"


async def synthesize_speech_edge(
    text: str,
    voice_id: str = "david-martinez",
    cadence: float = 1.0,
) -> Optional[bytes]:
    """Synthesize speech audio using edge-tts with authentic Cyberpunk character prosody."""
    clean_text = text.strip()
    if not clean_text:
        return None

    # Unknown voice ids fall back to the default voice with neutral prosody.
    profile = CHARACTER_AUDIO_PROFILES.get(voice_id)
    edge_voice = profile["voice"] if profile is not None else DEFAULT_EDGE_VOICE

    # Cadence combined with character rate offset
    base_rate_offset = profile["rate_offset"] if profile is not None else 0
    cadence_offset = int(round((cadence - 1.0) * 100))
    total_rate = base_rate_offset + cadence_offset
    rate_str = f"{total_rate:+d}%"

    # Edge-TTS regex pitch format strictly requires ^[+-]\d+Hz$
    pitch_hz = profile["pitch_hz"] if profile is not None else 0
    pitch_str = f"{pitch_hz:+d}Hz"

    try:
        import edge_tts
        communicate = edge_tts.Communicate(
            clean_text,
            edge_voice,
            rate=rate_str,
            pitch=pitch_str,
        )
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        if audio_chunks:
            return b"".join(audio_chunks)
    except Exception as exc:  # noqa: BLE001  # S3.3 RC4
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
                except Exception:  # noqa: BLE001, S110  # S3.3 RC4
                    pass

    except Exception as exc:  # noqa: BLE001  # S3.3 RC4
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
