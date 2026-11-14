"""Speech synthesis for the narration loop: one call that runs the configured TTS chain.

The chain (`voice_agent.tts.backends`) tries the primary backend, then the other one. When every backend fails the caller gets
(None, ""), so the narration skips the send and still advances its cursor.

# removed in S3.3 when narration takes the TTS chain directly
"""

from voice_agent.settings import load_settings
from voice_agent.tts.backends import ChainedTts, build_tts

_FORMAT_BY_MIME = {"audio/mpeg": "mp3", "audio/wav": "wav"}

# Built on first use, so importing the module loads no TTS engine and reads no settings.
_chain: ChainedTts | None = None


def _tts_chain() -> ChainedTts:
    global _chain
    if _chain is None:
        _chain = build_tts(load_settings().tts_primary)
    return _chain


async def synthesize_sentence(
    text: str,
    voice_id: str = "david-martinez",
    cadence: float = 1.0,
) -> tuple[bytes | None, str]:
    """Returns (audio bytes, "mp3" or "wav"), or (None, "") when no backend produced audio."""
    audio = await _tts_chain().synthesize(text, voice_id, cadence)
    if audio is None:
        return None, ""
    return audio.data, _FORMAT_BY_MIME[audio.mime]
