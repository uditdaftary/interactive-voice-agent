"""Central runtime settings: endpoint defaults and every narration timing knob.

Defaults equal the values that used to be hardcoded in the server modules, so
introducing Settings changes no behaviour. Tests build `Settings.fast()` to run
the narration loop without real sleeps.
"""

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from voice_agent.stt.backends import STT_BACKEND_NAMES
from voice_agent.tts.backends import TTS_BACKEND_NAMES

_FALSE_SPELLINGS = frozenset({"0", "false", "no"})


@dataclass(frozen=True)
class Settings:
    # LLM endpoint
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "voice-agent"
    llm_api_key: str = field(default="local", repr=False)  # a secret: keep it out of logged settings
    web_search_enabled: bool = True

    # Narration loop
    idle_poll_s: float = 0.5
    inter_sentence_pause_s: float = 0.35  # divided by cadence at the call site
    words_per_s: float = 2.5  # reading rate at cadence 1.0; a divisor, not a duration
    min_sentence_s: float = 1.2

    # Reply playback and turn-taking after a barge-in answer
    min_reply_playback_s: float = 2.0
    reply_words_per_s: float = 2.2  # divisor, like words_per_s
    followup_window_s: float = 2.5
    followup_poll_s: float = 0.1
    preroll_pause_s: float = 0.8
    unrecognized_pause_s: float = 2.0

    @property
    def followup_poll_count(self) -> int:
        """Polls in the follow-up window. 2.5 s / 0.1 s gives the legacy 25."""
        if self.followup_poll_s <= 0:
            return 0
        return round(self.followup_window_s / self.followup_poll_s)

    @classmethod
    def fast(cls) -> "Settings":
        """Every wait collapses to zero. Rates become infinite so derived durations are exactly 0."""
        return cls(
            idle_poll_s=0.0,
            inter_sentence_pause_s=0.0,
            words_per_s=float("inf"),
            min_sentence_s=0.0,
            min_reply_playback_s=0.0,
            reply_words_per_s=float("inf"),
            followup_window_s=0.0,
            followup_poll_s=0.0,
            preroll_pause_s=0.0,
            unrecognized_pause_s=0.0,
        )

    # STT: Whisper's initial prompt, which biases decoding toward names the user cares about. Empty means none.
    stt_initial_prompt: str = ""

    # TTS: the backend tried first; the other one is the fallback. "edge", "pyttsx3", or the optional "pocket".
    tts_primary: str = "edge"

    # STT: "whisper" (default) or the optional "moonshine" worker, and where its model files are.
    stt_backend: str = "whisper"
    moonshine_model_dir: str = "models/stt/moonshine-tiny-en-int8"


def _one_of(env_name: str, raw: str | None, default: str, names: tuple[str, ...]) -> str:
    """Case-insensitive, stored lower case. Any other name fails here, at load, not at the first use."""
    if raw is None:
        return default
    name = raw.strip().lower()
    if name not in names:
        raise ValueError(f"{env_name} must be one of {', '.join(names)}; got {raw!r}")
    return name


def _env_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_SPELLINGS


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `VOICE_AGENT_*` variables; anything unset keeps its default."""
    source = os.environ if env is None else env
    defaults = Settings()
    return dataclasses.replace(
        defaults,
        llm_base_url=source.get("VOICE_AGENT_LLM_BASE_URL", defaults.llm_base_url),
        llm_model=source.get("VOICE_AGENT_LLM_MODEL", defaults.llm_model),
        llm_api_key=source.get("VOICE_AGENT_LLM_API_KEY", defaults.llm_api_key),
        web_search_enabled=_env_bool(source.get("VOICE_AGENT_WEB_SEARCH"), defaults.web_search_enabled),
        stt_initial_prompt=source.get("VOICE_AGENT_STT_INITIAL_PROMPT", defaults.stt_initial_prompt),
        tts_primary=_one_of("VOICE_AGENT_TTS_PRIMARY", source.get("VOICE_AGENT_TTS_PRIMARY"), defaults.tts_primary, TTS_BACKEND_NAMES),
        stt_backend=_one_of("VOICE_AGENT_STT_BACKEND", source.get("VOICE_AGENT_STT_BACKEND"), defaults.stt_backend, STT_BACKEND_NAMES),
        moonshine_model_dir=source.get("VOICE_AGENT_MOONSHINE_MODEL_DIR", defaults.moonshine_model_dir),
    )
