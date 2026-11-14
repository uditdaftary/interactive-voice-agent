"""Central runtime settings: endpoint defaults and every narration timing knob.

Defaults equal the values that used to be hardcoded in the server modules, so
introducing Settings changes no behaviour. Tests build `Settings.fast()` to run
the narration loop without real sleeps.
"""

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

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
    )
