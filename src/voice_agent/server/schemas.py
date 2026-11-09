"""WebSocket Protocol Message Schemas for Interactive Voice Agent HUD.

Provides typed Pydantic models for client actions and server events,
strictly conforming to Section 4.2 of PROJECT.md.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import json
from pydantic import BaseModel, Field, ConfigDict


class BaseProtocolMessage(BaseModel):
    """Base model for all WebSocket protocol messages."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def to_dict(self) -> Dict[str, Any]:
        """Backwards-compatible dict export."""
        return self.model_dump()

    def to_json(self) -> str:
        """Backwards-compatible JSON string export."""
        return self.model_dump_json()


# ============================================================================
# Client -> Server Action Models
# ============================================================================


class SelectVoiceMessage(BaseProtocolMessage):
    """Client request to switch character persona voice."""

    type: Literal["select_voice"] = "select_voice"
    voice_id: str = Field(
        ...,
        description="Character voice identifier (e.g. david-martinez, lucy-kushinada, v-male, v-female, johnny-silverhand)",
    )


class SetMicSourceMessage(BaseProtocolMessage):
    """Client request to switch microphone ingestion source."""

    type: Literal["set_mic_source"] = "set_mic_source"
    source: str = Field(
        ...,
        description="Microphone source: 'browser' (Web Audio PCM) or 'host' (sounddevice) or 'mock'",
    )


class BargeInTriggerMessage(BaseProtocolMessage):
    """Client request to trigger an instantaneous barge-in interruption."""

    type: Literal["barge_in_trigger"] = "barge_in_trigger"


class SetCadenceMessage(BaseProtocolMessage):
    """Client request to update playback reading speed cadence multiplier."""

    type: Literal["set_cadence"] = "set_cadence"
    cadence: float = Field(
        ...,
        ge=0.1,
        le=3.0,
        description="Playback speed cadence multiplier (e.g. 0.8, 1.0, 1.2)",
    )


class PlaybackControlMessage(BaseProtocolMessage):
    """Client request to control narration playback state."""

    type: Literal["playback_control"] = "playback_control"
    action: str = Field(
        ...,
        description="Playback action: 'pause', 'resume', 'next', 'prev'",
    )


class QueryInterruptionMessage(BaseProtocolMessage):
    """User speech query or text prompt sent to interrupt narration with a question."""

    type: Literal["query_interruption"] = "query_interruption"
    query: str = Field(
        ...,
        description="Question or prompt asked by the user during interruption",
    )


class UpdateLLMConfigMessage(BaseProtocolMessage):
    """Update local or remote LLM endpoint provider and web search configuration."""

    type: Literal["update_config"] = "update_config"
    provider: Optional[str] = Field(default=None, description="LLM provider ('ollama', 'openai', 'groq', 'gemini')")
    base_url: Optional[str] = Field(default=None, description="API Base URL (e.g. 'http://localhost:11434')")
    api_key: Optional[str] = Field(default=None, description="API Key for remote LLM provider")
    model: Optional[str] = Field(default=None, description="Model identifier (e.g. 'llama3.2', 'gpt-4o-mini')")
    web_search_enabled: Optional[bool] = Field(default=None, description="Enable live internet web search grounding")


ClientMessage = Union[
    SelectVoiceMessage,
    SetMicSourceMessage,
    BargeInTriggerMessage,
    SetCadenceMessage,
    PlaybackControlMessage,
    QueryInterruptionMessage,
    UpdateLLMConfigMessage,
]


# ============================================================================
# Server -> Client Event Models
# ============================================================================


class StateSyncMessage(BaseProtocolMessage):
    """Server state synchronization snapshot sent on handshake and state changes."""

    type: Literal["state_sync"] = "state_sync"
    state: str = Field(
        default="NARRATING",
        description="Current agent state (NARRATING, LISTENING, REPLYING, PAUSED)",
    )
    active_voice: str = Field(
        default="david-martinez",
        description="Currently active character persona voice",
    )
    mic_source: str = Field(
        default="browser",
        description="Currently active microphone ingestion source (browser, host, mock)",
    )
    cadence: float = Field(
        default=1.0,
        description="Current reading speed multiplier",
    )
    sentence_idx: int = Field(
        default=0,
        description="Current sentence index in narrative text",
    )
    word_offset: int = Field(
        default=0,
        description="Active word index offset within current sentence",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Latest measured round-trip or drain latency in milliseconds",
    )


class VADEventMessage(BaseProtocolMessage):
    """Voice Activity Detection telemetry event emitted for each 30ms frame."""

    type: Literal["vad_event"] = "vad_event"
    speech_active: bool = Field(
        ...,
        description="True if user speech is currently detected in this 30ms window",
    )
    dbfs: float = Field(
        ...,
        description="Observed signal level in dBFS",
    )
    noise_floor_dbfs: float = Field(
        ...,
        description="Current dynamic background noise floor in dBFS",
    )
    threshold_dbfs: float = Field(
        ...,
        description="Current adaptive speech decision threshold in dBFS",
    )


class BargeInEventMessage(BaseProtocolMessage):
    """Sub-50ms barge-in interruption event dispatched to client HUD."""

    type: Literal["barge_in"] = "barge_in"
    latency_ms: float = Field(
        ...,
        description="Atomic buffer drain execution latency in milliseconds (< 0.05ms target)",
    )
    generation: int = Field(
        ...,
        description="Monotonically incremented generation counter invalidating stale audio",
    )
    flushed_samples: int = Field(
        ...,
        description="Number of audio samples purged from playback ring buffer",
    )


class CursorUpdateMessage(BaseProtocolMessage):
    """Word-level reading cursor update synchronized with audio DAC playhead."""

    type: Literal["cursor_update"] = "cursor_update"
    sentence_idx: int = Field(
        ...,
        description="Current zero-based index of sentence being narrated",
    )
    word_offset: int = Field(
        ...,
        description="Zero-based index of active word token within current sentence",
    )
    char_offset: int = Field(
        default=0,
        description="Character offset in full narrative text",
    )
    total_sentences: int = Field(
        ...,
        description="Total number of sentences in the narrative document",
    )
    sentence_text: str = Field(
        ...,
        description="Full text string of the current sentence",
    )


class PromptEnvelopeMessage(BaseProtocolMessage):
    """Q&A prompt context envelope and dialogue telemetry."""

    type: Literal["prompt_envelope"] = "prompt_envelope"
    state: str = Field(
        ...,
        description="Dialogue state ('LISTENING', 'REPLYING', etc.)",
    )
    context: str = Field(
        default="",
        description="Sanitized background narrative context provided to LLM",
    )
    query: str = Field(
        default="",
        description="User speech query transcribed from voice input",
    )
    reply: str = Field(
        default="",
        description="Streaming or completed LLM response text",
    )
    pre_roll_words: int = Field(
        default=3,
        description="Number of pre-roll words used for seamless resumption",
    )


class TelemetryMessage(BaseProtocolMessage):
    """Real-time audio DSP and buffer telemetry for HUD visualizers."""

    type: Literal["telemetry"] = "telemetry"
    buffer_samples: int = Field(
        ...,
        description="Current count of queued unread audio samples in ring buffer",
    )
    buffer_capacity: int = Field(
        ...,
        description="Total capacity of ring buffer in samples (e.g. 240,000 for 10s @ 24kHz)",
    )
    drain_latency_ms: float = Field(
        ...,
        description="Latency of last atomic ring buffer drain in milliseconds",
    )
    rms_db: float = Field(
        ...,
        description="Root mean square audio energy level in dBFS",
    )
    fft_bins: List[float] = Field(
        default_factory=list,
        description="16-band normalized frequency spectrum magnitudes for HUD visualizer",
    )


ServerMessage = Union[
    StateSyncMessage,
    VADEventMessage,
    BargeInEventMessage,
    CursorUpdateMessage,
    PromptEnvelopeMessage,
    TelemetryMessage,
]


# ============================================================================
# Parser Utility
# ============================================================================


def parse_client_message(raw: Union[str, bytes, Dict[str, Any]]) -> Optional[ClientMessage]:
    """Parse raw client message payload into a typed client message model.

    Supports canonical snake_case types as well as common uppercase aliases.
    Returns None if payload cannot be parsed or message type is unrecognized.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        return None

    if not isinstance(payload, dict):
        return None

    msg_type = str(payload.get("type", "")).strip().lower()

    try:
        if msg_type in ("select_voice", "selectvoice"):
            voice_id = payload.get("voice_id")
            if voice_id is None:
                voice_id = payload.get("voice", "")
            return SelectVoiceMessage(voice_id=str(voice_id))

        if msg_type in ("set_mic_source", "switch_mic_source", "setmicsource"):
            source = payload.get("source") or payload.get("mic_source") or "browser"
            return SetMicSourceMessage(source=str(source))

        if msg_type in ("barge_in_trigger", "barge_in", "bargeintrigger"):
            return BargeInTriggerMessage()

        if msg_type in ("set_cadence", "setcadence"):
            cadence_val = float(payload.get("cadence", 1.0))
            return SetCadenceMessage(cadence=cadence_val)

        if msg_type in ("playback_control", "playbackcontrol", "start_narration", "pause_narration", "resume_narration"):
            action = payload.get("action")
            if not action:
                if msg_type == "start_narration":
                    action = "resume"
                elif msg_type == "pause_narration":
                    action = "pause"
                elif msg_type == "resume_narration":
                    action = "resume"
                else:
                    return None
            return PlaybackControlMessage(action=str(action))

        if msg_type in ("query_interruption", "user_query", "ask_question", "query"):
            query = str(payload.get("query") or payload.get("text") or "").strip()
            if query:
                return QueryInterruptionMessage(query=query)
            return None

        if msg_type in ("update_config", "set_config", "configure_llm"):
            return UpdateLLMConfigMessage(
                provider=payload.get("provider"),
                base_url=payload.get("base_url"),
                api_key=payload.get("api_key"),
                model=payload.get("model"),
                web_search_enabled=payload.get("web_search_enabled"),
            )

    except Exception:
        return None

    return None
