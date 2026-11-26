"""WebSocket Protocol Message Schemas for Interactive Voice Agent HUD.

Provides typed Pydantic models for client actions and server events,
strictly conforming to Section 4.2 of PROJECT.md.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
import json
from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from voice_agent.orchestrator.state_machine import SessionStateName

# ============================================================================
# Shared field constraints (REST bodies and WebSocket client messages)
# ============================================================================

# An identifier shape, not an enumerated set: the voice registry arrives in sprint S4.2.
VOICE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
MAX_QUERY_CHARS = 2000
# The HUD's default "builtin" provider sends this placeholder instead of a URL (removed with that provider in S3.1).
BUILTIN_ENDPOINT = "in-process"

MicSource = Literal["browser", "host"]
PlaybackAction = Literal["resume", "start", "pause", "stop", "next", "prev"]

_HTTP_URL = TypeAdapter(AnyHttpUrl)


def _check_base_url(value: str) -> str:
    value = value.strip()
    if value == BUILTIN_ENDPOINT:
        return value
    try:
        _HTTP_URL.validate_python(value)
    except ValidationError:
        # Do not chain: the original message echoes the rejected input.
        raise ValueError("must be an http(s) URL") from None
    return value


VoiceId = Annotated[str, StringConstraints(pattern=VOICE_ID_PATTERN)]
Cadence = Annotated[float, Field(ge=0.1, le=3.0, description="Playback speed cadence multiplier (e.g. 0.8, 1.0, 1.2)")]
QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)]
BaseUrl = Annotated[str, AfterValidator(_check_base_url)]
ApiKey = Annotated[str, StringConstraints(max_length=512)]
ProviderName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_-]{1,32}$")]
ModelName = Annotated[str, StringConstraints(min_length=1, max_length=128)]


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
    voice_id: VoiceId = Field(
        ...,
        description="Character voice identifier (e.g. david-martinez, lucy-kushinada, v-male, v-female, johnny-silverhand)",
    )


class SetMicSourceMessage(BaseProtocolMessage):
    """Client request to switch microphone ingestion source."""

    type: Literal["set_mic_source"] = "set_mic_source"
    source: MicSource = Field(
        ...,
        description="Microphone source: 'browser' (Web Audio PCM) or 'host' (sounddevice)",
    )


class BargeInTriggerMessage(BaseProtocolMessage):
    """Client request to trigger an instantaneous barge-in interruption."""

    type: Literal["barge_in_trigger"] = "barge_in_trigger"


class SetCadenceMessage(BaseProtocolMessage):
    """Client request to update playback reading speed cadence multiplier."""

    type: Literal["set_cadence"] = "set_cadence"
    cadence: Cadence


class PlaybackControlMessage(BaseProtocolMessage):
    """Client request to control narration playback state."""

    type: Literal["playback_control"] = "playback_control"
    action: PlaybackAction = Field(
        ...,
        description="Playback action: 'pause', 'resume', 'next', 'prev'",
    )


class QueryInterruptionMessage(BaseProtocolMessage):
    """User speech query or text prompt sent to interrupt narration with a question."""

    type: Literal["query_interruption"] = "query_interruption"
    query: QueryText = Field(
        ...,
        description="Question or prompt asked by the user during interruption",
    )


class UpdateLLMConfigMessage(BaseProtocolMessage):
    """Update local or remote LLM endpoint provider and web search configuration."""

    type: Literal["update_config"] = "update_config"
    provider: Optional[ProviderName] = Field(default=None, description="LLM provider ('ollama', 'openai', 'groq', 'gemini')")
    base_url: Optional[BaseUrl] = Field(default=None, description="API Base URL (e.g. 'http://localhost:11434')")
    api_key: Optional[ApiKey] = Field(default=None, description="API Key for remote LLM provider")
    model: Optional[ModelName] = Field(default=None, description="Model identifier (e.g. 'llama3.2', 'gpt-4o-mini')")
    web_search_enabled: Optional[bool] = Field(default=None, description="Enable live internet web search grounding")


class PlaybackDoneMessage(BaseProtocolMessage):
    """The browser finished playing a clip the server announced with `audio_segment`."""

    type: Literal["playback_done"] = "playback_done"
    segment_id: Annotated[int, Field(strict=True, ge=0, description="The `segment_id` of the clip that finished")]
    generation: Annotated[int, Field(strict=True, ge=0, description="The `generation` that clip was sent with")]


class VoiceEndMessage(BaseProtocolMessage):
    """Client notification that user voice utterance has concluded."""

    type: Literal["voice_end"] = "voice_end"
    fallback_query: Optional[str] = Field(default=None, description="Optional client preview transcript if available")


# ============================================================================
# REST request bodies
# ============================================================================


class ConfigUpdateRequest(BaseModel):
    """POST /api/config. Every field is optional; a request with any invalid field applies nothing."""

    model_config = ConfigDict(extra="forbid")

    base_url: Optional[BaseUrl] = None
    api_key: Optional[ApiKey] = None
    provider: Optional[ProviderName] = None
    model: Optional[ModelName] = None
    web_search_enabled: Optional[bool] = None
    active_voice: Optional[VoiceId] = None
    cadence: Optional[Cadence] = None
    mic_source: Optional[MicSource] = None

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_aliases(cls, data: Any) -> Any:
        """The HUD still sends `endpoint` and `model_name` next to `base_url` and `model`."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for legacy, current in (("endpoint", "base_url"), ("model_name", "model")):
            if legacy in data:
                legacy_value = data.pop(legacy)
                if data.get(current) is None:
                    data[current] = legacy_value
        return data


class AskRequest(BaseModel):
    """POST /api/ask."""

    query: QueryText


class PlaybackControlRequest(BaseModel):
    """POST /api/narration/control."""

    action: PlaybackAction = "resume"


class TextDocumentRequest(BaseModel):
    """POST /api/upload-document with a JSON body instead of a file."""

    text: str = Field(min_length=1)
    title: str = Field(default="Custom Document", max_length=200)


ClientMessage = Union[
    SelectVoiceMessage,
    SetMicSourceMessage,
    BargeInTriggerMessage,
    SetCadenceMessage,
    PlaybackControlMessage,
    QueryInterruptionMessage,
    UpdateLLMConfigMessage,
    VoiceEndMessage,
    PlaybackDoneMessage,
]


# ============================================================================
# Server -> Client Event Models
# ============================================================================


class StateSyncMessage(BaseProtocolMessage):
    """Server state synchronization snapshot sent on handshake and state changes."""

    type: Literal["state_sync"] = "state_sync"
    state: SessionStateName = Field(
        default="PAUSED",
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
    latency_ms: Optional[float] = Field(
        default=None,
        description="Latest measured round-trip or drain latency in milliseconds; null when nothing was measured",
    )
    mic_owner: bool = Field(
        default=True,
        description="True when this connection owns the microphone and the audio; false for every other tab",
    )
    llm_status: str = Field(
        default="ok",
        description="How the last model request ended: ok, offline, auth, error, interrupted; none when no model is configured",
    )
    llm_provider: str = Field(default="", description="Provider name of the language model endpoint; none is the no-model mode")
    tts_status: str = Field(
        default="ok",
        description="ok, fallback:<backend> when a later backend spoke, unavailable when every backend failed",
    )
    stt_status: str = Field(
        default="ok",
        description="ok, loading while the recognizer starts, unavailable when it cannot run, rejected when the last audio was not speech",
    )
    web_speech_preview: bool = Field(
        default=False,
        description="True when the browser's Web Speech preview (which sends audio to the browser vendor) may be used",
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
    latency_ms: Optional[float] = Field(
        default=None,
        description="Measured time to stop the audio, in milliseconds; null when nothing was measured",
    )
    generation: int = Field(
        ...,
        description="Monotonically incremented generation counter invalidating stale audio",
    )
    flushed_samples: Optional[int] = Field(
        default=None,
        description="Number of audio samples purged from the playback buffer; null when nothing was measured",
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


class UserSpeechTranscriptMessage(BaseProtocolMessage):
    """Real-time user speech transcript event broadcast to HUD."""

    type: Literal["user_speech_transcript"] = "user_speech_transcript"
    transcript: str = Field(
        ...,
        description="Transcribed user speech query or command",
    )
    is_final: bool = Field(
        default=True,
        description="True if utterance transcription is finalized",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Transcription execution latency in milliseconds",
    )


class AudioSegmentMessage(BaseProtocolMessage):
    """Announces the audio blob that follows it on the socket, which only the owner connection receives."""

    type: Literal["audio_segment"] = "audio_segment"
    segment_id: int = Field(..., description="Echoed back in `playback_done` when the clip has finished playing")
    generation: int = Field(..., description="A clip older than the newest `audio_stop` or `barge_in` must not play")
    gap_s: float = Field(default=0.0, description="Silence the browser adds before the clip starts")


class AudioStopMessage(BaseProtocolMessage):
    """Audio was stopped (a pause, a navigation, a new document, a new owner): drop every clip older than `generation`."""

    type: Literal["audio_stop"] = "audio_stop"
    generation: int = Field(..., description="Clips with a smaller generation are dead, queued, decoding or playing")


class ErrorMessage(BaseProtocolMessage):
    """Reply to a client message that was rejected. The session stays open."""

    type: Literal["error"] = "error"
    detail: str = Field(
        ...,
        description="Short reason, never an echo of the client's payload",
    )


class ReplyErrorMessage(BaseProtocolMessage):
    """A reply could not be produced or spoken. The session carries on; nothing else about the failure is exposed."""

    type: Literal["reply_error"] = "reply_error"
    detail: str = Field(
        ...,
        description="The exception type name only, never its message",
    )


ServerMessage = Union[
    StateSyncMessage,
    VADEventMessage,
    BargeInEventMessage,
    CursorUpdateMessage,
    PromptEnvelopeMessage,
    TelemetryMessage,
    UserSpeechTranscriptMessage,
    ReplyErrorMessage,
    AudioSegmentMessage,
    AudioStopMessage,
    ErrorMessage,
]


# ============================================================================
# Parser Utility
# ============================================================================


class InvalidClientMessage(ValueError):
    """A client message that cannot be dispatched. `str(exc)` is a short reason safe to send back and to log."""


def _first_validation_problem(msg_type: str, error: ValidationError) -> InvalidClientMessage:
    # loc and msg only: `input` (and msg for some validators) would echo what the client sent.
    problem = error.errors(include_url=False, include_context=False, include_input=False)[0]
    field = ".".join(str(part) for part in problem["loc"]) or "message"
    return InvalidClientMessage(f"invalid {msg_type}: {field}: {problem['msg']}")


def parse_client_message(raw: Union[str, bytes, Dict[str, Any]]) -> ClientMessage:
    """Parse raw client message payload into a typed client message model.

    Supports canonical snake_case types as well as common uppercase aliases.
    Raises InvalidClientMessage when the payload is not JSON, not an object, has an unknown type, or
    fails a field constraint. The reason never contains the payload text.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise InvalidClientMessage("message is not valid UTF-8") from None

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise InvalidClientMessage("message is not valid JSON") from None
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        raise InvalidClientMessage("message must be a JSON object")

    if not isinstance(payload, dict):
        raise InvalidClientMessage("message must be a JSON object")

    msg_type = str(payload.get("type", "")).strip().lower()

    try:
        if msg_type in ("select_voice", "selectvoice"):
            voice_id = payload.get("voice_id")
            if voice_id is None:
                voice_id = payload.get("voice", "")
            return SelectVoiceMessage.model_validate({"voice_id": voice_id})

        if msg_type in ("set_mic_source", "switch_mic_source", "setmicsource"):
            source = payload.get("source") or payload.get("mic_source") or "browser"
            return SetMicSourceMessage.model_validate({"source": source})

        if msg_type in ("barge_in_trigger", "barge_in", "bargeintrigger"):
            return BargeInTriggerMessage()

        if msg_type in ("set_cadence", "setcadence"):
            return SetCadenceMessage.model_validate({"cadence": payload.get("cadence", 1.0)})

        if msg_type in ("playback_control", "playbackcontrol", "start_narration", "pause_narration", "resume_narration"):
            action = payload.get("action")
            if not action:
                action = {"start_narration": "resume", "pause_narration": "pause", "resume_narration": "resume"}.get(msg_type)
            if not action:
                raise InvalidClientMessage("invalid playback_control: action: Field required")
            return PlaybackControlMessage.model_validate({"action": action})

        if msg_type in ("query_interruption", "user_query", "ask_question", "query"):
            return QueryInterruptionMessage.model_validate({"query": payload.get("query") or payload.get("text") or ""})

        if msg_type in ("voice_end", "voiceend", "speech_end", "speechend", "voice_stop"):
            return VoiceEndMessage()

        if msg_type == "playback_done":
            return PlaybackDoneMessage.model_validate(
                {"segment_id": payload.get("segment_id"), "generation": payload.get("generation")}
            )

        if msg_type in ("update_config", "set_config", "configure_llm"):
            return UpdateLLMConfigMessage.model_validate(
                {
                    key: payload.get(key)
                    for key in ("provider", "base_url", "api_key", "model", "web_search_enabled")
                }
            )

    except ValidationError as error:
        raise _first_validation_problem(msg_type, error) from None

    raise InvalidClientMessage("unknown message type")
