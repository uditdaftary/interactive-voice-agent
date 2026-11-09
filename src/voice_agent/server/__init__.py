"""Unified FastAPI Server and WebSocket Protocol Engine package.

Exposes application factory, WebSocket router, connection manager, and
typed protocol message models.
"""

from voice_agent.server.app import app, create_app, run
from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.schemas import (
    BargeInEventMessage,
    BargeInTriggerMessage,
    CursorUpdateMessage,
    PlaybackControlMessage,
    PromptEnvelopeMessage,
    SelectVoiceMessage,
    ServerMessage,
    SetCadenceMessage,
    SetMicSourceMessage,
    StateSyncMessage,
    TelemetryMessage,
    VADEventMessage,
    parse_client_message,
)
from voice_agent.server.websocket import router as websocket_router

__all__ = [
    "app",
    "create_app",
    "run",
    "ConnectionManager",
    "default_connection_manager",
    "websocket_router",
    "SelectVoiceMessage",
    "SetMicSourceMessage",
    "BargeInTriggerMessage",
    "SetCadenceMessage",
    "PlaybackControlMessage",
    "StateSyncMessage",
    "VADEventMessage",
    "BargeInEventMessage",
    "CursorUpdateMessage",
    "PromptEnvelopeMessage",
    "TelemetryMessage",
    "ServerMessage",
    "parse_client_message",
]
