"""WebSocket Router and Protocol Handler for Interactive Voice Agent HUD.

Exposes bi-directional streaming endpoint at `/ws` multiplexing JSON control / telemetry
and raw binary PCM audio frames.
"""

from typing import Optional
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.schemas import (
    BargeInTriggerMessage,
    PlaybackControlMessage,
    SelectVoiceMessage,
    SetCadenceMessage,
    SetMicSourceMessage,
    parse_client_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


async def handle_client_text_message(
    text: str,
    websocket: WebSocket,
    manager: ConnectionManager,
) -> None:
    """Parse and dispatch an incoming client JSON action message."""
    msg = parse_client_message(text)
    if msg is None:
        logger.debug(f"Unrecognized or unparseable client payload: {text[:100]}")
        return

    if isinstance(msg, SelectVoiceMessage):
        manager.set_voice(msg.voice_id)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, SetMicSourceMessage):
        manager.set_mic_source(msg.source)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, BargeInTriggerMessage):
        barge_in_event = manager.trigger_barge_in()
        await manager.broadcast_json(barge_in_event)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, SetCadenceMessage):
        manager.set_cadence(msg.cadence)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, PlaybackControlMessage):
        if msg.action == "pause":
            manager.set_state("PAUSED")
        elif msg.action == "resume":
            manager.set_state("NARRATING")
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Bi-directional WebSocket streaming endpoint.

    On connect:
      1. Accepts connection.
      2. Sends initial state_sync handshake message.

    In loop:
      - Text frames: parsed and dispatched to update session state or trigger barge-in.
      - Binary frames: routed to audio ingestion sink (browser Web Audio PCM).

    On disconnect:
      - Traps WebSocketDisconnect cleanly.
      - Deregisters connection without dangling tasks or memory leaks.
    """
    manager: ConnectionManager = getattr(
        websocket.app.state, "connection_manager", default_connection_manager
    )

    await manager.connect(websocket)

    try:
        # Handshake: transmit initial state synchronization snapshot
        await manager.send_state_sync(websocket)

        while True:
            # Receive raw frame (multiplexes text and binary frames)
            raw_msg = await websocket.receive()
            msg_type = raw_msg.get("type")

            if msg_type == "websocket.disconnect":
                break

            # Handle text frame (JSON protocol actions)
            if "text" in raw_msg and raw_msg["text"] is not None:
                await handle_client_text_message(raw_msg["text"], websocket, manager)

            # Handle binary frame (24kHz 16-bit PCM from browser mic)
            elif "bytes" in raw_msg and raw_msg["bytes"] is not None:
                manager.on_audio_frame(raw_msg["bytes"])

    except WebSocketDisconnect:
        logger.info("Client disconnected via WebSocketDisconnect.")
    except Exception as exc:
        logger.warning(f"WebSocket session terminated with exception: {exc}")
    finally:
        manager.disconnect(websocket)
