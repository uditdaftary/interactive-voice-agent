"""WebSocket Router and Protocol Handler for Interactive Voice Agent HUD.

Exposes bi-directional streaming endpoint at `/ws` multiplexing JSON control / telemetry
and raw binary audio frames.
"""

import asyncio
import time
from typing import Optional, Any
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
    QueryInterruptionMessage,
    SelectVoiceMessage,
    SetCadenceMessage,
    SetMicSourceMessage,
    UpdateLLMConfigMessage,
    UserSpeechTranscriptMessage,
    VoiceEndMessage,
    parse_client_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


async def process_stt_utterance(
    utterance: bytes,
    manager: ConnectionManager,
    stt_engine: Any,
    engine: Optional[Any] = None,
) -> Optional[str]:
    """Transcribe accumulated utterance audio, broadcast transcript, and trigger interruption."""
    if not utterance:
        return None
    start_t = time.perf_counter()
    try:
        transcript = stt_engine.transcribe(utterance)
    except Exception as exc:
        logger.warning(f"STT transcription failed: {exc}")
        return None

    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    if transcript and transcript.strip():
        clean_text = transcript.strip()
        msg = UserSpeechTranscriptMessage(
            transcript=clean_text,
            is_final=True,
            latency_ms=latency_ms,
        )
        await manager.broadcast_json(msg)
        if engine is not None and hasattr(engine, "handle_user_interruption"):
            asyncio.create_task(engine.handle_user_interruption(clean_text))
        return clean_text
    return None


async def handle_client_text_message(
    text: str,
    websocket: WebSocket,
    manager: ConnectionManager,
    engine: Optional[Any] = None,
    stt_engine: Optional[Any] = None,
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
        if engine is not None:
            engine.pause_narration()
        barge_in_event = manager.trigger_barge_in()
        await manager.broadcast_json(barge_in_event)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, SetCadenceMessage):
        manager.set_cadence(msg.cadence)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, PlaybackControlMessage):
        if msg.action in ("pause", "stop"):
            manager.set_state("PAUSED")
            if engine is not None:
                engine.pause_narration()
        elif msg.action in ("resume", "start"):
            manager.set_state("NARRATING")
        elif msg.action == "next":
            if engine is not None and getattr(engine, "sentences", None):
                engine.current_sentence_idx = (engine.current_sentence_idx + 1) % max(1, len(engine.sentences))
                engine.current_word_idx = 0
                manager.update_cursor(engine.current_sentence_idx, 0)
        elif msg.action == "prev":
            if engine is not None and getattr(engine, "sentences", None):
                engine.current_sentence_idx = (engine.current_sentence_idx - 1 + len(engine.sentences)) % max(1, len(engine.sentences))
                engine.current_word_idx = 0
                manager.update_cursor(engine.current_sentence_idx, 0)

        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

        if msg.action in ("resume", "start") and engine is not None:
            engine.start_narration()

    elif isinstance(msg, QueryInterruptionMessage):
        if engine is not None:
            asyncio.create_task(engine.handle_user_interruption(msg.query))
        else:
            barge_in_event = manager.trigger_barge_in()
            await manager.broadcast_json(barge_in_event)
            manager.set_state("REPLYING")
            await manager.broadcast_json(manager.get_state_sync_message())

    elif isinstance(msg, UpdateLLMConfigMessage):
        if engine is not None and getattr(engine, "llm", None) is not None:
            engine.llm.update_config(
                base_url=msg.base_url,
                api_key=msg.api_key,
                provider=msg.provider,
                model=msg.model,
                web_search_enabled=msg.web_search_enabled,
            )
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, VoiceEndMessage):
        if manager.state in ("LISTENING", "BARGE_IN"):
            acc = getattr(manager, "accumulator", None)
            active_stt = stt_engine or getattr(manager, "stt_engine", None)
            utterance = acc.flush() if acc is not None else None
            clean_text = None
            if utterance and active_stt is not None:
                clean_text = await process_stt_utterance(utterance, manager, active_stt, engine)

            if not clean_text and getattr(msg, "fallback_query", None):
                fb = msg.fallback_query.strip()
                if fb:
                    out_msg = UserSpeechTranscriptMessage(
                        transcript=fb,
                        is_final=True,
                        latency_ms=0.0,
                    )
                    await manager.broadcast_json(out_msg)
                    if engine is not None and hasattr(engine, "handle_user_interruption"):
                        asyncio.create_task(engine.handle_user_interruption(fb))


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
    engine = getattr(websocket.app.state, "narration_engine", None)
    stt_engine = getattr(websocket.app.state, "stt_engine", None)

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
                await handle_client_text_message(raw_msg["text"], websocket, manager, engine, stt_engine)

            # Handle binary frame (24kHz / 16kHz 16-bit PCM from browser mic)
            elif "bytes" in raw_msg and raw_msg["bytes"] is not None:
                pcm_data = raw_msg["bytes"]
                manager.on_audio_frame(pcm_data)
                if manager.state in ("LISTENING", "BARGE_IN"):
                    acc = getattr(manager, "accumulator", None)
                    if acc is not None:
                        utterance = acc.feed(pcm_data)
                        if utterance:
                            active_stt = stt_engine or getattr(manager, "stt_engine", None)
                            if active_stt is not None:
                                await process_stt_utterance(utterance, manager, active_stt, engine)

    except WebSocketDisconnect:
        logger.info("Client disconnected via WebSocketDisconnect.")
    except Exception as exc:
        logger.warning(f"WebSocket session terminated with exception: {exc}")
    finally:
        manager.disconnect(websocket)
