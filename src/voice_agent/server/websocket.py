"""WebSocket Router and Protocol Handler for Interactive Voice Agent HUD.

Exposes bi-directional streaming endpoint at `/ws` multiplexing JSON control / telemetry
and raw binary audio frames.
"""

import asyncio
import time
from typing import Any, Mapping, Optional
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.security import origin_allowed
from voice_agent.server.schemas import (
    BargeInTriggerMessage,
    PlaybackControlMessage,
    QueryInterruptionMessage,
    SelectVoiceMessage,
    SetCadenceMessage,
    SetMicSourceMessage,
    UpdateLLMConfigMessage,
    InvalidClientMessage,
    UserSpeechTranscriptMessage,
    VoiceEndMessage,
    parse_client_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


async def _ensure_stt_engine(app: Any) -> Any:
    # Imported here because app.py imports this module's router at load time.
    from voice_agent.server.app import ensure_stt_engine

    return await ensure_stt_engine(app)


async def process_stt_utterance(
    utterance: bytes,
    manager: ConnectionManager,
    stt_engine: Any,
    engine: Optional[Any] = None,
    fallback_query: Optional[str] = None,
) -> Optional[str]:
    """Transcribe accumulated utterance audio, broadcast transcript, and trigger interruption."""
    if not utterance:
        if fallback_query and fallback_query.strip():
            clean_text = fallback_query.strip()
            logger.info("Empty utterance audio; routing fallback query: %s", clean_text)
            msg = UserSpeechTranscriptMessage(
                transcript=clean_text,
                is_final=True,
                latency_ms=0.0,
            )
            await manager.broadcast_json(msg)
            if engine is not None and hasattr(engine, "handle_user_interruption"):
                asyncio.create_task(engine.handle_user_interruption(clean_text))  # noqa: RUF006  # S2.1
            return clean_text
        return None

    start_t = time.perf_counter()
    try:
        transcript = await asyncio.to_thread(stt_engine.transcribe, utterance)
    except Exception as exc:  # noqa: BLE001  # S3.3 RC4
        logger.warning(f"STT transcription failed: {exc}")
        transcript = ""

    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
    clean_text = transcript.strip() if transcript else ""

    if not clean_text and fallback_query and fallback_query.strip():
        clean_text = fallback_query.strip()
        logger.info("Using fallback query following empty STT transcript: %s", clean_text)

    if clean_text:
        msg = UserSpeechTranscriptMessage(
            transcript=clean_text,
            is_final=True,
            latency_ms=latency_ms,
        )
        if engine is not None and hasattr(engine, "handle_user_interruption"):
            asyncio.create_task(engine.handle_user_interruption(clean_text))  # noqa: RUF006  # S2.1
        await manager.broadcast_json(msg)
        return clean_text
    else:
        logger.debug(
            "STT returned empty transcript for utterance (%d bytes, %.2f ms latency)",
            len(utterance),
            latency_ms,
        )
        msg = UserSpeechTranscriptMessage(
            transcript="",
            is_final=True,
            latency_ms=latency_ms,
        )
        await manager.broadcast_json(msg)
        if engine is not None and hasattr(engine, "handle_unrecognized_interruption"):
            asyncio.create_task(engine.handle_unrecognized_interruption())  # noqa: RUF006  # S2.1
        elif manager.state in ("LISTENING", "BARGE_IN"):
            manager.set_state("NARRATING")
            await manager.broadcast_json(manager.get_state_sync_message())

    return None


async def handle_client_text_message(
    text: str,
    websocket: WebSocket,
    manager: ConnectionManager,
    engine: Optional[Any] = None,
    stt_engine: Optional[Any] = None,
) -> None:
    """Parse and dispatch an incoming client JSON action message."""
    try:
        msg = parse_client_message(text)
    except InvalidClientMessage as exc:
        # The reason names the field, never the value, so neither the log nor the reply echoes the payload.
        logger.debug("Rejected client message: %s", exc)
        await manager.send_error(websocket, str(exc))
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
            asyncio.create_task(engine.handle_user_interruption(msg.query))  # noqa: RUF006  # S2.1
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
            utterance = acc.flush() if acc is not None else None
            fb = getattr(msg, "fallback_query", None)
            if utterance:
                active_stt = stt_engine or await _ensure_stt_engine(websocket.app)
                await process_stt_utterance(
                    utterance, manager, active_stt, engine, fallback_query=fb
                )
            elif fb and fb.strip():
                clean_fb = fb.strip()
                logger.info(f"Direct fallback query routing (no accumulated utterance): {clean_fb}")
                out_msg = UserSpeechTranscriptMessage(
                    transcript=clean_fb,
                    is_final=True,
                    latency_ms=0.0,
                )
                await manager.broadcast_json(out_msg)
                if engine is not None and hasattr(engine, "handle_user_interruption"):
                    asyncio.create_task(engine.handle_user_interruption(clean_fb))  # noqa: RUF006  # S2.1


async def _handle_frame(
    raw_msg: Mapping[str, Any],
    websocket: WebSocket,
    manager: ConnectionManager,
    engine: Optional[Any],
    stt_engine: Optional[Any],
) -> None:
    """Dispatch one received text or binary frame."""
    # Text frame (JSON protocol actions)
    if raw_msg.get("text") is not None:
        await handle_client_text_message(raw_msg["text"], websocket, manager, engine, stt_engine)

    # Binary frame (24kHz / 16kHz 16-bit PCM from browser mic)
    elif raw_msg.get("bytes") is not None:
        pcm_data = raw_msg["bytes"]
        manager.on_audio_frame(pcm_data)
        acc = getattr(manager, "accumulator", None)
        if acc is not None:
            if manager.state in ("LISTENING", "BARGE_IN"):
                acc.arm()
                utterance = acc.feed(pcm_data)
                if utterance:
                    active_stt = stt_engine or await _ensure_stt_engine(websocket.app)
                    await process_stt_utterance(utterance, manager, active_stt, engine)
            else:
                # Disarmed, the accumulator only refreshes its pre-roll (so initial consonants are never lost)
                # and drops any half-built turn: an hour of narration must not buffer an hour of microphone.
                acc.disarm()
                acc.feed(pcm_data)


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
    # A WebSocket handshake is never preflighted, so any page could otherwise open this socket.
    if not origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        logger.warning("Rejected WebSocket handshake from origin %r", websocket.headers.get("origin"))
        await websocket.close(code=1008)
        return

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

            try:
                await _handle_frame(raw_msg, websocket, manager, engine, stt_engine)
            except WebSocketDisconnect:
                raise
            except Exception:  # noqa: BLE001  # one failing message must not end the session; logged below and answered
                logger.exception("Error while handling a WebSocket message")
                await manager.send_error(websocket, "internal error while handling the message")

    except WebSocketDisconnect:
        logger.info("Client disconnected via WebSocketDisconnect.")
    except Exception as exc:  # noqa: BLE001  # S3.3 RC4
        logger.warning(f"WebSocket session terminated with exception: {exc}")
    finally:
        manager.disconnect(websocket)
