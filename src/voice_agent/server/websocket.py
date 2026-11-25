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
    LISTENING_STATES,
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.security import origin_allowed
from voice_agent.stt.accumulator import MAX_UTTERANCE_S
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


def submit_turn(
    websocket: WebSocket,
    manager: ConnectionManager,
    engine: Optional[Any],
    stt_engine: Optional[Any],
    utterance: Optional[bytes],
    fallback_query: Optional[str],
) -> None:
    """A turn has ended: answer it, once, without holding up the receive loop.

    Only the first end of the current turn is taken (a silence-detected utterance and the client's own
    `voice_end` can both arrive for one LISTENING entry); every later one is dropped. The recording goes to
    the single `stt` task. A turn with no recording is answered from its fallback text, or, with neither,
    given back to the session at once rather than at the deadline.
    """
    session = manager.session
    turn = session.turn_id
    if not session.claim_turn(turn):
        logger.debug("Dropping a turn end: turn %d is already answered or the session is %s", turn, session.state)
        return
    fallback = (fallback_query or "").strip()
    if not utterance:
        if fallback:
            logger.info("No audio for this turn; answering its fallback text: %s", fallback)
            _deliver(manager, engine, turn, fallback, 0.0)
        else:
            session.resume("empty_turn")
        return
    # Suspended here, in the same step as the claim, so the clock cannot run out before the task starts.
    session.suspend_deadline(turn)
    manager.tasks.spawn(
        _transcribe(websocket, manager, engine, stt_engine, turn, utterance, fallback), "stt", replace=True
    )


async def _transcribe(
    websocket: WebSocket,
    manager: ConnectionManager,
    engine: Optional[Any],
    stt_engine: Optional[Any],
    turn: int,
    utterance: bytes,
    fallback: str,
) -> None:
    """The `stt` task: transcribe one claimed turn, then hand the transcript on. Building the recognizer on the first
    utterance can take seconds, and the LISTENING deadline stays suspended for all of it."""
    started = time.perf_counter()
    try:
        try:
            active = stt_engine if stt_engine else await _ensure_stt_engine(websocket.app)
            transcript = await asyncio.to_thread(active.transcribe, utterance)
        except Exception:  # noqa: BLE001  # logged with its traceback; the user hears "I didn't catch that"
            logger.exception("STT transcription failed; treating the utterance as unrecognized")
            transcript = ""
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        text = (transcript or "").strip()
        if not text and fallback:
            logger.info("Using fallback query following empty STT transcript: %s", fallback)
            text = fallback
        _deliver(manager, engine, turn, text, latency_ms)
    finally:
        # Scoped to this turn: if the task was cancelled because a newer turn began, that turn's own clock stays as it is.
        manager.session.resume_deadline(turn)


def _deliver(manager: ConnectionManager, engine: Optional[Any], turn: int, text: str, latency_ms: float) -> None:
    """Show and answer a transcript ("" means nothing was understood), if its turn is still the one being listened to.

    There is no await between the check and the state change, so nothing can slip in between, and the transcript
    is queued before the answer's own messages so the HUD shows the question first. Whoever moved the session while
    the recognizer worked (a pause, a typed question, a newer barge-in) owns it now: the late transcript is dropped.
    """
    session = manager.session
    if session.state not in LISTENING_STATES or session.turn_id != turn:
        logger.debug("Dropping a transcript: turn %d is over (session is %s, turn %d)", turn, session.state, session.turn_id)
        return
    manager.broadcast_json_sync(UserSpeechTranscriptMessage(transcript=text, is_final=True, latency_ms=latency_ms))
    if engine is None:
        if not text:
            session.resume("stt_rejected")
    elif text:
        engine.ask(text)
    else:
        engine.report_unrecognized()


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
        # Narration stops at once. The client records until it sends voice_end, and quiet speech never registers
        # with the server, so the deadline waits for that message, but no longer than one utterance can last.
        manager.trigger_barge_in()
        manager.session.hold_deadline(MAX_UTTERANCE_S)

    elif isinstance(msg, SetCadenceMessage):
        manager.set_cadence(msg.cadence)
        sync_msg = manager.get_state_sync_message()
        await manager.broadcast_json(sync_msg)

    elif isinstance(msg, PlaybackControlMessage):
        state_before = manager.state
        if msg.action in ("pause", "stop"):
            if engine is not None:
                engine.pause_narration()
            else:
                manager.session.transition("PAUSED", "pause")
        elif msg.action in ("resume", "start"):
            if engine is not None:
                engine.start_narration()
            else:
                manager.session.transition("NARRATING", "resume")
        elif msg.action == "next":
            if engine is not None:
                engine.next_sentence()
        elif msg.action == "prev":
            if engine is not None:
                engine.previous_sentence()

        if manager.state == state_before:
            # No transition happened, so the state listener announced nothing; still answer the request.
            await manager.broadcast_json(manager.get_state_sync_message())

    elif isinstance(msg, QueryInterruptionMessage):
        if engine is not None:
            engine.ask(msg.query)
        else:
            manager.trigger_barge_in()
            manager.session.transition("REPLYING", "question")

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
        # The client is done recording, so the hold ends. The turn ends here, on Stop; the normal deadline only runs
        # if Stop never comes and the hold's cap passes.
        manager.session.release_deadline_hold()
        if manager.state not in LISTENING_STATES:
            # The turn ended before the client's Stop (silence, a typed question, a navigation). The client is waiting
            # on a state change that already happened, so tell it where the session is now.
            await manager.send_state_sync(websocket)
        elif not manager.session.turn_claimed:
            acc = getattr(manager, "accumulator", None)
            utterance = acc.flush() if acc is not None else None
            submit_turn(websocket, manager, engine, stt_engine, utterance, getattr(msg, "fallback_query", None))


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
            session = manager.session
            if manager.state in LISTENING_STATES and not session.turn_claimed:
                acc.arm()
                # A held turn (push-to-talk) ends on the client's voice_end, not on a pause in the speech.
                utterance = acc.feed(pcm_data, end_on_silence=not session.deadline_held)
                if acc.is_speaking:
                    session.extend_deadline()  # the user is talking: the wait must not end on them
                if utterance:
                    submit_turn(websocket, manager, engine, stt_engine, utterance, None)
            else:
                # Disarmed, the accumulator only refreshes its pre-roll (so initial consonants are never lost):
                # an hour of narration must not buffer an hour of microphone. The session's state listener already
                # dropped any half-built turn when LISTENING ended; disarming here also covers an accumulator that
                # was created (armed) while the session was not listening, and a turn already handed to the recognizer.
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
        logger.warning(
            "Rejected WebSocket handshake: Origin %.100r, Host %.100r",
            websocket.headers.get("origin"),
            websocket.headers.get("host"),
        )
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
