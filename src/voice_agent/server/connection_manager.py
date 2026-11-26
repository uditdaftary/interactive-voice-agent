"""Connection and Session State Manager for Interactive Voice Agent WebSocket Server.

Manages active client connections, session state synchronization,
thread-safe broadcasting, and clean disconnection cleanup without memory leaks.
"""

from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Union
import asyncio
import concurrent.futures
import json
import logging
import threading
from dataclasses import dataclass
from fastapi import WebSocket
from pydantic import BaseModel

from voice_agent.orchestrator.state_machine import SessionState, SessionStateName
from voice_agent.server.playback import PlaybackTracker
from voice_agent.server.schemas import (
    AudioSegmentMessage,
    AudioStopMessage,
    BargeInEventMessage,
    ErrorMessage,
    StateSyncMessage,
)
from voice_agent.server.tasks import TaskRegistry
from voice_agent.settings import Settings

logger = logging.getLogger(__name__)

# A client that cannot take a frame within this long is dropped rather than holding up every broadcast.
SEND_TIMEOUT_S = 2.0

# Session states in which microphone audio is recorded as a user turn.
LISTENING_STATES = frozenset({"LISTENING"})


@dataclass(frozen=True)
class Statuses:
    """What the HUD is told about the pieces the session depends on. Compared as a whole to see whether anything changed."""

    llm_status: str = "ok"
    llm_provider: str = ""
    tts_status: str = "ok"
    stt_status: str = "ok"
    web_speech_preview: bool = False


class ConnectionManager:
    """WebSocket connections and session settings. The session state itself lives in `session`.

    State changes and task spawns happen on the event loop thread; the lock guards the plain settings
    that other threads may read.
    """

    def __init__(
        self,
        stt_engine: Optional[Any] = None,
        accumulator: Optional[Any] = None,
        *,
        session: Optional[SessionState] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.active_connections: List[WebSocket] = []
        # The connection that claimed the microphone and the audio. None means the oldest connection has them.
        self._explicit_owner: Optional[WebSocket] = None
        self._owner_listeners: List[Callable[[], None]] = []
        self._ignored_logged: set[WebSocket] = set()
        self.tasks = TaskRegistry()
        # Audio: one clip at a time goes out, each announced just before its blob; `_generation` orders everything
        # audible (see `stop_audio`).
        self.playback = PlaybackTracker()
        self._audio_lock = asyncio.Lock()

        if session is None:
            # Defaults only: the NarrationEngine applies the configured durations when it attaches.
            defaults = Settings()
            session = SessionState(
                listen_deadline_s=defaults.barge_in_listen_s, followup_window_s=defaults.followup_window_s
            )
        self.session = session
        # Registered once, in this order: drop a half-recorded turn, then tell the clients.
        self.session.add_listener(self._disarm_accumulator)
        self.session.add_listener(self._announce_state)

        # Session defaults
        self._active_voice: str = "david-martinez"
        self._mic_source: str = "browser"
        self._cadence: float = 1.0
        self._sentence_idx: int = 0
        self._word_offset: int = 0
        self._latency_ms: Optional[float] = None  # None until something measures it
        self._generation: int = 0

        # Reports the language model, speech and recognizer statuses; the narration engine registers it.
        self._status_provider: Callable[[], Statuses] = Statuses
        self._announced_statuses = Statuses()

        # STT & Speech Turn-taking
        self._stt_engine: Optional[Any] = stt_engine
        self._accumulator: Optional[Any] = accumulator

        # Optional callbacks for DSP / Orchestrator integration
        self._audio_sink: Optional[Callable[[bytes], Any]] = None
        self._barge_in_callback: Optional[Callable[[Optional[int], Optional[float]], Any]] = None
        self._voice_change_callback: Optional[Callable[[str], Any]] = None
        self._mic_source_change_callback: Optional[Callable[[str], Any]] = None

    # ========================================================================
    # Connection Lifecycle
    # ========================================================================

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and register it in active connections."""
        await websocket.accept()
        with self._lock:
            if websocket not in self.active_connections:
                self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Deregister a WebSocket client cleanly without memory leaks."""
        remaining = self._forget(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {remaining}")
        if remaining == 0:
            # Nobody is listening: stop speaking to an empty room. A client that connects later is told PAUSED.
            self.session.transition("PAUSED", "no_clients")

    async def disconnect_all(self) -> None:
        """Close and disconnect all active connections during shutdown."""
        with self._lock:
            sockets = list(self.active_connections)
            self.active_connections.clear()
            self._explicit_owner = None
            self._ignored_logged.clear()

        for ws in sockets:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001, S110  # S3.3 RC4
                pass

    @property
    def connection_count(self) -> int:
        """Current number of active WebSocket connections."""
        with self._lock:
            return len(self.active_connections)

    # ========================================================================
    # Owner connection
    # ========================================================================

    @property
    def owner(self) -> Optional[WebSocket]:
        """The one connection whose microphone, push-to-talk and playback reports count, and that gets the audio:
        the connection that claimed it, else the oldest, so a tab without microphone permission still hears the
        narration. None only when nobody is connected."""
        with self._lock:
            return self._owner_locked()

    def _owner_locked(self) -> Optional[WebSocket]:
        if self._explicit_owner is not None:
            return self._explicit_owner
        return self.active_connections[0] if self.active_connections else None

    def is_owner(self, websocket: WebSocket) -> bool:
        return self.owner is websocket

    def claim_ownership(self, websocket: WebSocket, *, explicit: bool) -> bool:
        """Offer `websocket` the microphone and the audio; True if it owns them afterwards.

        An explicit claim (the user chose this tab as the microphone) always takes over. An implicit one (a tab
        sent audio) counts only while nobody has claimed it yet: the first tab to send audio keeps it.
        """
        with self._lock:
            if websocket not in self.active_connections:
                return False
            before = self._owner_locked()
            if explicit or self._explicit_owner is None:
                self._explicit_owner = websocket
            owner = self._owner_locked()
        if owner is not before:
            self._owner_changed()
        return owner is websocket

    def add_owner_listener(self, listener: Callable[[], None]) -> None:
        """Call `listener()` whenever the owner becomes a different connection."""
        self._owner_listeners.append(listener)

    def ignore_from_non_owner(self, websocket: WebSocket, what: str) -> None:
        """Note that `what` from a connection that does not own the microphone was dropped. One INFO line per
        connection: a second tab streams audio the whole time, and a line per frame would bury the log."""
        with self._lock:
            first = websocket not in self._ignored_logged
            self._ignored_logged.add(websocket)
        if first:
            logger.info("Ignoring %s from a connection that is not the microphone owner; later ones are not logged.", what)

    def _forget(self, websocket: WebSocket) -> int:
        """Remove a connection, whether it left or was dropped after a failed send, and hand its ownership on to the
        oldest one left. Returns how many connections remain."""
        with self._lock:
            before = self._owner_locked()
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if self._explicit_owner is websocket:
                self._explicit_owner = None
            self._ignored_logged.discard(websocket)
            after = self._owner_locked()
            remaining = len(self.active_connections)
        if after is not None and after is not before:
            self._owner_changed()
        return remaining

    def _owner_changed(self) -> None:
        """A different connection owns the microphone now. Whatever the old owner had started (a push-to-talk hold, a
        half-recorded turn) was its own and is dropped; then every tab is told and the listeners react. Event loop only."""
        if self.state in LISTENING_STATES:
            self.session.release_deadline_hold()
            self.clear_audio_turn()
        self.broadcast_state_sync()
        for listener in list(self._owner_listeners):
            try:
                listener()
            except Exception:  # noqa: BLE001  # one broken listener must not keep the others from running; logged here
                logger.exception("owner listener failed")

    # ========================================================================
    # Session State Accessors & Mutators
    # ========================================================================

    @property
    def state(self) -> SessionStateName:
        """Read-only view of `session.state`; every change goes through `session`."""
        return self.session.state

    def _disarm_accumulator(self, old: str, new: str, reason: str) -> None:
        if new not in LISTENING_STATES and self._accumulator is not None:
            # Drop the half-recorded turn on every exit from LISTENING, here and not when the next
            # audio frame happens to arrive: until then a stale tail could be transcribed together
            # with the next question. Runs on the event loop, like the accumulator's feed().
            self._accumulator.disarm()

    def _announce_state(self, old: str, new: str, reason: str) -> None:
        """The one place a state change is broadcast. With nobody connected there is nobody to tell."""
        if self.connection_count:
            self.broadcast_state_sync()

    @property
    def active_voice(self) -> str:
        with self._lock:
            return self._active_voice

    def set_voice(self, voice_id: str) -> None:
        cb = None
        with self._lock:
            self._active_voice = voice_id
            cb = self._voice_change_callback
        if cb is not None:
            try:
                cb(voice_id)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error executing voice_change_callback: {e}")

    @property
    def mic_source(self) -> str:
        with self._lock:
            return self._mic_source

    def set_mic_source(self, source: str) -> None:
        cb = None
        with self._lock:
            self._mic_source = source
            cb = self._mic_source_change_callback
        if cb is not None:
            try:
                cb(source)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error executing mic_source_change_callback: {e}")

    @property
    def cadence(self) -> float:
        with self._lock:
            return self._cadence

    def set_cadence(self, cadence: float) -> None:
        with self._lock:
            self._cadence = cadence

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def stt_engine(self) -> Any:
        """The stored STT engine, or None. Never constructs one: see `ensure_stt_engine`."""
        with self._lock:
            return self._stt_engine

    @stt_engine.setter
    def stt_engine(self, engine: Any) -> None:
        with self._lock:
            self._stt_engine = engine

    @property
    def accumulator(self) -> Any:
        with self._lock:
            if self._accumulator is None:
                from voice_agent.stt.accumulator import SpeechTurnAccumulator
                self._accumulator = SpeechTurnAccumulator()
            return self._accumulator

    @accumulator.setter
    def accumulator(self, acc: Any) -> None:
        with self._lock:
            self._accumulator = acc

    def update_cursor(
        self,
        sentence_idx: int,
        word_offset: int,
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            self._sentence_idx = sentence_idx
            self._word_offset = word_offset
            if latency_ms > 0:
                self._latency_ms = latency_ms

    def announce_barge_in(
        self,
        flushed_samples: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> BargeInEventMessage:
        """Tell everyone the user has cut in: bump the generation, drop the half-recorded turn (the pre-roll stays,
        because the speech that triggered this has already begun), queue the `barge_in` event for the clients and
        invoke the registered pipeline callback. The state does not change here; a caller that moves the session
        makes the move after this, so clients see the event first. Call it on the event loop.
        """
        cb = None
        with self._lock:
            self._generation += 1
            self._latency_ms = latency_ms
            current_gen = self._generation
            cb = self._barge_in_callback
            if self._accumulator is not None:
                self._accumulator.reset(preserve_pre_roll=True)
        self.playback.drop_all()  # whatever was being waited for was flushed with the audio

        event = BargeInEventMessage(
            latency_ms=latency_ms,
            generation=current_gen,
            flushed_samples=flushed_samples,
        )
        if self.connection_count:
            self.broadcast_json_sync(event)

        if cb is not None:
            try:
                cb(flushed_samples, latency_ms)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error executing barge_in_callback: {e}")

        return event

    def trigger_barge_in(
        self,
        flushed_samples: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> BargeInEventMessage:
        """Trigger an instantaneous barge-in: announce it, drop a transcription that is still pending (the user is
        starting a newer turn, and the newest turn wins), and switch to LISTENING, which arms its deadline and stops
        narration and any reply. Call it on the event loop.
        """
        event = self.announce_barge_in(flushed_samples, latency_ms)
        self.tasks.cancel("stt")
        self.session.enter_listening("barge_in")
        return event

    def stop_audio(self) -> None:
        """Stop all audio now: bump the generation, drop every segment still outstanding (their waiters are released
        with False) and tell every tab with `audio_stop`, so the old owner stops too. The caller cancels whatever
        was producing the audio. A barge-in does the same through its own event. Call it on the event loop.

        The message takes its place in the same queue as the clips, so it reaches the browser before any clip sent
        after the stop, and a clip handed over before the stop but not yet sent is never sent (see `send_audio`).
        """
        with self._lock:
            self._generation += 1
            generation = self._generation
        self.playback.drop_all()
        if self.connection_count:
            self._post(lambda: self._send_stop(generation), None, "stop_audio")

    async def _send_stop(self, generation: int) -> None:
        async with self._audio_lock:
            message = AudioStopMessage(generation=generation).model_dump_json()
            await self._fan_out(lambda ws: ws.send_text(message))

    async def send_audio(self, data: bytes, *, gap_s: float = 0.0, generation: Optional[int] = None) -> Optional[int]:
        """Send one clip to the owner, announced by an `audio_segment` right before it, and return its segment id.
        None when nobody owns the audio (the caller paces by an estimate) or the send failed.

        The pair is sent to the end even if the caller is cancelled in the middle: a clip announced without its blob
        would be paired with the next blob, and a clip of the old generation that is sent anyway is dropped by the
        browser. `gap_s` is the silence the browser adds before the clip.

        The clip belongs to `generation`, by default the one current when this is called; a caller that made the clip
        over a longer time passes the one it started in. If audio is stopped before it is sent, it is dropped and None
        is returned: sending it under the newer generation would play it after the stop.
        """
        if self.owner is None:
            return None
        if generation is None:
            generation = self.generation
        return await asyncio.shield(self.tasks.spawn(self._send_audio(data, gap_s, generation), "audio"))

    async def _send_audio(self, data: bytes, gap_s: float, generation: int) -> Optional[int]:
        async with self._audio_lock:
            owner = self.owner
            if owner is None or generation != self.generation:
                return None
            segment_id = self.playback.register(generation, owner)
            announcement = AudioSegmentMessage(segment_id=segment_id, generation=generation, gap_s=gap_s)
            try:
                await asyncio.wait_for(self._send_clip(owner, announcement.model_dump_json(), data), SEND_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001  # the owner is gone or stalled: dropped and logged here
                logger.info("Dropping WebSocket client after failed audio send: %s", type(exc).__name__)
                self.playback.forget(segment_id)
                self._forget(owner)
                await self._close_dropped(owner)
                return None
            return segment_id

    @staticmethod
    async def _send_clip(owner: WebSocket, announcement: str, data: bytes) -> None:
        await owner.send_text(announcement)
        await owner.send_bytes(data)

    def clear_audio_turn(self) -> None:
        """Empty the accumulator, pre-roll included. For the start of the follow-up window: what the microphone
        heard while the answer played is the answer's own echo, not the next question."""
        with self._lock:
            if self._accumulator is not None:
                self._accumulator.reset()

    def set_status_provider(self, provider: Callable[[], Statuses]) -> None:
        """Where every snapshot gets its statuses from. What the provider reports now counts as already announced: each
        client learns it from the snapshot it receives on connecting."""
        self._status_provider = provider
        self._announced_statuses = provider()

    def refresh_statuses(self) -> None:
        """Tell every client when a status has changed since the last time they were told, and only then. Call it
        wherever a status may have changed (after a request ends, never per token). Event loop only."""
        current = self._status_provider()
        if current == self._announced_statuses:
            return
        self._announced_statuses = current
        if self.connection_count:
            self.broadcast_state_sync()

    def get_state_sync_message(self, mic_owner: bool = True) -> StateSyncMessage:
        """Generate a StateSyncMessage snapshot from current session state."""
        statuses = self._status_provider()  # outside the lock: it reads the engine, which may take this lock
        with self._lock:
            return StateSyncMessage(
                mic_owner=mic_owner,
                state=self.session.state,
                active_voice=self._active_voice,
                mic_source=self._mic_source,
                cadence=self._cadence,
                sentence_idx=self._sentence_idx,
                word_offset=self._word_offset,
                latency_ms=self._latency_ms,
                llm_status=statuses.llm_status,
                llm_provider=statuses.llm_provider,
                tts_status=statuses.tts_status,
                stt_status=statuses.stt_status,
                web_speech_preview=statuses.web_speech_preview,
            )

    @staticmethod
    def _statuses_in(snapshot: StateSyncMessage) -> Statuses:
        return Statuses(
            llm_status=snapshot.llm_status,
            llm_provider=snapshot.llm_provider,
            tts_status=snapshot.tts_status,
            stt_status=snapshot.stt_status,
            web_speech_preview=snapshot.web_speech_preview,
        )

    # ========================================================================
    # Audio Ingestion & Callback Hooks
    # ========================================================================

    def register_audio_sink(self, callback: Callable[[bytes], Any]) -> None:
        """Register a callback sink to receive client Web Audio PCM frames."""
        with self._lock:
            self._audio_sink = callback

    def register_barge_in_callback(self, callback: Callable[[Optional[int], Optional[float]], Any]) -> None:
        """Register a callback invoked when barge-in is triggered."""
        with self._lock:
            self._barge_in_callback = callback

    def register_voice_change_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback invoked when character voice is changed."""
        with self._lock:
            self._voice_change_callback = callback

    def register_mic_source_change_callback(self, callback: Callable[[str], Any]) -> None:
        """Register a callback invoked when mic ingestion source changes."""
        with self._lock:
            self._mic_source_change_callback = callback

    def on_audio_frame(self, frame_bytes: bytes) -> None:
        """Route incoming raw client PCM audio frame to the registered audio sink."""
        sink = None
        with self._lock:
            sink = self._audio_sink
        if sink is not None:
            try:
                sink(frame_bytes)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error in audio sink callback: {e}")

    # ========================================================================
    # Messaging & Broadcasting
    # ========================================================================

    async def send_state_sync(self, websocket: WebSocket) -> None:
        """Send current state sync snapshot to a specific client connection."""
        msg = self.get_state_sync_message(mic_owner=self.is_owner(websocket))
        await websocket.send_text(msg.model_dump_json())

    async def broadcast_state(self) -> None:
        """Send every connection the state snapshot, each with its own `mic_owner`."""
        snapshot = self.get_state_sync_message()
        self._announced_statuses = self._statuses_in(snapshot)
        await self._send_snapshot(snapshot)

    def broadcast_state_sync(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """`broadcast_state` from synchronous code. The snapshot is taken now and queued behind earlier messages, like
        `broadcast_json_sync`, so a later change does not leak into a message about this one."""
        snapshot = self.get_state_sync_message()
        self._announced_statuses = self._statuses_in(snapshot)
        self._post(lambda: self._send_snapshot(snapshot), loop, "broadcast_state_sync")

    async def _send_snapshot(self, snapshot: StateSyncMessage) -> None:
        await self._fan_out(
            lambda ws: ws.send_text(snapshot.model_copy(update={"mic_owner": self.is_owner(ws)}).model_dump_json())
        )

    async def send_error(self, websocket: WebSocket, detail: str) -> None:
        """Tell one client its last message was rejected. `detail` must not echo the client's payload."""
        message = ErrorMessage(detail=detail)
        await asyncio.wait_for(websocket.send_text(message.model_dump_json()), timeout=SEND_TIMEOUT_S)

    async def _fan_out(self, send_one: Callable[[WebSocket], Awaitable[None]]) -> None:
        """Run `send_one` for every client concurrently; drop any client whose send fails or stalls.

        One slow client must not hold up the others (no head-of-line blocking), nor stall the caller
        indefinitely, so each send gets SEND_TIMEOUT_S.
        """
        with self._lock:
            sockets = list(self.active_connections)

        if not sockets:
            return

        async def guarded_send(ws: WebSocket) -> Optional[WebSocket]:
            try:
                await asyncio.wait_for(send_one(ws), timeout=SEND_TIMEOUT_S)
                return None
            except Exception as exc:  # noqa: BLE001  # any failed or timed-out send drops that client; logged here
                # Type name only: the message of a send error can carry peer addresses or payload fragments.
                logger.info("Dropping WebSocket client after failed send: %s", type(exc).__name__)
                return ws

        results = await asyncio.gather(*[guarded_send(ws) for ws in sockets])
        dead_set = {ws for ws in results if ws is not None}

        if dead_set:
            for ws in dead_set:
                self._forget(ws)
            await asyncio.gather(*[self._close_dropped(ws) for ws in dead_set])

    @staticmethod
    async def _close_dropped(ws: WebSocket) -> None:
        """Close a dropped client so its page sees the loss and reconnects.

        Without this the socket is merely forgotten: its endpoint loop keeps answering that client's
        messages, but it never receives another broadcast, so its HUD would freeze silently.
        """
        try:
            await asyncio.wait_for(ws.close(code=1011), timeout=SEND_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001  # best effort on a client that is already failing; logged
            logger.debug("Closing a dropped WebSocket client failed: %s", type(exc).__name__)

    async def broadcast_text(self, text: str) -> None:
        """Broadcast a text payload to all active connections."""
        await self._fan_out(lambda ws: ws.send_text(text))

    async def broadcast_json(self, message: Union[BaseModel, Dict[str, Any], str]) -> None:
        """Serialize and broadcast a JSON message to all active clients."""
        if isinstance(message, BaseModel):
            payload_text = message.model_dump_json()
        elif isinstance(message, dict):
            payload_text = json.dumps(message)
        elif isinstance(message, str):
            payload_text = message
        else:
            payload_text = json.dumps(str(message))

        await self.broadcast_text(payload_text)

    @staticmethod
    def _log_future_exception(fut: Any) -> None:
        try:
            exc = fut.exception()
            if exc is not None:
                logger.warning(f"Background WebSocket broadcast error: {exc}")
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            pass
        except Exception as e:  # noqa: BLE001  # S3.3 RC4
            logger.debug(f"Error inspecting broadcast future: {e}")

    @staticmethod
    def _usable_loop(loop: Optional[asyncio.AbstractEventLoop], caller: str) -> Optional[asyncio.AbstractEventLoop]:
        """The loop to post to, or None after a WARNING: a dropped broadcast must never be silent."""
        target_loop = loop
        if target_loop is None:
            try:
                target_loop = asyncio.get_running_loop()
            except RuntimeError:
                target_loop = None
        if target_loop is None or not target_loop.is_running() or target_loop.is_closed():
            logger.warning("%s dropped a message: no running event loop to deliver it on.", caller)
            return None
        return target_loop

    def broadcast_json_sync(
        self,
        message: Union[BaseModel, Dict[str, Any], str],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Synchronous broadcast helper: safe from background audio threads and from the loop itself.

        On the loop thread the broadcast becomes a tracked task right away, so it is queued ahead of anything
        started after it: a state_sync reaches clients before the messages the new state causes. From another
        thread it is posted to the loop, in the order the loop receives it.
        """
        self._post(lambda: self.broadcast_json(message), loop, "broadcast_json_sync")

    def _post(
        self,
        make_coroutine: Callable[[], Coroutine[Any, Any, None]],
        loop: Optional[asyncio.AbstractEventLoop],
        caller: str,
    ) -> None:
        """Run a broadcast on the loop without waiting for it. The coroutine is made only once a loop is known to take
        it, so a dropped message never leaves an unawaited coroutine behind."""
        target_loop = self._usable_loop(loop, caller)
        if target_loop is None:
            return
        try:
            on_target_loop = asyncio.get_running_loop() is target_loop
        except RuntimeError:  # a thread with no loop of its own
            on_target_loop = False
        if on_target_loop:
            self.tasks.spawn(make_coroutine(), "broadcast")
        else:
            fut = asyncio.run_coroutine_threadsafe(make_coroutine(), target_loop)
            fut.add_done_callback(self._log_future_exception)


# Module-level default singleton instance
default_connection_manager = ConnectionManager()
