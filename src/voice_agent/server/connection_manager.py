"""Connection and Session State Manager for Interactive Voice Agent WebSocket Server.

Manages active client connections, session state synchronization,
thread-safe broadcasting, and clean disconnection cleanup without memory leaks.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
import asyncio
import concurrent.futures
import json
import logging
import threading
from fastapi import WebSocket
from pydantic import BaseModel

from voice_agent.server.schemas import (
    BargeInEventMessage,
    ErrorMessage,
    StateSyncMessage,
)

logger = logging.getLogger(__name__)

# A client that cannot take a frame within this long is dropped rather than holding up every broadcast.
SEND_TIMEOUT_S = 2.0


class ConnectionManager:
    """Thread-safe WebSocket connection and session state manager."""

    def __init__(
        self,
        stt_engine: Optional[Any] = None,
        accumulator: Optional[Any] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.active_connections: List[WebSocket] = []

        # Session state defaults
        self._state: str = "NARRATING"
        self._active_voice: str = "david-martinez"
        self._mic_source: str = "browser"
        self._cadence: float = 1.0
        self._sentence_idx: int = 0
        self._word_offset: int = 0
        self._latency_ms: float = 0.0
        self._generation: int = 0

        # STT & Speech Turn-taking
        self._stt_engine: Optional[Any] = stt_engine
        self._accumulator: Optional[Any] = accumulator

        # Optional callbacks for DSP / Orchestrator integration
        self._audio_sink: Optional[Callable[[bytes], Any]] = None
        self._barge_in_callback: Optional[Callable[[int, float], Any]] = None
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
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def disconnect_all(self) -> None:
        """Close and disconnect all active connections during shutdown."""
        with self._lock:
            sockets = list(self.active_connections)
            self.active_connections.clear()

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
    # Session State Accessors & Mutators
    # ========================================================================

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

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

    def trigger_barge_in(
        self,
        flushed_samples: int = 0,
        latency_ms: float = 0.04,
    ) -> BargeInEventMessage:
        """Trigger an instantaneous barge-in interruption.

        Increments generation counter, switches state to LISTENING,
        and invokes registered pipeline callback.
        """
        cb = None
        with self._lock:
            self._generation += 1
            self._state = "LISTENING"
            self._latency_ms = latency_ms
            current_gen = self._generation
            cb = self._barge_in_callback
            if self._accumulator is not None:
                self._accumulator.reset(preserve_pre_roll=True)

        if cb is not None:
            try:
                cb(flushed_samples, latency_ms)
            except Exception as e:  # noqa: BLE001  # S3.3 RC4
                logger.error(f"Error executing barge_in_callback: {e}")

        return BargeInEventMessage(
            latency_ms=latency_ms,
            generation=current_gen,
            flushed_samples=flushed_samples,
        )

    def get_state_sync_message(self) -> StateSyncMessage:
        """Generate a StateSyncMessage snapshot from current session state."""
        with self._lock:
            return StateSyncMessage(
                state=self._state,
                active_voice=self._active_voice,
                mic_source=self._mic_source,
                cadence=self._cadence,
                sentence_idx=self._sentence_idx,
                word_offset=self._word_offset,
                latency_ms=self._latency_ms,
            )

    # ========================================================================
    # Audio Ingestion & Callback Hooks
    # ========================================================================

    def register_audio_sink(self, callback: Callable[[bytes], Any]) -> None:
        """Register a callback sink to receive client Web Audio PCM frames."""
        with self._lock:
            self._audio_sink = callback

    def register_barge_in_callback(self, callback: Callable[[int, float], Any]) -> None:
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
        msg = self.get_state_sync_message()
        await websocket.send_text(msg.model_dump_json())

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
            with self._lock:
                self.active_connections = [ws for ws in self.active_connections if ws not in dead_set]

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

    async def broadcast_bytes(self, data: bytes) -> None:
        """Broadcast raw binary PCM audio frame to all active clients."""
        await self._fan_out(lambda ws: ws.send_bytes(data))

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
        """Thread-safe synchronous broadcast helper for background audio threads."""
        target_loop = self._usable_loop(loop, "broadcast_json_sync")
        if target_loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.broadcast_json(message), target_loop)
            fut.add_done_callback(self._log_future_exception)

    def broadcast_bytes_sync(
        self,
        data: bytes,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Thread-safe synchronous binary broadcast helper for background audio threads."""
        target_loop = self._usable_loop(loop, "broadcast_bytes_sync")
        if target_loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self.broadcast_bytes(data), target_loop)
            fut.add_done_callback(self._log_future_exception)


# Module-level default singleton instance
default_connection_manager = ConnectionManager()
