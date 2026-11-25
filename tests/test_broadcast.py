"""Broadcast fan-out survives stalled and broken clients, and says so (M5, L2)."""

import asyncio
import json
import logging
from typing import Any

import pytest

from voice_agent.server import connection_manager as connection_manager_module
from voice_agent.server.connection_manager import ConnectionManager


class RecordingSocket:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.blobs: list[bytes] = []
        self.close_codes: list[int] = []

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.blobs.append(data)


class StalledSocket(RecordingSocket):
    """A client whose TCP window is full: the send never completes."""

    async def send_text(self, text: str) -> None:
        await asyncio.Event().wait()

    async def send_bytes(self, data: bytes) -> None:
        await asyncio.Event().wait()


class BrokenSocket(RecordingSocket):
    async def close(self, code: int = 1000) -> None:
        raise RuntimeError("already closed")

    async def send_text(self, text: str) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')


def manager_with(*sockets: Any) -> ConnectionManager:
    manager = ConnectionManager()
    manager.active_connections.extend(sockets)
    return manager


@pytest.fixture
def short_send_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection_manager_module, "SEND_TIMEOUT_S", 0.05)


def test_send_timeout_defaults_to_two_seconds() -> None:
    assert connection_manager_module.SEND_TIMEOUT_S == 2.0


async def test_stalled_socket_dropped_after_timeout(short_send_timeout: None, caplog: pytest.LogCaptureFixture) -> None:
    healthy, stalled = RecordingSocket(), StalledSocket()
    manager = manager_with(healthy, stalled)

    with caplog.at_level(logging.INFO):
        async with asyncio.timeout(2):
            await manager.broadcast_text("hello")

    assert healthy.texts == ["hello"]
    assert manager.active_connections == [healthy]
    assert "TimeoutError" in caplog.text
    # The dropped client is told, so its page reconnects instead of freezing on a socket nobody broadcasts to.
    assert stalled.close_codes == [1011]
    assert healthy.close_codes == []


async def test_stalled_owner_dropped_when_a_clip_is_sent(short_send_timeout: None, caplog: pytest.LogCaptureFixture) -> None:
    healthy, stalled = RecordingSocket(), StalledSocket()
    manager = manager_with(stalled, healthy)  # the oldest connection owns the audio

    with caplog.at_level(logging.INFO):
        async with asyncio.timeout(2):
            assert await manager.send_audio(b"\x01\x02") is None  # not delivered: nobody is waiting for it

    assert manager.active_connections == [healthy]
    assert manager.owner is healthy  # the next clip goes to the connection that is still there
    assert healthy.blobs == []
    assert "TimeoutError" in caplog.text
    assert stalled.close_codes == [1011]
    await manager.tasks.cancel_all()


async def test_broken_socket_dropped_and_logged_by_type(short_send_timeout: None, caplog: pytest.LogCaptureFixture) -> None:
    healthy, broken = RecordingSocket(), BrokenSocket()
    manager = manager_with(broken, healthy)

    with caplog.at_level(logging.INFO):
        await manager.broadcast_text("hello")
        await manager.send_audio(b"x")

    assert manager.active_connections == [healthy]
    assert healthy.texts[0] == "hello"  # the broken socket was the oldest, so the survivor is also told it owns the audio
    assert json.loads(healthy.texts[1])["mic_owner"] is True
    assert "RuntimeError" in caplog.text
    assert "Cannot call" not in caplog.text  # type name only, never the exception text


async def test_slow_client_does_not_delay_the_others(short_send_timeout: None) -> None:
    """Asserts the order of events, not elapsed time: no wall-clock bound to flake on a loaded runner.

    Sent one after the other, the healthy client would be served only after the stalled one's send
    was abandoned. Sent together, it is served while the stalled send is still pending.
    """
    events: list[str] = []

    class OrderedStalledSocket(RecordingSocket):
        async def send_text(self, text: str) -> None:
            events.append("stalled send started")
            try:
                await asyncio.Event().wait()
            finally:
                events.append("stalled send abandoned")  # runs when the send timeout cancels it

    class OrderedHealthySocket(RecordingSocket):
        async def send_text(self, text: str) -> None:
            events.append("healthy delivered")
            await super().send_text(text)

    manager = manager_with(OrderedStalledSocket(), OrderedHealthySocket())

    await manager.broadcast_text("hello")

    # The last delivery is the survivor being told it owns the audio now: the stalled client was the oldest.
    assert events == ["stalled send started", "healthy delivered", "stalled send abandoned", "healthy delivered"]


async def test_close_that_also_stalls_does_not_hang_the_broadcast(short_send_timeout: None) -> None:
    class UncloseableSocket(StalledSocket):
        async def close(self, code: int = 1000) -> None:
            await asyncio.Event().wait()

    manager = manager_with(UncloseableSocket())

    async with asyncio.timeout(2):
        await manager.broadcast_text("hello")

    assert manager.active_connections == []


def test_sync_broadcast_without_loop_warns(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()

    with caplog.at_level(logging.WARNING):
        manager.broadcast_json_sync({"type": "ping"})
        manager.broadcast_state_sync()

    assert [record.levelname for record in caplog.records] == ["WARNING", "WARNING"]
    assert "no running event loop" in caplog.records[0].getMessage()
    assert "no running event loop" in caplog.records[1].getMessage()


def test_sync_broadcast_with_closed_loop_warns(caplog: pytest.LogCaptureFixture) -> None:
    loop = asyncio.new_event_loop()
    loop.close()

    with caplog.at_level(logging.WARNING):
        ConnectionManager().broadcast_json_sync({"type": "ping"}, loop=loop)

    assert [record.levelname for record in caplog.records] == ["WARNING"]
