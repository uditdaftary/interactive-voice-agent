"""Broadcast fan-out survives stalled and broken clients, and says so (M5, L2)."""

import asyncio
import logging
from typing import Any

import pytest

from voice_agent.server import connection_manager as connection_manager_module
from voice_agent.server.connection_manager import ConnectionManager


class RecordingSocket:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.blobs: list[bytes] = []

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


async def test_stalled_socket_dropped_for_binary_frames(short_send_timeout: None, caplog: pytest.LogCaptureFixture) -> None:
    healthy, stalled = RecordingSocket(), StalledSocket()
    manager = manager_with(stalled, healthy)

    with caplog.at_level(logging.INFO):
        async with asyncio.timeout(2):
            await manager.broadcast_bytes(b"\x01\x02")

    assert healthy.blobs == [b"\x01\x02"]
    assert manager.active_connections == [healthy]
    assert "TimeoutError" in caplog.text


async def test_broken_socket_dropped_and_logged_by_type(short_send_timeout: None, caplog: pytest.LogCaptureFixture) -> None:
    healthy, broken = RecordingSocket(), BrokenSocket()
    manager = manager_with(broken, healthy)

    with caplog.at_level(logging.INFO):
        await manager.broadcast_text("hello")
        await manager.broadcast_bytes(b"x")

    assert manager.active_connections == [healthy]
    assert healthy.texts == ["hello"]
    assert "RuntimeError" in caplog.text
    assert "Cannot call" not in caplog.text  # type name only, never the exception text


async def test_slow_client_does_not_delay_the_others(short_send_timeout: None) -> None:
    healthy, stalled = RecordingSocket(), StalledSocket()
    manager = manager_with(stalled, healthy)

    delivered_at: list[float] = []
    loop = asyncio.get_running_loop()
    original_send = healthy.send_text

    async def stamped(text: str) -> None:
        delivered_at.append(loop.time())
        await original_send(text)

    healthy.send_text = stamped  # type: ignore[method-assign]
    started = loop.time()
    await manager.broadcast_text("hello")

    assert delivered_at and delivered_at[0] - started < 0.03


def test_sync_broadcast_without_loop_warns(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()

    with caplog.at_level(logging.WARNING):
        manager.broadcast_json_sync({"type": "ping"})
        manager.broadcast_bytes_sync(b"\x00\x00")

    assert [record.levelname for record in caplog.records] == ["WARNING", "WARNING"]
    assert "no running event loop" in caplog.records[0].getMessage()
    assert "no running event loop" in caplog.records[1].getMessage()


def test_sync_broadcast_with_closed_loop_warns(caplog: pytest.LogCaptureFixture) -> None:
    loop = asyncio.new_event_loop()
    loop.close()

    with caplog.at_level(logging.WARNING):
        ConnectionManager().broadcast_json_sync({"type": "ping"}, loop=loop)

    assert [record.levelname for record in caplog.records] == ["WARNING"]
