"""Test-suite guards: no outbound network, offline Hugging Face Hub, platform markers.

Tests that really need the internet carry `@pytest.mark.network` and are deselected by default
(see `addopts` in pyproject.toml). Everything else must pass with no route off this machine.
"""

import ipaddress
import os
import socket
import sys
from asyncio import base_events, proactor_events
from collections.abc import Iterator
from typing import Any

import pytest

# Set before any test module imports faster_whisper or huggingface_hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(host: object) -> bool:
    if host is None:
        return True  # wildcard bind / local service lookup, no remote peer
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    name = str(host).strip("[]").lower()
    if name in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return False
    # Dual-stack servers show up as ::ffff:127.0.0.1, which IPv6Address.is_loopback rejects on 3.11.
    return (getattr(address, "ipv4_mapped", None) or address).is_loopback


def _peer_host(address: Any) -> object:
    # AF_INET and AF_INET6 addresses are tuples led by the host; AF_UNIX is a path and never leaves the machine.
    return address[0] if isinstance(address, tuple) and address else None


@pytest.fixture(autouse=True)
def block_outbound_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail any test that opens a non-loopback connection or resolves a non-loopback name.

    Production code catches broad exceptions around network calls and falls back, so raising at the
    call site is not enough: each blocked attempt is recorded and the test is failed after it ends.
    Yields the record of blocked attempts so test_network_guard.py can assert on it and clear it.
    """
    blocked: list[str] = []
    if request.node.get_closest_marker("network"):
        yield blocked
        return

    def deny(what: str, target: object) -> None:
        blocked.append(f"{what} {target!r}")
        raise RuntimeError(f"network access blocked in tests: {target!r} ({what})")

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> None:
        if not _is_loopback(_peer_host(address)):
            deny("connect", address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_loopback(_peer_host(address)):
            deny("connect_ex", address)
        return real_connect_ex(self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not _is_loopback(_peer_host(address)):
            deny("create_connection", address)
        return real_create_connection(address, *args, **kwargs)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback(host):
            deny("getaddrinfo", host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guard_sock_connect(real: Any) -> Any:
        async def sock_connect(self: Any, sock: socket.socket, address: Any) -> None:
            if not _is_loopback(_peer_host(address)):
                deny("sock_connect", address)
            return await real(self, sock, address)

        return sock_connect

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # The Windows proactor loop connects through overlapped ConnectEx, bypassing socket.connect,
    # and overrides sock_connect, so each loop base class gets its own guard.
    for loop_base in (base_events.BaseEventLoop, proactor_events.BaseProactorEventLoop):
        monkeypatch.setattr(loop_base, "sock_connect", guard_sock_connect(loop_base.sock_connect))

    # ddgs talks through primp, a native HTTP client that no Python socket patch can see.
    try:
        import ddgs
    except ImportError:
        pass
    else:

        def ddgs_text(self: object, query: str, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
            deny("ddgs.DDGS.text", query)
            return []

        monkeypatch.setattr(ddgs.DDGS, "text", ddgs_text)

    yield blocked

    if blocked:
        pytest.fail("network access blocked in tests:\n  " + "\n  ".join(blocked), pytrace=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if sys.platform == "win32":
        return
    skip_windows = pytest.mark.skip(reason="needs Windows SAPI")
    for item in items:
        if item.get_closest_marker("windows"):
            item.add_marker(skip_windows)


@pytest.fixture
def offline_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a real `STTEngine` run with no Whisper model and no Google recognizer.

    Loading the model needs the Hugging Face cache (absent on CI) and `recognize_google` is a network
    call, so tests that exercise the engine's own logic opt in to this fixture.
    """
    from voice_agent.stt.recognizer import STTEngine

    monkeypatch.setattr(STTEngine, "_init_engine", lambda self: None)
    try:
        import speech_recognition as sr
    except ImportError:
        return  # not installed (it is not a declared dependency), so the fallback is already skipped

    def recognize_google(self: object, *args: Any, **kwargs: Any) -> str:
        raise sr.RequestError("offline in tests")

    monkeypatch.setattr(sr.Recognizer, "recognize_google", recognize_google)
