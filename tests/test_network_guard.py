"""The autouse network guard in conftest.py blocks non-loopback traffic and still allows loopback."""

import asyncio
import os
import socket
import threading

import pytest


def test_connect_to_remote_address_is_blocked(block_outbound_network: list[str]) -> None:
    with socket.socket() as sock, pytest.raises(RuntimeError, match="network access blocked"):
        sock.connect(("192.0.2.1", 80))
    assert len(block_outbound_network) == 1
    block_outbound_network.clear()  # expected here, so it must not fail the test at teardown


def test_create_connection_to_remote_address_is_blocked(block_outbound_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.create_connection(("192.0.2.1", 80), timeout=1)
    assert len(block_outbound_network) == 1
    block_outbound_network.clear()


def test_dns_lookup_of_remote_name_is_blocked(block_outbound_network: list[str]) -> None:
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.getaddrinfo("example.com", 443)
    assert len(block_outbound_network) == 1
    block_outbound_network.clear()


async def test_asyncio_connect_to_remote_address_is_blocked(block_outbound_network: list[str]) -> None:
    # A literal IP skips DNS, so this reaches the event loop's own connect path.
    with pytest.raises(RuntimeError, match="network access blocked"):
        await asyncio.open_connection("192.0.2.1", 80)
    assert block_outbound_network
    block_outbound_network.clear()


def _ipv6_socket() -> socket.socket:
    try:
        return socket.socket(socket.AF_INET6)
    except OSError:
        pytest.skip("IPv6 sockets are unavailable on this machine")


def test_ipv4_mapped_loopback_is_allowed(block_outbound_network: list[str]) -> None:
    # Dual-stack local servers appear as ::ffff:127.0.0.1; Python 3.11 does not call that loopback.
    with _ipv6_socket() as sock:
        sock.settimeout(1)
        try:
            sock.connect(("::ffff:127.0.0.1", 9))
        except OSError:
            pass  # refused or unsupported by this OS: the guard let it through, which is the point
    assert block_outbound_network == []


def test_ipv4_mapped_remote_address_is_still_blocked(block_outbound_network: list[str]) -> None:
    with _ipv6_socket() as sock, pytest.raises(RuntimeError, match="network access blocked"):
        sock.connect(("::ffff:192.0.2.1", 80))
    assert len(block_outbound_network) == 1
    block_outbound_network.clear()


def test_hub_is_offline_unless_live_tests_are_selected(request: pytest.FixtureRequest) -> None:
    live = any(item.get_closest_marker("slow") or item.get_closest_marker("network") for item in request.session.items)
    if live:
        pytest.skip("this run selects slow or network tests, so the Hugging Face Hub is left reachable")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_ddgs_search_is_blocked(block_outbound_network: list[str]) -> None:
    ddgs = pytest.importorskip("ddgs")
    with pytest.raises(RuntimeError, match="network access blocked"):
        ddgs.DDGS().text("anything")
    assert len(block_outbound_network) == 1
    block_outbound_network.clear()


def test_loopback_socket_still_works(block_outbound_network: list[str]) -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received: list[bytes] = []

    def serve() -> None:
        conn, _ = server.accept()
        with conn:
            received.append(conn.recv(4))

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(b"ping")
        worker.join(timeout=5)
    finally:
        server.close()
    assert received == [b"ping"]
    assert socket.getaddrinfo("localhost", port)
    assert block_outbound_network == []


async def test_loopback_asyncio_connection_still_works(block_outbound_network: list[str]) -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.readline())
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"hello\n")
        assert await reader.readline() == b"hello\n"
        writer.close()
        await writer.wait_closed()
    assert block_outbound_network == []
