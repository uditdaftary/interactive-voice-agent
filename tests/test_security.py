"""Host allowlist and same-origin checks on the HTTP and WebSocket entry points (H4, N7)."""

import inspect
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fakes import FakeLLM, FakeSearch, FakeTts
from voice_agent.server import security
from voice_agent.server.app import create_app, run
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.security import origin_allowed

ALLOWED_HOSTS = ["testserver", "127.0.0.1", "localhost"]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(llm: FakeLLM) -> Iterator[TestClient]:
    app = create_app(
        connection_manager=ConnectionManager(),
        llm_client=llm,
        tts=FakeTts(),
        search=FakeSearch(),
        allowed_hosts=ALLOWED_HOSTS,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("origin", "host", "expected"),
    [
        (None, "x", True),
        ("null", "127.0.0.1:8000", False),
        ("http://127.0.0.1:8000", "127.0.0.1:8000", True),
        ("http://LOCALHOST:8000", "localhost:8000", True),
        ("http://localhost:8000", "LocalHost:8000", True),
        ("http://evil.com", "127.0.0.1:8000", False),
        ("http://127.0.0.1", "127.0.0.1:80", True),
        ("https://127.0.0.1", "127.0.0.1:443", True),
        ("http://127.0.0.1", "127.0.0.1:8000", False),
        ("http://127.0.0.1:8001", "127.0.0.1:8000", False),
        ("https://127.0.0.1:8000", "127.0.0.1:8000", True),
        ("http://[::1]:8000", "[::1]:8000", True),
        ("http://127.0.0.1:8000", None, False),
        ("", "127.0.0.1:8000", False),
        ("http://127.0.0.1:notaport", "127.0.0.1:8000", False),
    ],
)
def test_origin_allowed_cases(origin: str | None, host: str | None, expected: bool) -> None:
    assert origin_allowed(origin, host) is expected


def test_cross_origin_json_post_rejected(client: TestClient, llm: FakeLLM) -> None:
    before = llm.get_config()

    resp = client.post("/api/config", json={"model": "pwned"}, headers={"Origin": "http://evil.com"})

    assert resp.status_code == 403
    assert resp.json() == {"detail": "cross-origin request rejected"}
    assert llm.get_config() == before


def test_cross_origin_multipart_upload_rejected(client: TestClient) -> None:
    title_before = client.get("/api/document").json()["title"]
    files = {"file": ("note.txt", b"Injected sentence one. Injected sentence two.", "text/plain")}

    resp = client.post("/api/upload-document", files=files, headers={"Origin": "http://evil.com"})

    assert resp.status_code == 403
    assert client.get("/api/document").json()["title"] == title_before


def test_cross_origin_multipart_transcribe_rejected(client: TestClient) -> None:
    files = {"file": ("a.pcm", b"\x00\x00" * 160, "application/octet-stream")}

    resp = client.post("/api/transcribe", files=files, headers={"Origin": "http://evil.com"})

    assert resp.status_code == 403


def test_null_origin_post_rejected(client: TestClient) -> None:
    assert client.post("/api/config", json={}, headers={"Origin": "null"}).status_code == 403


def test_same_origin_post_allowed(client: TestClient) -> None:
    resp = client.post("/api/config", json={}, headers={"Origin": "http://testserver"})

    assert resp.status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "LOCALHOST:8000"])
def test_hud_origins_are_allowed(client: TestClient, host: str) -> None:
    """The UI is served by this app, so the browser sends its own origin back on POST."""
    resp = client.post("/api/config", json={}, headers={"Host": host, "Origin": f"http://{host}"})

    assert resp.status_code == 200


def test_cross_origin_get_is_not_blocked(client: TestClient) -> None:
    """GET stays readable: a foreign page cannot read the response without CORS headers."""
    assert client.get("/api/config", headers={"Origin": "http://evil.com"}).status_code == 200


def test_bad_host_rejected(client: TestClient) -> None:
    assert client.get("/api/config", headers={"Host": "evil.com"}).status_code == 400


def test_default_allowlist_is_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest widens the default for TestClient's "testserver" host; undo that to test the shipped default.
    monkeypatch.setattr(security, "DEFAULT_ALLOWED_HOSTS", ("127.0.0.1", "localhost"))
    app = create_app(connection_manager=ConnectionManager(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    with TestClient(app) as default_client:
        assert default_client.get("/api/config").status_code == 400
        assert default_client.get("/api/config", headers={"Host": "127.0.0.1:8000"}).status_code == 200
        assert default_client.get("/api/config", headers={"Host": "192.168.1.20:8000"}).status_code == 400


def test_ws_foreign_origin_closed(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect("/ws", headers={"origin": "http://evil.com"}):
        pass

    assert closed.value.code == 1008


def test_ws_same_origin_accepted(client: TestClient) -> None:
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "state_sync"


def test_ws_without_origin_accepted(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state_sync"


def test_ws_bad_host_rejected(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers={"host": "evil.com"}):
        pass


def test_run_defaults_to_loopback() -> None:
    assert inspect.signature(run).parameters["host"].default == "127.0.0.1"


def test_run_warns_on_non_loopback_host(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import uvicorn

    started: list[dict[str, object]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: started.append(kwargs))

    with caplog.at_level("WARNING"):
        run(host="127.0.0.1")
        assert not caplog.records
        run(host="0.0.0.0")

    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert [call["host"] for call in started] == ["127.0.0.1", "0.0.0.0"]


# ---------------------------------------------------------------------------
# Rejections are logged (#65): who was turned away, never what they sent
# ---------------------------------------------------------------------------


def warnings_from(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


def test_cross_origin_post_rejection_is_logged_without_the_body(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    secret = "SECRET-API-KEY-0123"

    with caplog.at_level(logging.WARNING):
        resp = client.post("/api/config", json={"api_key": secret}, headers={"Origin": "http://evil.com"})

    assert resp.status_code == 403
    [message] = warnings_from(caplog)
    assert "http://evil.com" in message  # the rejected Origin
    assert "testserver" in message  # the Host it was compared with
    assert "/api/config" in message
    assert "POST" in message
    assert secret not in caplog.text


def test_cross_origin_multipart_rejection_is_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        client.post(
            "/api/upload-document",
            files={"file": ("note.txt", b"Injected sentence one. Two.", "text/plain")},
            headers={"Origin": "http://evil.com"},
        )

    [message] = warnings_from(caplog)
    assert "http://evil.com" in message
    assert "Injected sentence" not in caplog.text


def test_null_origin_rejection_is_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        client.post("/api/config", json={}, headers={"Origin": "null"})

    [message] = warnings_from(caplog)
    assert "'null'" in message


def test_bad_host_rejection_is_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        resp = client.get("/api/config", headers={"Host": "evil.com"})

    assert resp.status_code == 400
    [message] = warnings_from(caplog)
    assert "evil.com" in message
    assert "/api/config" in message


def test_bad_host_on_a_post_is_logged_once_and_the_body_is_not(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        resp = client.post("/api/config", json={"api_key": "SECRET-API-KEY-0123"}, headers={"Host": "192.168.1.20:8000"})

    assert resp.status_code == 400
    [message] = warnings_from(caplog)
    assert "192.168.1.20:8000" in message
    assert "SECRET-API-KEY-0123" not in caplog.text


def test_ws_bad_host_rejection_is_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING), pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers={"host": "evil.com"}):
        pass

    [message] = warnings_from(caplog)
    assert "evil.com" in message


def test_ws_foreign_origin_rejection_logs_origin_and_host(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING), pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers={"origin": "http://evil.com"}):
        pass

    [message] = warnings_from(caplog)
    assert "http://evil.com" in message
    assert "testserver" in message


def test_allowed_requests_log_no_warning(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        client.get("/api/config")
        client.get("/api/config", headers={"Origin": "http://evil.com"})  # GET is not origin-checked
        client.post("/api/config", json={}, headers={"Origin": "http://testserver"})
        client.post("/api/config", json={})  # no Origin: CLI and TestClient
        with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
            ws.receive_json()

    assert warnings_from(caplog) == []


def test_logged_header_values_are_escaped_and_truncated(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """Headers are attacker-controlled: quotes are escaped by repr and a huge value cannot flood the log."""
    with caplog.at_level(logging.WARNING):
        client.post("/api/config", json={}, headers={"Origin": 'http://evil.com/"x" ' + "a" * 5000})

    [message] = warnings_from(caplog)
    assert len(message) < 600
    assert len(message.splitlines()) == 1
