"""A bad WebSocket message gets an error reply and the session stays open (L3, M5 follow-up)."""

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeTts
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import InvalidClientMessage, parse_client_message


@pytest.fixture
def manager() -> ConnectionManager:
    return ConnectionManager()


@pytest.fixture
def client(manager: ConnectionManager) -> Iterator[TestClient]:
    app = create_app(connection_manager=manager, llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    with TestClient(app) as test_client:
        yield test_client


def test_ws_bad_message_keeps_session(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state_sync"

        ws.send_text("{not json")
        first = ws.receive_json()
        ws.send_json({"type": "set_cadence", "cadence": 99})
        second = ws.receive_json()
        ws.send_json({"type": "select_voice", "voice_id": "lucy-kushinada"})
        third = ws.receive_json()

    assert first == {"type": "error", "detail": "message is not valid JSON"}
    assert second["type"] == "error"
    assert "cadence" in second["detail"]
    assert third["type"] == "state_sync"
    assert third["active_voice"] == "lucy-kushinada"


def test_ws_cadence_out_of_range_ignored_with_error_reply(client: TestClient, manager: ConnectionManager) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for bad in (99, 0.01, "fast"):
            ws.send_json({"type": "set_cadence", "cadence": bad})
            assert ws.receive_json()["type"] == "error"

    assert manager.cadence == 1.0


@pytest.mark.parametrize(
    ("message", "expected_fragment"),
    [
        ({"type": "select_voice", "voice_id": "Bad Voice!"}, "voice_id"),
        ({"type": "set_mic_source", "source": "quantum"}, "source"),
        ({"type": "playback_control", "action": "explode"}, "action"),
        ({"type": "query_interruption", "query": "q" * 2001}, "query"),
        ({"type": "update_config", "base_url": "ftp://example.com"}, "base_url"),
        ({"type": "update_config", "provider": "Not Valid"}, "provider"),
        ({"type": "no_such_message"}, "unknown message type"),
        ({}, "unknown message type"),
    ],
)
def test_ws_invalid_messages_get_a_specific_error(client: TestClient, message: dict[str, Any], expected_fragment: str) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json(message)
        reply = ws.receive_json()

    assert reply["type"] == "error"
    assert expected_fragment in reply["detail"]


@pytest.mark.parametrize("payload", ["[1, 2]", "42", "null", '"text"', "", "\x00\x01"])
def test_ws_non_object_payloads_get_error_replies(client: TestClient, payload: str) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(payload)
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "set_cadence", "cadence": 1.1})
        assert ws.receive_json()["type"] == "state_sync"


def test_error_reply_and_log_never_echo_the_payload(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    secret = "SECRET-TOKEN-1234"
    with caplog.at_level(logging.DEBUG), client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for payload in (
            f'{{"type": "set_cadence", "cadence": "{secret}"}}',
            f'{{"type": "select_voice", "voice_id": "{secret}"}}',
            f'{{"type": "update_config", "base_url": "{secret}"}}',
            f"not json {secret}",
        ):
            ws.send_text(payload)
            reply = ws.receive_json()
            assert reply["type"] == "error"
            assert secret not in reply["detail"]

    assert secret not in caplog.text


def test_handler_exception_replies_with_error_and_keeps_session(
    client: TestClient, manager: ConnectionManager, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real_set_cadence = manager.set_cadence

    def exploding_set_cadence(cadence: float) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(manager, "set_cadence", exploding_set_cadence)
    with caplog.at_level(logging.ERROR), client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_cadence", "cadence": 1.5})
        reply = ws.receive_json()

        monkeypatch.setattr(manager, "set_cadence", real_set_cadence)
        ws.send_json({"type": "set_cadence", "cadence": 1.5})
        recovered = ws.receive_json()

    assert reply["type"] == "error"
    assert "synthetic failure" not in reply["detail"]
    assert recovered["type"] == "state_sync"
    assert recovered["cadence"] == 1.5
    traceback_records = [record for record in caplog.records if record.exc_info]
    assert traceback_records and "synthetic failure" in str(traceback_records[0].exc_info)


def test_parse_client_message_raises_with_a_short_reason() -> None:
    with pytest.raises(InvalidClientMessage, match="not valid JSON"):
        parse_client_message("{nope")
    with pytest.raises(InvalidClientMessage, match="JSON object"):
        parse_client_message("[1]")
    with pytest.raises(InvalidClientMessage, match="unknown message type"):
        parse_client_message({"type": "zzz"})
    with pytest.raises(InvalidClientMessage, match="cadence"):
        parse_client_message({"type": "set_cadence", "cadence": 50})
