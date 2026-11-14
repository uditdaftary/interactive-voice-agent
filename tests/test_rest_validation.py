"""REST bodies are validated by Pydantic models before any handler code applies them (M8, M10)."""

from collections.abc import Iterator
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeTts
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager


class Rig(NamedTuple):
    client: TestClient
    llm: FakeLLM
    manager: ConnectionManager


@pytest.fixture
def rig() -> Iterator[Rig]:
    llm = FakeLLM()
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, llm_client=llm, tts=FakeTts(), search=FakeSearch())
    with TestClient(app) as client:
        yield Rig(client, llm, manager)


def test_config_rejects_out_of_range_cadence_and_applies_nothing(rig: Rig) -> None:
    llm_before = rig.llm.get_config()

    resp = rig.client.post("/api/config", json={"cadence": 1e-9, "model": "x", "active_voice": "lucy-kushinada"})

    assert resp.status_code == 422
    assert rig.llm.get_config() == llm_before
    assert rig.manager.cadence == 1.0
    assert rig.manager.active_voice == "david-martinez"


@pytest.mark.parametrize(
    "bad_field",
    [
        {"modle": "x"},
        {"base_url": "ftp://example.com"},
        {"base_url": "not a url"},
        {"base_url": ""},
        {"provider": "Open AI!"},
        {"provider": "x" * 33},
        {"model": "m" * 129},
        {"api_key": "k" * 513},
        {"web_search_enabled": "sometimes"},
        {"active_voice": "Lucy Kushinada"},
        {"active_voice": "../etc/passwd"},
        {"active_voice": ""},
        {"cadence": 3.01},
        {"cadence": "fast"},
        {"mic_source": "quantum"},
    ],
)
def test_config_rejects_invalid_field_and_applies_nothing(rig: Rig, bad_field: dict[str, Any]) -> None:
    """The valid `model` rides along with every bad field, so a partial update would be visible."""
    llm_before = rig.llm.get_config()

    resp = rig.client.post("/api/config", json={"model": "ok-model", **bad_field})

    assert resp.status_code == 422
    assert rig.llm.get_config() == llm_before


def test_config_rejects_unknown_field(rig: Rig) -> None:
    resp = rig.client.post("/api/config", json={"unexpected": 1})

    assert resp.status_code == 422


def test_config_text_plain_rejected(rig: Rig) -> None:
    llm_before = rig.llm.get_config()

    resp = rig.client.post("/api/config", content='{"model": "x"}', headers={"Content-Type": "text/plain"})

    assert resp.status_code == 415
    assert rig.llm.get_config() == llm_before


JSON_ROUTE_BODIES = {
    "/api/config": '{"model": "x"}',
    "/api/ask": '{"query": "Who is Lucy?"}',
    "/api/narration/control": '{"action": "pause"}',
}


@pytest.mark.parametrize("path", list(JSON_ROUTE_BODIES))
@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "text/html", "application/jsonp"])
def test_json_routes_reject_a_non_json_content_type_with_415(rig: Rig, path: str, content_type: str) -> None:
    """A cross-site form can send these types without a preflight; the body must never be read as JSON."""
    llm_before = rig.llm.get_config()
    state_before = rig.manager.state

    resp = rig.client.post(path, content=JSON_ROUTE_BODIES[path], headers={"Content-Type": content_type})

    assert resp.status_code == 415
    assert "application/json" in resp.json()["detail"]
    assert rig.llm.get_config() == llm_before
    assert rig.manager.state == state_before


@pytest.mark.parametrize("path", list(JSON_ROUTE_BODIES))
def test_json_routes_reject_a_missing_content_type_with_415(rig: Rig, path: str) -> None:
    request = rig.client.build_request("POST", path, content=JSON_ROUTE_BODIES[path])
    request.headers.pop("content-type", None)

    assert rig.client.send(request).status_code == 415


@pytest.mark.parametrize("path", list(JSON_ROUTE_BODIES))
@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8", "Application/JSON", "application/vnd.api+json"])
def test_json_routes_accept_json_media_types(rig: Rig, path: str, content_type: str) -> None:
    resp = rig.client.post(path, content=JSON_ROUTE_BODIES[path], headers={"Content-Type": content_type})

    assert resp.status_code == 200


def test_content_type_is_checked_before_the_body(rig: Rig) -> None:
    """An invalid body with the wrong type is a 415 (wrong kind of request), not a 422."""
    assert rig.client.post("/api/config", content="not json", headers={"Content-Type": "text/plain"}).status_code == 415
    assert rig.client.post("/api/config", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize("body", ["[1, 2]", '"text"', "null", "42"])
def test_config_non_object_json_rejected(rig: Rig, body: str) -> None:
    resp = rig.client.post("/api/config", content=body, headers={"Content-Type": "application/json"})

    assert resp.status_code == 422


def test_config_applies_valid_body_including_legacy_aliases(rig: Rig) -> None:
    """The HUD sends base_url and endpoint (and model and model_name) together; both spellings must pass."""
    body = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "endpoint": "http://localhost:11434",
        "model": "llama3.2",
        "model_name": "llama3.2",
        "api_key": "",
        "web_search_enabled": True,
        "active_voice": "lucy-kushinada",
        "cadence": 1.5,
        "mic_source": "host",
    }

    resp = rig.client.post("/api/config", json=body)

    assert resp.status_code == 200
    config = rig.llm.get_config()
    assert (config["provider"], config["base_url"], config["model"]) == ("ollama", "http://localhost:11434", "llama3.2")
    assert (rig.manager.active_voice, rig.manager.cadence, rig.manager.mic_source) == ("lucy-kushinada", 1.5, "host")


def test_config_alias_alone_is_enough(rig: Rig) -> None:
    resp = rig.client.post("/api/config", json={"endpoint": "https://api.groq.com/openai/v1", "model_name": "m"})

    assert resp.status_code == 200
    assert rig.llm.get_config()["base_url"] == "https://api.groq.com/openai/v1"
    assert rig.llm.get_config()["model"] == "m"


def test_config_accepts_builtin_provider_sentinel(rig: Rig) -> None:
    """The HUD's default 'builtin' provider sends the placeholder endpoint 'in-process'."""
    resp = rig.client.post(
        "/api/config",
        json={"provider": "builtin", "base_url": "in-process", "model": "builtin-assistant"},
    )

    assert resp.status_code == 200
    assert rig.llm.get_config()["base_url"] == "in-process"


def test_playback_rejects_unknown_action(rig: Rig) -> None:
    resp = rig.client.post("/api/narration/control", json={"action": "explode"})

    assert resp.status_code == 422


def test_playback_defaults_to_resume_and_accepts_known_actions(rig: Rig) -> None:
    for action in ("pause", "next", "prev", "stop"):
        assert rig.client.post("/api/narration/control", json={"action": action}).status_code == 200


@pytest.mark.parametrize("query", ["", "   ", "x" * 2001])
def test_ask_rejects_empty_and_overlong(rig: Rig, query: str) -> None:
    assert rig.client.post("/api/ask", json={"query": query}).status_code == 422


def test_ask_rejects_missing_query_and_non_string(rig: Rig) -> None:
    assert rig.client.post("/api/ask", json={}).status_code == 422
    assert rig.client.post("/api/ask", json={"query": ["a"]}).status_code == 422


def test_mic_source_enum(rig: Rig) -> None:
    for source in ("browser", "host"):
        assert rig.client.post("/api/config", json={"mic_source": source}).status_code == 200
        assert rig.manager.mic_source == source
    assert rig.client.post("/api/config", json={"mic_source": "usb-sdr"}).status_code == 422
    assert rig.manager.mic_source == "host"


def test_empty_document_returns_422_and_keeps_old(rig: Rig) -> None:
    title_before = rig.client.get("/api/document").json()["title"]

    for files in (
        {"file": ("blank.txt", b"   \n\n\t  ", "text/plain")},
        {"file": ("blank.md", b"\n\n", "text/markdown")},
    ):
        resp = rig.client.post("/api/upload-document", files=files)
        assert resp.status_code == 422
        assert resp.json() == {"detail": "document contains no readable text"}

    resp = rig.client.post("/api/upload-document", json={"text": "   \n  ", "title": "Blank"})
    assert resp.status_code == 422
    assert resp.json() == {"detail": "document contains no readable text"}

    after = rig.client.get("/api/document").json()
    assert after["title"] == title_before
    assert after["sentence_count"] > 1


def test_text_document_validation(rig: Rig) -> None:
    assert rig.client.post("/api/upload-document", json={"text": ""}).status_code == 422
    assert rig.client.post("/api/upload-document", json={}).status_code == 422
    assert rig.client.post("/api/upload-document", json={"text": "Hi there.", "title": "t" * 201}).status_code == 422
    assert rig.client.post("/api/upload-document", json={"text": ["a"]}).status_code == 422


def test_text_document_loads(rig: Rig) -> None:
    resp = rig.client.post("/api/upload-document", json={"text": "First line. Second line.", "title": "Mine"})

    assert resp.status_code == 200
    assert resp.json()["title"] == "Mine"
    assert resp.json()["skipped_parts"] == []
    assert rig.client.get("/api/document").json()["title"] == "Mine"


def test_upload_text_plain_body_rejected(rig: Rig) -> None:
    resp = rig.client.post(
        "/api/upload-document", content='{"text": "Hello there."}', headers={"Content-Type": "text/plain"}
    )

    assert resp.status_code == 415


def test_ws_invalid_values_rejected_without_state_change(rig: Rig) -> None:
    """Each bad control message gets an error reply and changes nothing."""
    with rig.client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state_sync"
        for bad in (
            {"type": "set_cadence", "cadence": 99},
            {"type": "select_voice", "voice_id": "Bad Voice!"},
            {"type": "set_mic_source", "source": "quantum"},
            {"type": "playback_control", "action": "explode"},
        ):
            ws.send_json(bad)
            assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "set_cadence", "cadence": 1.2})
        reply = ws.receive_json()

    assert reply["type"] == "state_sync"
    assert reply["cadence"] == 1.2
    assert reply["active_voice"] == "david-martinez"
    assert reply["mic_source"] == "browser"
