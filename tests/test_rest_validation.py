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

    assert resp.status_code == 422
    assert rig.llm.get_config() == llm_before


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


def test_ws_invalid_values_ignored_without_state_change(rig: Rig) -> None:
    """Each bad control message is dropped; the first reply on the socket is for the later valid one."""
    with rig.client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state_sync"
        ws.send_json({"type": "set_cadence", "cadence": 99})
        ws.send_json({"type": "select_voice", "voice_id": "Bad Voice!"})
        ws.send_json({"type": "set_mic_source", "source": "quantum"})
        ws.send_json({"type": "playback_control", "action": "explode"})
        ws.send_json({"type": "set_cadence", "cadence": 1.2})

        reply = ws.receive_json()

    assert reply["type"] == "state_sync"
    assert reply["cadence"] == 1.2
    assert reply["active_voice"] == "david-martinez"
    assert reply["mic_source"] == "browser"
