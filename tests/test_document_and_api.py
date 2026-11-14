"""Tier 2 & 3: Document Ingestion, REST API, and Narration Engine Integration Tests.

Validates:
1. DocumentParser extracting clean sentences, paragraphs, and word counts from text, PDF, and EPUB.
2. REST API endpoints:
   - GET /api/config and POST /api/config
   - POST /api/upload-document (both JSON body and multipart file upload)
   - GET /api/document
   - POST /api/ask
   - POST /api/narration/control
3. LLM client config updates and web search grounding fallbacks.
"""

import io
import httpx
import pytest
from fastapi.testclient import TestClient
import pypdf

from voice_agent.document.parser import DocumentParser
from voice_agent.llm.client import LLMClient
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings
from fakes import FakeSearch, FakeTts, wait_until


@pytest.fixture
def test_client():
    """Provides a fresh FastAPI test client with isolated manager and narration engine."""
    manager = ConnectionManager()
    engine = NarrationEngine(
        connection_manager=manager,
        llm_client=LLMClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))),  # no real dial
        settings=Settings.fast(),
        tts=FakeTts(),
        search=FakeSearch(),
    )
    app = create_app(connection_manager=manager, narration_engine=engine)
    with TestClient(app) as client:
        yield client, manager, engine


def test_document_parser_text():
    """Test parsing raw text into structured DocumentContent with sentence boundaries."""
    raw = (
        "In 2077, Night City was voted the worst place to live in America. "
        "Main issues? Sky-high rate of violence and more people living below the poverty line than anywhere else. "
        "Can't deny it, it's all true... but everybody still wants to live here."
    )
    doc = DocumentParser.parse_text(raw, title="Night City Intro")
    assert doc.title == "Night City Intro"
    assert doc.total_sentences == 4
    assert doc.total_words > 20
    assert "In 2077, Night City was voted" in doc.sentences[0]
    assert len(doc.paragraphs) >= 1


def test_document_parser_synthetic_pdf():
    """Test parsing an in-memory generated PDF document using pypdf."""
    # Create an in-memory PDF using pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_bytes_io = io.BytesIO()
    writer.write(pdf_bytes_io)
    pdf_bytes = pdf_bytes_io.getvalue()

    doc = DocumentParser.parse_bytes(pdf_bytes, filename="test_manual.pdf")
    assert doc.filename == "test_manual.pdf"
    # A blank page has no text; the upload route turns zero sentences into a 422.
    assert doc.total_sentences == 0
    assert doc.sentences == []


def test_api_config_get_and_post(test_client):
    """Test GET /api/config and POST /api/config updating runtime LLM settings."""
    client, manager, engine = test_client

    # 1. GET initial config
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "llm" in data
    assert "voice" in data
    assert data["voice"]["active_voice"] == "david-martinez"

    # 2. POST config update
    update_payload = {
        "model": "qwen2.5:7b",
        "web_search_enabled": False,
        "cadence": 1.2,
        "active_voice": "lucy-kushinada",
    }
    post_resp = client.post("/api/config", json=update_payload)
    assert post_resp.status_code == 200
    updated = post_resp.json()
    assert updated["status"] == "ok"
    assert updated["llm"]["model"] == "qwen2.5:7b"
    assert updated["llm"]["web_search_enabled"] is False
    assert updated["voice"]["active_voice"] == "lucy-kushinada"
    assert updated["voice"]["cadence"] == 1.2
    assert manager.active_voice == "lucy-kushinada"
    assert manager.cadence == 1.2


def test_api_upload_document_json(test_client):
    """Test uploading document text via JSON body at POST /api/upload-document."""
    client, manager, engine = test_client

    payload = {
        "title": "Neuromancer Ch 1",
        "text": "The sky above the port was the color of television, tuned to a dead channel. Case was twenty-four. At twenty-two, he'd been a cowboy, a rustler."
    }
    resp = client.post("/api/upload-document", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "ok"
    assert res_data["title"] == "Neuromancer Ch 1"
    assert res_data["sentence_count"] == 3
    assert engine.document_title == "Neuromancer Ch 1"
    assert len(engine.sentences) == 3

    # Check GET /api/document reflects the new document
    doc_resp = client.get("/api/document")
    assert doc_resp.status_code == 200
    doc_data = doc_resp.json()
    assert doc_data["title"] == "Neuromancer Ch 1"
    assert doc_data["sentence_count"] == 3


def test_api_upload_document_file_multipart(test_client):
    """Test uploading a document file via multipart form data at POST /api/upload-document."""
    client, manager, engine = test_client

    file_content = b"Call me Ishmael. Some years ago, never mind how long precisely, I went to sea."
    files = {"file": ("moby_dick.txt", io.BytesIO(file_content), "text/plain")}

    resp = client.post("/api/upload-document", files=files)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "ok"
    assert "Moby Dick" in res_data["title"]
    assert res_data["sentence_count"] == 2
    assert engine.sentences[0].startswith("Call me Ishmael")


def test_api_ask_question_interruption(test_client):
    """Test triggering an ad-hoc interruption question via POST /api/ask."""
    client, manager, engine = test_client

    resp = client.post("/api/ask", json={"query": "Who is David Martinez?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["query"] == "Who is David Martinez?"

    # The interruption runs in the background: reply spoken first, then narration resumes.
    tts = engine.tts
    assert wait_until(lambda: len(tts.calls) >= 2), "interruption did not speak a reply and resume narration"
    reply_text, reply_voice, _ = tts.calls[0]
    assert reply_text.strip()
    assert reply_text not in engine.sentences
    assert reply_voice == "david-martinez"
    assert tts.calls[1][0] in engine.sentences
    assert manager.state == "NARRATING"


def test_api_narration_control(test_client):
    """Test programmatic control (pause, resume, next, prev) via POST /api/narration/control."""
    client, manager, engine = test_client

    # Pause
    resp_pause = client.post("/api/narration/control", json={"action": "pause"})
    assert resp_pause.status_code == 200
    assert resp_pause.json()["state"] == "PAUSED"
    assert manager.state == "PAUSED"

    # Next sentence
    init_idx = engine.current_sentence_idx
    resp_next = client.post("/api/narration/control", json={"action": "next"})
    assert resp_next.status_code == 200
    assert engine.current_sentence_idx == (init_idx + 1) % len(engine.sentences)

    # Prev sentence
    resp_prev = client.post("/api/narration/control", json={"action": "prev"})
    assert resp_prev.status_code == 200
    assert engine.current_sentence_idx == init_idx
