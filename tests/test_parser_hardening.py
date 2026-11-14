"""Bounded, ordered and honest document parsing (H5, M14, N5)."""

import io
import logging
import time
import tracemalloc
import zipfile
from collections.abc import Iterator

import pypdf
import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeSearch, FakeTts
from voice_agent.document import parser
from voice_agent.document.parser import DocumentParseError, DocumentParser
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager

CONTAINER_XML = (
    '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def chapter(text: str) -> str:
    return f"<html><body><p>{text}</p></body></html>"


def make_epub(
    files: dict[str, bytes | str],
    manifest: dict[str, str],
    spine: list[str] | None,
    *,
    title: str = "Test Book",
) -> bytes:
    """`files` maps zip paths to content, `manifest` maps item ids to hrefs relative to OEBPS/."""
    items = "".join(f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>' for item_id, href in manifest.items())
    spine_xml = "" if spine is None else "<spine>" + "".join(f'<itemref idref="{i}"/>' for i in spine) + "</spine>"
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></metadata>'
        f"<manifest>{items}</manifest>{spine_xml}</package>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", opf)
        for path, content in files.items():
            archive.writestr(path, content)
    return buffer.getvalue()


def two_chapter_epub(spine: list[str] | None) -> bytes:
    return make_epub(
        {
            "OEBPS/ch1.xhtml": chapter("Alpha one is here. Alpha two follows."),
            "OEBPS/ch2.xhtml": chapter("Beta one is here. Beta two follows."),
        },
        {"c1": "ch1.xhtml", "c2": "ch2.xhtml"},
        spine,
    )


def test_epub_spine_order() -> None:
    doc = DocumentParser.parse_bytes(two_chapter_epub(spine=["c2", "c1"]), "book.epub")

    assert doc.title == "Test Book"
    assert doc.sentences == ["Beta one is here.", "Beta two follows.", "Alpha one is here.", "Alpha two follows."]
    assert doc.skipped_parts == []


def test_epub_without_spine_reads_sorted_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        doc = DocumentParser.parse_bytes(two_chapter_epub(spine=None), "book.epub")

    assert doc.sentences[0] == "Alpha one is here."
    assert len(doc.skipped_parts) == 1
    assert "reading order" in doc.skipped_parts[0]
    assert "book.epub" in caplog.text


def test_epub_spine_with_missing_member_loads_the_rest() -> None:
    data = make_epub(
        {"OEBPS/ch1.xhtml": chapter("Alpha one is here. Alpha two follows.")},
        {"c1": "ch1.xhtml", "c2": "gone.xhtml", "c3": "ch1.xhtml"},
        ["c2", "c1", "c3"],
    )

    doc = DocumentParser.parse_bytes(data, "book.epub")

    assert "Alpha one is here." in doc.sentences
    assert doc.sentences.count("Alpha one is here.") == 1  # a chapter listed twice is read once
    assert doc.skipped_parts == ["chapter OEBPS/gone.xhtml: not in the archive"]


def test_epub_percent_encoded_and_nested_hrefs_resolve() -> None:
    data = make_epub(
        {"OEBPS/text/my chapter.xhtml": chapter("Gamma one is here. Gamma two follows.")},
        {"c1": "text/my%20chapter.xhtml#start"},
        ["c1"],
    )

    doc = DocumentParser.parse_bytes(data, "book.epub")

    assert "Gamma one is here." in doc.sentences
    assert doc.skipped_parts == []


def test_epub_oversized_member_skipped(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(parser, "MAX_EPUB_MEMBER_BYTES", 800)  # above container.xml and the OPF, below ch2
    data = make_epub(
        {
            "OEBPS/ch1.xhtml": "<p>Short one. Short two.</p>",
            "OEBPS/ch2.xhtml": "<p>" + "Long sentence here. " * 100 + "</p>",
        },
        {"c1": "ch1.xhtml", "c2": "ch2.xhtml"},
        ["c1", "c2"],
    )

    with caplog.at_level(logging.WARNING):
        doc = DocumentParser.parse_bytes(data, "book.epub")

    assert doc.sentences == ["Short one.", "Short two."]
    assert len(doc.skipped_parts) == 1
    assert "OEBPS/ch2.xhtml" in doc.skipped_parts[0]
    assert "uncompressed" in doc.skipped_parts[0]
    assert "book.epub" in caplog.text


def test_epub_total_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser, "MAX_EPUB_TOTAL_BYTES", 400)
    body = "<p>" + "Word here. " * 15 + "</p>"  # ~170 bytes
    data = make_epub(
        {f"OEBPS/ch{i}.xhtml": body for i in range(1, 5)},
        {f"c{i}": f"ch{i}.xhtml" for i in range(1, 5)},
        [f"c{i}" for i in range(1, 5)],
    )

    doc = DocumentParser.parse_bytes(data, "book.epub")

    assert doc.sentences
    assert any("total" in reason for reason in doc.skipped_parts)


def test_epub_zip_bomb_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser, "MAX_EPUB_MEMBER_BYTES", 1024 * 1024)
    bomb = b"0" * (50 * 1024 * 1024)
    data = make_epub(
        {"OEBPS/bomb.xhtml": bomb, "OEBPS/ch1.xhtml": chapter("Alpha one is here. Alpha two follows.")},
        {"c1": "bomb.xhtml", "c2": "ch1.xhtml"},
        ["c1", "c2"],
    )
    del bomb
    assert len(data) < 1024 * 1024

    tracemalloc.start()
    try:
        doc = DocumentParser.parse_bytes(data, "bomb.epub")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 20 * 1024 * 1024
    assert "Alpha one is here." in doc.sentences
    assert any("bomb.xhtml" in reason and "uncompressed" in reason for reason in doc.skipped_parts)


def test_epub_member_count_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser, "MAX_EPUB_MEMBERS", 4)

    with pytest.raises(DocumentParseError, match="members"):
        DocumentParser.parse_bytes(two_chapter_epub(spine=["c1", "c2"]), "book.epub")


@pytest.mark.parametrize(
    "markup",
    ["<script>" * 150_000, "<p" * 500_000, "<style>x" * 100_000, "<" * 1_000_000],
    ids=["unclosed-script", "unterminated-tag", "unclosed-style", "bare-brackets"],
)
def test_epub_pathological_markup_parses_in_linear_time(markup: str) -> None:
    data = make_epub({"OEBPS/ch1.xhtml": markup}, {"c1": "ch1.xhtml"}, ["c1"])

    started = time.perf_counter()
    DocumentParser.parse_bytes(data, "evil.epub")

    assert time.perf_counter() - started < 5.0


@pytest.mark.parametrize("payload", [b"not a zip at all", b"", b"PK\x03\x04truncated"])
def test_corrupt_epub_raises(payload: bytes) -> None:
    with pytest.raises(DocumentParseError) as raised:
        DocumentParser.parse_bytes(payload, "broken.epub")

    assert "EPUB" in str(raised.value)


def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentParseError, match="PDF"):
        DocumentParser.parse_bytes(b"%PDF-1.4 this is not a pdf", "broken.pdf")


def test_pdf_page_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parser, "MAX_PDF_PAGES", 2)
    writer = pypdf.PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)

    doc = DocumentParser.parse_bytes(buffer.getvalue(), "long.pdf")

    assert doc.skipped_parts == ["pages 3-5: not read, limit is 2 pages"]


def test_pdf_page_extract_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = pypdf.PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)

    def broken_extract(self: pypdf.PageObject, *args: object, **kwargs: object) -> str:
        raise ValueError("bad content stream")

    monkeypatch.setattr(pypdf.PageObject, "extract_text", broken_extract)

    doc = DocumentParser.parse_bytes(buffer.getvalue(), "bad.pdf")

    assert doc.skipped_parts == [
        "page 1: extract failed: ValueError: bad content stream",
        "page 2: extract failed: ValueError: bad content stream",
    ]


# ---------------------------------------------------------------------------
# Upload route
# ---------------------------------------------------------------------------


def build_app():
    return create_app(connection_manager=ConnectionManager(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(build_app()) as test_client:
        yield test_client


