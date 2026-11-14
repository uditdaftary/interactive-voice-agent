"""Bounded, ordered and honest document parsing (H5, M14, N5)."""

import asyncio
import io
import logging
import random
import struct
import time
import tracemalloc
import zipfile
from collections.abc import Iterator

import httpx
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


def central_header(data: bytes, member: str) -> int:
    """Offset of a member's central-directory header, for crafting corrupt archives."""
    position = 0
    while True:
        position = data.index(b"PK", position)
        name_length = struct.unpack_from("<H", data, position + 28)[0]
        if data[position + 46 : position + 46 + name_length] == member.encode():
            return position
        position += 4


def with_future_zip_version(data: bytes) -> bytes:
    """`zipfile` refuses to open an archive that needs a newer extractor (NotImplementedError)."""
    crafted = bytearray(data)
    crafted[central_header(data, "OEBPS/ch1.xhtml") + 6] = 0xF0
    return bytes(crafted)


def with_invalid_utf8_member_name(data: bytes) -> bytes:
    """Flag a name as UTF-8, then break it: opening the archive raises UnicodeDecodeError."""
    crafted = bytearray(data)
    header = central_header(data, "OEBPS/ch1.xhtml")
    crafted[header + 9] |= 0x08
    crafted[header + 46] = 0xFF
    return bytes(crafted)


def with_unsupported_compression(data: bytes, member: str) -> bytes:
    """Reading that one member raises NotImplementedError; the others are untouched."""
    crafted = bytearray(data)
    struct.pack_into("<H", crafted, central_header(data, member) + 10, 99)
    return bytes(crafted)


def with_misplaced_directory(data: bytes) -> bytes:
    """A wrong central-directory offset shifts every member offset negative: reads raise ValueError."""
    crafted = bytearray(data)
    end_of_directory = data.rindex(b"PK")
    offset = struct.unpack_from("<I", data, end_of_directory + 16)[0]
    struct.pack_into("<I", crafted, end_of_directory + 16, offset + 5000)
    return bytes(crafted)


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


@pytest.mark.parametrize("corrupt", [with_future_zip_version, with_invalid_utf8_member_name], ids=["future-version", "bad-utf8-name"])
def test_epub_that_cannot_be_opened_raises_parse_error(corrupt) -> None:
    """Stdlib zipfile raises NotImplementedError and UnicodeDecodeError here; neither may escape the parser."""
    with pytest.raises(DocumentParseError, match="EPUB"):
        DocumentParser.parse_bytes(corrupt(two_chapter_epub(["c1", "c2"])), "broken.epub")


def test_epub_unsupported_compression_skips_only_that_chapter() -> None:
    data = with_unsupported_compression(two_chapter_epub(["c1", "c2"]), "OEBPS/ch2.xhtml")

    doc = DocumentParser.parse_bytes(data, "book.epub")

    assert doc.sentences == ["Alpha one is here.", "Alpha two follows."]
    assert len(doc.skipped_parts) == 1
    assert "OEBPS/ch2.xhtml" in doc.skipped_parts[0]
    assert "NotImplementedError" in doc.skipped_parts[0]


def test_epub_with_misplaced_directory_reports_every_member_instead_of_raising() -> None:
    """ValueError from a negative seek is per member too, so it lands in skipped_parts."""
    doc = DocumentParser.parse_bytes(with_misplaced_directory(two_chapter_epub(["c1", "c2"])), "book.epub")

    assert doc.sentences == []
    assert any("ValueError" in reason for reason in doc.skipped_parts)


def test_epub_random_corruption_only_ever_raises_parse_error() -> None:
    """Seeded fuzz over a valid book: every outcome is a document or a DocumentParseError, never another type."""
    valid = two_chapter_epub(["c1", "c2"])
    rng = random.Random(20261114)
    outcomes = {"document": 0, "parse_error": 0}

    for _ in range(600):
        data = bytearray(valid)
        for _ in range(rng.randint(1, 6)):
            data[rng.randrange(len(data))] = rng.randrange(256)
        try:
            DocumentParser.parse_bytes(bytes(data), "fuzz.epub")
        except DocumentParseError:
            outcomes["parse_error"] += 1
        else:
            outcomes["document"] += 1

    assert outcomes["document"] > 0 and outcomes["parse_error"] > 0  # the seed exercises both paths


def test_skipped_part_log_escapes_member_names_from_the_archive(caplog: pytest.LogCaptureFixture) -> None:
    """A crafted href with a line break must not forge a log line."""
    data = make_epub(
        {"OEBPS/ch1.xhtml": chapter("Alpha one is here. Alpha two follows.")},
        {"c1": "ch1.xhtml", "c2": "x%0A2026-11-14 CRITICAL forged.xhtml"},
        ["c1", "c2"],
    )

    with caplog.at_level(logging.WARNING):
        doc = DocumentParser.parse_bytes(data, "book.epub")

    assert any("forged" in reason for reason in doc.skipped_parts)
    assert caplog.records
    assert not any(len(record.getMessage().splitlines()) > 1 for record in caplog.records)


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


def test_corrupt_pdf_upload_returns_422(client: TestClient) -> None:
    title_before = client.get("/api/document").json()["title"]

    resp = client.post("/api/upload-document", files={"file": ("broken.pdf", b"%PDF-1.4 nope", "application/pdf")})

    assert resp.status_code == 422
    assert "PDF" in resp.json()["detail"]
    assert client.get("/api/document").json()["title"] == title_before


def test_corrupt_epub_upload_returns_422_and_ingests_no_error_text(client: TestClient) -> None:
    resp = client.post("/api/upload-document", files={"file": ("broken.epub", b"not a zip", "application/epub+zip")})

    assert resp.status_code == 422
    assert "Failed to read" not in " ".join(client.get("/api/document").json()["sentences"])


@pytest.mark.parametrize("corrupt", [with_future_zip_version, with_invalid_utf8_member_name], ids=["future-version", "bad-utf8-name"])
def test_unopenable_epub_upload_returns_422_not_500(client: TestClient, corrupt) -> None:
    title_before = client.get("/api/document").json()["title"]
    data = corrupt(two_chapter_epub(["c1", "c2"]))

    resp = client.post("/api/upload-document", files={"file": ("broken.epub", data, "application/epub+zip")})

    assert resp.status_code == 422
    assert "EPUB" in resp.json()["detail"]
    assert client.get("/api/document").json()["title"] == title_before


def test_epub_upload_with_one_bad_chapter_loads_the_rest(client: TestClient) -> None:
    data = with_unsupported_compression(two_chapter_epub(["c1", "c2"]), "OEBPS/ch2.xhtml")

    resp = client.post("/api/upload-document", files={"file": ("book.epub", data, "application/epub+zip")})

    assert resp.status_code == 200
    assert resp.json()["sentences"] == ["Alpha one is here.", "Alpha two follows."]
    assert "OEBPS/ch2.xhtml" in resp.json()["skipped_parts"][0]


@pytest.mark.parametrize("name", ["notes.docx", "run.exe", "page.html", "archive.tar.gz", "noextension"])
def test_unknown_extension_415(client: TestClient, name: str) -> None:
    resp = client.post("/api/upload-document", files={"file": (name, b"Some text here.", "application/octet-stream")})

    assert resp.status_code == 415


@pytest.mark.parametrize("name", ["Book.EPUB", "notes.TXT", "readme.md"])
def test_allowed_extensions_are_case_insensitive(client: TestClient, name: str) -> None:
    content = two_chapter_epub(["c1", "c2"]) if name.lower().endswith("epub") else b"Hello there. General Kenobi."

    resp = client.post("/api/upload-document", files={"file": (name, content, "application/octet-stream")})

    assert resp.status_code == 200


def test_upload_reports_skipped_parts(client: TestClient) -> None:
    data = make_epub(
        {"OEBPS/ch1.xhtml": chapter("Alpha one is here. Alpha two follows.")},
        {"c1": "ch1.xhtml", "c2": "gone.xhtml"},
        ["c1", "c2"],
    )

    resp = client.post("/api/upload-document", files={"file": ("book.epub", data, "application/epub+zip")})

    assert resp.status_code == 200
    assert resp.json()["skipped_parts"] == ["chapter OEBPS/gone.xhtml: not in the archive"]


def test_upload_with_nothing_readable_still_reports_skipped_parts(client: TestClient) -> None:
    data = make_epub({}, {"c1": "gone.xhtml"}, ["c1"])

    resp = client.post("/api/upload-document", files={"file": ("book.epub", data, "application/epub+zip")})

    assert resp.status_code == 422
    assert resp.json()["detail"] == "document contains no readable text"
    assert resp.json()["skipped_parts"] == ["chapter OEBPS/gone.xhtml: not in the archive"]


def test_zip_bomb_epub_upload_is_rejected_in_bounded_memory(client: TestClient) -> None:
    """The P1 exit gate, at the route and with the shipped limits: a 50 MB chapter inside a ~50 KB file."""
    bomb = make_epub({"OEBPS/bomb.xhtml": b"0" * (50 * 1024 * 1024)}, {"c1": "bomb.xhtml"}, ["c1"])
    assert len(bomb) < 1024 * 1024
    title_before = client.get("/api/document").json()["title"]

    tracemalloc.start()
    try:
        resp = client.post("/api/upload-document", files={"file": ("bomb.epub", bomb, "application/epub+zip")})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert resp.status_code == 422
    assert resp.json()["detail"] == "document contains no readable text"
    assert any("bomb.xhtml" in reason and "10 MB uncompressed" in reason for reason in resp.json()["skipped_parts"])
    assert peak < 20 * 1024 * 1024
    assert client.get("/api/document").json()["title"] == title_before


async def test_upload_parses_off_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_parse(data: bytes, filename: str = "document.txt") -> parser.DocumentContent:
        time.sleep(0.3)
        return parser.parse_text_bytes(data, filename=filename)

    monkeypatch.setattr(DocumentParser, "parse_bytes", staticmethod(slow_parse))
    transport = httpx.ASGITransport(app=build_app())

    async def longest_loop_stall(until: "asyncio.Task[httpx.Response]") -> float:
        """Tick every 10 ms while the upload runs; a blocked loop shows up as one long gap between ticks."""
        worst, last = 0.0, time.perf_counter()
        while not until.done():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            worst, last = max(worst, now - last), now
        return worst

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        upload = asyncio.create_task(
            http.post("/api/upload-document", files={"file": ("slow.txt", b"One sentence. Two sentences.", "text/plain")})
        )
        stall = await longest_loop_stall(upload)
        resp = await upload

    assert resp.status_code == 200
    assert stall < 0.1
