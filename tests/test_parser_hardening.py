"""Bounded, ordered and honest document parsing (H5, M14, N5)."""

import io
import logging
import zipfile

import pytest

from voice_agent.document.parser import DocumentParser

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


