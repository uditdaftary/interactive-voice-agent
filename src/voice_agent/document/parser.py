"""Document ingestion and parsing module for Interactive Voice Agent.

Supports extracting clean narrative text from PDF, EPUB, Markdown, and Plain Text files,
segmenting into sentences using token-shielded boundary detection.
"""

from dataclasses import dataclass, field
import html
import io
import logging
from pathlib import Path
import posixpath
import re
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote
import zipfile
import zlib

import pypdf

from voice_agent.cursor.splitter import split_sentences
from voice_agent.document.bm25 import build_rolling_windows

logger = logging.getLogger(__name__)

DOCUMENT_EXTENSIONS = frozenset({".pdf", ".epub", ".txt", ".md"})

# Bounds on what a small crafted file can expand to. Request size itself is deliberately not capped.
MAX_EPUB_MEMBERS = 2000
MAX_EPUB_MEMBER_BYTES = 10 * 1024 * 1024
MAX_EPUB_TOTAL_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 2000


class DocumentParseError(ValueError):
    """The file cannot be read at all: a corrupt archive or PDF, or too many archive members."""


@dataclass
class DocumentContent:
    """Structured representation of an ingested document."""
    title: str
    filename: str
    total_words: int
    total_sentences: int
    sentences: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    passages: List[str] = field(default_factory=list)
    # Human-readable reasons for parts of the file that were not read (corrupt page, oversized chapter, ...).
    skipped_parts: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "title": self.title,
            "filename": self.filename,
            "total_words": self.total_words,
            "total_sentences": self.total_sentences,
            "sentences": self.sentences,
            "paragraphs": self.paragraphs,
            "passages": self.passages,
            "skipped_parts": self.skipped_parts,
        }


def _clean_extracted_text(text: str) -> str:
    """Normalize whitespace and strip artifact control characters."""
    if not text:
        return ""
    # Normalize unicode spaces and quotes
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_document_content(raw_text: str, title: str, filename: str) -> DocumentContent:
    """Build DocumentContent by segmenting raw text into sentences and paragraphs."""
    cleaned = _clean_extracted_text(raw_text)
    # No placeholder text for an empty document: callers see zero sentences and reject it.
    sentences = split_sentences(cleaned) if cleaned else []
    if cleaned and not sentences:
        sentences = [cleaned]

    # Reconstruct readable paragraphs (grouping every 2-4 sentences or keeping headers intact)
    paragraphs: List[str] = []
    current_para: List[str] = []
    for s in sentences:
        if s.startswith("#"):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            paragraphs.append(s)
        else:
            current_para.append(s)
            if len(current_para) >= 3:
                paragraphs.append(" ".join(current_para))
                current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))

    if not paragraphs:
        paragraphs = sentences

    total_words = sum(len(s.split()) for s in sentences)
    passages = build_rolling_windows(sentences)

    return DocumentContent(
        title=title or Path(filename).stem.replace("_", " ").title(),
        filename=filename,
        total_words=total_words,
        total_sentences=len(sentences),
        sentences=sentences,
        paragraphs=paragraphs,
        passages=passages,
    )


def _skip(skipped: List[str], filename: str, reason: str) -> None:
    """Record a part of the file that was not read: logged, and returned to the caller."""
    logger.warning("%r: skipped %s", filename, reason)
    skipped.append(reason)


def _format_bytes(size: int) -> str:
    return f"{size / (1024 * 1024):g} MB" if size >= 1024 * 1024 else f"{size} bytes"


def parse_pdf_bytes(data: bytes, filename: str = "document.pdf") -> DocumentContent:
    """Extract text from PDF file bytes using pypdf.

    Raises DocumentParseError when the file cannot be opened. Pages that fail to extract, and pages
    beyond MAX_PDF_PAGES, are listed in `skipped_parts`.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        total_pages = len(reader.pages)
        title = str(reader.metadata.title).strip() if reader.metadata and reader.metadata.title else ""
    except Exception as exc:  # noqa: BLE001  # pypdf raises arbitrary types on hostile input; re-raised as one
        raise DocumentParseError(f"not a readable PDF: {type(exc).__name__}: {exc}") from exc

    skipped: List[str] = []
    page_texts: List[str] = []
    for index in range(min(total_pages, MAX_PDF_PAGES)):
        try:
            page_text = reader.pages[index].extract_text()
        except Exception as exc:  # noqa: BLE001  # one hostile page must not lose the rest; recorded and logged
            _skip(skipped, filename, f"page {index + 1}: extract failed: {type(exc).__name__}: {exc}")
            continue
        if page_text and page_text.strip():
            page_texts.append(page_text.strip())
    if total_pages > MAX_PDF_PAGES:
        _skip(skipped, filename, f"pages {MAX_PDF_PAGES + 1}-{total_pages}: not read, limit is {MAX_PDF_PAGES} pages")

    document = _build_document_content("\n\n".join(page_texts), title=title, filename=filename)
    document.skipped_parts = skipped
    return document


def _attribute(tag_attributes: str, name: str) -> Optional[str]:
    match = re.search(rf"(?<![\w:-]){name}\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", tag_attributes, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _epub_opf_path(names: List[str], read_member: Callable[[str, str], Optional[bytes]]) -> Optional[str]:
    """The package document named by META-INF/container.xml, else any .opf in the archive."""
    if "META-INF/container.xml" in names:
        container = read_member("META-INF/container.xml", "package file")
        rootfile = re.search(r"<rootfile\b([^<>]*)>", container.decode("utf-8", errors="ignore"), re.IGNORECASE) if container else None
        full_path = _attribute(rootfile.group(1), "full-path") if rootfile else None
        if full_path:
            return full_path
    return next((name for name in names if name.lower().endswith(".opf")), None)


def _epub_spine(opf: str, opf_path: str) -> List[str]:
    """Chapter paths in reading order: spine itemrefs resolved through the manifest, relative to the OPF."""
    manifest: Dict[str, str] = {}
    for item in re.finditer(r"<item\b([^<>]*)>", opf, re.IGNORECASE):
        item_id, href = _attribute(item.group(1), "id"), _attribute(item.group(1), "href")
        if item_id and href:
            manifest[item_id] = href
    spine_open = re.compile(r"<spine\b[^<>]*>", re.IGNORECASE).search(opf)
    if not spine_open:
        return []
    spine_close = re.compile(r"</spine\s*>", re.IGNORECASE).search(opf, spine_open.end())
    spine_block = opf[spine_open.end() : spine_close.start() if spine_close else len(opf)]
    base = posixpath.dirname(opf_path)
    chapters: List[str] = []
    for itemref in re.finditer(r"<itemref\b([^<>]*)>", spine_block, re.IGNORECASE):
        href = manifest.get(_attribute(itemref.group(1), "idref") or "")
        if href is None:
            continue
        path = posixpath.normpath(posixpath.join(base, unquote(href.split("#", 1)[0])))
        if path not in chapters:
            chapters.append(path)
    return chapters


# Every pattern below must stay linear on hostile input: `[^<>]` (not `[^>]`) stops a scan at the next tag
# start, and script/style blocks are cut with two forward searches instead of a lazy `.*?` per opening tag.
_LINE_BREAK_TAG = re.compile(r"<(?:h[1-6]|p|div|br)\b[^<>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^<>]*>")
_DROPPED_BLOCKS = [
    (re.compile(rf"<{tag}\b", re.IGNORECASE), re.compile(rf"</{tag}\s*>", re.IGNORECASE)) for tag in ("script", "style")
]


def _drop_blocks(markup: str, opening: re.Pattern[str], closing: re.Pattern[str]) -> str:
    """Remove every opening..closing block in one pass. An unclosed block swallows the rest, as in a browser."""
    pieces: List[str] = []
    position = 0
    while (start := opening.search(markup, position)) is not None:
        pieces.append(markup[position : start.start()])
        end = closing.search(markup, start.end())
        if end is None:
            return "".join(pieces)
        position = end.end()
    pieces.append(markup[position:])
    return "".join(pieces)


def _chapter_text(markup: str) -> str:
    for opening, closing in _DROPPED_BLOCKS:
        markup = _drop_blocks(markup, opening, closing)
    markup = _LINE_BREAK_TAG.sub("\n", markup)
    markup = _ANY_TAG.sub(" ", markup)
    return re.sub(r"\s+", " ", html.unescape(markup)).strip()


def parse_epub_bytes(data: bytes, filename: str = "document.epub") -> DocumentContent:
    """Extract text from EPUB ebook bytes by parsing XHTML/HTML chapters in spine order.

    Raises DocumentParseError for a file that is not a readable zip or has too many members. A chapter that
    is missing, too large, or unreadable is skipped and listed in `skipped_parts`; the rest still load.
    Sizes are checked against `ZipInfo.file_size` before reading, and `ZipFile.read` never returns more
    than that declared size, so a zip bomb cannot expand past the limits.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data), "r")
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise DocumentParseError(f"not a readable EPUB (zip) file: {type(exc).__name__}: {exc}") from exc

    skipped: List[str] = []
    with archive:
        infos = {info.filename: info for info in archive.infolist()}
        if len(infos) > MAX_EPUB_MEMBERS:
            raise DocumentParseError(f"EPUB has {len(infos)} members, the limit is {MAX_EPUB_MEMBERS}")
        total_read = 0

        def read_member(name: str, label: str) -> Optional[bytes]:
            nonlocal total_read
            info = infos.get(name)
            if info is None:
                _skip(skipped, filename, f"{label} {name}: not in the archive")
            elif info.file_size > MAX_EPUB_MEMBER_BYTES:
                _skip(skipped, filename, f"{label} {name}: exceeds {_format_bytes(MAX_EPUB_MEMBER_BYTES)} uncompressed")
            elif total_read + info.file_size > MAX_EPUB_TOTAL_BYTES:
                _skip(skipped, filename, f"{label} {name}: total uncompressed size would exceed {_format_bytes(MAX_EPUB_TOTAL_BYTES)}")
            else:
                try:
                    content = archive.read(info)
                except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, OSError, EOFError) as exc:
                    _skip(skipped, filename, f"{label} {name}: read failed: {type(exc).__name__}: {exc}")
                else:
                    total_read += len(content)
                    return content
            return None

        title = ""
        chapters: List[str] = []
        opf_path = _epub_opf_path(list(infos), read_member)
        opf_bytes = read_member(opf_path, "package file") if opf_path else None
        if opf_path and opf_bytes is not None:
            opf = opf_bytes.decode("utf-8", errors="ignore")
            match = re.search(r"<dc:title\b[^<>]*>([^<]*)</dc:title\s*>", opf, re.IGNORECASE)
            if match:
                title = html.unescape(match.group(1).strip())
            chapters = _epub_spine(opf, opf_path)

        if not chapters:
            _skip(skipped, filename, "no reading order found in the package document; chapters read in file-name order")
            chapters = sorted(
                name for name in infos if name.lower().endswith((".xhtml", ".html", ".htm")) and not name.startswith("__MACOSX")
            )

        chapter_texts: List[str] = []
        for chapter in chapters:
            content = read_member(chapter, "chapter")
            text = _chapter_text(content.decode("utf-8", errors="ignore")) if content is not None else ""
            if text:
                chapter_texts.append(text)

    document = _build_document_content("\n\n".join(chapter_texts), title=title, filename=filename)
    document.skipped_parts = skipped
    return document


def parse_text_bytes(data: bytes, filename: str = "document.txt") -> DocumentContent:
    """Extract text from plain text or markdown bytes."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="ignore")

    return _build_document_content(text, title=Path(filename).stem.replace("_", " ").title(), filename=filename)


def parse_uploaded_file(data: bytes, filename: str) -> DocumentContent:
    """Autodetect document type and extract structured content."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return parse_pdf_bytes(data, filename=filename)
    elif lower_name.endswith(".epub"):
        return parse_epub_bytes(data, filename=filename)
    else:
        return parse_text_bytes(data, filename=filename)


class DocumentParser:
    """Unified document parser interface supporting PDF, EPUB, TXT, and Markdown."""

    @staticmethod
    def parse_bytes(data: bytes, filename: str = "document.txt") -> DocumentContent:
        return parse_uploaded_file(data, filename)

    @staticmethod
    def parse_text(text: str, title: str = "Custom Document", filename: str = "document.txt") -> DocumentContent:
        return _build_document_content(text, title=title, filename=filename)

