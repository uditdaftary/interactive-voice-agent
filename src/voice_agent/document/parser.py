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
from typing import Dict, List, Optional
from urllib.parse import unquote
import zipfile

import pypdf

from voice_agent.cursor.splitter import split_sentences
from voice_agent.document.bm25 import build_rolling_windows

logger = logging.getLogger(__name__)


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


def parse_pdf_bytes(data: bytes, filename: str = "document.pdf") -> DocumentContent:
    """Extract text from PDF file bytes using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()

    page_texts: List[str] = []
    for page in reader.pages:
        try:
            pt = page.extract_text()
            if pt and pt.strip():
                page_texts.append(pt.strip())
        except Exception:  # noqa: BLE001  # S3.3 RC4
            continue

    raw_text = "\n\n".join(page_texts)
    return _build_document_content(raw_text, title=title, filename=filename)


def _skip(skipped: List[str], filename: str, reason: str) -> None:
    """Record a part of the file that was not read: logged, and returned to the caller."""
    logger.warning("%r: skipped %s", filename, reason)
    skipped.append(reason)


def _attribute(tag_attributes: str, name: str) -> Optional[str]:
    match = re.search(rf"(?<![\w:-]){name}\s*=\s*([\"'])(.*?)\1", tag_attributes, re.IGNORECASE | re.DOTALL)
    return match.group(2) if match else None


def _epub_opf_path(archive: zipfile.ZipFile) -> Optional[str]:
    """The package document named by META-INF/container.xml, else any .opf in the archive."""
    names = archive.namelist()
    if "META-INF/container.xml" in names:
        container = archive.read("META-INF/container.xml").decode("utf-8", errors="ignore")
        rootfile = re.search(r"<rootfile\b([^<>]*)>", container, re.IGNORECASE)
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
    spine_block = re.search(r"<spine\b[^<>]*>(.*?)</spine\s*>", opf, re.IGNORECASE | re.DOTALL)
    if not spine_block:
        return []
    base = posixpath.dirname(opf_path)
    chapters: List[str] = []
    for itemref in re.finditer(r"<itemref\b([^<>]*)>", spine_block.group(1), re.IGNORECASE):
        href = manifest.get(_attribute(itemref.group(1), "idref") or "")
        if href is None:
            continue
        path = posixpath.normpath(posixpath.join(base, unquote(href.split("#", 1)[0])))
        if path not in chapters:
            chapters.append(path)
    return chapters


def parse_epub_bytes(data: bytes, filename: str = "document.epub") -> DocumentContent:
    """Extract text from EPUB ebook bytes by parsing XHTML/HTML chapters in spine order."""
    raw_texts: List[str] = []
    skipped: List[str] = []
    title = ""

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            namelist = z.namelist()
            chapters: List[str] = []

            opf_path = _epub_opf_path(z)
            if opf_path in namelist:
                opf_content = z.read(opf_path).decode("utf-8", errors="ignore")
                match = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf_content, re.IGNORECASE | re.DOTALL)
                if match:
                    title = html.unescape(match.group(1).strip())
                chapters = _epub_spine(opf_content, opf_path)

            if not chapters:
                _skip(skipped, filename, "no reading order found in the package document; chapters read in file-name order")
                chapters = sorted(
                    n for n in namelist if n.lower().endswith((".xhtml", ".html", ".htm")) and not n.startswith("__MACOSX")
                )

            for hfile in chapters:
                if hfile not in namelist:
                    _skip(skipped, filename, f"chapter {hfile}: not in the archive")
                    continue
                try:
                    content = z.read(hfile).decode("utf-8", errors="ignore")
                    # Strip script and style blocks
                    content = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
                    # Convert headers and paragraph tags to linebreaks
                    content = re.sub(r"<(h[1-6]|p|div|br)[^>]*>", "\n", content, flags=re.IGNORECASE)
                    # Strip remaining HTML tags
                    content = re.sub(r"<[^>]+>", " ", content)
                    content = html.unescape(content)
                    cleaned_block = re.sub(r"\s+", " ", content).strip()
                    if cleaned_block:
                        raw_texts.append(cleaned_block)
                except Exception:  # noqa: BLE001  # S3.3 RC4
                    continue
    except Exception as e:  # noqa: BLE001  # S3.3 RC4
        raw_texts.append(f"Failed to read EPUB: {e}")

    raw_text = "\n\n".join(raw_texts)
    document = _build_document_content(raw_text, title=title, filename=filename)
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

