"""Document ingestion and parsing module for Interactive Voice Agent.

Supports extracting clean narrative text from PDF, EPUB, Markdown, and Plain Text files,
segmenting into sentences using token-shielded boundary detection.
"""

from dataclasses import dataclass, field
import html
import io
from pathlib import Path
import re
from typing import List, Optional, Union
import zipfile

import pypdf

from voice_agent.cursor.splitter import split_sentences


@dataclass
class DocumentContent:
    """Structured representation of an ingested document."""
    title: str
    filename: str
    total_words: int
    total_sentences: int
    sentences: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "title": self.title,
            "filename": self.filename,
            "total_words": self.total_words,
            "total_sentences": self.total_sentences,
            "sentences": self.sentences,
            "paragraphs": self.paragraphs,
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
    if not cleaned:
        cleaned = "Empty document."

    sentences = split_sentences(cleaned)
    if not sentences:
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

    return DocumentContent(
        title=title or Path(filename).stem.replace("_", " ").title(),
        filename=filename,
        total_words=total_words,
        total_sentences=len(sentences),
        sentences=sentences,
        paragraphs=paragraphs,
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
        except Exception:
            continue

    raw_text = "\n\n".join(page_texts)
    return _build_document_content(raw_text, title=title, filename=filename)


def parse_epub_bytes(data: bytes, filename: str = "document.epub") -> DocumentContent:
    """Extract text from EPUB ebook bytes by parsing XHTML/HTML chapters."""
    raw_texts: List[str] = []
    title = ""

    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            namelist = z.namelist()

            # Attempt to find title in metadata (content.opf)
            for name in namelist:
                if name.endswith(".opf"):
                    try:
                        opf_content = z.read(name).decode("utf-8", errors="ignore")
                        match = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf_content, re.IGNORECASE | re.DOTALL)
                        if match:
                            title = html.unescape(match.group(1).strip())
                            break
                    except Exception:
                        pass

            # Read all xhtml / html documents in reading order
            html_files = [n for n in namelist if n.lower().endswith((".xhtml", ".html", ".htm")) and not n.startswith("__MACOSX")]
            # Sort files naturally
            html_files.sort()

            for hfile in html_files:
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
                except Exception:
                    continue
    except Exception as e:
        raw_texts.append(f"Failed to read EPUB: {e}")

    raw_text = "\n\n".join(raw_texts)
    return _build_document_content(raw_text, title=title, filename=filename)


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

