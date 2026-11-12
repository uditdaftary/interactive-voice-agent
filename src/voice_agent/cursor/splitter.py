import re
from typing import List

SENTINEL_DOT = "\uE000"

# Common abbreviations to shield (excluding etc., which has special sentence-end semantics)
_ABBREVIATIONS = (
    r"Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|"
    r"approx|apt|dept|est|govt|inc|ltd|corp|vol|rev|hon|gen|col|maj|capt|lt|sgt|rep|sen"
)
_ABBREVIATION_PATTERN = re.compile(rf"\b({_ABBREVIATIONS})\.", re.IGNORECASE)

# etc. abbreviation: shielded when NOT followed by a capital letter (i.e. not ending a sentence)
_ETC_PATTERN = re.compile(r"\betc\.(?!\s+[A-Z])", re.IGNORECASE)

# Multi-dot abbreviations like e.g., i.e., D.C., U.S.
_MULTI_DOT_ABBREVIATIONS = re.compile(r"\b([eE]\.[gG]\.|[iI]\.[eE]\.|[dD]\.[cC]\.|[uU]\.[sS]\.)")

# Decimal numbers (e.g., 1.5, 3.14159)
_DECIMAL_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")

# Ellipses (... or ..)
_ELLIPSIS_PATTERN = re.compile(r"\.{2,}")

# Common sentence starters and label words to distinguish initials from sentence end
_SENTENCE_STARTERS = {
    "He", "She", "It", "They", "We", "You", "I",
    "The", "This", "That", "These", "Those", "There",
    "Then", "When", "What", "Who", "Why", "How", "Where",
    "If", "As", "In", "On", "At", "But", "And", "So", "However", "Meanwhile"
}

_LABEL_WORDS = {
    "Sentence", "Section", "Chapter", "Paragraph", "Part",
    "Step", "Option", "Item", "Exhibit", "Figure", "Table",
    "Phase", "Track", "Appendix", "Rule"
}

# Initials followed by another initial (e.g. J. K.)
_INITIAL_PAIR_PATTERN = re.compile(r"\b([A-Z])\.(?=\s+[A-Z]\.)")

# Initials followed by a capitalized name (e.g. A. Lincoln, J. Watson), with preceding context check
_INITIAL_NAME_PATTERN = re.compile(r"(?:(\b[A-Za-z]+)\s+)?\b([A-Z])\.\s+([A-Z][a-z]+)")

# Sentence boundary regex:
# Lookbehind supports:
# 1) Dialogue quotes with punctuation inside (width 2): (?<=[.!?]["'”’])
# 2) Dialogue quotes with punctuation outside (width 2): (?<=["'”’][.!?])
# 3) Standalone punctuation (width 1): (?<=[.!?])
# Followed by whitespace \s+
# Lookahead: next token starts with capital letter, digit, or quote
_SENTENCE_BOUNDARY = re.compile(
    r'(?:(?<=[.!?]["\'”’])|(?<=["\'”’][.!?])|(?<=[.!?]))\s+(?=[A-Z0-9"\'“‘])'
)


def normalize_text(text: str) -> str:
    """Normalizes whitespace within a text segment."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _shield_tokens(text: str) -> str:
    """Shields periods in abbreviations, initials, decimals, and ellipses with sentinel character."""
    # Shield ellipses
    text = _ELLIPSIS_PATTERN.sub(lambda m: SENTINEL_DOT * len(m.group(0)), text)
    
    # Shield decimals
    text = _DECIMAL_PATTERN.sub(SENTINEL_DOT, text)
    
    # Shield multi-dot abbreviations (e.g. e.g., i.e., D.C.)
    text = _MULTI_DOT_ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", SENTINEL_DOT), text)
    
    # Shield standard word abbreviations (e.g. Dr., Mr., St., vs.)
    text = _ABBREVIATION_PATTERN.sub(lambda m: m.group(1) + SENTINEL_DOT, text)

    # Shield etc. when inside sentence
    text = _ETC_PATTERN.sub("etc" + SENTINEL_DOT, text)
    
    # Shield initial pairs (e.g. J. K.)
    text = _INITIAL_PAIR_PATTERN.sub(lambda m: m.group(1) + SENTINEL_DOT, text)
    
    # Shield single initials before names if not a label or common sentence starter
    def _shield_initial_name(match: re.Match) -> str:
        prev_word = match.group(1) or ""
        initial = match.group(2)
        next_word = match.group(3)
        
        if (
            prev_word in _LABEL_WORDS
            or next_word in _LABEL_WORDS
            or next_word in _SENTENCE_STARTERS
        ):
            return match.group(0)
            
        prefix = f"{prev_word} " if prev_word else ""
        return f"{prefix}{initial}{SENTINEL_DOT} {next_word}"

    text = _INITIAL_NAME_PATTERN.sub(_shield_initial_name, text)
    
    return text


def _unshield_tokens(text: str) -> str:
    """Restores shielded sentinels back to periods."""
    return text.replace(SENTINEL_DOT, ".")


def _split_paragraph_sentences(paragraph_text: str) -> List[str]:
    """Splits a single paragraph into individual sentences using token-shielding."""
    cleaned = normalize_text(paragraph_text)
    if not cleaned:
        return []
        
    shielded = _shield_tokens(cleaned)
    raw_sentences = _SENTENCE_BOUNDARY.split(shielded)
    
    sentences = []
    for s in raw_sentences:
        unshielded = _unshield_tokens(s).strip()
        if unshielded:
            sentences.append(unshielded)
    return sentences


def split_sentences(text: str) -> List[str]:
    """Splits text into sentences, preserving markdown headers and paragraph boundaries."""
    if not text or not text.strip():
        return []

    # Normalize line breaks
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Split across paragraph boundaries (\n\s*\n+)
    raw_paragraphs = re.split(r"\n\s*\n+", normalized_text)
    
    results: List[str] = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
            
        lines = para.split("\n")
        body_lines: List[str] = []
        
        def flush_body(pending: List[str]) -> None:
            if pending:
                block_text = " ".join(line.strip() for line in pending if line.strip())
                if block_text:
                    results.extend(_split_paragraph_sentences(block_text))
                pending.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^#{1,6}\s+", stripped):
                flush_body(body_lines)
                results.append(stripped)
            else:
                body_lines.append(stripped)
        flush_body(body_lines)

    return results
