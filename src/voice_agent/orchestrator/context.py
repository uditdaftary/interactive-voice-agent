import re
import unicodedata
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a companion narrator. Provide precise, brief answers (< 3 sentences). "
    "Be direct, conversational, and answer based on the story context provided."
)

# Regex targeting ChatML, Llama-style instruction delimiters, and control tokens
_CHATML_CONTROL_TOKENS = re.compile(
    r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|im_sep\|>|'
    r'\[INST\]|\[/INST\]|<s>|</s>|<<SYS>>|<</SYS>>|<\|.*?\|>',
    re.IGNORECASE
)

# The fence tags build_messages wraps untrusted text in. Callers collapse whitespace after sanitising, which
# turns "</book_passages\n>" into "</book_passages >", so a tag that is only blanks and well-formed attributes
# must match whatever the whitespace between its parts (newlines, any number of blanks). Junk after the tag
# name is different: a stray "<current_sentence" must not swallow the text after it, so a junk body only counts
# when it closes with '>' on the same line within 200 characters and without another '<' in between. A bare tag
# start followed only by whitespace up to the next '<' or the end of the text (a tag cut off) is removed too;
# the lookahead lets one pass remove many of them, where matching only at the very end took one pass each.
_FENCE_ATTRIBUTE = r'\s+[\w:.-]+\s*=\s*(?:"[^"<>]{0,200}"|\'[^\'<>]{0,200}\'|[^\s<>"\']{1,200})'
_FENCE_TAGS = re.compile(
    r'<\s*/?\s*(?:book_passages|current_sentence|web_results)\b'
    r'(?:(?:' + _FENCE_ATTRIBUTE + r')*\s*/?>|[^<>\n]{0,200}>|\s*(?=<|$))',
    re.IGNORECASE,
)

# Characters that show nothing but can sit inside a tag name, so "</book_\u00adpassages>" is not a tag to the
# regexes yet reads as one to a model whose tokenizer drops or merges them. str.split() does not treat them as
# blanks either. Control characters (preserves newline, carriage return, and tab) plus the invisible marks that
# are not category Cf: the combining grapheme joiner and the variation selectors, which are category Mn.
_HIDDEN_CHARS = re.compile('[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\u034f\ufe00-\ufe0f\U000e0100-\U000e01ef]')


def _strip_hidden(text: str) -> str:
    """Drop control characters and every Unicode format character (category Cf), instead of listing ranges.

    Cf covers soft hyphen, zero-width and bidi controls, invisible math operators and the tag characters, and new
    ones arrive with each Unicode version. The cost is that a zero-width joiner or non-joiner (also Cf) no longer
    joins emoji sequences or shapes some Persian and Indic words; that does not matter in a prompt.
    """
    text = _HIDDEN_CHARS.sub("", text)
    if text.isascii():  # no format characters in ASCII: skip the per-character pass for the common case
        return text
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def sanitize_chatml(text: str) -> str:
    """Strip ChatML / instruction delimiters, fence tags and hidden characters to prevent prompt injection."""
    if not text:
        return ""
    # Hidden characters go first so none can sit inside a tag name. Then repeat until nothing changes:
    # removing an inner tag can join its neighbours into a new one ("</book_pas<web_results>sages>").
    cleaned = _strip_hidden(text)
    while True:
        stripped = _FENCE_TAGS.sub("", _CHATML_CONTROL_TOKENS.sub("", cleaned))
        if stripped == cleaned:
            return cleaned.strip()
        cleaned = stripped


def estimate_tokens(text: str) -> int:
    """Estimate token count for LLM budgeting without requiring heavy external tokenizers.
    
    Standard approximation: English text averages ~1.33 tokens per word or ~3.8 characters per token.
    Uses conservative maximum to ensure strict hardware VRAM limits are respected.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int(max(words * 1.33, chars / 3.8)))


def truncate_to_tokens(text: str, max_tokens: int, from_end: bool = False) -> str:
    """Truncates text to fit inside max_tokens.
    
    Args:
        text: Input string.
        max_tokens: Maximum allowed token count.
        from_end: If True, retains trailing words (best for local scene context near playhead).
                  If False, retains leading words (best for queries and summaries).
    """
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    words = text.split()
    if not words:
        return ""

    if from_end:
        # Keep trailing words
        for idx in range(len(words)):
            candidate = " ".join(words[idx:])
            if estimate_tokens(candidate) <= max_tokens:
                return candidate
        return words[-1]
    else:
        # Keep leading words
        for idx in range(len(words), 0, -1):
            candidate = " ".join(words[:idx])
            if estimate_tokens(candidate) <= max_tokens:
                return candidate
        return words[0]


def assemble_qa_prompt(
    document_summary: str,
    local_context: str,
    user_query: str,
    sentence_idx: int = 0,
    max_context_tokens: int = 2048
) -> List[Dict[str, str]]:
    """Assemble sanitized, token-bounded prompt envelope adhering to 4GB VRAM hardware budget.
    
    Guarantees:
    1. ChatML control tokens are stripped from all dynamic variables to block prompt injection.
    2. User query is clamped to a maximum of 256 tokens.
    3. Document summary and local context are bounded such that total prompt tokens strictly
       do not exceed the 2048-token context budget (protecting RTX 3050 4GB VRAM from KV cache OOM).
    """
    # 1. Sanitize all dynamic inputs against prompt injection
    clean_query = sanitize_chatml(user_query)
    clean_context = sanitize_chatml(local_context)
    clean_summary = sanitize_chatml(document_summary)

    # 2. System prompt overhead and output generation headroom
    system_tokens = estimate_tokens(SYSTEM_PROMPT) + 30  # Envelope formatting allowance
    output_headroom = 256  # Reserved for 3-sentence LLM generation
    available_user_envelope_tokens = max(256, max_context_tokens - system_tokens - output_headroom)

    # 3. Clamp user query to max 256 tokens
    clean_query = truncate_to_tokens(clean_query, max_tokens=256, from_end=False)
    query_tokens = estimate_tokens(clean_query)

    # 4. Allocate remaining context budget between local scene context and document summary
    remaining_context_budget = max(100, available_user_envelope_tokens - query_tokens)
    
    # Prioritize local scene context (up to ~900 tokens, preserving scene around cursor)
    max_local_context_tokens = min(900, remaining_context_budget)
    clean_context = truncate_to_tokens(clean_context, max_tokens=max_local_context_tokens, from_end=True)
    used_context_tokens = estimate_tokens(clean_context)

    # Remainder goes to book summary
    max_summary_tokens = max(50, remaining_context_budget - used_context_tokens)
    clean_summary = truncate_to_tokens(clean_summary, max_tokens=max_summary_tokens, from_end=False)

    user_envelope = (
        f"Active Book Context: {clean_summary}\n"
        f"Surrounding Scene: {clean_context}\n"
        f"User Query: {clean_query}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_envelope}
    ]
