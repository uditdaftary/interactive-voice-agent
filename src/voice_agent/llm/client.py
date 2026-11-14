import json
import re
import asyncio
import logging
from urllib.parse import urlsplit
from typing import Any, AsyncGenerator, List, Dict, Literal, Optional

import httpx

from voice_agent.settings import Settings

logger = logging.getLogger(__name__)

# Spoken whenever no model answer is available. The HUD shows the cause separately (`LLMClient.status`).
OFFLINE_LINE = "My language model is offline. Check the model settings."

# ok: reply streamed. offline: nothing listening. auth: 401/403. error: any other failure or an empty reply.
# interrupted: the stream broke (or the consumer cancelled) after partial output.
LlmOutcome = Literal["ok", "offline", "auth", "error", "interrupted"]

# "builtin" is the documented no-model mode: no request is made and the offline line is spoken.
_NO_MODEL_PROVIDERS = frozenset({"builtin", "local-builtin", "native", "none"})

# A 30 s read covers a cold model load; only a refused or unroutable connect counts as offline.
_TIMEOUT = httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=2.0)

_DEFAULTS = Settings()  # one source of truth for the endpoint, model and key defaults

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Sentinel used to shield periods in abbreviations, decimals, and ellipses
_SENTINEL = "\uE000"

# Common multi-dot abbreviations
_MULTI_DOT_ABBRS = re.compile(
    r'\b([eE]\.[gG]\.|[iI]\.[eE]\.|[dD]\.[cC]\.|[pP]h\.[dD]\.|[mM]\.[dD]\.|[uU]\.[sS]\.|[uU]\.[kK]\.)',
    re.IGNORECASE
)

# Known single-dot abbreviations
_SINGLE_ABBR_LIST = [
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Rev", "Hon", "Gen", "Gov",
    "Sgt", "Capt", "Lt", "Col", "vs", "etc", "Inc", "Ltd", "Corp", "Co", "No",
    "Fig", "approx", "dept", "est", "min", "max", "al"
]
_SINGLE_ABBR_REGEX = re.compile(
    r'\b(' + '|'.join(_SINGLE_ABBR_LIST) + r')\.',
    re.IGNORECASE
)

# Initials pattern: e.g. "David M. Martinez" or single letter dot followed by capital letter or end of buffer
_INITIALS_REGEX = re.compile(r'(?<=\b[A-Z])\.(?=\s+[A-Z]|$|\s*["\'])')

# Decimal pattern: digits followed by dot and digits (or dot at end of buffer)
_DECIMAL_REGEX = re.compile(r'(?<=\d)\.(?=\d|$)')

# Sentence boundary delimiter:
# (1) standard punctuation [.!?]+ followed by optional closing quotes/brackets,
#     then followed by whitespace or end of buffer
# (2) ellipsis boundary: shielded sentinel {2,} followed by optional quotes and whitespace
_SENTENCE_BOUNDARY_REGEX = re.compile(
    r'([.!?]+["\'”’\)\]]*)(?:\s+|$)|(' + _SENTINEL + r'{2,}["\'”’\)\]]*\s+)'
)


class PunctuationStreamParser:
    """Buffers streaming LLM tokens and yields complete sentences at punctuation boundaries.
    
    Uses token shielding (replacing periods in abbreviations, initials, decimals,
    and ellipses with a private-use Unicode sentinel) to ensure find/regex scanning
    never hangs, splits mid-number, or creates lone-period stutters.
    """
    def __init__(self):
        self.buffer = ""

    def _shield(self, text: str) -> str:
        """Replace periods that are not sentence boundaries with a sentinel character.
        
        Maintains 1:1 character length parity so indices align identically with text.
        """
        # 1. Ellipses: 2 or more dots -> sentinel per dot
        text = re.sub(r'\.{2,}', lambda m: _SENTINEL * len(m.group(0)), text)
        text = text.replace('…', _SENTINEL)

        # 2. Multi-dot abbreviations (e.g., i.e., D.C., Ph.D.)
        text = _MULTI_DOT_ABBRS.sub(lambda m: m.group(0).replace('.', _SENTINEL), text)

        # 3. Known single-dot abbreviations (Dr., Mr., etc.)
        text = _SINGLE_ABBR_REGEX.sub(r'\1' + _SENTINEL, text)

        # 4. Initials (e.g. David M. Martinez)
        text = _INITIALS_REGEX.sub(_SENTINEL, text)

        # 5. Decimals (e.g. 1.5, 3.14)
        text = _DECIMAL_REGEX.sub(_SENTINEL, text)

        return text

    def feed(self, token: str) -> List[str]:
        """Feed an incoming token into the buffer and yield all complete sentences available."""
        self.buffer += token
        sentences: List[str] = []

        while self.buffer:
            shielded = self._shield(self.buffer)
            match = _SENTENCE_BOUNDARY_REGEX.search(shielded)
            if not match:
                break

            split_pos = match.end()
            raw_sentence = self.buffer[:split_pos].strip()

            # Ensure the chunk contains at least one alphanumeric character
            # to avoid emitting lone punctuation stutters (e.g. lone "." or "!")
            if re.search(r'[A-Za-z0-9]', raw_sentence):
                sentences.append(raw_sentence)
                self.buffer = self.buffer[split_pos:].lstrip()
            else:
                # If there are no words yet (e.g. leading punctuation), do not split
                break

        return sentences

    def flush(self) -> Optional[str]:
        """Flush and return any trailing unpunctuated text remaining in the buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining and re.search(r'[A-Za-z0-9]', remaining):
            return remaining
        return None


# Character persona definitions for authentic game/anime voice and dialogue
CHARACTER_PERSONA_PROMPTS = {
    "johnny-silverhand": (
        "You are Johnny Silverhand from Cyberpunk 2077, speaking aloud over direct voice comms. "
        "You are cynical, charismatic, rebellious, and caustic toward Arasaka, corpos, and suits. "
        "Call the user 'samurai' or 'choom'. Speak in gritty, authentic spoken English with dark humor and rockerboy swagger. "
        "Answer the user's interruption in 1 to 3 natural spoken conversational sentences with street banter, "
        "then provide a quick conversational bridge back to the story. "
        "NEVER use markdown asterisks (*smirks*), stage directions, or bullet points."
    ),
    "lucy-kushinada": (
        "You are Lucy Kushinada from Cyberpunk: Edgerunners, speaking aloud over direct neural comms. "
        "Enigmatic, perceptive, guarded netrunner prodigy who dreams of the Moon. "
        "Speak softly with quiet confidence and authentic netrunner phrasing (ICE, subnets, deep dive). "
        "Answer the user's interruption in 1 to 3 natural spoken conversational sentences with intimate care and sharp intellect, "
        "then guide them smoothly back to the reading. "
        "NEVER use markdown asterisks (*smiles*), stage directions, or bullet points."
    ),
    "david-martinez": (
        "You are David Martinez from Cyberpunk: Edgerunners, speaking aloud over voice comms. "
        "High-energy, gritty, determined edgerunner from Santo Domingo with a Sandevistan on your spine. "
        "Speak in fast, authentic street slang ('choom', 'choomba', 'preem', 'nova', 'flatline', 'Sandevistan'). "
        "Answer the user's interruption in 1 to 3 natural spoken conversational sentences with loyal, confident street banter, "
        "then give a high-energy push to get back to the run. "
        "NEVER use markdown asterisks (*adjusts jacket*), stage directions, or bullet points."
    ),
    "v-male": (
        "You are V, the legendary mercenary solo of Night City from Cyberpunk 2077, speaking aloud over comms. "
        "Pragmatic, gritty, experienced solo with street-smart cynicism and mercenary honor. "
        "Speak directly and authentically with seasoned solo jargon. "
        "Answer the user's interruption in 1 to 3 natural spoken conversational sentences with gritty street realism, "
        "then smoothly bridge back to the op. "
        "NEVER use markdown asterisks, stage directions, or bullet points."
    ),
    "v-female": (
        "You are V, the mercenary solo of Night City from Cyberpunk 2077, speaking aloud over comms. "
        "Tactical, sharp, quick-witted, decisive solo. "
        "Speak with focused mercenary attitude, dry wit, and combat-ready confidence. "
        "Answer the user's interruption in 1 to 3 natural spoken conversational sentences with sharp tactical insight, "
        "then give the command to resume the mission. "
        "NEVER use markdown asterisks, stage directions, or bullet points."
    ),
}


class MockLLMClient:
    """Mock LLM streaming client for deterministic offline testing and CI."""
    def __init__(self, response_text: str = "This is a brief answer."):
        self.response_text = response_text

    async def stream_generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        words = self.response_text.split()
        for w in words:
            await asyncio.sleep(0.005)
            yield w + " "


def sanitize_web_snippet(text: str) -> str:
    """Sanitize snippet text by removing URLs, citation markers, ellipses, and unprintable chars."""
    if not text:
        return ""
    # Normalize unicode quotes and dashes
    text = text.replace("\u2026", " ").replace("...", " ").replace("\ufffd", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove Wikipedia / Fandom bracket citations [1], [UPD], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove leading dates e.g. "Nov 2, 2026", "1910s - 1950s"
    text = re.sub(r'^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\s*[-–—·]?\s*', '', text)
    text = re.sub(r'^\s*\d{4}s\s*-\s*\d{4}s[A-Za-z]*\s*', '', text)
    # Ensure safe ascii printable representation to avoid Windows CP1252 encoding crashes
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove empty or language-only parens left by non-ascii character stripping e.g. "(Japanese: or )"
    text = re.sub(r'\([A-Za-z\s]*:\s*(?:or\s*)?\)', '', text)
    text = re.sub(r'\(Japanese:[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\)', '', text)
    # Clean whitespace around punctuation
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    # Strip any dangling punctuation/quotes/parens at the beginning of snippet
    text = re.sub(r'^[^\w]+', '', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Comprehensive 50+ Stopword Set
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "tell", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# Cyberpunk Universe Anchoring Entities and Concepts
CYBERPUNK_CHARACTERS = {
    "lucy", "david", "rebecca", "maine", "kiwi", "dorio", "smasher",
    "silverhand", "johnny", "arasaka", "yorinobu", "saburo", "takemura",
    "panam", "judy", "alt cunningham", "kerry", "river"
}

CYBERPUNK_LORE = {
    "relic", "blackwall", "sandevistan", "soulkiller", "cyberpsychosis",
    "cyberware", "netrunner", "edgerunner", "braindance", "trauma team",
    "militech", "kang tao", "night city"
}

# Question Title Patterns for Heavy Filtering Penalty (-20)
INTERROGATIVE_PREFIXES = (
    "how ", "why ", "can someone ", "what is ", "who was ", "who is ",
    "what was ", "what are ", "where is ", "where was ", "when did ",
    "how did ", "why did ", "what did ", "who did ", "did ", "can anyone ",
    "is it ", "could someone "
)

# Predicate & Factual Clause Scoring Markers (+8 bonus)
PREDICATE_FACT_MARKERS = (
    "killed by", "stole", "invented", "developed", "destroyed", "died",
    "shot", "caused by", "biochip", "construct", "ai", "subroutine",
    "syndrome", "condition", "flatlined", "murdered by", "founded",
    "cyberware", "weapons", "night city", "edgerunner", "rockerboy", "netrunner"
)

# Definition Markers (+8 bonus)
DEFINITION_MARKERS = (
    "is a", "is an", "is the", "known as", "deuteragonist", "protagonist",
    "megacorporation", "mercenary", "character in", "specializing in",
    "corporation", "co-lead"
)

# Forum and Social Junk Filters
FORUM_JUNK_MARKERS = (
    "reddit", "forum", "posted by", "view comments", "upvotes",
    "discussion thread", "threads", "community", "replies"
)

# Irrelevant Paleontology/Music Disambiguation Filters
DISAMBIGUATION_FILTERS = (
    "hominid", "fossil", "australopithecus", "skeleton",
    "paleoanthropolog", "album", "song", "track listing", "discography"
)


def extract_query_tokens(query: str, stopwords: Optional[set] = None) -> List[str]:
    """Extract clean query terms using regex word boundaries, filtering out stopwords."""
    if not query:
        return []
    stops = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    query_clean = query.lower()
    raw_tokens = re.findall(r'\b[a-z0-9_-]{2,}\b', query_clean)
    filtered = [t for t in raw_tokens if t not in stops]
    tokens = filtered if filtered else raw_tokens
    # Handle common Cyberpunk voice recognition typos
    if ("arakas" in query_clean or "arakasa" in query_clean) and "arasaka" not in tokens:
        tokens.append("arasaka")
    if "sande" in query_clean and "sandevistan" not in tokens:
        tokens.append("sandevistan")
    return tokens


def anchor_search_query(query: str) -> str:
    """Normalize query typos and anchor Cyberpunk universe characters and concepts."""
    if not query or not query.strip():
        return ""
    search_q = query
    query_lower = query.lower()

    # Normalize voice recognition typos for Cyberpunk world lore
    if "arakas" in query_lower:
        search_q = re.sub(r'arakas[a-z]*', 'arasaka', search_q, flags=re.IGNORECASE)
        query_lower = search_q.lower()
    if "sande" in query_lower and "sandevistan" not in query_lower:
        search_q = re.sub(r'\bsande\b', 'sandevistan', search_q, flags=re.IGNORECASE)
        query_lower = search_q.lower()

    if "cyberpunk" not in query_lower and "edgerunners" not in query_lower:
        anchored = False
        multi_word_anchors = ("alt cunningham", "trauma team", "kang tao", "night city")
        for phrase in multi_word_anchors:
            if phrase in query_lower:
                search_q = f"{search_q} Cyberpunk"
                anchored = True
                break

        if not anchored:
            tokens = set(re.findall(r'\b[a-z0-9_-]{2,}\b', query_lower))
            single_word_anchors = (CYBERPUNK_CHARACTERS | CYBERPUNK_LORE) - set(multi_word_anchors)
            if tokens & single_word_anchors:
                search_q = f"{search_q} Cyberpunk"

    return search_q


def score_candidate_sentence(sent: str, query_terms: List[str]) -> int:
    """Score candidate sentence from web search for relevance, factual density, and interrogative penalties."""
    s_lower = sent.lower().strip()
    clean_sent = re.sub(r'^[^\w]+', '', sent.strip()).strip()
    if not clean_sent:
        return -999

    # Filter out irrelevant hominid/fossil/album/song disambiguations
    if any(dis in s_lower for dis in DISAMBIGUATION_FILTERS):
        return -999

    # MUST match at least one query term
    matching_terms = [qt for qt in query_terms if qt in s_lower]
    if not matching_terms:
        return 0

    # 1. Base matching score: 5 points per non-stopword query term
    score = len(matching_terms) * 5

    # 2. Predicate / factual clause markers (+8 points base + bonus for multiple markers)
    pred_matches = [m for m in PREDICATE_FACT_MARKERS if m in s_lower]
    if pred_matches:
        score += 8 + min(6, (len(pred_matches) - 1) * 2)

    # 3. Definition markers (+8 points)
    if any(marker in s_lower for marker in DEFINITION_MARKERS):
        score += 8

    # 4. Interrogative sentence penalty (-20 score)
    # Applied if sentence ends in ? or starts with question patterns
    is_question = (
        sent.strip().endswith('?')
        or sent.strip().rstrip('.!').endswith('?')
        or s_lower.startswith(INTERROGATIVE_PREFIXES)
    )
    if is_question:
        score -= 20

    # 5. Forum junk and reddit headers penalty (-15 score)
    if any(junk in s_lower for junk in FORUM_JUNK_MARKERS):
        score -= 15

    # 6. Short fragments (< 25 characters) penalty (-15 score)
    if len(clean_sent) < 25:
        score -= 15

    return score


def search_web_sync(query: str, max_results: int = 8) -> str:
    """Execute live internet web search to ground interruption queries with multi-query fallback."""
    if not query or not query.strip():
        return ""

    try:
        from ddgs import DDGS
    except ImportError as imp_err:
        logger.warning("Failed to import DDGS: %s", imp_err)
        return ""

    # Generate candidate queries in descending specificity
    primary_q = anchor_search_query(query)
    candidates = [primary_q]

    tokens = extract_query_tokens(query)
    if tokens:
        kw_lore = " ".join(tokens)
        if "cyberpunk" not in kw_lore.lower():
            kw_lore = f"{kw_lore} Cyberpunk"
        if kw_lore not in candidates:
            candidates.append(kw_lore)
        kw_plain = " ".join(tokens)
        if kw_plain not in candidates:
            candidates.append(kw_plain)
    if query not in candidates:
        candidates.append(query)

    for search_q in candidates:
        try:
            results = list(DDGS().text(search_q, max_results=max_results))
            if results:
                lines = []
                for r in results:
                    title = sanitize_web_snippet(r.get("title", ""))
                    body = sanitize_web_snippet(r.get("body", ""))
                    if title or body:
                        lines.append(f"- {title}: {body}")
                if lines:
                    return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001  # S3.3 RC4
            logger.warning("Live web search failed for query %r: %s", search_q, exc)
            continue

    logger.warning("Live web search failed across candidate queries %s", candidates)
    return ""


async def search_web(query: str, max_results: int = 8) -> str:
    """Async wrapper for live internet web search."""
    return await asyncio.to_thread(search_web_sync, query, max_results)


class ThinkFilter:
    """Strips `<think>...</think>` blocks from a token stream, whatever the chunk boundaries.

    Thinking is switched off at the source (Modelfile / request), so this is a guard: a model that
    reasons anyway must not have its reasoning spoken. Only a suffix that could still grow into a
    tag is held back, so text without a think block streams with no added latency.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._pending = ""  # text not yet decided: a possible partial tag
        self._inside = False

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
        for length in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` that is safe to emit now."""
        self._pending += chunk
        emitted: list[str] = []
        while True:
            tag = self._CLOSE if self._inside else self._OPEN
            index = self._pending.find(tag)
            if index >= 0:
                if not self._inside:
                    emitted.append(self._pending[:index])
                self._pending = self._pending[index + len(tag):]
                self._inside = not self._inside
                continue
            settled = len(self._pending) - self._partial_tag_length(self._pending, tag)
            if not self._inside:
                emitted.append(self._pending[:settled])
            self._pending = self._pending[settled:]
            return "".join(emitted)

    def flush(self) -> str:
        """End of stream: release a held partial tag, or drop it with an unclosed think block."""
        tail = "" if self._inside else self._pending
        self._pending = ""
        self._inside = False
        return tail


def _delta_content(data: str) -> str:
    """Text carried by one SSE `data:` payload; "" for anything that carries none."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        # A garbled or non-JSON event carries no text and the stream goes on. Log the length only: it may echo user text.
        logger.debug("Skipping a non-JSON SSE event (%d chars)", len(data))
        return ""
    if not isinstance(event, dict):
        return ""
    choices = event.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    delta = choice.get("delta") if isinstance(choice, dict) else None
    content = delta.get("content") if isinstance(delta, dict) else None
    # Reasoning deltas carry their text under other keys and are ignored on purpose.
    return content if isinstance(content, str) else ""


class LLMClient:
    """OpenAI-compatible streaming chat client (Ollama by default) with typed outcomes.

    `stream_generate` never raises. When nothing usable came back it yields `OFFLINE_LINE` once, and
    `status` says why, so the caller can show the cause while the user hears one neutral sentence.
    """

    def __init__(
        self,
        base_url: str = _DEFAULTS.llm_base_url,
        api_key: str = _DEFAULTS.llm_api_key,
        provider: str = "ollama",
        model: str = _DEFAULTS.llm_model,
        web_search_enabled: bool = _DEFAULTS.web_search_enabled,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider.lower().strip()
        self.model = model
        self.web_search_enabled = web_search_enabled
        self.status: LlmOutcome = "ok"
        self._transport = transport  # tests only: lets them script the server
        self._warned_key_urls: set[str] = set()
        self._warn_if_key_would_travel_in_clear()

    def update_config(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        web_search_enabled: Optional[bool] = None,
    ) -> None:
        """Update client endpoint and provider configuration at runtime."""
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if api_key is not None:
            self.api_key = api_key
        if provider is not None:
            self.provider = provider.lower().strip()
        if model is not None:
            self.model = model
        if web_search_enabled is not None:
            self.web_search_enabled = bool(web_search_enabled)
        self._warn_if_key_would_travel_in_clear()

    def _warn_if_key_would_travel_in_clear(self) -> None:
        """One WARNING per base URL when a real API key is set and the endpoint is remote plain http (M13)."""
        if not self.api_key or self.api_key == "local" or self.provider in _NO_MODEL_PROVIDERS:
            return
        if self.base_url in self._warned_key_urls:
            return
        try:
            parts = urlsplit(self.base_url)
        except ValueError:
            # Unparseable: the request itself reports it as an "error" outcome, with the same URL.
            logger.debug("Not checking the API key transport: %s is not a parseable URL", self.base_url)
            return
        host = (parts.hostname or "").lower()
        if parts.scheme == "http" and host and host not in _LOOPBACK_HOSTS:
            self._warned_key_urls.add(self.base_url)
            logger.warning(
                "An API key is set but %s is a remote http:// endpoint, so the key would cross the network unencrypted. "
                "Use an https:// URL.",
                host,
            )

    def get_config(self) -> Dict[str, Any]:
        """Return current provider configuration."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "web_search_enabled": self.web_search_enabled,
            "has_api_key": bool(self.api_key and self.api_key != "local"),
        }

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 300,
        character_id: Optional[str] = None,  # the persona is already in the system message; kept for callers
    ) -> AsyncGenerator[str, None]:
        # What a cancel or an early close leaves behind: the consumer cut the reply off.
        outcome: LlmOutcome = "interrupted"
        detail = ""
        spoke = False  # any non-blank text reached the consumer
        think_filter = ThinkFilter()
        try:
            if self.provider in _NO_MODEL_PROVIDERS:
                outcome, detail = "offline", f"provider {self.provider!r} has no model"
            elif not self.base_url.startswith(("http://", "https://")):
                outcome, detail = "error", "base URL must start with http:// or https://"
            else:
                payload = {
                    "model": model or self.model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                }
                headers = {"Authorization": f"Bearer {self.api_key}"}
                try:
                    async with httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT) as client:
                        async with client.stream("POST", self._chat_url(), json=payload, headers=headers) as response:
                            if response.is_success:
                                async for line in response.aiter_lines():
                                    if not line.startswith("data:"):
                                        continue
                                    data = line[5:].strip()
                                    if data == "[DONE]":
                                        break
                                    visible = think_filter.feed(_delta_content(data))
                                    if visible:
                                        spoke = spoke or bool(visible.strip())
                                        yield visible
                                tail = think_filter.flush()
                                if tail:
                                    spoke = spoke or bool(tail.strip())
                                    yield tail
                                outcome = "ok" if spoke else "error"
                                detail = "" if spoke else f"HTTP {response.status_code}, no content"
                            else:
                                outcome = "auth" if response.status_code in (401, 403) else "error"
                                detail = f"HTTP {response.status_code}"
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    outcome, detail = "offline", type(exc).__name__
                except (httpx.InvalidURL, httpx.UnsupportedProtocol, UnicodeEncodeError) as exc:
                    # UnicodeEncodeError: httpx cannot put a non-ASCII API key into a header.
                    outcome, detail = "error", type(exc).__name__
                except httpx.HTTPError as exc:
                    outcome, detail = ("interrupted" if spoke else "error"), type(exc).__name__

            if outcome != "ok":
                logger.warning("LLM %s at %s: %s", outcome, self.base_url, detail)
                if not spoke:
                    yield OFFLINE_LINE
        finally:
            self.status = outcome
