import re
import asyncio
import logging
from typing import Any, AsyncGenerator, List, Dict, Optional

logger = logging.getLogger(__name__)

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
        "You are Johnny Silverhand from Cyberpunk 2077. Cynical, charismatic rockerboy, anti-corpo rebel. "
        "Speak with gritty swagger, call the user 'samurai' or 'choom', make caustic remarks about Night City and corpos. "
        "Answer the question concisely in 1 to 3 spoken sentences, then tell them to get back to the narrative."
    ),
    "lucy-kushinada": (
        "You are Lucy Kushinada from Cyberpunk: Edgerunners. Enigmatic, sharp-witted netrunner prodigy. "
        "Speak softly, introspectively, with quiet confidence. Use netrunner terms like ICE, subnets, deep dive. "
        "Answer the question concisely in 1 to 3 spoken sentences, then return to our reading."
    ),
    "david-martinez": (
        "You are David Martinez from Cyberpunk: Edgerunners. Fast, gritty, determined edgerunner from Santo Domingo. "
        "Talk with street slang like 'choom', 'preem', 'nova', 'flatline', 'Sandevistan'. Confident and loyal. "
        "Answer concisely in 1 to 3 spoken sentences, then let's get back to the run."
    ),
    "v-male": (
        "You are V, the legendary mercenary solo of Night City from Cyberpunk 2077. Pragmatic, gritty, professional survivor. "
        "Speak directly and street-smart like a seasoned edgerunner. "
        "Answer the question concisely in 1 to 3 spoken sentences, then let's get back to the story."
    ),
    "v-female": (
        "You are V, the mercenary solo of Night City from Cyberpunk 2077. Tactical, sharp, quick-witted, decisive. "
        "Speak with focused mercenary attitude and dry wit. "
        "Answer the question concisely in 1 to 3 spoken sentences, then let's get back to business."
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
    """Execute live internet web search to ground interruption queries."""
    if not query or not query.strip():
        return ""
    search_q = anchor_search_query(query)

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(search_q, max_results=max_results))
        if not results:
            return ""
        lines = []
        for r in results:
            title = sanitize_web_snippet(r.get("title", ""))
            body = sanitize_web_snippet(r.get("body", ""))
            if title or body:
                lines.append(f"- {title}: {body}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Live web search failed for query %r: %s", search_q, exc, exc_info=True)
        return ""


async def search_web(query: str, max_results: int = 8) -> str:
    """Async wrapper for live internet web search."""
    return await asyncio.to_thread(search_web_sync, query, max_results)


class LLMClient:
    """Multi-provider client supporting Built-in Zero-Setup extraction, Ollama,
    OpenAI-compatible APIs (Groq, Together, DeepSeek), and live internet web search grounding.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "local",
        provider: str = "builtin",
        model: str = "builtin-assistant",
        web_search_enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider.lower().strip()
        self.model = model
        self.web_search_enabled = web_search_enabled
        self._fallback_mock = MockLLMClient("Built-in model responding.")

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

    def get_config(self) -> Dict[str, Any]:
        """Return current provider configuration."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "web_search_enabled": self.web_search_enabled,
            "has_api_key": bool(self.api_key and self.api_key != "local"),
        }

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 150,
        character_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        # Determine character context
        chosen_char = character_id
        if not chosen_char:
            sys_content = " ".join([m.get("content", "") for m in messages if m.get("role") == "system"]).lower()
            for c_id in ["johnny-silverhand", "lucy-kushinada", "david-martinez", "v-female", "v-male"]:
                if c_id in sys_content or c_id.replace("-", " ") in sys_content:
                    chosen_char = c_id
                    break

        # Fast path: built-in native zero-setup question extraction
        if self.provider in ("builtin", "local-builtin", "native", "none"):
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break
            fallback_text = self._synthesize_fallback_answer(user_msg, character_id=chosen_char)
            for word in fallback_text.split():
                await asyncio.sleep(0.015)
                yield word + " "
            return

        import json
        import urllib.request
        import urllib.error

        chosen_model = model or self.model

        # Determine target endpoint
        target_url = self.base_url
        if not target_url.endswith("/chat/completions"):
            if target_url.endswith("/v1"):
                target_url = f"{target_url}/chat/completions"
            elif ":11434" in target_url:
                target_url = f"{target_url}/v1/chat/completions"
            else:
                target_url = f"{target_url}/v1/chat/completions"

        payload = json.dumps({
            "model": chosen_model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(target_url, data=payload, headers=headers, method="POST")

        yielded_any = False
        try:
            def _open_stream():
                return urllib.request.urlopen(req, timeout=3.5)

            response = await asyncio.to_thread(_open_stream)

            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, response.readline)
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str == "data: [DONE]":
                    break
                if line_str.startswith("data: "):
                    data_json = line_str[6:]
                    try:
                        data = json.loads(data_json)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yielded_any = True
                                yield content
                    except json.JSONDecodeError:
                        continue
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            yielded_any = False

        if not yielded_any:
            # Intelligent fallback: formulate a concise answer based on user query and context
            user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break

            fallback_text = self._synthesize_fallback_answer(user_msg, character_id=chosen_char)
            for word in fallback_text.split():
                await asyncio.sleep(0.015)
                yield word + " "

    def _synthesize_fallback_answer(self, user_msg: str, character_id: Optional[str] = None) -> str:
        """Construct a direct, context-aware spoken response in authentic Cyberpunk character voice
        from book text or web search grounding without requiring Ollama or external LLM setup.
        """
        char = (character_id or "david-martinez").lower()

        # 1. Extract query
        query = ""
        if "interrupted your narration with this question: '" in user_msg:
            query = user_msg.split("interrupted your narration with this question: '")[1].split("'")[0].strip()
        elif "User Question:" in user_msg:
            after = user_msg.split("User Question:")[-1]
            lines = [l.strip() for l in after.split("\n") if l.strip()]
            query = lines[0] if lines else ""
        elif "Query:" in user_msg:
            after = user_msg.split("Query:")[-1]
            lines = [l.strip() for l in after.split("\n") if l.strip()]
            query = lines[0] if lines else ""
        else:
            lines = [l.strip() for l in user_msg.split("\n") if l.strip()]
            query = lines[0] if lines else ""

        # Clean query tokens
        query_terms = extract_query_tokens(query)

        # 2. Extract web search results if present; or trigger live search if enabled
        web_section = ""
        if "Live Internet Web Search Results:" in user_msg:
            after_web = user_msg.split("Live Internet Web Search Results:")[1]
            if "User Question:" in after_web:
                web_section = after_web.split("User Question:")[0].strip()
            else:
                web_section = after_web.strip()
        elif getattr(self, "web_search_enabled", True) and query:
            web_section = search_web_sync(query, max_results=8)

        # 3. Extract document / book passages if present
        book_context = ""
        for tag in [
            "Authoritative Document Passages (Primary Source of Truth):",
            "Authoritative Document Passages:",
            "Relevant Passages from Loaded Document:",
            "Passages from Document:",
            "Book Context:",
            "Currently Spoken Line:",
            "Context:",
        ]:
            if tag in user_msg:
                after_tag = user_msg.split(tag)[1]
                for stop_marker in ["Live Internet Web Search Results:", "User Question:", "Provide a direct"]:
                    if stop_marker in after_tag:
                        after_tag = after_tag.split(stop_marker)[0]
                book_context = after_tag.strip()
                break

        # 4. Find most relevant sentence from the book / uploaded document
        best_book_sent = ""
        if book_context:
            candidate_sents = [s.strip().lstrip("- ").strip() for s in re.split(r'(?<=[.!?])\s+|\n+', book_context) if len(s.strip()) > 10]
            best_score = -1
            for s in candidate_sents:
                s_lower = s.lower()
                score = 0
                for qt in query_terms:
                    if qt in s_lower:
                        score += 3
                    elif len(qt) >= 4 and qt[:4] in s_lower:
                        score += 1
                if score > best_score and score > 0:
                    best_score = score
                    best_book_sent = s

        # 5. Extract substantive, non-truncated fact from Web Search
        best_web_fact = ""
        if web_section and web_section.strip():
            candidate_web = []
            for line in web_section.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if not line:
                    continue
                if ":" in line:
                    title, body = line.split(":", 1)
                    title = title.strip()
                    body = body.strip()
                    if title.endswith(('.', '!', '?')):
                        line = f"{title} {body}"
                    else:
                        line = f"{title}. {body}"
                clean = sanitize_web_snippet(line)
                for sent in re.split(r'(?<=[.!?])\s+', clean):
                    sent = sent.strip()
                    # Filter basic junk
                    if len(sent) > 10 and not any(skip in sent.lower() for skip in ['login', 'sign up', 'all rights reserved', 'cookie', 'wikipedia is']):
                        candidate_web.append(sent)

            scored_web = []
            for sent in candidate_web:
                clean_sent = re.sub(r'^[^\w]+', '', sent.strip()).strip()
                if not clean_sent:
                    continue
                score = score_candidate_sentence(clean_sent, query_terms)
                if score > 0:
                    scored_web.append((score, clean_sent))

            scored_web.sort(key=lambda x: x[0], reverse=True)
            if scored_web:
                # Select top sentence ensuring no trailing '...' and full thought
                top_sent = scored_web[0][1].strip()
                if not top_sent.endswith(('.', '!', '?')):
                    top_sent += '.'
                best_web_fact = top_sent

                # If short, add second relevant sentence if available and substantive
                if len(top_sent) < 90 and len(scored_web) > 1:
                    second_sent = scored_web[1][1].strip()
                    if (
                        second_sent
                        and second_sent != top_sent
                        and len(second_sent.split()) >= 6
                        and not second_sent.endswith('?')
                        and not second_sent.lower().startswith(INTERROGATIVE_PREFIXES)
                    ):
                        if not second_sent.endswith(('.', '!', '?')):
                            second_sent += '.'
                        best_web_fact = f"{top_sent} {second_sent}"

        # 6. Synthesize the integrated response
        is_book_query = bool(
            "Authoritative Document Passages" in user_msg
            or "Primary Source of Truth" in user_msg
            or any(w in query.lower() for w in ["story", "book", "chapter", "scene", "passage", "narrative", "sentence", "reading"])
            or (best_book_sent and not web_section)
        )

        # Context provenance hierarchy: prioritize book context when query is about book contents
        if is_book_query and best_book_sent:
            if "johnny" in char:
                return f"Straight from the scene, samurai: {best_book_sent} Keep your wits sharp."
            elif "lucy" in char:
                return f"In this passage: {best_book_sent} Don't lose focus, David."
            elif "v-female" in char:
                return f"Target intel from the scene: {best_book_sent} Let's move."
            elif "v-male" in char or char == "v":
                return f"Right here in the op, choom: {best_book_sent} Back to it."
            else:
                return f"Check it out, choom: {best_book_sent} Sandevistan primed, let's roll."

        if best_web_fact:
            if best_book_sent and any(qt in best_book_sent.lower() for qt in query_terms):
                clean_book = best_book_sent.lstrip("\"'")
                full_body = f"{best_web_fact} In our narrative, {clean_book}"
            else:
                full_body = best_web_fact

            if "johnny" in char:
                return f"Listen up samurai, tapped into the net: {full_body} Don't let the corpos spin it. Back to the story."
            elif "lucy" in char:
                return f"Dived into the subnet. {full_body} Stay quiet, let's keep reading."
            elif "v-female" in char:
                return f"Intel verified: {full_body} Eyes forward, resuming the mission."
            elif "v-male" in char or char == "v":
                return f"Scanned the feed, choom: {full_body} Let's get back to business."
            else:
                return f"Yo choomba, pulled this off the net: {full_body} Preem stuff. Let's roll."

        # If only document / notes context is available
        if best_book_sent:
            if "johnny" in char:
                return f"Straight from the scene, samurai: {best_book_sent} Keep your wits sharp."
            elif "lucy" in char:
                return f"In this passage: {best_book_sent} Don't lose focus, David."
            elif "v-female" in char:
                return f"Target intel from the scene: {best_book_sent} Let's move."
            elif "v-male" in char or char == "v":
                return f"Right here in the op, choom: {best_book_sent} Back to it."
            else:
                return f"Check it out, choom: {best_book_sent} Sandevistan primed, let's roll."

        clean = query.lower()
        if "who" in clean or "what" in clean or "where" in clean:
            if "johnny" in char:
                return f"Night City doesn't give free handouts samurai, but that unfolds right up ahead. Back to the run."
            elif "lucy" in char:
                return f"That unfolds as we dive deeper. Stay with me."
            elif "v-female" in char:
                return f"Keep your eyes on the board, details are developing now. Moving on."
            elif "v-male" in char or char == "v":
                return f"Answers are coming up in the next op, choom. Let's keep moving."
            else:
                return f"Don't worry choomba, the story develops that right ahead. Let's keep going."

        if "johnny" in char:
            return "Got it, samurai. Resuming the run."
        elif "lucy" in char:
            return "Understood. Resuming our dive."
        elif "v-female" in char:
            return "Copy that. Resuming playback."
        elif "v-male" in char or char == "v":
            return "Understood choom. Back to the op."
        return "Understood choomba. Resuming narration with your question in mind."

