import re
import asyncio
from typing import Any, AsyncGenerator, List, Dict, Optional

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


def search_web_sync(query: str, max_results: int = 3) -> str:
    """Execute live internet web search to ground interruption queries."""
    if not query or not query.strip():
        return ""
    # Normalize query typos for Cyberpunk world lore
    search_q = query
    query_lower = query.lower()
    if "arakas" in query_lower:
        search_q = query_lower.replace("arakas", "arasaka")
    elif "sande" in query_lower and "sandevistan" not in query_lower:
        search_q = query_lower.replace("sande", "sandevistan")

    # Disambiguate short character queries (e.g. "who is lucy", "who is david", "what is arasaka")
    words = [w for w in query_lower.split() if w not in {"who", "is", "what", "tell", "me", "about", "the"}]
    for name in ["lucy", "david", "rebecca", "maine", "kiwi", "dorio", "smasher", "silverhand", "johnny", "arasaka"]:
        if name in words and "cyberpunk" not in query_lower and "edgerunners" not in query_lower:
            search_q = f"{search_q} Cyberpunk"
            break

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
    except Exception:
        return ""


async def search_web(query: str, max_results: int = 3) -> str:
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
        query_terms = [t for t in re.findall(r'\b[a-z]{3,}\b', query.lower()) if t not in {'what', 'who', 'where', 'when', 'tell', 'about', 'this', 'that', 'from', 'with'}]
        # Handle common Cyberpunk voice recognition typos
        if "arakas" in query.lower() or "arakasa" in query.lower():
            query_terms.append("arasaka")
        if "sande" in query.lower():
            query_terms.append("sandevistan")

        # 2. Extract web search results if present; or trigger live search if enabled
        web_section = ""
        if "Live Internet Web Search Results:" in user_msg:
            after_web = user_msg.split("Live Internet Web Search Results:")[1]
            if "User Question:" in after_web:
                web_section = after_web.split("User Question:")[0].strip()
            else:
                web_section = after_web.strip()
        elif getattr(self, "web_search_enabled", True) and query:
            web_section = search_web_sync(query)

        # 3. Extract document / book passages if present
        book_context = ""
        for tag in ["Relevant Passages from Loaded Document:", "Passages from Document:", "Book Context:", "Currently Spoken Line:", "Context:"]:
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
                    line = f"{title}. {body}"
                clean = sanitize_web_snippet(line)
                for sent in re.split(r'(?<=[.!?])\s+', clean):
                    sent = sent.strip()
                    # Filter junk
                    if len(sent) > 15 and not any(skip in sent.lower() for skip in ['login', 'sign up', 'all rights reserved', 'cookie', 'wikipedia is']):
                        candidate_web.append(sent)

            scored_web = []
            for sent in candidate_web:
                s_lower = sent.lower()
                # MUST match at least one query term
                matching_terms = [qt for qt in query_terms if qt in s_lower]
                if not matching_terms:
                    continue

                # Filter out irrelevant hominid/fossil/album/song disambiguations
                if any(dis in s_lower for dis in ['hominid', 'fossil', 'australopithecus', 'skeleton', 'paleoanthropolog', 'album', 'song', 'track listing', 'discography']):
                    continue

                score = len(matching_terms) * 5

                # Strong bonus for definition / role markers
                if any(kw in s_lower for kw in ['is a', 'is an', 'is the', 'netrunner', 'megacorporation', 'mercenary', 'character in', 'specializing in', 'deuteragonist', 'co-lead']):
                    score += 8
                elif any(kw in s_lower for kw in ['corporation', 'founded', 'weapons', 'cyberware', 'crew', 'night city', 'edgerunner']):
                    score += 4

                clean_sent = re.sub(r'^[^\w]+', '', sent.strip()).strip()
                if clean_sent:
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
                    if second_sent and second_sent != top_sent and len(second_sent.split()) >= 6:
                        if not second_sent.endswith(('.', '!', '?')):
                            second_sent += '.'
                        best_web_fact = f"{top_sent} {second_sent}"

        # 6. Synthesize the integrated response
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

