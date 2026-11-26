import json
import re
import asyncio
import logging
from urllib.parse import urlsplit
from typing import Any, AsyncGenerator, List, Dict, Literal, Optional

import httpx

from voice_agent.settings import Settings

logger = logging.getLogger(__name__)

# Spoken when there is no model answer and no passage of the book to read instead. The HUD shows the cause
# separately (`LLMClient.status`).
OFFLINE_LINE = "My language model is offline. Check the model settings."

# ok: reply streamed. offline: nothing listening. auth: 401/403. error: any other failure or an empty reply.
# interrupted: the stream broke (or the consumer cancelled) after partial output.
LlmOutcome = Literal["ok", "offline", "auth", "error", "interrupted"]

# "none" is the no-model mode: no request is made, and the server reads the closest passage of the book instead.
_NO_MODEL_PROVIDERS = frozenset({"none"})

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


def search_web_sync(query: str, max_results: int = 8) -> str:
    """Search the web once, for the user's own words, and return the results as bullet lines ("" when there are none).

    Blocking: call it through `search_web`. The query is sent as spoken or typed; nothing is added to it."""
    if not query or not query.strip():
        return ""

    try:
        from ddgs import DDGS
    except ImportError as imp_err:
        logger.warning("Failed to import DDGS: %s", imp_err)
        return ""

    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001  # S3.3 RC4
        logger.warning("Live web search failed for query %r: %s", query, exc)
        return ""
    lines = []
    for result in results:
        title = sanitize_web_snippet(result.get("title", ""))
        body = sanitize_web_snippet(result.get("body", ""))
        if title or body:
            lines.append(f"- {title}: {body}")
    if not lines:
        logger.warning("Live web search found nothing for query %r", query)
    return "\n".join(lines)


async def search_web(query: str, max_results: int = 8) -> str:
    """Async wrapper for live internet web search."""
    return await asyncio.to_thread(search_web_sync, query, max_results)


class ThinkFilter:
    """Strips `<think>...</think>` blocks from a token stream, whatever the chunk boundaries.

    Thinking is NOT disabled at the source yet: neither `models/Modelfile` nor the request turns it off
    (how to is unverified, see the Modelfile). This filter is therefore the only defence against the
    model's reasoning being spoken, and reasoning still spends the reply's token budget.

    Two stream shapes are handled:
    - `<think>reasoning</think>Answer`: the block is removed.
    - `reasoning</think>Answer`, which is what a chat template that prefills `<think>` makes the model
      stream: everything before a `</think>` that has no `<think>` before it is reasoning.

    The second shape cannot be told from a plain reply until `</think>` shows up, so the start of the
    stream is held back. The choice is `LEAD_HOLD_CHARS` = 120 characters (about 30 tokens, under a second
    at local-model speeds):
    - A stream with no think tag STARTING in its first 120 characters is a plain reply. It is released from
      there on, never altered, and a reply shorter than that is released whole when the stream ends, so
      the first words of a reply arrive at most 120 characters late (127 when a tag has started inside the
      hold and is still incomplete). A tag that starts after the hold is not looked for in the lead, so
      the result does not depend on how the server batches chunks.
    - Reasoning that ends inside the hold is removed, together with the whitespace after its `</think>`.
      Reasoning longer than the hold (the usual case for a thinking model) is NOT caught: it has already
      been released, and a later lone `</think>` is ordinary text. Holding longer would only delay every
      plain reply while still missing most reasoning, because reasoning tends to outrun the 300-token
      reply budget. Disabling thinking at the source is the real fix.
    After the hold, only a suffix that could still grow into `<think>` is held back.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    LEAD_HOLD_CHARS = 120

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._pending = ""  # text not yet decided: the held start of the stream or a possible partial tag
        self._inside = False
        self._undecided = True  # the start of the stream has not been classified yet
        self._spoken = False  # some text has been released
        self._trim = False  # drop the whitespace after a think block that opened the reply

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
        for length in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def _release(self, emitted: list[str]) -> str:
        text = "".join(emitted)
        if self._trim:
            text = text.lstrip()
        if text:
            self._spoken = True
            self._trim = False
        return text

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` that is safe to emit now."""
        self._pending += chunk
        emitted: list[str] = []
        while True:
            if self._undecided:
                # Decide on a tag that STARTS inside the first LEAD_HOLD_CHARS only. Searching the whole buffer
                # would make the result depend on how the server batches its chunks.
                window = self._pending[: self.LEAD_HOLD_CHARS + len(self._CLOSE) - 1]
                opened = window.find(self._OPEN)
                closed = window.find(self._CLOSE)
                if closed >= 0 and (opened < 0 or closed < opened):
                    # The stream began inside a think block whose opening tag was in the prompt.
                    self._pending = self._pending[closed + len(self._CLOSE):]
                    self._undecided = False
                    self._trim = True
                    continue
                if opened < 0:
                    if len(self._pending) < self.LEAD_HOLD_CHARS:
                        return ""
                    # A tag that has started inside the hold but is not complete yet may still arrive.
                    partial = max(
                        self._partial_tag_length(self._pending, self._OPEN),
                        self._partial_tag_length(self._pending, self._CLOSE),
                    )
                    if len(self._pending) - partial < self.LEAD_HOLD_CHARS:
                        return ""
                self._undecided = False
            tag = self._CLOSE if self._inside else self._OPEN
            index = self._pending.find(tag)
            if index >= 0:
                if not self._inside:
                    emitted.append(self._pending[:index])
                elif not self._spoken and not "".join(emitted).strip():
                    self._trim = True  # a block that opens the reply: the whitespace after it is not the answer
                self._pending = self._pending[index + len(tag):]
                self._inside = not self._inside
                continue
            settled = len(self._pending) - self._partial_tag_length(self._pending, tag)
            if not self._inside:
                emitted.append(self._pending[:settled])
            self._pending = self._pending[settled:]
            return self._release(emitted)

    def flush(self) -> str:
        """End of stream: release whatever is held, or drop it with an unclosed think block. Resets the filter."""
        tail = "" if self._inside else self._release([self._pending])
        self._reset()
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

    `stream_generate` never raises. When nothing usable came back it yields nothing, and `status` (set once the
    stream has ended) says why; the caller decides what the user hears. After a broken stream the text that did
    arrive was yielded, and `status` is "interrupted".
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
            elif not self.base_url.lower().startswith(("http://", "https://")):  # URL schemes are case-insensitive
                outcome, detail = "error", "base URL must start with http:// or https://"
            else:
                payload = {
                    "model": model or self.model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                }
                headers = {"Authorization": f"Bearer {self.api_key}"}
                failure: tuple[LlmOutcome, str] | None = None
                status_code = 0
                try:
                    async with httpx.AsyncClient(transport=self._transport, timeout=_TIMEOUT) as client:
                        async with client.stream("POST", self._chat_url(), json=payload, headers=headers) as response:
                            status_code = response.status_code
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
                            else:
                                failure = ("auth" if status_code in (401, 403) else "error", f"HTTP {status_code}")
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    failure = ("offline", type(exc).__name__)
                except (httpx.InvalidURL, httpx.UnsupportedProtocol, UnicodeEncodeError) as exc:
                    # UnicodeEncodeError: httpx cannot put a non-ASCII API key into a header.
                    failure = ("error", type(exc).__name__)
                except httpx.HTTPError as exc:
                    failure = ("interrupted", type(exc).__name__)

                # Release what the filter still holds, also after a broken stream: a partial reply is spoken (N3).
                tail = think_filter.flush()
                if tail:
                    spoke = spoke or bool(tail.strip())
                    yield tail
                if failure is None:
                    outcome, detail = ("ok", "") if spoke else ("error", f"HTTP {status_code}, no content")
                else:
                    outcome, detail = failure
                    if outcome == "interrupted" and not spoke:
                        outcome = "error"

            if outcome != "ok":
                logger.warning("LLM %s at %s: %s", outcome, self.base_url, detail)
        finally:
            self.status = outcome
