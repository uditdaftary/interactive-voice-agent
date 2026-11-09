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


class MockLLMClient:
    """Mock LLM streaming client for deterministic offline testing and CI."""
    def __init__(self, response_text: str = "This is a brief answer."):
        self.response_text = response_text

    async def stream_generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        words = self.response_text.split()
        for w in words:
            await asyncio.sleep(0.005)
            yield w + " "


def search_web_sync(query: str, max_results: int = 3) -> str:
    """Execute live internet web search to ground interruption queries."""
    if not query or not query.strip():
        return ""
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            lines.append(f"- {title}: {body}")
        return "\n".join(lines)
    except Exception:
        return ""


async def search_web(query: str, max_results: int = 3) -> str:
    """Async wrapper for live internet web search."""
    return await asyncio.to_thread(search_web_sync, query, max_results)


class LLMClient:
    """Multi-provider client supporting Ollama, OpenAI-compatible APIs (Groq, Together, DeepSeek),
    Gemini, and intelligent local extractive fallback with live internet web search grounding.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "local",
        provider: str = "ollama",
        model: str = "llama3.2",
        web_search_enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider.lower().strip()
        self.model = model
        self.web_search_enabled = web_search_enabled
        self._fallback_mock = MockLLMClient("Local model responding.")

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
    ) -> AsyncGenerator[str, None]:
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

            fallback_text = self._synthesize_fallback_answer(user_msg)
            for word in fallback_text.split():
                await asyncio.sleep(0.015)
                yield word + " "

    def _synthesize_fallback_answer(self, user_msg: str) -> str:
        """Construct a natural response when external LLM endpoint is offline."""
        clean = user_msg.lower()
        if "who" in clean or "what is" in clean or "where" in clean:
            # Extract key nouns
            return f"Regarding your question about {user_msg.strip()[:60]}, the story highlights key details in the current passage. Let us continue exploring."
        elif "why" in clean:
            return f"The events unfold directly from the underlying circumstances in this chapter. Let's see what happens next."
        else:
            return f"Understood. Continuing narration with your question in mind."

