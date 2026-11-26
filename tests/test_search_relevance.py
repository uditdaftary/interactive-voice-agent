import logging
from unittest.mock import MagicMock, patch

import pytest

from voice_agent.llm.client import sanitize_web_snippet, search_web, search_web_sync


def ddgs_module(results: list[dict[str, str]] | Exception) -> tuple[MagicMock, MagicMock]:
    """A stand-in `ddgs` module whose `DDGS().text` returns `results` (or raises it), and the `text` mock to inspect."""
    text = MagicMock()
    if isinstance(results, Exception):
        text.side_effect = results
    else:
        text.return_value = results
    module = MagicMock()
    module.DDGS.return_value.text = text
    return module, text


def test_search_web_sync_depth_and_warning_logging(caplog):
    """Verify search_web_sync defaults to max_results=8 and logs warnings on exceptions."""
    module, text = ddgs_module(RuntimeError("DDGS rate limit exceeded"))
    with patch.dict("sys.modules", {"ddgs": module}):
        with caplog.at_level(logging.WARNING):
            res = search_web_sync("who is johnny")

    assert res == ""
    assert any("web search failed" in rec.message.lower() for rec in caplog.records)
    text.assert_called_once_with("who is johnny", max_results=8)


@pytest.mark.parametrize("query", ["who is johnny", "what is the weather in Tokyo?", "tell me about Arasaka"])
def test_the_query_is_sent_as_given_and_only_once(query: str):
    """Nothing is added to what the user asked, and an empty result does not trigger a second request."""
    module, text = ddgs_module([])
    with patch.dict("sys.modules", {"ddgs": module}):
        assert search_web_sync(query) == ""

    text.assert_called_once_with(query, max_results=8)
    assert "cyberpunk" not in text.call_args.args[0].lower()


def test_results_become_bullet_lines():
    module, _ = ddgs_module([{"title": "A page", "body": "Some text [1] here."}, {"title": "", "body": ""}])
    with patch.dict("sys.modules", {"ddgs": module}):
        assert search_web_sync("anything") == "- A page: Some text here."


def test_a_blank_query_is_not_searched():
    module, text = ddgs_module([])
    with patch.dict("sys.modules", {"ddgs": module}):
        assert search_web_sync("   ") == ""

    text.assert_not_called()


def test_sanitize_web_snippet_strips_urls_and_citations():
    assert sanitize_web_snippet("See https://example.com now [1]. Done...") == "See now. Done"


@pytest.mark.asyncio
async def test_search_web_async_wrapper():
    """Verify search_web async wrapper uses max_results=8 by default."""
    with patch("voice_agent.llm.client.search_web_sync") as mock_sync:
        mock_sync.return_value = "- Title: Body"
        res = await search_web("test query")
        assert res == "- Title: Body"
        mock_sync.assert_called_once_with("test query", 8)
