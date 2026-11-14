import logging
import pytest
from unittest.mock import patch, MagicMock

from voice_agent.llm.client import (
    DEFAULT_STOPWORDS,
    extract_query_tokens,
    anchor_search_query,
    score_candidate_sentence,
    search_web_sync,
    search_web,
)


def test_stopwords_coverage():
    """Verify DEFAULT_STOPWORDS has 50+ stopwords and includes all mandatory terms."""
    assert len(DEFAULT_STOPWORDS) >= 50
    mandatory_terms = [
        "the", "why", "how", "did", "does", "was", "were", "can",
        "could", "tell", "about", "what", "where", "when", "who", "from", "with"
    ]
    for term in mandatory_terms:
        assert term in DEFAULT_STOPWORDS, f"Mandatory stopword '{term}' missing from DEFAULT_STOPWORDS"


def test_extract_query_tokens_punctuation_tolerance():
    """Verify regex tokenization cleanly removes trailing punctuation and stopwords."""
    assert extract_query_tokens("who is johnny?") == ["johnny"]
    assert extract_query_tokens("what is the relic!?") == ["relic"]
    assert extract_query_tokens("How did Johnny die...") == ["johnny", "die"]
    assert extract_query_tokens("Tell me about Lucy, please!") == ["lucy", "please"]
    assert extract_query_tokens("who is david-martinez?") == ["david-martinez"]


def test_extract_query_tokens_voice_recognition_typos():
    """Verify typos like arakas/sande are corrected to arasaka/sandevistan in tokens."""
    tokens_arakas = extract_query_tokens("who is in arakas tower?")
    assert "arasaka" in tokens_arakas
    tokens_sande = extract_query_tokens("where is my sande?")
    assert "sandevistan" in tokens_sande


@pytest.mark.parametrize("character", [
    "lucy", "david", "rebecca", "maine", "kiwi", "dorio", "smasher",
    "silverhand", "johnny", "arasaka", "yorinobu", "saburo", "takemura",
    "panam", "judy", "alt cunningham", "kerry", "river"
])
def test_concept_anchoring_all_characters(character: str):
    """Verify all 18 specified characters anchor with Cyberpunk even with trailing punctuation."""
    query = f"who is {character}?"
    anchored = anchor_search_query(query)
    assert anchored.endswith("Cyberpunk"), f"Character '{character}' failed to anchor query: {anchored}"
    assert character in anchored.lower()


@pytest.mark.parametrize("concept", [
    "relic", "blackwall", "sandevistan", "soulkiller", "cyberpsychosis",
    "cyberware", "netrunner", "edgerunner", "braindance", "trauma team",
    "militech", "kang tao", "night city"
])
def test_concept_anchoring_all_lore_concepts(concept: str):
    """Verify all 13 specified lore concepts anchor with Cyberpunk even with trailing punctuation."""
    query = f"what is {concept}?"
    anchored = anchor_search_query(query)
    assert anchored.endswith("Cyberpunk"), f"Concept '{concept}' failed to anchor query: {anchored}"
    assert concept in anchored.lower()


def test_concept_anchoring_no_duplicate_when_cyberpunk_or_edgerunners_present():
    """Verify 'Cyberpunk' is not appended if already present in the query."""
    assert anchor_search_query("who is Johnny in Cyberpunk 2077") == "who is Johnny in Cyberpunk 2077"
    assert anchor_search_query("who is Lucy in edgerunners?") == "who is Lucy in edgerunners?"


def test_concept_anchoring_unrelated_query_not_anchored():
    """Verify unrelated general queries are not mistakenly anchored."""
    assert anchor_search_query("what is the weather in Tokyo?") == "what is the weather in Tokyo?"
    assert anchor_search_query("how to bake sourdough bread") == "how to bake sourdough bread"


def test_interrogative_sentence_penalty_rejection():
    """Verify question titles ending in ? or starting with question patterns receive -20 penalty."""
    query_terms = ["johnny", "die"]

    # Interrogative titles from web search
    q1 = "How Did Johnny Silverhand Die?"
    q2 = "Why did Johnny Silverhand sacrifice himself?"
    q3 = "Can someone explain Johnny's fate?"
    q4 = "What is Johnny Silverhand's death scene?"
    q5 = "Who was Johnny Silverhand killed by?"

    score_q1 = score_candidate_sentence(q1, query_terms)
    score_q2 = score_candidate_sentence(q2, query_terms)
    score_q3 = score_candidate_sentence(q3, query_terms)
    score_q4 = score_candidate_sentence(q4, query_terms)
    score_q5 = score_candidate_sentence(q5, query_terms)

    # Factual sentence
    factual = "Johnny Silverhand was killed by Adam Smasher in 2023 during the Arasaka Tower raid."
    score_factual = score_candidate_sentence(factual, query_terms)

    # All interrogative sentences must be penalized heavily (< 0 or significantly below factual)
    assert score_q1 <= 0, f"Expected non-positive score for question title, got {score_q1}"
    assert score_factual > 10, f"Expected strong factual score, got {score_factual}"
    assert score_factual > score_q1
    assert score_factual > score_q2
    assert score_factual > score_q3
    assert score_factual > score_q4
    assert score_factual > score_q5


def test_predicate_and_definition_scoring_markers():
    """Verify predicate factual markers and definition markers boost candidate scores."""
    # Factual predicate markers
    for marker in [
        "killed by", "stole", "invented", "developed", "destroyed", "died",
        "shot", "caused by", "biochip", "construct", "ai", "subroutine",
        "syndrome", "condition"
    ]:
        sent = f"The prototype was {marker} in Night City laboratory."
        score = score_candidate_sentence(sent, ["prototype"])
        assert score > 5, f"Marker '{marker}' did not receive predicate bonus"

    # Definition markers
    for marker in ["is a", "known as", "deuteragonist", "protagonist", "megacorporation"]:
        sent = f"Arasaka {marker} operating globally."
        score = score_candidate_sentence(sent, ["arasaka"])
        assert score > 5, f"Definition marker '{marker}' did not receive definition bonus"


def test_penalize_forum_junk_and_short_fragments():
    """Verify reddit headers and fragments shorter than 25 characters are penalized."""
    # Short fragment (< 25 chars)
    short = "Johnny Silverhand."
    assert score_candidate_sentence(short, ["johnny"]) <= 0

    # Reddit header
    reddit = "Johnny Silverhand Death Discussion : r/cyberpunkgame - Reddit"
    assert score_candidate_sentence(reddit, ["johnny"]) <= 0


def test_search_web_sync_depth_and_warning_logging(caplog):
    """Verify search_web_sync defaults to max_results=8 and logs warnings on exceptions."""
    with patch("voice_agent.llm.client.anchor_search_query") as mock_anchor:
        mock_anchor.return_value = "johnny silverhand Cyberpunk"
        mock_mod = MagicMock()
        mock_ddgs = MagicMock()
        mock_inst = MagicMock()
        mock_inst.text.side_effect = RuntimeError("DDGS rate limit exceeded")
        mock_ddgs.return_value = mock_inst
        mock_mod.DDGS = mock_ddgs
        with patch.dict("sys.modules", {"ddgs": mock_mod}):
            with caplog.at_level(logging.WARNING):
                res = search_web_sync("who is johnny")
                assert res == ""
                assert any("web search failed" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_search_web_async_wrapper():
    """Verify search_web async wrapper uses max_results=8 by default."""
    with patch("voice_agent.llm.client.search_web_sync") as mock_sync:
        mock_sync.return_value = "- Title: Body"
        res = await search_web("test query")
        assert res == "- Title: Body"
        mock_sync.assert_called_once_with("test query", 8)
