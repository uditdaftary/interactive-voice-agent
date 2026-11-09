import pytest
from voice_agent.llm.client import MockLLMClient, PunctuationStreamParser

def test_punctuation_stream_parser():
    parser = PunctuationStreamParser()
    tokens = ["The ", "clock ", "chimed ", "midnight. ", "It ", "echoed ", "loudly!"]
    
    chunks = []
    for token in tokens:
        sentences = parser.feed(token)
        chunks.extend(sentences)
    
    assert len(chunks) == 2
    assert chunks[0] == "The clock chimed midnight."
    assert chunks[1] == "It echoed loudly!"

@pytest.mark.asyncio
async def test_mock_llm_stream():
    client = MockLLMClient(response_text="The clock represents the passage of time. It warns the characters.")
    collected = []
    async for chunk in client.stream_generate([{"role": "user", "content": "Explain clock"}]):
        collected.append(chunk)
    assert "".join(collected).strip() == "The clock represents the passage of time. It warns the characters."

def test_parser_abbreviations_no_hang():
    """Verify Dr., Prof., e.g., i.e., D.C. never freeze the parser or split mid-abbreviation."""
    parser = PunctuationStreamParser()
    tokens = [
        "Dr. ", "Smith ", "and ", "Prof. ", "Jones ", "traveled ", "to ", "Washington ", "D.C. ",
        "yesterday. ", "They ", "brought ", "e.g. ", "apples ", "and ", "pears."
    ]
    
    chunks = []
    for token in tokens:
        chunks.extend(parser.feed(token))
    
    assert len(chunks) == 2
    assert chunks[0] == "Dr. Smith and Prof. Jones traveled to Washington D.C. yesterday."
    assert chunks[1] == "They brought e.g. apples and pears."

def test_parser_decimals_no_mid_number_split():
    """Verify 1.5, 3.14, 0.75 do not split into '1.' and '5' even across token boundaries."""
    parser = PunctuationStreamParser()
    tokens = [
        "The ", "engine ", "displacement ", "is ", "1.", "5 ", "liters. ",
        "Pi ", "is ", "approximately ", "3.14159. ", "Confirmed!"
    ]
    
    chunks = []
    for token in tokens:
        chunks.extend(parser.feed(token))
        
    assert len(chunks) == 3
    assert chunks[0] == "The engine displacement is 1.5 liters."
    assert chunks[1] == "Pi is approximately 3.14159."
    assert chunks[2] == "Confirmed!"

def test_parser_ellipses_no_lone_dot_stutter():
    """Verify ellipses do not yield lone '.' punctuation tokens to TTS."""
    parser = PunctuationStreamParser()
    tokens = ["The ", "shadows ", "lengthened... ", "A ", "sound ", "echoed."]
    
    chunks = []
    for token in tokens:
        chunks.extend(parser.feed(token))
    
    # Ensure no lone dot chunks exist
    for c in chunks:
        assert c != "."
        assert c != ".."
        assert c != "..."
    
    # Sentences are split cleanly
    assert len(chunks) == 2
    assert chunks[0] == "The shadows lengthened..."
    assert chunks[1] == "A sound echoed."

def test_parser_multi_sentence_single_feed():
    """Verify feed() returns all available complete sentences in a single call."""
    parser = PunctuationStreamParser()
    sentence_block = "First sentence. Second sentence! Third sentence? Fourth trailing"
    
    chunks = parser.feed(sentence_block)
    assert len(chunks) == 3
    assert chunks[0] == "First sentence."
    assert chunks[1] == "Second sentence!"
    assert chunks[2] == "Third sentence?"
    
    # Remaining buffer has the trailing unpunctuated text
    trailing = parser.flush()
    assert trailing == "Fourth trailing"

def test_parser_flush_empty_and_trailing():
    """Verify flush() retrieves unpunctuated text and returns None when empty."""
    parser = PunctuationStreamParser()
    assert parser.flush() is None
    
    parser.feed("Unpunctuated trailing response")
    assert parser.flush() == "Unpunctuated trailing response"
    assert parser.flush() is None
