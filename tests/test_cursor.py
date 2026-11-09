import pytest
from voice_agent.cursor.tracker import CursorTracker, CursorPosition
from voice_agent.cursor.splitter import split_sentences, normalize_text


def test_split_sentences_basic():
    text = "Hello there! Dr. Smith went to Washington D.C. yesterday. He said, 'Wait, is that true?'"
    sentences = split_sentences(text)
    assert len(sentences) >= 2
    assert sentences[0] == "Hello there!"
    assert any("Washington" in s for s in sentences)


def test_abbreviation_splitting():
    text = "Dr. Watson and Mr. Holmes met at 221B Baker St. in London."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == "Dr. Watson and Mr. Holmes met at 221B Baker St. in London."

    text2 = "Prof. Plum vs. Col. Mustard in the library, etc. It was a mystery."
    sentences2 = split_sentences(text2)
    assert len(sentences2) == 2
    assert sentences2[0] == "Prof. Plum vs. Col. Mustard in the library, etc."
    assert sentences2[1] == "It was a mystery."

    text3 = "For e.g. fruit or i.e. vegetables, prices vary. They always do."
    sentences3 = split_sentences(text3)
    assert len(sentences3) == 2
    assert "e.g." in sentences3[0]
    assert "i.e." in sentences3[0]
    assert sentences3[1] == "They always do."


def test_dialogue_quote_splitting():
    text = '"Stop!" shouted V. "I need help."'
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == '"Stop!" shouted V.'
    assert sentences[1] == '"I need help."'

    # Single quotes
    text_single = "'Stop!' shouted V. 'I need help.'"
    sentences_single = split_sentences(text_single)
    assert len(sentences_single) == 2
    assert sentences_single[0] == "'Stop!' shouted V."
    assert sentences_single[1] == "'I need help.'"

    # Terminal quote ending a sentence before new sentence
    text_terminal = '"I have arrived." Holmes sat down by the fireplace.'
    sentences_terminal = split_sentences(text_terminal)
    assert len(sentences_terminal) == 2
    assert sentences_terminal[0] == '"I have arrived."'
    assert sentences_terminal[1] == "Holmes sat down by the fireplace."


def test_decimal_numbers_and_ellipses():
    text = "The ratio was 1.5 to 3.14159... or so they thought. That was surprising."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "1.5" in sentences[0]
    assert "3.14159" in sentences[0]
    assert "..." in sentences[0]
    assert sentences[1] == "That was surprising."

    # Decimal in middle of sentence should not split
    text2 = "He paid $2.50 for 1.25 kg of apples... or perhaps it was 3.0 kg."
    sentences2 = split_sentences(text2)
    assert len(sentences2) == 1
    assert "2.50" in sentences2[0]
    assert "1.25" in sentences2[0]
    assert "3.0" in sentences2[0]
    assert "..." in sentences2[0]


def test_markdown_headers_and_paragraphs():
    doc = """# Chapter 1: The Gathering
In the dimly lit parlor, three cloaked figures sat in silence.

The grandfather clock chimed midnight with an ominous reverberation.

## Section 2
"Stop!" shouted V. "I need help."
"""
    sentences = split_sentences(doc)
    assert sentences[0] == "# Chapter 1: The Gathering"
    assert sentences[1] == "In the dimly lit parlor, three cloaked figures sat in silence."
    assert sentences[2] == "The grandfather clock chimed midnight with an ominous reverberation."
    assert sentences[3] == "## Section 2"
    assert sentences[4] == '"Stop!" shouted V.'
    assert sentences[5] == '"I need help."'


def test_initials_splitting():
    text = "J. K. Rowling and George W. Bush visited A. Lincoln."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert "J. K. Rowling" in sentences[0]
    assert "George W. Bush" in sentences[0]
    assert "A. Lincoln" in sentences[0]


def test_cursor_polymorphic_initialization():
    # String input with multiple sentences
    tracker_str = CursorTracker("Sentence one. Sentence two. Sentence three.")
    assert len(tracker_str.sentences) == 3
    assert tracker_str.sentences[0] == "Sentence one."

    # List of strings with multiple sentences per element
    tracker_list = CursorTracker([
        "Chapter 1: The Beginning.",
        "Sentence A. Sentence B."
    ])
    assert len(tracker_list.sentences) == 3
    assert tracker_list.sentences[0] == "Chapter 1: The Beginning."
    assert tracker_list.sentences[1] == "Sentence A."
    assert tracker_list.sentences[2] == "Sentence B."

    # Empty inputs
    assert CursorTracker("").sentences == []
    assert CursorTracker([]).sentences == []
    assert CursorTracker(None).sentences == []


def test_cursor_tracking_and_preroll():
    doc = [
        "Chapter 1: The Gathering.",
        "In the dimly lit parlor, three cloaked figures sat in silence.",
        "The grandfather clock chimed midnight with an ominous reverberation."
    ]
    tracker = CursorTracker(doc)
    assert tracker.current_sentence() == "Chapter 1: The Gathering."
    
    tracker.advance()
    assert tracker.current_sentence() == "In the dimly lit parlor, three cloaked figures sat in silence."
    
    # Interrupted at word 4
    tracker.record_progress(word_offset=4)
    resumption = tracker.get_resumption_text(pre_roll_words=2)
    assert "dimly" in resumption or "parlor" in resumption


def test_cursor_offset_clamping_and_resumption():
    sentence = "In the dimly lit parlor, three cloaked figures sat in silence."
    words = sentence.split()
    total_words = len(words)
    max_offset = total_words - 1
    tracker = CursorTracker(sentence)
    
    # Overflow word offset: clamped to len(words) - 1
    tracker.record_progress(word_offset=500)
    assert tracker.pos.word_offset == max_offset
    resumption = tracker.get_resumption_text(pre_roll_words=3)
    assert resumption != ""
    assert resumption == "figures sat in silence."

    # Zero pre-roll on overflow: returns last word, never empty
    res_zero = tracker.get_resumption_text(pre_roll_words=0)
    assert res_zero == "silence."

    # Underflow word offset: clamped to 0
    tracker.record_progress(word_offset=-50)
    assert tracker.pos.word_offset == 0
    res_underflow = tracker.get_resumption_text(pre_roll_words=3)
    assert res_underflow == sentence

    # Negative pre-roll clamped to 0
    tracker.record_progress(word_offset=2)
    res_neg_preroll = tracker.get_resumption_text(pre_roll_words=-5)
    assert res_neg_preroll == "dimly lit parlor, three cloaked figures sat in silence."
