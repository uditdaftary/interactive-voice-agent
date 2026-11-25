"""Local command and greeting recognition: whole-utterance matches only (D2)."""

import pytest

from voice_agent.llm.client import CHARACTER_PERSONA_PROMPTS
from voice_agent.llm.intents import LocalIntent, classify_local, greeting_reply


@pytest.mark.parametrize(
    ("utterance", "intent"),
    [
        ("pause", "pause"),
        ("wait", "pause"),
        ("hold on", "pause"),
        ("stop", "pause"),
        ("resume", "resume"),
        ("continue", "resume"),
        ("go on", "resume"),
        ("keep going", "resume"),
        ("repeat", "repeat"),
        ("say that again", "repeat"),
        ("what did you say", "repeat"),
        ("hi", "greeting"),
        ("hello", "greeting"),
        ("hey", "greeting"),
        ("good morning", "greeting"),
        ("good evening", "greeting"),
        # Speech recognition capitalises, punctuates and pads.
        ("Pause.", "pause"),
        ("  Hold   on!  ", "pause"),
        ("Hello,", "greeting"),
        ("WHAT DID YOU SAY?", "repeat"),
        ("Keep going...", "resume"),
    ],
)
def test_classify_commands(utterance: str, intent: LocalIntent) -> None:
    assert classify_local(utterance) == intent


@pytest.mark.parametrize(
    "utterance",
    [
        "why did she stop",
        "wait, who is Lucy",
        "hello there, who is David",
        "can you repeat the part about Arasaka",
        "pause the story and tell me about the Relic",
        "continue reading chapter two",
        "who is the narrator",
        "",
        "   ",
        "...",
    ],
)
def test_question_with_command_word_is_none(utterance: str) -> None:
    assert classify_local(utterance) is None


def test_greeting_reply_per_persona() -> None:
    default = greeting_reply(None)
    assert default.strip()

    replies = {persona: greeting_reply(persona) for persona in CHARACTER_PERSONA_PROMPTS}

    assert all(reply.strip() for reply in replies.values())
    assert any(reply != default for reply in replies.values())
    # A persona added to CHARACTER_PERSONA_PROMPTS needs its own line, or it falls back to the default.
    assert all(reply != default for reply in replies.values())


def test_unknown_persona_gets_the_default_greeting() -> None:
    assert greeting_reply("someone-else") == greeting_reply(None)
