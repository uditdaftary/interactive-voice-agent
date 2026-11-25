"""Commands and greetings the agent handles locally, whether or not the language model is up (D2).

Matching is deliberately narrow: the whole utterance, after lowercasing and dropping punctuation, must be
one of a few phrases. "why did she stop" is a question for the model, not a pause command.
"""

import re
from typing import Literal

LocalIntent = Literal["pause", "resume", "repeat", "greeting"]

_PHRASES: dict[LocalIntent, frozenset[str]] = {
    "pause": frozenset({"pause", "wait", "hold on", "stop"}),
    "resume": frozenset({"resume", "continue", "go on", "keep going"}),
    "repeat": frozenset({"repeat", "say that again", "what did you say"}),
    "greeting": frozenset({"hi", "hello", "hey", "good morning", "good evening"}),
}

# Short and free of lore: the model answers anything that needs knowledge. Keys are persona ids.
_GREETINGS = {
    "johnny-silverhand": "Yeah, I'm here, samurai.",
    "lucy-kushinada": "Hi. I'm here.",
    "david-martinez": "Hey, choom. I'm here.",
    "v-male": "Hey, choom. I'm listening.",
    "v-female": "Hey. I'm listening.",
}
_DEFAULT_GREETING = "Hello. I'm listening."


def classify_local(query: str) -> LocalIntent | None:
    """The local intent for a whole utterance, or None when it should go to the model."""
    words = " ".join(re.sub(r"\W+", " ", query.lower()).split())
    for intent, phrases in _PHRASES.items():
        if words in phrases:
            return intent
    return None


def greeting_reply(character_id: str | None) -> str:
    return _GREETINGS.get(character_id or "", _DEFAULT_GREETING)
