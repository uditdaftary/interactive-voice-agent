from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
from voice_agent.cursor.splitter import split_sentences


@dataclass
class CursorPosition:
    sentence_idx: int = 0
    word_offset: int = 0
    char_offset: int = 0


class CursorTracker:
    """Tracks reading position across sentences with polymorphic initialization and clamped offsets."""

    def __init__(self, sentences_or_text: Union[str, Sequence[Any], None] = None):
        sentences: List[str] = []
        if isinstance(sentences_or_text, str):
            sentences = split_sentences(sentences_or_text)
        elif sentences_or_text is not None and hasattr(sentences_or_text, "__iter__"):
            for item in sentences_or_text:
                if isinstance(item, str):
                    sentences.extend(split_sentences(item))
                elif hasattr(item, "__iter__") and not isinstance(item, (bytes, bytearray)):
                    for sub in item:
                        if isinstance(sub, str):
                            sentences.extend(split_sentences(sub))
                        elif sub is not None:
                            sentences.extend(split_sentences(str(sub)))
                elif item is not None:
                    sentences.extend(split_sentences(str(item)))
        elif sentences_or_text is not None:
            sentences = split_sentences(str(sentences_or_text))

        self.sentences: List[str] = sentences
        self.pos = CursorPosition()

    def current_sentence(self) -> Optional[str]:
        if 0 <= self.pos.sentence_idx < len(self.sentences):
            return self.sentences[self.pos.sentence_idx]
        return None

    def advance(self) -> bool:
        if self.pos.sentence_idx < len(self.sentences):
            self.pos.sentence_idx += 1
            self.pos.word_offset = 0
            self.pos.char_offset = 0
            return self.pos.sentence_idx < len(self.sentences)
        return False

    def record_progress(self, word_offset: int):
        """Records word progress within current sentence, clamping to [0, total_words - 1]."""
        current = self.current_sentence()
        if current is not None:
            words = current.split()
            total_words = len(words)
            if total_words > 0:
                self.pos.word_offset = max(0, min(word_offset, total_words - 1))
                return
        self.pos.word_offset = max(0, word_offset)

    def get_resumption_text(self, pre_roll_words: int = 3) -> str:
        """Returns resumption text rewound by pre_roll_words, guaranteed non-empty if words exist."""
        current = self.current_sentence()
        if not current:
            return ""
        words = current.split()
        if not words:
            return current.strip()

        effective_pre_roll = max(0, pre_roll_words)
        offset = max(0, min(self.pos.word_offset, len(words) - 1))
        start_idx = max(0, offset - effective_pre_roll)
        return " ".join(words[start_idx:])
