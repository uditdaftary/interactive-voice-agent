"""High-performance pure-Python Okapi BM25 ranking engine for document retrieval.

Key specifications:
- Parameters: k1 = 1.5, b = 0.75.
- Tokenization: regex word boundaries r'\b[a-z0-9_-]{2,}\b'.
- 150+ English stopwords set.
- Robertson-Spärck Jones / Lucene non-negative Okapi BM25 IDF weighting.
- Exact word boundary matching (eliminates substring false positives like 'car' in 'scars').
- Overlapping 3-sentence rolling context windows.
- Playhead proximity Gaussian boost for active narration grounding.
"""

from collections import Counter
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

# Comprehensive set of 160+ English stopwords
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "almost", "also", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "even", "few", "for", "from", "further", "get", "got", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "just", "let's", "like", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "say",
    "said", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
    # Common conversational / filler stopwords
    "tell", "know", "going", "much", "many", "well", "back", "still", "way", "take",
    "come", "see", "make", "made", "now", "always", "around", "another", "every", "first"
})

_WORD_REGEX = re.compile(r'\b[a-z0-9_-]{2,}\b')


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Extract lowercase alphanumeric tokens of length >= 2 using regex word boundaries.
    
    Optionally strips English stopwords to focus on content-bearing terms.
    """
    if not text:
        return []
    tokens = _WORD_REGEX.findall(text.lower())
    if remove_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS]
        return filtered if filtered else tokens  # Fall back to all tokens if only stopwords present
    return tokens


def build_rolling_windows(sentences: Sequence[str], deduplicate: bool = True) -> List[str]:
    """Build overlapping 3-sentence rolling context windows:
    W_i = ' '.join([s_{max(0, i-1)}, s_i, s_{min(N-1, i+1)}])
    
    Preserves subject antecedents and coreference across adjacent sentences.
    """
    n = len(sentences)
    if n == 0:
        return []
    windows = []
    for i in range(n):
        if deduplicate:
            start = max(0, i - 1)
            end = min(n - 1, i + 1)
            windows.append(" ".join(sentences[start : end + 1]))
        else:
            w = [sentences[max(0, i - 1)], sentences[i], sentences[min(n - 1, i + 1)]]
            windows.append(" ".join(w))
    return windows


def compute_gaussian_proximity_boost(
    sentence_idx: int,
    current_sentence_idx: Optional[int],
    sigma: float = 10.0,
    max_boost: float = 0.35,
) -> float:
    """Compute Gaussian proximity boost breaking ties in favor of active reading location:
    boost = 1.0 + 0.35 * exp(-|i - current_idx|^2 / (2 * 10^2))
    """
    if current_sentence_idx is None:
        return 1.0
    dist_sq = float((sentence_idx - current_sentence_idx) ** 2)
    denom = 2.0 * (sigma ** 2)
    return 1.0 + max_boost * math.exp(-dist_sq / denom)


class BM25Index:
    """Pure-Python Okapi BM25 ranking engine ($k_1 = 1.5, b = 0.75$).
    
    Provides exact-word matching, Lucene-style IDF weighting, rolling context window
    indexing, and playhead proximity boosting.
    """

    def __init__(
        self,
        corpus: Optional[Sequence[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        sentence_indices: Optional[Sequence[int]] = None,
    ):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.df: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.sentence_indices: Optional[List[int]] = None

        if corpus:
            self.index(corpus, sentence_indices=sentence_indices)

    def index(
        self,
        corpus: Sequence[str],
        sentence_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Index a corpus of document strings."""
        self.corpus = list(corpus)
        self.sentence_indices = list(sentence_indices) if sentence_indices is not None else None
        self.doc_tokens = [tokenize(doc, remove_stopwords=True) for doc in self.corpus]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        total_docs = len(self.corpus)
        self.avgdl = sum(self.doc_lengths) / total_docs if total_docs > 0 else 1.0
        if self.avgdl == 0:
            self.avgdl = 1.0

        # Calculate document frequency df for each unique term
        self.df = {}
        for tf in self.doc_term_freqs:
            for term in tf:
                self.df[term] = self.df.get(term, 0) + 1

        self.idf_cache.clear()

    def idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones / Lucene BM25 IDF for a term:
        IDF(t) = ln(1.0 + (N - n(t) + 0.5) / (n(t) + 0.5))
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        n_t = self.df.get(term, 0)
        n = len(self.corpus)
        idf_score = math.log(1.0 + (n - n_t + 0.5) / (n_t + 0.5))
        self.idf_cache[term] = idf_score
        return idf_score

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for all documents in the index."""
        if not self.corpus or not query.strip():
            return [0.0] * len(self.corpus)

        query_tokens = tokenize(query, remove_stopwords=True)
        if not query_tokens:
            return [0.0] * len(self.corpus)

        scores = [0.0] * len(self.corpus)
        for i, tf in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lengths[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)

            score = 0.0
            for term in query_tokens:
                f_td = tf.get(term, 0)
                if f_td == 0:
                    continue
                term_idf = self.idf(term)
                term_score = term_idf * (f_td * (self.k1 + 1.0)) / (f_td + self.k1 * len_norm)
                score += term_score

            scores[i] = score

        return scores

    def search(
        self,
        query: str,
        top_k: int = 3,
        current_sentence_idx: Optional[int] = None,
    ) -> List[Tuple[int, float, str]]:
        """Search index and return top_k results: (doc_index, score, doc_text).
        
        Applies playhead proximity Gaussian boost if current_sentence_idx is provided.
        """
        if not self.corpus or not query.strip():
            return []

        base_scores = self.get_scores(query)
        scored_docs: List[Tuple[int, float, str]] = []

        for idx, base_score in enumerate(base_scores):
            if base_score <= 0.0:
                continue

            # Determine sentence index for playhead proximity boost
            s_idx = self.sentence_indices[idx] if self.sentence_indices is not None else idx
            boost = compute_gaussian_proximity_boost(s_idx, current_sentence_idx)
            final_score = base_score * boost
            scored_docs.append((idx, final_score, self.corpus[idx]))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    @classmethod
    def from_sentences(
        cls,
        sentences: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build an index from narrative sentences, indexing both individual sentences
        and rolling 3-sentence context windows to preserve pronoun antecedents.
        """
        n = len(sentences)
        if n == 0:
            return cls([], k1=k1, b=b)

        corpus: List[str] = []
        indices: List[int] = []

        # 1. Single sentences
        for i, s in enumerate(sentences):
            corpus.append(s)
            indices.append(i)

        # 2. Overlapping 3-sentence rolling windows
        windows = build_rolling_windows(sentences)
        for i, w in enumerate(windows):
            corpus.append(w)
            indices.append(i)

        return cls(corpus=corpus, k1=k1, b=b, sentence_indices=indices)
