# Task 3 Brief: Pure-Python Okapi BM25 Ranking Engine & Rolling Context Windows (Issue #18)

## Objective
Resolve Issue #18: Replace naive substring matching in document retrieval with a pure-Python, zero-dependency Okapi BM25 ranking engine, index overlapping 3-sentence rolling context windows to preserve pronouns and antecedents, and establish a clear context provenance hierarchy.

## Assigned Files
- `src/voice_agent/document/bm25.py` (new module)
- `src/voice_agent/document/parser.py`
- `src/voice_agent/server/narration.py`
- `tests/test_bm25.py` (new test file)
- `tests/test_document_and_api.py`

## Requirements
1. **Pure-Python Okapi BM25 Ranking Engine (`src/voice_agent/document/bm25.py`)**:
   - Implement `BM25Index`:
     - Parameters: $k_1 = 1.5, b = 0.75$.
     - Tokenization: regex word boundaries `\b[a-z0-9_-]{2,}\b`.
     - 150+ English stopwords.
     - Formula:
       $$\text{Score}(Q, D) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
     - Exact word matching (eliminates "car" in "scars", "art" in "department", "has" in "purchased").
2. **Rolling 3-Sentence Context Windows**:
   - When building passages for retrieval from `sentences`:
     - For each sentence $s_i$, passage $W_i = \text{" ".join}([s_{\max(0, i-1)}, s_i, s_{\min(N-1, i+1)}])$.
     - Index both single sentences and 3-sentence windows.
     - Preserves subject antecedents for pronouns ("David pulled out the Sandevistan. He activated the unit.").
3. **Playhead Proximity Boost**:
   - In `search_document_passages(query, max_passages=3, current_sentence_idx=None)`:
     - If `current_sentence_idx` is provided, apply a Gaussian proximity boost:
       $\text{boost} = 1.0 + 0.35 \cdot \exp\left(-\frac{|i - \text{current\_idx}|^2}{2 \cdot 10^2}\right)$
       to break ties in favor of the active reading location.
4. **Context Provenance Hierarchy in `NarrationEngine.handle_user_interruption`**:
   - If the best document passage has a high BM25 score ($\ge 2.5$), prioritize document passages as the primary source of truth.
   - In `_synthesize_fallback_answer` (`client.py`), prioritize `best_book_sent` over `best_web_fact` when the query is about book contents.
5. **Context Budgeting & ChatML Sanitization**:
   - Re-integrate `assemble_qa_prompt` or `sanitize_chatml` from `orchestrator/context.py` in `handle_user_interruption` to enforce the 2048-token context budget on RTX 3050 4GB VRAM.
6. **Testing**:
   - Create `tests/test_bm25.py` testing BM25 tokenization, scoring, IDF weighting, and rolling windows.
   - Run `pytest tests/test_bm25.py tests/test_document_and_api.py`.
