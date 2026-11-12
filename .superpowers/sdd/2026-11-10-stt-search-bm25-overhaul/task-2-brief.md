# Task 2 Brief: Web Search Grounding Relevance Overhaul & Interrogative Filtering (Issue #17)

## Objective
Resolve Issue #17: Fix the punctuation token disambiguation bug, expand concept/entity anchoring, increase search depth, eliminate interrogative title answers, and replace rigid definition keyword bias with information-density scoring.

## Assigned Files
- `src/voice_agent/llm/client.py`
- `tests/test_search_relevance.py` (new test file)
- `tests/test_llm.py`

## Requirements
1. **Punctuation-Free Tokenizer & Comprehensive Stopwords**:
   - In `search_web_sync` and `_synthesize_fallback_answer`:
     - Extract query tokens using regex word boundaries: `re.findall(r'\b[a-z0-9_-]{2,}\b', query_clean)` so trailing punctuation (`?`, `!`, `.`) never contaminates token matching.
     - Include a full 50+ stopword set (including `the`, `why`, `how`, `did`, `does`, `was`, `were`, `can`, `could`, `tell`, `about`, `what`, `where`, `when`, `who`, `from`, `with`).
2. **Context-Aware Entity Anchoring**:
   - Disambiguate not just 10 names, but Cyberpunk universe concepts:
     - Characters: `lucy`, `david`, `rebecca`, `maine`, `kiwi`, `dorio`, `smasher`, `silverhand`, `johnny`, `arasaka`, `yorinobu`, `saburo`, `takemura`, `panam`, `judy`, `alt cunningham`, `kerry`, `river`.
     - Lore concepts: `relic`, `blackwall`, `sandevistan`, `soulkiller`, `cyberpsychosis`, `cyberware`, `netrunner`, `edgerunner`, `braindance`, `trauma team`, `militech`, `kang tao`, `night city`.
     - When any of these tokens appear, ensure `"Cyberpunk"` is appended if not present.
3. **Retrieval Depth & Clean Snippet Extraction**:
   - Increase DuckDuckGo `max_results` in `search_web_sync` from 3 to **8**.
   - Log warnings on exceptions with details instead of silent swallowing.
4. **Interrogative Sentence Penalty & Question Title Filtering**:
   - If a candidate sentence from web snippets ends in `?` or starts with question patterns (`How `, `Why `, `Can someone `, `What is `, `Who was `), apply a heavy penalty (-20 score) so question titles are never returned as answers.
5. **Information-Density & Fact Scoring**:
   - Score candidate sentences based on:
     - Matching non-stopword query terms (e.g. 5 points per term).
     - Predicate / factual clause markers (`killed by`, `stole`, `invented`, `developed`, `destroyed`, `died`, `shot`, `caused by`, `biochip`, `construct`, `ai`, `subroutine`, `syndrome`, `condition`).
     - Definition markers (`is a`, `known as`, `deuteragonist`, `protagonist`, `megacorporation`).
   - Penalize forum junk, reddit headers, and short fragments (< 25 characters).
6. **Testing**:
   - Write comprehensive tests in `tests/test_search_relevance.py` verifying:
     - Punctuation tolerance (`"who is johnny?"` matches Johnny Silverhand).
     - Concept anchoring (`"what is the relic"` anchors with Cyberpunk).
     - Interrogative penalty (`"How Did Johnny Silverhand Die?"` is rejected).
     - Factual answer synthesis for causal questions (`"how did Johnny die"`).
   - Run `pytest tests/test_search_relevance.py tests/test_llm.py`.
