# Remediation roadmap: sprints, dependencies and orchestration protocol

**Spec:** `docs/superpowers/specs/2026-11-12-remediation-design.md`. Every sprint plan argues from it; builders read both.

Each sprint gets its own plan, `docs/superpowers/plans/2026-11-13-sprint-<id>-<slug>.md`, written by the orchestrator just before dispatch. Each sprint ends in one PR to `main`.

## Sprints

Small sprints are merged where they share files or are mechanical (owner instruction, 2026-11-13). Haiku builds the mechanical sprints (lint sweeps, docs, small backends); Sonnet builds the coupled ones.

| Sprint | Phase | Scope (spec sections / matrix IDs) | Builder | Depends on | Can run in parallel with |
|---|---|---|---|---|---|
| S0.1 tooling | P0 | sec. 6 deps, ruff, mypy, CI (lint + typecheck jobs); H8, M22; L1 one-line fix | Haiku | none | none (touches every file) |
| S0.2 isolation | P0 | sec. 6 settings.py, seams, conftest guard, fakes, markers, CI pytest job; M2, M19, M20, M21, L15 | Sonnet | S0.1 | S3.2 |
| S1 safety | P1 | 5.8 in full, 5.5 `/api/transcribe`, sec. 8 interim accumulator; H3, H4, H5, M5, M8, M9, M10, M14, N5, N7, L2, L3 | Sonnet | S0.2 | S3.1, S3.2 |
| S2.1 state | P2 | 5.1 state machine, resume_to, deadlines, TaskRegistry; M4, M6, L5, part of L6 | Sonnet | S1 | S3.1, S3.2 |
| S2.2 interruption | P2 | 5.2, 5.3 minimal ack, owner connection, sec. 8 turn dedupe; H1, H2, D7, M3, M7, M11, N16-N20 | Sonnet | S2.1 | S3.1, S3.2 |
| S3.1 llm | P3 | 5.7; D2, H6, H7, M12, M13, N3, N9, L13, L14 | Sonnet | S0.2 | P1, P2, S3.2 |
| S3.2 backends | P3 | 5.5 (no Moonshine), 5.6 (no Pocket); M16, L9, L10, L12 | Haiku | S0.2 | P1, P2, S3.1 |
| S3.3 integration + ui | P3 | wire S3.1/S3.2 into narration, statuses, neutral lines, RC4 logging sweep (removes legacy ruff ignores), 5.9 P3 UI items; N1, N2, N8, N15, M1, L16, L17 | Sonnet | P2, S3.1, S3.2 | none |
| S3.4 docs | P3 | sec. 7; D4, D5, D6, L18 | Haiku | S3.3 | S4.2 |
| S4.1 audiohub | P4 | 5.4 AudioHub, server VAD barge-in, `voice_start`, client segmenter removal; M17, N10, N11, N13, L8, L11 | Sonnet | P3 merged | S4.2 |
| S4.2 optional backends | P4 | 5.6 PocketTTSBackend + delete the vendored TTS workers; 5.5 MoonshineBackend; H9 | Haiku | S3.2 | S3.4, S4.1, S4.3 |
| S4.3 audio out | P4 | 5.3 browser `word_times` + `playback_stopped`, host I/O, `host_output`, ring-buffer fixes; D3, D8, L4, L7, M18, N12, N14 | Sonnet | S4.1 | S4.2 |
| S4.4 retire | P4 | sec. 4 pipeline/run_agent, import-graph check, full-pipeline test, final doc touch-up; D1, M15 | Haiku | S4.1-S4.3 | none |

Merge order follows the "Depends on" column. A sprint built in parallel rebases on `main` after its dependencies merge.

## Orchestration protocol (every subagent follows this)

### Concurrency
- At most 3 subagents *running* at once, Opus reviewers included, and at most 2 of them Opus.
- A builder waiting for review instructions is idle and does not count.
- The Haiku issue-poster is exempt from the cap (owner instruction, 2026-11-13).

### Worktrees and branches
- The main checkout `C:\Projects\interactive-voice-agent` stays on `main` and is never edited by builders.
- Each sprint works in its own worktree: from the main checkout, run `git fetch origin && git worktree add -b sprint/<id>-<slug> C:/Projects/iva-wt/<id> origin/main`.
- Use only that worktree. Never touch another sprint's worktree or branch.
- Python: the shared venv `C:\Projects\interactive-voice-agent\.venv\Scripts\python.exe`. `pytest` puts the worktree's `src` first via `pythonpath`.
- Installing a new dependency into the shared venv is allowed, but only one that the sprint's plan adds to `pyproject.toml`.
- Before opening the PR: `git fetch origin && git rebase origin/main`, then re-run the gates.

### Commits
- One commit per logical change, using Conventional Commits (`feat(scope): ...`, `fix(scope): ...`, `test(scope): ...`, `chore(ci): ...`, `docs: ...`).
- Write the failing test first; the test and the fix may share a commit when they are one logical change.
- **Strict rule (owner, 2026-11-13): no attribution of any kind.** No `Co-Authored-By` or other trailers in commits, and no "Generated with Claude Code" or similar line or sentence in commits, PR bodies, review comments or issues.
- Never force-push a branch after its PR has review comments. Add fix commits instead.

### Gates (must pass before opening the PR, and before reporting fixes done)
- `ruff check .`
- `mypy` (configuration in `pyproject.toml`)
- `python -m pytest -q`: the default selection, which excludes the `network` and `slow` markers once S0.2 lands.
- Paste the tail of each command's output into the PR description under "Verification".
- If a gate cannot run, say so in the PR. Never skip a gate silently.

### Pull requests
- Open with `gh pr create --base main --head sprint/<id>-<slug>`.
- Title: conventional, e.g. `chore(ci): sprint S0.1 tooling and lint gates`.
- Body sections:
  - Summary
  - Spec items closed (matrix IDs)
  - Verification (gate output)
  - Not done / deviations
- Add one line to `CHANGELOG.md` under `## [Unreleased]`. Resolve changelog conflicts on rebase by keeping both lines.
- The builder's final report to the orchestrator: PR number, branch, worktree path, gate results, deviations from the plan.

### Review cycle (run by the orchestrator)
1. The phase's Opus reviewer runs `/code-review high --comment <PR#>`, which posts inline comments, and returns its findings.
2. The orchestrator selects the findings worth fixing.
3. The Haiku issue-poster opens one GitHub issue per selected finding:
   - labels `review-finding` and `sprint-<id>`;
   - the body links the PR and the comment, and quotes the finding.
4. The orchestrator sends the original builder the issue numbers. The builder:
   - fixes each issue on the same branch, one commit per issue, with `Refs #N` in the commit message;
   - pushes;
   - closes each issue with `gh issue close N --comment "Fixed in <sha>"`.
5. The same Opus reviewer re-reviews the fixes once (mandatory). Further rounds happen only if serious findings remain: correctness, security or data-loss level.
6. The orchestrator verifies locally on Windows (all three gates, plus the full suite including `windows`-marked tests), checks CI is green, then merges with `gh pr merge <PR#> --merge --delete-branch` (merge commit, preserving the granular commits) and removes the worktree.

### Peer communication without the orchestrator
- Agents talk to each other through temporary files at the main checkout root: `C:\Projects\interactive-voice-agent\AGENT_COMMS_<from>-to-<to>.txt`, for example `AGENT_COMMS_S3.1-to-S2.2.txt`. These are git-ignored.
- Check for files addressed to you at the start of each task and before opening the PR.
- Append messages, each dated, with the headers `Context:`, `Content:` and `Action:`.
- The recipient deletes the file once the message is resolved.
- Use it to settle shared-file edits: agree who owns which lines, and the second one rebases.
- Escalate to the orchestrator only for a scope or spec question.

### Scope discipline
- Implement the sprint plan. If the plan contradicts the code, stop that task and report it.
- Out-of-scope bugs go in the PR's "Not done / deviations" section, not into the diff.
