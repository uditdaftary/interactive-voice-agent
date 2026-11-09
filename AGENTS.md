# Multi-Agent Topology & Operational Registry

*Documentation of all agent personas, subagent clusters, communications protocols, and operational workflows for the Interactive Voice Agent project.*

---

## 1. Multi-Agent System Overview

This project was built and hardened using a coordinated multi-agent collective. To avoid mono-agent blind spots, work is divided across specialized roles with strict separation of concerns, formal reviews, and cross-agent communication protocols.

```
                              +--------------------------+
                              |   Lead Coordinator Agent |
                              |   (Antigravity Assistant)|
                              +-------------+------------+
                                            |
         +--------------------+-------------+--------------------+--------------------+
         |                    |                                  |                    |
         v                    v                                  v                    v
  +--------------+    +----------------+                 +----------------+    +--------------+
  | AUDITOR TEAM |    |   TEAM ALPHA   |                 |   TEAM BETA    |    |  FIXER TEAM  |
  |  (4 Agents)  |    | (2 Architects) |                 | (2 Architects) |    |  (4 Agents)  |
  +--------------+    +----------------+                 +----------------+    +--------------+
  - Auditor 1: Audio  - A1: Lead Audio/DSP                - B1: Hardware/Sys    - Fixer 1: Audio
  - Auditor 2: VAD    - A2: Orchestration                 - B2: Concurrency     - Fixer 2: VAD
  - Auditor 3: Cursor          |                                  |             - Fixer 3: Cursor
  - Auditor 4: LLM/HW          +------------ send_message --------+             - Fixer 4: LLM
```

---

## 2. Agent Teams & Specializations

### 2.1. Adversarial Scrutiny Team (Phase 1)
*Applied `/code-review-and-quality` (5-axis: correctness, readability, architecture, security, performance) and `/bugs-are-annoying` (adversarial defect hunting).*

1. **Auditor 1 (`Audio & Concurrency Auditor`)**:
   - *Domain*: `src/voice_agent/audio/ring_buffer.py`, `pipeline.py`.
   - *Key Findings*: Float truncation into digital silence (`BUG-A01`), 2D array broadcast crashes (`BUG-A02`), integer overflow acoustic shock (`BUG-A05`), heap allocations in hot path mutex (`BUG-A06`), zombie audio re-injection post-drain (`BUG-A07`), step-discontinuity pop on drain (`BUG-A08`).
2. **Auditor 2 (`VAD & Acoustic Auditor`)**:
   - *Domain*: `src/voice_agent/vad/energy_vad.py`, microphone streaming.
   - *Key Findings*: Disconnection from real microphone (`BUG-V01`), missing AEC causing self-barge-in loops (`BUG-V02`), static threshold failure in noise/quiet (`BUG-V03`), permanent lockup on hardware DC bias (`BUG-V04`), float32 scale mismatch (`BUG-V05`), unbuffered chunk hangover corruption (`BUG-V06`), transient click sensitivity (`BUG-V07`).
3. **Auditor 3 (`Orchestrator & Cursor Auditor`)**:
   - *Domain*: `src/voice_agent/orchestrator/state_machine.py`, `cursor/tracker.py`, `cursor/splitter.py`.
   - *Key Findings*: User unable to interrupt during `REPLYING` state (`BUG-O01`), state machine deadlock on callback exception (`BUG-O02`), missing thread safety (`BUG-O03`), disconnected audio-cursor progress (`BUG-O04`), type hazards with magic 100-char threshold (`BUG-O05`), abbreviations broken into fragments (`BUG-O06`), dialogue quotes unhandled (`BUG-O07`), invalid regex `[\b]` backspace token (`BUG-O08`).
4. **Auditor 4 (`LLM & Hardware Auditor`)**:
   - *Domain*: `src/voice_agent/llm/client.py`, `orchestrator/context.py`, hardware budgets.
   - *Key Findings*: Infinite loop in stream parser on abbreviations (`BUG-L01`), decimals split mid-number (`BUG-L03`), ellipses emitting single dots (`BUG-L04`), trailing text lost without flush (`BUG-L05`), prompt injection vulnerabilities (`BUG-L06`), 4GB VRAM budget violations (`BUG-L07`, `BUG-L09`), subprocess worker zombies (`BUG-L08`), pipe buffer deadlocks (`BUG-L14`).

---

### 2.2. Solution Architecture Teams (Phase 2)
*Cross-communicated via `send_message` to design competing, united architectural specifications.*

#### Team Alpha
- **Subagent A1**: Lead Audio & DSP Systems Engineer
- **Subagent A2**: Lead Orchestrator & State Machine Architect
- *United Deliverable*: `docs/TEAM_ALPHA_UNITED_PROPOSAL.md`
- *Core Contributions*: Designed the token-shielding architecture (`\uE000` sentinels), half-cosine micro-fadeout window on drain, and re-entrant `threading.RLock` state machine.

#### Team Beta
- **Subagent B1**: Lead Real-Time Hardware & Systems Architect
- **Subagent B2**: Lead Pipeline & Stream Concurrency Engineer
- *United Deliverable*: `docs/TEAM_BETA_UNITED_SOLUTION_PROPOSAL.md`
- *Core Contributions*: Designed the 4GB RTX 3050 VRAM budget envelope, zero-allocation `read_into` with silence fill, Win32 Job Object subprocess lifecycle management, and non-blocking stderr drain threads.

---

### 2.3. Implementation & Fixer Team (Phase 3)
*Applied `/test-driven-development` and `/verification-before-completion` to execute code fixes and verify tests.*

1. **Fixer 1 (`Audio & DSP Engine`)**:
   - *Target*: `src/voice_agent/audio/ring_buffer.py`, `tests/test_audio_buffer.py`.
   - *Outcome*: 15 comprehensive unit tests written and passed (100%).
2. **Fixer 2 (`VAD & Acoustic Ingestion`)**:
   - *Target*: `src/voice_agent/vad/energy_vad.py`, `tests/test_vad.py`.
   - *Outcome*: 9 comprehensive unit tests written and passed (100%).
3. **Fixer 3 (`Cursor & State Machine`)**:
   - *Target*: `src/voice_agent/cursor/splitter.py`, `cursor/tracker.py`, `orchestrator/state_machine.py`, `tests/test_cursor.py`, `tests/test_state_machine.py`.
   - *Outcome*: 15 comprehensive unit tests written and passed (100%).
4. **Fixer 4 (`LLM & Pipeline Coordinator`)**:
   - *Target*: `src/voice_agent/llm/client.py`, `orchestrator/context.py`, `pipeline.py`, `run_agent.py`, `tests/test_llm.py`, `tests/test_context.py`, `tests/test_full_pipeline.py`.
   - *Outcome*: 17 comprehensive unit/integration tests written and passed (100%).

---

## 3. Dedicated Interactive UI/UX Agent Session

Per architectural recommendation, the high-fidelity UI/UX and visual design work is decoupled into a **dedicated interactive chat session** directly with the user.

- **Workspace**: `C:\Projects\interactive-voice-agent`
- **Active Skills**: `/impeccable`, `/brandkit`, `/high-end-visual-design`, `/animejs-animation`, `/motion-graphics`
- **Artifacts Managed**: `ui/index.html`, `ui/app.css`, `ui/serve.py`, `DESIGN.md`
- **Focus Areas**:
  - Cyberpunk-themed Audio Visualizer HUD (neon oscilloscope, frequency spectrum bars).
  - Narrative Reader panel with real-time word highlighting synced to audio playhead.
  - Sub-50ms Barge-In Wave indicator (flashing instantaneous visual feedback upon user speech detection).
  - Character Persona Voice Selector (David, Lucy, Johnny Silverhand, V Female, V Male).

---

## 4. Operational Guidelines for Future Subagents

1. **No Shared State Without Locks**: Any cross-thread communication must be guarded by `threading.Lock` or `threading.RLock`.
2. **Deterministic Budgets**:
   - Audio buffer drain: $< 5\text{ms}$.
   - Prompt context: $\le 2048\text{ tokens}$.
   - VRAM usage: $\le 3100\text{ MB}$ (leaving $\ge 500\text{ MB}$ headroom on RTX 3050).
3. **Verification Before Completion**: Never declare a task complete without executing `pytest` on the relevant test suite and confirming 100% test pass evidence.
