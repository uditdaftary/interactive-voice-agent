# Team Beta: United Architectural Solution Proposal (Sprint 2 Hardening)

> **Architects:** High-Throughput Network & Concurrency Lead (B1) & Frontend Performance Architect (B2)  
> **Target Scope:** Comprehensive resolution of `BUG-001` through `BUG-010` in `bugs.md`  
> **Status:** PROPOSED & READY FOR LEAD EVALUATION  

---

## 1. Executive Summary & Design Philosophy

Team Beta approaches the Sprint 2 defects from a **network throughput, client-side resilience, and non-blocking event loop** perspective:
1. Sockets are treated as ephemeral streams with non-blocking drain queues.
2. Browser Web Audio ingestion incorporates defensive chunk alignment and soft acoustic cross-fading.
3. Client-side HUD uses Page Visibility API (`document.visibilityState`) to pause audio queues cleanly without memory bloat.

---

## 2. Technical Remediation Comparison

| Finding ID | Team Alpha Approach | Team Beta Alternative | Team Beta Advantage |
|---|---|---|---|
| **BUG-001** (Dead socket pruning) | Atomic set-difference comprehension in RLock | Pre-filtered set tracking + Asyncio gather exception isolation | Prevents slow client writes from delaying broadcast loop |
| **BUG-002** (FIFO audio contamination) | Hard buffer clear + VAD state reset | Half-cosine 5ms crossfade on switch + DC reset | Eliminates abrupt clicks on audio device handover |
| **BUG-003** (TestClient warning) | Direct `fastapi.testclient` import | Direct `fastapi.testclient` import | Identical |
| **BUG-004** (Background broadcast future) | Static done-callback logging | Daemon worker background task queue with asyncio.Queue | Completely decouples real-time audio threads from event loops |
| **BUG-005** (Playhead clock drift) | Advance by `samples_to_read` | Clock synchronization timestamp header in WebSocket audio packets | Network-agnostic synchronization |
| **BUG-006** (Unaligned byte buffer) | Truncate odd byte with slice | Truncate odd byte with slice | Identical |
| **BUG-007** (Overlapping cadence timers) | Clear timeout on trigger | Single requestAnimationFrame ticker driven by monotonic clock | Zero timer drift |
| **BUG-008** (Resampler empty array) | Early guard check | Early guard check | Identical |
| **BUG-009** (Canvas background tab) | Clamp buffer to 240 samples | Listen to `visibilitychange` to freeze queue | Zero CPU cycles when tab is hidden |
| **BUG-010** (Font fallback) | CSS variable fallbacks | CSS font-display: swap + font-face fallbacks | Zero FOIT |

---

## 3. Team Beta Recommendation

While Team Beta's async queue decoupling and crossfade transitions provide excellent long-term scale, **Team Alpha's immediate atomic set comprehension (BUG-001), FIFO purge (BUG-002), and playhead advance (BUG-005) represent the cleanest, zero-risk, mathematically deterministic solutions** within our existing codebase architecture.
