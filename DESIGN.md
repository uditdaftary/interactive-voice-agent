---
name: Voice Agent HUD Dashboard
description: A numbered catalog sheet on matte black, one white hairline plot per view, for live voice interaction, continuous narrative streaming, and sub-50ms barge-in orchestration.
colors:
  ground: "#0b0b0b"
  sleeve: "#121211"
  raised: "#191918"
  ink: "#efede6"
  ink-2: "#b0aea7"
  ink-3: "#8a8882"
  hair: "rgba(239, 237, 230, 0.15)"
  hair-2: "rgba(239, 237, 230, 0.32)"
  advisory: "#8d8c87"
  medium: "#2f6fd6"
  high: "#e8b622"
  critical: "#e03a2f"
  critical-ink: "#ff6a5c"
typography:
  display:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "64px"
    fontWeight: 500
    lineHeight: 0.85
    letterSpacing: "-0.025em"
    fontVariation: "'wdth' 125"
  headline:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: "-0.005em"
  title:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 500
    lineHeight: 1.55
  body:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.55
  mono:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.55
    fontVariation: "'wdth' 90"
rounded:
  square: "0px"
spacing:
  inset-tight: "12px"
  inset: "16px"
  gap: "32px"
  page: "48px"
  page-mobile: "16px"
components:
  button:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.square}"
    height: "44px"
    padding: "0 0 0 16px"
  button-primary-tab:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    rounded: "{rounded.square}"
    width: "44px"
  button-live-tab:
    backgroundColor: "{colors.high}"
    textColor: "{colors.ground}"
    rounded: "{rounded.square}"
    width: "44px"
---

# Design System: Voice Agent HUD Dashboard

## 1. Overview & Creative North Star

**Creative North Star: "The Catalog Sheet"** (Aligned with `C:\Projects\KGCR_Cloud_Project_2026`)

The interface reads as a numbered technical catalog sheet printed on matte black. Every view, subsystem, and telemetry cluster carries a catalog number (`VA 010`, `VA 020`, `VA 030`, `VA 040`), set in Archivo at 125 percent width or in Martian Mono. One white hairline plot owns each view; everything else is hairline rules, warm-white text and square controls.

The product refuses the generic SaaS pill-card template: there are no rounded cards, no filled panels at rest, no decorative gradients, and no blur overkill.

Color is near-monochrome. Warm white ink (`#efede6`) carries structure and selection on a matte black ground (`#0b0b0b`). Four severity/state colors carry all chromatic meaning:
- **Plot White** (`#efede6`): Primary foreground accent and selection line.
- **Signal Yellow** (`#e8b622`): Active audio playback, in-flight narration, speech detected.
- **Fault Red** (`#e03a2f`): Critical barge-in buffer flush, failed states, error notices.
- **Medium Blue** (`#2f6fd6`): Secondary audio processing stages.
- **Advisory Grey** (`#8d8c87`): Informational metadata.

---

## 2. Catalog Taxonomy

- **VA 010: Acoustic Ring Buffer Sink & Polar Estate Plot**  
  Real-time audio telemetry, atomic buffer drainage verification, and polar frequency chord mapping under dual-mode continuous narrative execution.
- **VA 020: Continuous Narrative Stream & Cursor Tracker**  
  Sentence boundary chunking, word-level pointer highlight, and automated audio resumption pre-roll protocol.
- **VA 030: Sub-50ms Barge-In Disruption Waveform**  
  Voice activity detection with instant circular ring buffer purge (< 0.05ms) and contextual QA prompt assembly.
- **VA 040: Resident Character Voice Roster**  
  Offline zero-shot voice profile adapters, speech rate cadences, and resident timbre frequency models.

---

## 3. Component Architecture

### Buttons
- **Shape:** Square, 44px minimum height, 1px Strong Hairline border (`rgba(239, 237, 230, 0.32)`), label inset 16px from left, 44px right-edge tab.
- **Primary Tab:** Fills Plot White with Matte Black icon.
- **Live / Stream Tab:** Fills Signal Yellow (`#e8b622`).
- **Danger Tab:** Takes Fault Red Text (`#ff6a5c`).

### The Single Plot Rule
Each view is built around one white hairline plot, and the plot leads the view. For the voice agent, the primary signature is the **Polar Acoustic Estate Plot** with concentric harmonics, 24 polar spoke lines, and acoustic dependency chords bent through the core.
