# Consolidated Audit Findings: Interactive Voice Agent

Date: 2026-11-09
Scope: C:\Projects\interactive-voice-agent

## 1. Audio Streaming & Concurrency Defects
- **BUG-A01 (Critical): Destructive Truncation of Floating-Point Audio** (`ring_buffer.py:19-20`): `data.astype(np.int16)` truncates all float in `[-1.0, 1.0]` to `0` (silence). Must scale by 32767.0 and clip.
- **BUG-A02 (Critical): 2D Audio Array Broadcast Crash** (`ring_buffer.py:31,34`): `sounddevice` delivers `(N, 1)` or `(N, 2)` arrays; 1D slice assignment raises `ValueError`. Must downmix/flatten to 1D.
- **BUG-A03 (Critical): Negative samples_requested Crash** (`ring_buffer.py:42-46`): Negative values trigger `ValueError: negative dimensions are not allowed`. Must guard `samples_requested <= 0`.
- **BUG-A04 (Critical): Playback Underrun Callback Crash** (`ring_buffer.py:40-55`): Truncated returns during underflow cause `ValueError` in `outdata[:] = ...` in PortAudio callback. Need `read_into` or padded `read_frames`.
- **BUG-A05 (Critical): Integer Overflow & Polarity Inversion Shock** (`ring_buffer.py:20`): Unclipped casts wrap modulo 65536, creating violent acoustic clicks. Must clip before casting.
- **BUG-A06 (Intermediate): Heap Allocation Inside Mutex Hot Path** (`ring_buffer.py:44,46`): `np.empty` and `np.zeros` inside lock risk GC jitter and audio dropouts.
- **BUG-A07 (Intermediate): Drain Race Condition & Zombie Audio Re-Injection** (`ring_buffer.py:57-63`, `pipeline.py:23-25`): In-flight TTS chunks written after drain() re-inject stale pre-barge-in audio. Need generation fencing and task cancellation.
- **BUG-A08 (Intermediate): Acoustic Pop/Click on Drain** (`ring_buffer.py:57-63`): Immediate zeroing causes step discontinuity. Need 1-2ms rapid micro-fadeout.
- **BUG-A09 (Intermediate): Raw Byte Stream Crash** (`ring_buffer.py:19`): `bytes` object has no `dtype`. Support buffer protocol.
- **BUG-A10 (Intermediate): Missing Playhead Tracking** (`ring_buffer.py:9-16`): No sample counter to map elapsed playback to words for cursor tracking.

## 2. Voice Activity Detection (VAD) & Acoustic Ingestion Defects
- **BUG-V01 (Critical): Absence of Real Microphone Ingestion** (`run_agent.py`, `pipeline.py`): No `sounddevice.InputStream`, completely disconnected mock timers.
- **BUG-V02 (Critical): Acoustic Echo Cancellation Missing** (`energy_vad.py`, `pipeline.py`): Speaker output enters microphone, agent interrupts itself on its own speech in an infinite loop.
- **BUG-V03 (Critical): Fixed Energy Threshold Without Adaptive Noise Floor** (`energy_vad.py:4,9,25`): Static 300.0 threshold fails in noisy environments (permanent lockup) or quiet rooms (deafness).
- **BUG-V04 (Critical): DC Offset Bias Blindness** (`energy_vad.py:18-21`): Computes uncentered RMS; hardware DC bias causes permanent false positive speech lockup. Must center signal and high-pass filter.
- **BUG-V05 (Critical): Audio Scale / Dtype Mismatch in VAD** (`energy_vad.py:9,25`): Threshold 300.0 compared against float32 `[-1.0, 1.0]` never triggers. Standardize to dBFS or normalize.
- **BUG-V06 (Intermediate): Unbuffered Chunk Sizes Corrupt Hangover** (`energy_vad.py:13,23-31`): Hangover decrements per call, not per millisecond. Need internal FIFO buffer of exact 30ms frames.
- **BUG-V07 (Intermediate): Lack of Transient Impulse Suppression** (`energy_vad.py:23-31`): Single-frame clicks/keystrokes trigger full 270ms speech detection. Need multi-frame onset persistence window.
- **BUG-V08 (Intermediate): Trailing Hangover Latency Penalty** (`energy_vad.py:10,28-30`): Fixed 240ms hangover delays turn completion. Decouple pause bridging from finalization.
- **BUG-V09 (Intermediate): Input Validation Missing** (`energy_vad.py:18-24`): `None`, `NaN`, `Inf` crashes detector.

## 3. Orchestrator, Cursor Tracker & Sentence Splitter Defects
- **BUG-O01 (Critical): Inability to Barge-In During REPLYING State** (`state_machine.py:16-23`): Only checks `if state == NARRATING`. Cannot interrupt agent while answering.
- **BUG-O02 (Critical): Deadlock on Barge-In Callback Exception** (`state_machine.py:18-22`): If callback raises, state remains in `BARGE_IN_TRIGGERED` forever. Wrap in `try...finally`.
- **BUG-O03 (Critical): Lack of Thread Safety on State Transitions** (`state_machine.py`, `pipeline.py`): Unsynchronized mutations across async event loop and audio threads.
- **BUG-O04 (Critical): Disconnected Audio-Cursor Progress & Frozen Cursor** (`pipeline.py:13-60`, `tracker.py:32-33`): `record_progress()` and `advance()` never called in pipeline. Cursor never advances.
- **BUG-O05 (Critical): CursorTracker Type Hazard & 100-Char Magic Threshold** (`tracker.py:12-17`): Passing `str` unpacks into single chars; single items `<= 100` chars skip splitting entirely.
- **BUG-O06 (Intermediate): Sentence Splitter Breaks on Abbreviations** (`splitter.py:4`): Splits at `Dr.`, `Mr.`, `St.` into fragmented sentence stubs.
- **BUG-O07 (Intermediate): Failure to Split on Terminal Dialogue Quotes** (`splitter.py:4`): `!" ` or `." ` fails lookbehind, merging distinct sentences.
- **BUG-O08 (Intermediate): Invalid Regex Token `[\b]`** (`splitter.py:4`): Matches ASCII backspace `0x08` instead of word boundary.
- **BUG-O09 (Intermediate): Unbounded Word Offset Causes Silent Resumption** (`tracker.py:32-43`): Offset past end returns empty string. Clamp offset to word count.
- **BUG-O10 (Intermediate): Markdown Headers Coalesced** (`splitter.py:6-8`): Double newlines and headers squashed into single run-on sentences.

## 4. LLM Client, Context Packaging & Hardware Budget Defects
- **BUG-L01 (Critical): PunctuationStreamParser Infinite Loop on Abbreviations** (`client.py:18-24`): `find(".")` continually returns first abbreviation index without advancing buffer, freezing stream.
- **BUG-L02 (Critical): Multi-Dot Abbreviations Split Mid-Word** (`client.py:18-24`): Splitting `e.g.` yields `"For e."`.
- **BUG-L03 (Critical): Floating Point Decimals Split Mid-Number** (`client.py:18-24`): `1.5` splits into `"1."` and `"5"`.
- **BUG-L04 (Critical): Ellipses `...` Cause Single-Dot Stutter** (`client.py:18-24`): Yields lone `.` tokens to TTS.
- **BUG-L05 (Critical): Trailing Text Dropped / Missing Flush** (`pipeline.py:53-57`, `client.py:27-30`): Unpunctuated final tokens dropped because `flush()` is never invoked.
- **BUG-L06 (Critical): Prompt Injection Vulnerability** (`context.py:15-22`): Unsanitized strings allow ChatML delimiters `<|im_start|>` to override system prompt.
- **BUG-L07 (Critical): Unbounded Context Length Leading to RTX 3050 CUDA OOM** (`context.py:8-19`): 4GB VRAM limit requires prompt <= 2048 tokens to keep KV cache within ~350MB headroom.
- **BUG-L08 (Critical): STT Subprocess Worker Zombie Processes** (`local_stt_worker.py:515-525`): No watchdog or Job Object on Windows; orphaned on parent crash.
- **BUG-L09 (Critical): Fatal CUDA OOM if OmniVoice is Used** (`resident_voice_worker.py:1187`): OmniVoice needs ~2GB VRAM; Qwen 4B takes 2.7GB. Exceeds 4GB VRAM. TTS must strictly use PocketTTS on CPU.
- **BUG-L10 (Intermediate): Multi-Sentence Yield Latches Later Sentences** (`client.py:14-25`): Returns only one sentence per feed call, delaying downstream speech.
- **BUG-L11 (Intermediate): Non-Deterministic Punctuation Set Evaluation** (`client.py:12,16`): Set iteration order produces inconsistent chunk boundaries.
- **BUG-L12 (Intermediate): LLMClient is Non-Functional Stub** (`client.py:42-50`): Lacks actual HTTP/SSE connection to llama.cpp/Ollama.
- **BUG-L13 (Intermediate): Unenforced 3-Sentence Constraint** (`context.py:3-6`): Prompt constraint ignored by verbose LLMs; need token limit and stream stop guard.
- **BUG-L14 (Intermediate): Stderr Pipe Deadlock in Worker IPC** (`local_stt_worker.py:70-71`): Blocking stderr writes deadlock if parent pipe buffer fills.
