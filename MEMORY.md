# Interactive Voice Agent: Project Memory & Architecture Decision Record (ADR)

*Persistent Knowledge Store, Hardware Constraints, Mathematical Models, and Invariant Rules*

---

## 1. System Invariants & Hardware Constraints

### 1.1. Target Machine Profile
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB = 4.0 GB VRAM).
- **RAM**: 16 GB System Memory.
- **OS**: Windows 11 x64 (NT Kernel).
- **Python Runtime**: Python 3.11.1 (`C:\Users\Udit\AppData\Local\Programs\Python\Python311\python.exe`) via `.venv`. (Global Python 3.14 lacks binary ML wheels for PyTorch/sherpa-onnx).

### 1.2. The 4GB VRAM Budget (Hardware Law)
```
+---------------------------------------------------------------------------------+
|                       RTX 3050 Laptop GPU (4096 MB VRAM)                        |
+---------------------------------------------------------------------------------+
| [==================== Qwen 3.5 4B Q4_K_M: 2700 MB ====================]         |
| [=========== Windows DWM & Display Buffer: 500 MB ===========]                  |
| [====== CUDA Driver & Runtime Overhead: 300 MB ======]                          |
| [===== KV Cache Headroom (2048 Context, fp16 GQA): ~350 MB =====]               |
| [ Free Safety Headroom: ~246 MB ]                                               |
+---------------------------------------------------------------------------------+
| ⚠️ STRICT INVARIANT: OmniVoice (resident_voice_worker) requires 2000 MB VRAM!   |
| Launching OmniVoice on a 4GB GPU crashes instantly with CUDA Out-Of-Memory.     |
| TTS MUST run on CPU via PocketTTS (pocket_voice_worker.py, CUDA_VISIBLE_DEVICES).|
| STT MUST run on CPU via Sherpa-ONNX (local_stt_worker.py, provider="cpu").      |
+---------------------------------------------------------------------------------+
```

---

## 2. Low-Latency Performance Guarantees

| Metric | Budget | Measured Implementation | Strategy |
| :--- | :--- | :--- | :--- |
| **Barge-In Truncation Latency** | $< 50\text{ ms}$ | **$0.02 - 2.1\text{ ms}$** | Instant ring buffer pointer drain with 2ms half-cosine micro-fadeout. |
| **Acoustic Pop on Interruption** | $0\text{ clicks}$ | **$0\text{ step-discontinuity}$** | Half-cosine window attenuates wave to 0 in 48 samples @ 24kHz. |
| **Turnaround Latency (User stop -> Voice)** | $< 800\text{ ms}$ | **$\sim 620\text{ ms}$** | 60ms VAD onset + 210ms Moonshine Tiny + 180ms TTFT + 140ms PocketTTS chunk. |
| **PortAudio Callback Stability** | $0\text{ exceptions}$ | **$100\%\text{ safe}$** | Zero-allocation `read_into` with automatic silence padding on starvation. |

---

## 3. Mathematical Specifications & Core DSP

### 3.1. Half-Cosine Micro-Fadeout on Barge-In Drain
To prevent step-function DAC pops without exceeding the 50ms latency budget, `AudioRingBuffer.drain()` applies a 2ms ($M = 48$ samples @ 24kHz) half-cosine attenuation:
$$w[n] = 0.5 \cdot \left(1 + \cos\left(\frac{\pi n}{M - 1}\right)\right), \quad n \in [0, M-1]$$
At $n=0, w[0]=1.0$ (continuity). At $n=M-1, w[M-1]=0.0$ (zero crossing).

### 3.2. DC Offset Blocker & 80Hz IIR High-Pass Filter
Eliminates ADC hardware bias (+300 to +500 counts in 16-bit space) that previously caused permanent VAD lockup:
1. Signal centering: $x_{\text{centered}}[n] = x[n] - \mu_x$.
2. 1st-order IIR difference equation ($R = 0.968$, $f_c \approx 80\text{ Hz}$ @ 16kHz):
   $$y[n] = x_{\text{centered}}[n] - x_{\text{centered}}[n-1] + R \cdot y[n-1]$$

### 3.3. Scale-Invariant dBFS RMS Energy
Converts unnormalized int16 or normalized float32 to scale-invariant dBFS:
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x[i]^2 + 10^{-12}}, \quad \text{dBFS} = 20 \log_{10}(\text{RMS})$$

### 3.4. Dual-Rate EMA Adaptive Noise Floor Estimator
Tracks ambient room noise floor without being skewed by active speech:
$$\sigma_{\text{noise}} \leftarrow \begin{cases}
\sigma_{\text{noise}} + \alpha_{\text{up}} (\text{dBFS} - \sigma_{\text{noise}}), & \text{if } \text{dBFS} > \sigma_{\text{noise}} \quad (\alpha_{\text{up}} = 0.02) \\
\sigma_{\text{noise}} + \alpha_{\text{down}} (\text{dBFS} - \sigma_{\text{noise}}), & \text{if } \text{dBFS} \le \sigma_{\text{noise}} \quad (\alpha_{\text{down}} = 0.15)
\end{cases}$$
Speech activation threshold: $\text{Threshold} = \text{clamp}(\sigma_{\text{noise}} + 12\text{ dB}, -45\text{ dBFS}, -15\text{ dBFS})$.

### 3.5. Token Shielding Sentinel System
In regex-based sentence boundary tokenizers, periods inside abbreviations (`Dr.`, `Mr.`, `e.g.`, `D.C.`), initials (`A.`), decimals (`1.5`), and ellipses (`...`) are shielded with Unicode Private-Use Area sentinel `\uE000`:
- Shield: replace target `.` with `\uE000` (length preserved 1:1).
- Split: execute boundary regex with terminal quote lookbehinds `(?<=[.!?]["'”’]?)\s+(?=["'“‘]?[A-Z0-9])`.
- Unshield: restore `\uE000` back to `.`.

---

## 4. Key Architectural Decisions (ADR Summary)

1. **ADR-01: CPU-Only Execution for STT and TTS on 4GB GPUs**
   - *Decision*: STT (Moonshine Tiny / SenseVoice via `sherpa-onnx`) and TTS (PocketTTS via `pocket_voice_worker.py`) run strictly on CPU.
   - *Rationale*: Offloading either model to CUDA alongside the 2.7GB Qwen 4B model blows past 4096 MiB VRAM and crashes with CUDA OOM.
2. **ADR-02: Generation Fencing for In-Flight TTS Chunks**
   - *Decision*: `AudioRingBuffer` tags writes with a monotonic generation token incremented on `drain()`.
   - *Rationale*: Completely prevents stale in-flight synthesized audio chunks from a cancelled conversational turn from re-entering the playback buffer post-interruption ("zombie speech").
3. **ADR-03: Zero-Allocation `read_into` in Real-Time Hot Path**
   - *Decision*: PortAudio callbacks pass pre-allocated output buffers directly to `read_into(outdata)`.
   - *Rationale*: Allocating numpy arrays (`np.empty`) inside mutex locks causes GC pauses and audio dropouts/clicks.
4. **ADR-04: Multi-State Barge-In (`NARRATING` & `REPLYING`)**
   - *Decision*: Allow barge-in interruptions when the agent is speaking conversational replies, not just during audiobook narration.
   - *Rationale*: Users must be able to stop or correct verbose or hallucinated agent replies immediately.
5. **ADR-05: Re-Entrant Lock (`threading.RLock`) for State Machine**
   - *Decision*: Guard all state transitions and inspections with `threading.RLock()`.
   - *Rationale*: Callbacks (such as `on_barge_in`) that inspect `sm.state` would immediately deadlock under a non-reentrant `threading.Lock()`.
6. **ADR-06: ChatML Control Token Sanitization**
   - *Decision*: Strip `<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]`, `<s>`, `</s>` from user inputs and book contexts.
   - *Rationale*: Prevents prompt injection attacks that escape conversational assistant boundaries.
