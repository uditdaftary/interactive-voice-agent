"""Tier 4: Statistical Latency & Performance Benchmark Suite.

Benchmarks:
Every threshold is on the median of many runs with a generous bound, so one scheduler hiccup
cannot fail the suite. The measured values are printed; the bounds are regression tripwires.

1. Atomic Ring Buffer Drain Latency:
   - AudioRingBuffer.drain(fade_ms=2.0) median < 1ms across 1,000 iterations.
2. End-to-End Barge-In Interruption Latency:
   - Speech frame ingestion through VAD onset to playback buffer drainage, median < 50ms.
3. Audio Normalization & Resampling Throughput:
   - 24kHz to 16kHz linear resampling and normalization < 2ms per 30ms frame.
4. Generation Fencing Rejection Latency:
   - Stale TTS audio chunk rejection post-barge-in, median < 0.5ms across 1,000 iterations.
5. Real Whisper transcription (marked `slow` and `network`: a cold cache downloads tiny.en once):
   - Median of 3 runs on 2s of real speech < 2,000ms, with a non-empty transcript.
"""

import math
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from voice_agent.audio.hybrid import HybridMicrophoneBridge
from voice_agent.audio.loopback import VirtualAudioDevice
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.stt.backends import STT_UNAVAILABLE
from voice_agent.stt.recognizer import STTEngine
from voice_agent.vad.energy_vad import VoiceActivityDetector


VOICE_SAMPLE = Path(__file__).resolve().parent.parent / "models" / "voices" / "david-martinez.wav"


def test_atomic_ring_buffer_drain_latency_benchmark() -> None:
    """Benchmark verifying the median atomic ring buffer drain latency across 1,000 runs is < 1ms."""
    sample_rate = 24000
    capacity = sample_rate * 10  # 10 seconds = 240,000 samples
    rb = AudioRingBuffer(capacity_samples=capacity, sample_rate=sample_rate)

    test_audio = np.ones(48000, dtype=np.int16) * 1000  # 2 seconds of audio
    fade_ms = 2.0
    iterations = 1000

    # Warm up caches and JIT paths
    for _ in range(50):
        rb.write(test_audio)
        rb.drain(fade_ms=fade_ms)

    times_ms = []
    for _ in range(iterations):
        rb.write(test_audio)
        t0 = time.perf_counter()
        rb.drain(fade_ms=fade_ms)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(times_ms)
    mean_lat = float(latencies.mean())
    median_lat = float(np.median(latencies))
    min_lat = float(latencies.min())
    max_lat = float(latencies.max())
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    print("\n[BENCHMARK] 1,000 Atomic Buffer Drain Iterations:")
    print(f"  Min:    {min_lat:.5f} ms")
    print(f"  Max:    {max_lat:.5f} ms")
    print(f"  Mean:   {mean_lat:.5f} ms")
    print(f"  Median: {median_lat:.5f} ms  (Bound: < 1.0 ms)")
    print(f"  P95:    {p95_lat:.5f} ms")
    print(f"  P99:    {p99_lat:.5f} ms")

    assert median_lat < 1.0, f"Median drain latency {median_lat:.5f}ms exceeded the 1ms bound!"


def test_end_to_end_barge_in_latency_benchmark() -> None:
    """Benchmark verifying median end-to-end barge-in latency from speech detection to buffer drain is < 50ms."""
    # The bound is on the median, which 100 runs pin down as well as 1,000 do; each run builds a bridge, so the count sets the cost.
    iterations = 100
    e2e_latencies_ms = []

    for _ in range(iterations):
        rb = AudioRingBuffer(capacity_samples=24000 * 5, sample_rate=24000)
        vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
        virtual_dev = VirtualAudioDevice(sample_rate=24000, chunk_size=720)

        bridge = HybridMicrophoneBridge(
            source="browser",
            input_sample_rate=24000,
            target_sample_rate=16000,
            frame_duration_ms=30,
            ring_buffer=rb,
            vad=vad,
            virtual_device=virtual_dev,
        )

        # Pre-fill playback buffer with 2 seconds of audio
        rb.write(np.ones(48000, dtype=np.int16) * 2000)
        assert rb.size == 48000

        # Prime VAD with 1 speech frame so the next frame immediately triggers onset (onset_frames=2)
        t = np.linspace(0, 0.03, 720, endpoint=False)
        speech_frame = (0.8 * 32767 * np.sin(2 * np.pi * 350.0 * t)).astype(np.int16).tobytes()
        bridge.feed_browser_pcm(speech_frame)

        # Measure time from onset frame ingestion to playback drain and state halt
        t0 = time.perf_counter()
        bridge.feed_browser_pcm(speech_frame)
        t_drain_done = time.perf_counter()

        latency_ms = (t_drain_done - t0) * 1000.0
        e2e_latencies_ms.append(latency_ms)

        # Invariant: buffer must be emptied
        assert rb.size == 0
        bridge.close()

    arr = np.array(e2e_latencies_ms)
    mean_e2e = float(arr.mean())
    median_e2e = float(np.median(arr))
    max_e2e = float(arr.max())
    p95_e2e = float(np.percentile(arr, 95))
    p99_e2e = float(np.percentile(arr, 99))

    print(f"\n[BENCHMARK] {iterations} End-to-End Barge-In Iterations:")
    print(f"  Min:  {arr.min():.4f} ms")
    print(f"  Max:  {max_e2e:.4f} ms")
    print(f"  Mean: {mean_e2e:.4f} ms")
    print(f"  Median: {median_e2e:.4f} ms  (Bound: < 50.0 ms)")
    print(f"  P95:  {p95_e2e:.4f} ms")
    print(f"  P99:  {p99_e2e:.4f} ms")

    assert median_e2e < 50.0, f"Median barge-in latency {median_e2e:.4f}ms exceeded 50ms budget!"


def test_audio_normalization_and_resampling_throughput_benchmark() -> None:
    """Benchmark verifying 24kHz -> 16kHz audio normalization and resampling operates in < 2ms."""
    iterations = 500
    chunk_samples = 720  # 30ms @ 24kHz
    raw_bytes = np.random.randint(-10000, 10000, size=chunk_samples, dtype=np.int16).tobytes()

    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        normalized = HybridMicrophoneBridge.normalize_audio(raw_bytes)
        _ = HybridMicrophoneBridge.resample_audio(normalized, orig_sr=24000, target_sr=16000)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    mean_processing = float(arr.mean())
    p99_processing = float(np.percentile(arr, 99))

    print(f"\n[BENCHMARK] Normalization & 24k->16k Resampling ({iterations} frames):")
    print(f"  Mean: {mean_processing:.5f} ms per 30ms frame")
    print(f"  P99:  {p99_processing:.5f} ms")

    # Processing 30ms frame in < 2ms leaves > 93% CPU headroom
    assert mean_processing < 2.0, (
        f"Mean resampling time {mean_processing:.5f}ms exceeded 2.0ms budget!"
    )


def test_generation_fencing_rejection_overhead_benchmark() -> None:
    """Benchmark verifying median stale audio chunk rejection post-drain is < 0.5ms."""
    rb = AudioRingBuffer(capacity_samples=24000 * 5, sample_rate=24000)
    stale_chunk = np.ones(2400, dtype=np.int16)

    # Invalidate generation
    rb.drain()
    stale_gen = rb.generation - 1

    times_ms = []
    for _ in range(1000):
        t0 = time.perf_counter()
        written = rb.write(stale_chunk, generation=stale_gen)
        t1 = time.perf_counter()
        assert written == 0
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    median_fence = float(np.median(arr))
    print(f"\n[BENCHMARK] Generation Fencing Rejection: Median {median_fence:.6f} ms (Bound: < 0.5 ms)")
    assert median_fence < 0.5, f"Generation fence rejection took {median_fence:.6f}ms, expected < 0.5ms"


@pytest.mark.slow
@pytest.mark.network  # a cold Hugging Face cache needs a one-time tiny.en download, which the guard would block
def test_benchmark_whisper_real() -> None:
    """Transcribe 2s of real speech with the real tiny.en model; median of 3 runs must be < 2,000ms."""
    engine = STTEngine(model_size="tiny.en", device="cpu", compute_type="int8")

    with wave.open(str(VOICE_SAMPLE), "rb") as wf:
        source_rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(2 * source_rate), dtype=np.int16)
    divisor = math.gcd(source_rate, 16000)
    speech_16k = resample_poly(samples.astype(np.float32), 16000 // divisor, source_rate // divisor)
    speech_pcm = np.clip(speech_16k, -32768, 32767).astype(np.int16).tobytes()

    # The model loads on first use; that first decode is not part of the latency under test.
    if engine.backend.transcribe(speech_pcm, sample_rate=16000).rejected_reason == STT_UNAVAILABLE:
        pytest.skip("Whisper tiny.en could not be loaded (not in the Hugging Face cache and the Hub is unreachable)")

    times_ms = []
    transcripts = []
    for _ in range(3):
        t0 = time.perf_counter()
        transcripts.append(engine.transcribe(speech_pcm, sample_rate=16000))
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(times_ms))
    print(f"\n[BENCHMARK] Whisper tiny.en on 2s of speech: median {median_ms:.1f} ms of 3 runs {times_ms}")
    print(f"  Transcript: {transcripts[0]!r}")
    assert all(text.strip() for text in transcripts), f"Empty transcript from real speech: {transcripts}"
    assert median_ms < 2000.0, f"Median Whisper latency {median_ms:.1f}ms exceeded 2,000ms!"
