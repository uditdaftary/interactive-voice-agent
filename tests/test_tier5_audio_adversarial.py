"""Tier 5: Adversarial Stress Testing & Empirical Verification Suite.

Adversarial Auditing & Stress Vectors:
1. Boundary & Buffer Wrap-Around:
   - Exact capacity writes (240,000 samples) and full-buffer rejection.
   - Circular buffer wrap-around sequence integrity across read/write cycles.
   - 2ms half-cosine micro-fadeout continuity under boundary wrap-around.
   - Multi-channel and chunked read_into consumption of fadeout tail.
   - Zero, single-sample, and micro-fadeout boundary conditions.
2. Extreme Signals & DC Bias:
   - Extreme DC offsets (+10,000 counts) and transient step rejection.
   - Floating-point NaNs, +Infs, -Infs, and recovery to normal speech detection.
   - Full-scale square wave transitions, hangover bridging, and turn completion.
   - Ultra-low noise floor adaptation (-90 dBFS) and safe clamping (-60 dBFS).
3. Resampling Stress:
   - Multi-rate ingestion (8k, 11.025k, 16k, 22.05k, 24k, 44.1k, 48k, 96k, 192k).
   - Arbitrary fractional and unaligned chunk stream ingestion.
   - Bounded FIFO memory guarantees (no accumulation / leak).
   - Real-time throughput verification (< 2.0ms per 30ms frame).
4. Concurrency & Latency:
   - Rapid multi-threaded source switching ("browser", "virtual", "host") with concurrent feeds and drains.
   - Statistical re-verification of 1,000 atomic drain iterations (< 0.05ms).
   - End-to-end barge-in latency budget (< 50ms).
"""

import threading
import time
from typing import List, Tuple
import numpy as np
import pytest

from voice_agent.audio.hybrid import HybridMicrophoneBridge
from voice_agent.audio.loopback import VirtualAudioDevice
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.vad.energy_vad import VoiceActivityDetector


# =============================================================================
# 1. Boundary & Buffer Wrap-Around Stress Tests
# =============================================================================

def test_boundary_exact_capacity_writes_and_overflow() -> None:
    """Stress circular buffer writes and drains at exact capacity boundary (240,000 samples)."""
    capacity = 240000
    sample_rate = 24000
    rb = AudioRingBuffer(capacity_samples=capacity, sample_rate=sample_rate)

    assert rb.capacity == capacity
    assert rb.size == 0
    assert rb.available_write == capacity

    # Write exactly capacity samples
    data_full = np.ones(capacity, dtype=np.int16) * 1234
    written = rb.write(data_full)
    assert written == capacity
    assert rb.size == capacity
    assert rb.available_write == 0

    # Attempt write of 1 more sample when full
    written_extra_1 = rb.write(np.array([9999], dtype=np.int16))
    assert written_extra_1 == 0
    assert rb.size == capacity

    # Attempt write of 10,000 samples when full
    written_extra_bulk = rb.write(np.ones(10000, dtype=np.int16))
    assert written_extra_bulk == 0
    assert rb.size == capacity

    # Partial read of 100,000 samples
    chunk_read = rb.read(100000)
    assert len(chunk_read) == 100000
    assert np.all(chunk_read == 1234)
    assert rb.size == 140000
    assert rb.available_write == 100000

    # Partial write of 150,000 samples (should accept exactly 100,000 and reject remainder)
    data_overflow = np.ones(150000, dtype=np.int16) * 5678
    written_partial = rb.write(data_overflow)
    assert written_partial == 100000
    assert rb.size == capacity
    assert rb.available_write == 0

    # Read all remaining 240,000 samples and verify payload contents
    drained = rb.read(capacity)
    assert len(drained) == capacity
    assert np.all(drained[:140000] == 1234)
    assert np.all(drained[140000:] == 5678)
    assert rb.size == 0
    assert rb.available_write == capacity


def test_boundary_circular_wrap_around_sequence_integrity() -> None:
    """Verify that multiple circular wrap-around read/write cycles preserve sequence integrity."""
    capacity = 240000
    rb = AudioRingBuffer(capacity_samples=capacity, sample_rate=24000)

    # Cycle 1: Write 160,000 ramp samples
    seq_1 = np.arange(160000, dtype=np.int32)
    rb.write(seq_1.astype(np.int16))
    assert rb.size == 160000

    # Read 100,000 samples (read_pos=100,000, write_pos=160,000, size=60,000)
    read_1 = rb.read(100000)
    assert np.array_equal(read_1, seq_1[:100000].astype(np.int16))
    assert rb.size == 60000

    # Cycle 2: Write 160,000 samples, forcing write_pos to wrap around (160000 + 160000) % 240000 = 80000
    seq_2 = np.arange(160000, 320000, dtype=np.int32)
    rb.write(seq_2.astype(np.int16))
    assert rb.size == 220000

    # Read 220,000 samples across wrap-around boundary
    read_2 = rb.read(220000)
    assert len(read_2) == 220000
    assert rb.size == 0

    # Combined read sequence must exactly match original sequential integers
    combined_expected = np.concatenate([seq_1, seq_2]).astype(np.int16)
    combined_actual = np.concatenate([read_1, read_2])
    assert np.array_equal(combined_actual, combined_expected)


def test_micro_fadeout_wrap_around_boundary_continuity() -> None:
    """Verify 2ms half-cosine micro-fadeout continuity when read position wraps across buffer boundary."""
    capacity = 240000
    sample_rate = 24000
    rb = AudioRingBuffer(capacity_samples=capacity, sample_rate=sample_rate)

    fade_samples = int(round(sample_rate * 0.002))  # 48 samples
    assert fade_samples == 48

    # Fill buffer completely
    full_data = np.full(capacity, 10000, dtype=np.int16)
    rb.write(full_data)

    # Read until read_pos is exactly 20 samples before buffer wrap (239,980)
    rb.read(239980)
    assert rb.read_pos == 239980
    assert rb.size == 20

    # Write 100 samples to bridge across wrap boundary (wraps to index 80)
    rb.write(np.full(100, 10000, dtype=np.int16))
    assert rb.size == 120

    # Drain buffer: 2ms fadeout must take 20 samples from buffer end (239980..239999)
    # and 28 samples from buffer start (0..27)
    rb.drain(fade_ms=2.0)
    assert rb.fadeout_tail is not None
    assert len(rb.fadeout_tail) == fade_samples
    assert rb.size == 0

    # Validate half-cosine window decay
    t = np.linspace(0.0, np.pi, fade_samples, endpoint=True)
    expected_window = 0.5 * (1.0 + np.cos(t))
    expected_tail = np.round(10000 * expected_window).astype(np.int16)

    assert np.array_equal(rb.fadeout_tail, expected_tail)
    assert rb.fadeout_tail[0] == 10000
    assert rb.fadeout_tail[-1] == 0
    assert np.all(np.diff(rb.fadeout_tail) <= 0), "Fadeout tail must be monotonically non-increasing"


def test_micro_fadeout_chunked_reads_and_multichannel_shapes() -> None:
    """Verify fadeout tail is correctly consumed across chunked read_into calls and multi-channel shapes."""
    sample_rate = 24000
    rb = AudioRingBuffer(capacity_samples=48000, sample_rate=sample_rate)
    rb.write(np.full(1000, 8000, dtype=np.int16))
    rb.drain(fade_ms=2.0)  # 48 samples

    # Chunk 1: Read 15 samples into 1D float32
    buf1 = np.zeros(15, dtype=np.float32)
    n1 = rb.read_into(buf1)
    assert n1 == 15
    assert len(rb.fadeout_tail) == 33
    assert buf1[0] == pytest.approx(8000.0 / 32767.0, rel=1e-3)

    # Chunk 2: Read 15 samples into 2D stereo int16 (15, 2)
    buf2 = np.zeros((15, 2), dtype=np.int16)
    n2 = rb.read_into(buf2)
    assert n2 == 15
    assert len(rb.fadeout_tail) == 18
    assert np.all(buf2[:, 0] == buf2[:, 1]), "Stereo channels must be identical"

    # Chunk 3: Read 18 samples into 2D mono int16 (18, 1)
    buf3 = np.zeros((18, 1), dtype=np.int16)
    n3 = rb.read_into(buf3)
    assert n3 == 18
    assert rb.fadeout_tail is None

    # Chunk 4: Subsequent read when empty must silence-pad
    buf4 = np.ones(32, dtype=np.int16) * 9999
    n4 = rb.read_into(buf4)
    assert n4 == 0
    assert np.all(buf4 == 0), "Buffer deficit must be silence padded"


def test_micro_fadeout_edge_cases() -> None:
    """Verify micro-fadeout behavior with empty buffer, 1-sample buffer, and 0ms fadeout."""
    rb = AudioRingBuffer(capacity_samples=48000, sample_rate=24000)

    # 1. Empty buffer drain
    rb.drain(fade_ms=2.0)
    assert rb.fadeout_tail is None

    # 2. 1-sample buffer drain
    rb.write(np.array([12345], dtype=np.int16))
    assert rb.size == 1
    rb.drain(fade_ms=2.0)
    assert rb.fadeout_tail is not None
    assert len(rb.fadeout_tail) == 1
    assert rb.fadeout_tail[0] == 0  # 1-sample edge case sets zero

    # 3. 0ms fadeout drain
    rb.write(np.full(100, 5000, dtype=np.int16))
    rb.drain(fade_ms=0.0)
    assert rb.fadeout_tail is None
    assert rb.size == 0


# =============================================================================
# 2. Extreme Signals & DC Bias Stress Tests
# =============================================================================

def test_extreme_dc_offset_rejection() -> None:
    """Test extreme DC offset (+10,000 counts) rejection and speech detection under bias."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)

    # 20 frames of pure +10,000 DC offset (digital silence + DC bias)
    dc_frame = np.full(480, 10000, dtype=np.int16)
    decisions = [vad.process_frame(dc_frame) for _ in range(20)]

    assert not any(decisions), "Pure DC offset must never trigger speech activity"
    assert vad.noise_floor_dbfs <= -80.0, f"Noise floor adapted improperly: {vad.noise_floor_dbfs}"

    # Speech signal (+/- 8,000 amplitude sinusoid) biased with +10,000 DC offset
    t = np.linspace(0, 0.03, 480, endpoint=False)
    speech_frame = (8000.0 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.int16) + 10000

    speech_decisions = [vad.process_frame(speech_frame) for _ in range(5)]
    assert speech_decisions[-1] is True, "AC speech with DC offset must be reliably detected"


def test_transient_click_and_step_rejection() -> None:
    """Verify that isolated mid-frame DC steps and 1-sample impulse clicks do not trigger speech onset."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, onset_frames=2)

    # 1. Single sample full-scale impulse click (+32767) in pure silence
    click_frame = np.zeros(480, dtype=np.int16)
    click_frame[240] = 32767

    vad.process_frame(click_frame)
    assert vad.is_speech is False, "Single-sample impulse click must not trigger speech onset"

    # Interleaved silence frame to ensure onset counter resets
    silence = np.zeros(480, dtype=np.int16)
    vad.process_frame(silence)
    assert vad.is_speech is False

    # 2. Mid-frame DC step (0 to +10,000)
    step_frame = np.zeros(480, dtype=np.int16)
    step_frame[240:] = 10000
    vad.process_frame(step_frame)
    assert vad.is_speech is False, "Mid-frame step transient must not trigger speech onset"

    # Subsequent steady DC frame
    vad.process_frame(np.full(480, 10000, dtype=np.int16))
    assert vad.is_speech is False, "Settled DC must remain silence"


def test_extreme_floating_point_nans_infs_and_recovery() -> None:
    """Feed floating-point NaNs and Infs into HybridMicrophoneBridge and verify VAD state recovery."""
    bridge = HybridMicrophoneBridge(source="browser", input_sample_rate=24000)

    # 1. Ingest frame of all NaNs
    nan_chunk = np.full(720, np.nan, dtype=np.float32)
    frames_processed_nan = bridge.feed_browser_pcm(nan_chunk)
    assert frames_processed_nan == 1
    assert bridge.vad.is_speech is False
    assert not np.isnan(bridge.vad.noise_floor_dbfs)
    assert not np.isnan(bridge.vad.threshold_dbfs)

    # 2. Ingest frame of all +Inf and -Inf
    inf_chunk = np.array([np.inf if i % 2 == 0 else -np.inf for i in range(720)], dtype=np.float32)
    frames_processed_inf = bridge.feed_browser_pcm(inf_chunk)
    assert frames_processed_inf == 1
    assert not np.isinf(bridge.vad.noise_floor_dbfs)

    # 3. Verify clean recovery: feed genuine speech and assert reliable detection
    t = np.linspace(0, 0.03, 720, endpoint=False)
    speech_chunk = (0.7 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.float32)

    for _ in range(5):
        bridge.feed_browser_pcm(speech_chunk)

    assert bridge.vad.is_speech is True, "VAD must recover cleanly and detect speech following NaN/Inf feeds"


def test_full_scale_square_wave_transitions_and_hangover() -> None:
    """Test full-scale square wave transitions, 6-frame pause hangover, and 16-frame turn completion."""
    bridge = HybridMicrophoneBridge(source="browser", input_sample_rate=24000)

    # 400Hz full-scale square wave
    period = 60
    pattern = np.array([32767] * (period // 2) + [-32768] * (period // 2), dtype=np.int16)
    square_frame = np.tile(pattern, 720 // period)
    silence_frame = np.zeros(720, dtype=np.int16)

    # Feed 3 speech frames
    for _ in range(3):
        bridge.feed_browser_pcm(square_frame)

    assert bridge.vad.is_speech is True

    # Feed silence and count exact hangover frames
    hangover_trues = 0
    turn_completion_frame = -1

    for frame_idx in range(25):
        bridge.feed_browser_pcm(silence_frame)
        if bridge.vad.is_speech:
            hangover_trues += 1
        if bridge.vad.is_turn_complete and turn_completion_frame == -1:
            turn_completion_frame = frame_idx

    assert hangover_trues == 6, f"Expected exactly 6 hangover frames, got {hangover_trues}"
    assert turn_completion_frame == 15, f"Expected turn completion at frame 15 (16th frame), got {turn_completion_frame}"


def test_ultra_low_noise_floor_and_clamping() -> None:
    """Verify that ultra-low noise floors (-90 dBFS) adapt cleanly and threshold clamps at -60 dBFS."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        energy_threshold=None,  # Purely adaptive threshold mode
        min_noise_floor_dbfs=-90.0,
        snr_margin_db=12.0,
    )

    silence = np.zeros(480, dtype=np.int16)
    for _ in range(50):
        vad.process_frame(silence)

    assert vad.noise_floor_dbfs == pytest.approx(-90.0, abs=1.0)
    assert vad.threshold_dbfs == pytest.approx(-60.0, abs=0.1), "Threshold must clamp to lower limit of -60 dBFS"

    # Ambient noise at -75 dBFS (RMS ~ 5.8 counts in int16)
    # RMS 6 / 32768 = -74.7 dBFS
    low_noise = np.random.randint(-6, 7, size=480).astype(np.int16)
    is_speech = vad.process_frame(low_noise)
    assert is_speech is False, "Low-level noise at -75 dBFS must not exceed -60 dBFS threshold"


# =============================================================================
# 3. Resampling Stress & Rate Conversion Tests
# =============================================================================

@pytest.mark.parametrize("sr", [8000, 11025, 16000, 22050, 24000, 44100, 48000, 96000, 192000])
def test_resampling_stress_and_bounded_fifo(sr: int) -> None:
    """Ingest varied sampling rates into HybridMicrophoneBridge and verify valid frames, bounded FIFO, and throughput."""
    bridge = HybridMicrophoneBridge(source="browser", input_sample_rate=sr)
    chunk_len = int(round(sr * 0.03))  # 30ms frame
    t = np.linspace(0, 0.03, chunk_len, endpoint=False)
    sine_chunk = (0.6 * 32767 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.int16)

    # Warmup
    bridge.feed_browser_pcm(sine_chunk, sample_rate=sr)

    # Benchmark 100 consecutive 30ms frames (3 seconds of audio)
    t0 = time.perf_counter()
    total_frames = 0
    for _ in range(100):
        total_frames += bridge.feed_browser_pcm(sine_chunk, sample_rate=sr)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) * 1000.0) / 100.0

    assert 99 <= total_frames <= 101, f"Expected ~100 frames, got {total_frames} for {sr}Hz"
    assert len(bridge._fifo) < 480, f"FIFO accumulated unboundedly: {len(bridge._fifo)}"
    assert avg_ms < 2.0, f"Resampling throughput exceeded 2ms budget: {avg_ms:.3f}ms for {sr}Hz"
    assert bridge.vad.is_speech is True, f"Speech was not detected for {sr}Hz"


def test_resampling_arbitrary_unaligned_chunk_sizes() -> None:
    """Verify that arbitrary non-multiple chunk sizes stream through FIFO without drift or data loss."""
    bridge = HybridMicrophoneBridge(source="browser", input_sample_rate=44100)
    unaligned_sizes = [17, 100, 313, 777, 1200]

    total_samples = 0
    total_frames = 0

    for i in range(200):
        sz = unaligned_sizes[i % len(unaligned_sizes)]
        chunk = np.zeros(sz, dtype=np.int16)
        total_samples += sz
        total_frames += bridge.feed_browser_pcm(chunk, sample_rate=44100)

    # Total expected 16kHz frames: (total_samples / 44100) / 0.03
    expected_frames = int((total_samples * (16000.0 / 44100.0)) // 480)
    assert abs(total_frames - expected_frames) <= 1
    assert len(bridge._fifo) < 480


# =============================================================================
# 4. Concurrency & Latency Verification Tests
# =============================================================================

def test_concurrent_source_switching_and_barge_in_stress() -> None:
    """Stress multi-threaded race conditions during rapid source switching, feeding, and barge-in."""
    rb = AudioRingBuffer(capacity_samples=240000, sample_rate=24000)
    bridge = HybridMicrophoneBridge(source="browser", ring_buffer=rb)

    stop_event = threading.Event()
    errors: List[Tuple[str, Exception]] = []

    def source_switcher():
        sources = ["browser", "virtual", "host"]
        i = 0
        while not stop_event.is_set():
            try:
                bridge.set_source(sources[i % len(sources)])
                i += 1
                time.sleep(0.001)
            except Exception as e:
                errors.append(("source_switcher", e))

    def audio_feeder():
        chunk = np.zeros(720, dtype=np.int16)
        while not stop_event.is_set():
            try:
                bridge.feed_browser_pcm(chunk)
                bridge.step_virtual(720)
                time.sleep(0.001)
            except Exception as e:
                errors.append(("audio_feeder", e))

    def barge_in_spammer():
        while not stop_event.is_set():
            try:
                rb.write(np.ones(1000, dtype=np.int16) * 500)
                bridge.trigger_manual_barge_in()
                time.sleep(0.002)
            except Exception as e:
                errors.append(("barge_in_spammer", e))

    def ring_buffer_reader():
        out = np.zeros(256, dtype=np.int16)
        while not stop_event.is_set():
            try:
                rb.read_into(out)
                time.sleep(0.001)
            except Exception as e:
                errors.append(("ring_buffer_reader", e))

    threads = [
        threading.Thread(target=source_switcher),
        threading.Thread(target=audio_feeder),
        threading.Thread(target=barge_in_spammer),
        threading.Thread(target=ring_buffer_reader),
    ]

    for t in threads:
        t.start()

    time.sleep(2.0)
    stop_event.set()

    for t in threads:
        t.join(timeout=2.0)

    bridge.close()
    assert len(errors) == 0, f"Encountered thread safety errors: {errors}"


def test_atomic_buffer_drain_latency_stress_1000_iterations() -> None:
    """Stress test atomic ring buffer drain latency across 1,000 iterations (< 0.05ms budget)."""
    rb = AudioRingBuffer(capacity_samples=240000, sample_rate=24000)
    test_audio = np.ones(48000, dtype=np.int16) * 1000  # 2 seconds
    fade_ms = 2.0
    iterations = 1000

    # Warmup
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

    print(f"\n[STRESS VERIFICATION] 1,000 Atomic Buffer Drains:")
    print(f"  Min:    {min_lat:.5f} ms")
    print(f"  Max:    {max_lat:.5f} ms")
    print(f"  Mean:   {mean_lat:.5f} ms (Target: < 0.05000 ms)")
    print(f"  Median: {median_lat:.5f} ms")
    print(f"  P95:    {p95_lat:.5f} ms")
    print(f"  P99:    {p99_lat:.5f} ms")

    assert mean_lat < 0.05, f"Mean drain latency {mean_lat:.5f}ms exceeded sub-0.05ms requirement!"


def test_end_to_end_barge_in_latency_stress_100_iterations() -> None:
    """Stress test end-to-end speech onset barge-in latency across 100 iterations (< 50ms budget)."""
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

        rb.write(np.ones(48000, dtype=np.int16) * 2000)

        # Prime with 1 speech frame
        t = np.linspace(0, 0.03, 720, endpoint=False)
        speech_frame = (0.8 * 32767 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.int16).tobytes()
        bridge.feed_browser_pcm(speech_frame)

        # Measure end-to-end latency to halt
        t0 = time.perf_counter()
        bridge.feed_browser_pcm(speech_frame)
        t1 = time.perf_counter()

        e2e_ms = (t1 - t0) * 1000.0
        e2e_latencies_ms.append(e2e_ms)

        assert rb.size == 0
        bridge.close()

    arr = np.array(e2e_latencies_ms)
    mean_lat = float(arr.mean())
    max_lat = float(arr.max())
    p95_lat = float(np.percentile(arr, 95))
    p99_lat = float(np.percentile(arr, 99))

    print(f"\n[STRESS VERIFICATION] 100 End-to-End Barge-In Runs:")
    print(f"  Min:  {arr.min():.4f} ms")
    print(f"  Max:  {max_lat:.4f} ms (Target: < 50.0 ms)")
    print(f"  Mean: {mean_lat:.4f} ms")
    print(f"  P95:  {p95_lat:.4f} ms")
    print(f"  P99:  {p99_lat:.4f} ms")

    assert mean_lat < 5.0, f"Mean barge-in latency {mean_lat:.4f}ms exceeded 5ms expected"
    assert max_lat < 50.0, f"Max barge-in latency {max_lat:.4f}ms exceeded 50ms hard limit!"
