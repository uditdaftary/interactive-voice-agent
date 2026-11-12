import time
import threading
import numpy as np
import pytest
from voice_agent.audio.ring_buffer import AudioRingBuffer

def test_ring_buffer_write_read():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    data = np.ones(1600, dtype=np.int16) * 100
    written = buf.write(data)
    assert written == 1600
    assert buf.available_read() == 1600
    
    out = buf.read(800)
    assert len(out) == 800
    assert np.all(out == 100)
    assert buf.available_read() == 800

def test_instant_barge_in_drain():
    buf = AudioRingBuffer(capacity_samples=48000, sample_rate=24000)
    data = np.ones(24000, dtype=np.int16)
    elapsed_ms = []
    for _ in range(200):
        buf.write(data)
        assert buf.available_read() == 24000

        t0 = time.perf_counter()
        buf.drain()
        elapsed_ms.append((time.perf_counter() - t0) * 1000)

        assert buf.available_read() == 0

    # Median drain must stay < 5ms (well within the 50ms budget); one slow outlier must not fail the suite.
    assert float(np.median(elapsed_ms)) < 5.0

def test_circular_buffer_wrap_around():
    capacity = 1000
    buf = AudioRingBuffer(capacity_samples=capacity, sample_rate=16000)
    
    # Fill 700 samples
    chunk1 = np.arange(700, dtype=np.int16)
    assert buf.write(chunk1) == 700
    
    # Read 500 samples
    read1 = buf.read(500)
    assert np.array_equal(read1, chunk1[:500])
    assert buf.available_read() == 200
    
    # Write 600 samples (causing wrap-around: 200 + 600 <= 1000)
    chunk2 = np.arange(1000, 1600, dtype=np.int16)
    assert buf.write(chunk2) == 600
    assert buf.available_read() == 800
    
    # Read remaining 200 from chunk1 and 600 from chunk2
    read2 = buf.read(800)
    assert np.array_equal(read2[:200], chunk1[500:])
    assert np.array_equal(read2[200:], chunk2)
    assert buf.available_read() == 0

def test_float_to_int16_scaling_and_clipping():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    
    # Floats in [-1.0, 1.0] scaled by 32767.0
    float_data = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    buf.write(float_data)
    out = buf.read(5)
    assert out[0] == 0
    assert out[1] == 32767
    assert out[2] == -32767
    assert out[3] == 16384
    assert out[4] == -16384
    
    # Floats outside [-1.0, 1.0] clipped to [-32768, 32767]
    overdrive = np.array([2.5, -3.0], dtype=np.float32)
    buf.write(overdrive)
    out = buf.read(2)
    assert out[0] == 32767
    assert out[1] == -32768
    
    # Integer clipping
    int_overflow = np.array([40000, -50000], dtype=np.int32)
    buf.write(int_overflow)
    out = buf.read(2)
    assert out[0] == 32767
    assert out[1] == -32768
    
    # NaN and Inf handling
    with_nans = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    buf.write(with_nans)
    out_nan = buf.read(4)
    assert out_nan[0] == 0
    assert out_nan[1] == 32767
    assert out_nan[2] == -32767
    assert abs(out_nan[3] - 8192) <= 1

def test_2d_array_downmixing():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    
    # (N, 1) mono 2D array
    mono_2d = np.array([[100], [200], [300]], dtype=np.int16)
    buf.write(mono_2d)
    out = buf.read(3)
    assert np.array_equal(out, np.array([100, 200, 300], dtype=np.int16))
    
    # (N, 2) stereo 2D array downmixed to mono
    stereo = np.array([[100, 300], [500, 700]], dtype=np.int16)
    buf.write(stereo)
    out = buf.read(2)
    assert np.array_equal(out, np.array([200, 600], dtype=np.int16))
    
    # (N, 2) float stereo array
    stereo_float = np.array([[0.2, 0.4], [-0.6, -0.2]], dtype=np.float32)
    buf.write(stereo_float)
    out = buf.read(2)
    assert abs(out[0] - 9830) <= 1
    assert abs(out[1] - (-13107)) <= 1
    
    # (1, N) row vector array
    row_vector = np.array([[150, 250, 350, 450]], dtype=np.int16)
    buf.write(row_vector)
    out_row = buf.read(4)
    assert np.array_equal(out_row, np.array([150, 250, 350, 450], dtype=np.int16))

def test_input_types_bytes_bytearray_memoryview():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    
    # bytes
    raw_samples = np.array([500, -500, 1000], dtype=np.int16)
    raw_bytes = raw_samples.tobytes()
    buf.write(raw_bytes)
    assert np.array_equal(buf.read(3), raw_samples)
    
    # bytearray
    raw_ba = bytearray(raw_bytes)
    buf.write(raw_ba)
    assert np.array_equal(buf.read(3), raw_samples)
    
    # memoryview of bytes
    raw_mv = memoryview(raw_bytes)
    buf.write(raw_mv)
    assert np.array_equal(buf.read(3), raw_samples)
    
    # memoryview of numpy int16 array
    arr_mv = memoryview(raw_samples)
    buf.write(arr_mv)
    assert np.array_equal(buf.read(3), raw_samples)
    
    # odd byte count in bytes
    odd_bytes = raw_bytes + b'\x42'
    buf.write(odd_bytes)
    assert np.array_equal(buf.read(3), raw_samples)

def test_negative_and_zero_samples_requested():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    buf.write(np.ones(100, dtype=np.int16))
    
    out_neg = buf.read(-5)
    assert isinstance(out_neg, np.ndarray)
    assert len(out_neg) == 0
    assert out_neg.dtype == np.int16
    
    out_zero = buf.read(0)
    assert isinstance(out_zero, np.ndarray)
    assert len(out_zero) == 0
    
    # Buffer unchanged
    assert buf.available_read() == 100
    
    # read_into with empty / negative
    empty_out = np.empty(0, dtype=np.int16)
    assert buf.read_into(empty_out) == 0

def test_read_into_zero_allocation_and_silence_padding():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    samples = np.array([10, 20, 30, 40, 50], dtype=np.int16)
    buf.write(samples)
    
    # Underrun: request 10 samples when only 5 exist
    outdata = np.full(10, 999, dtype=np.int16)
    read_count = buf.read_into(outdata)
    
    assert read_count == 5
    assert np.array_equal(outdata[:5], samples)
    assert np.all(outdata[5:] == 0)  # Silence padded
    assert buf.available_read() == 0
    
    # 2D (N, 1) PortAudio callback buffer
    buf.write(samples)
    outdata_2d = np.full((10, 1), 999, dtype=np.int16)
    read_count_2d = buf.read_into(outdata_2d)
    assert read_count_2d == 5
    assert np.array_equal(outdata_2d[:5, 0], samples)
    assert np.all(outdata_2d[5:, 0] == 0)
    
    # 2D (N, 2) stereo PortAudio callback buffer
    buf.write(samples)
    outdata_stereo = np.full((10, 2), 999, dtype=np.int16)
    read_count_stereo = buf.read_into(outdata_stereo)
    assert read_count_stereo == 5
    assert np.array_equal(outdata_stereo[:5, 0], samples)
    assert np.array_equal(outdata_stereo[:5, 1], samples)
    assert np.all(outdata_stereo[5:, :] == 0)

def test_generation_fencing():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    chunk = np.ones(100, dtype=np.int16)
    
    # Current generation write succeeds
    assert buf.write(chunk, generation=0) == 100
    assert buf.available_read() == 100
    
    # Drain increments generation
    buf.drain()
    assert buf.available_read() == 0
    assert buf.generation == 1
    
    # Stale generation chunk is immediately discarded
    assert buf.write(chunk, generation=0) == 0
    assert buf.available_read() == 0
    
    # Current generation write succeeds
    assert buf.write(chunk, generation=1) == 100
    assert buf.available_read() == 100
    
    # Unversioned write (generation=None) succeeds
    assert buf.write(chunk, generation=None) == 100
    assert buf.available_read() == 200

def test_micro_fadeout_on_drain():
    # 24kHz sample rate: 2ms = 48 samples
    buf = AudioRingBuffer(capacity_samples=24000, sample_rate=24000)
    constant_val = 10000
    drain_times_ms = []
    for _ in range(200):
        buf.write(np.full(1000, constant_val, dtype=np.int16))
        t0 = time.perf_counter()
        buf.drain()
        drain_times_ms.append((time.perf_counter() - t0) * 1000)
    # Median drain must stay < 5ms; the fadeout assertions below inspect the last drain.
    assert float(np.median(drain_times_ms)) < 5.0
    assert buf.available_read() == 0
    
    # Inspect fadeout_tail
    assert buf.fadeout_tail is not None
    fade_len = len(buf.fadeout_tail)
    assert fade_len == 48  # 24000 * 0.002 = 48 samples
    
    # Half-cosine properties:
    # First sample should be close to constant_val * 1.0
    assert abs(buf.fadeout_tail[0] - constant_val) <= 10
    # Last sample should be exactly or very close to 0
    assert abs(buf.fadeout_tail[-1]) <= 1
    # Tail is monotonically non-increasing
    assert np.all(np.diff(buf.fadeout_tail) <= 0)
    # Derivative bounded (smooth curve, max step < 400 << 10000)
    assert np.max(np.abs(np.diff(buf.fadeout_tail))) < 400
    
    # When read_into is called, it outputs the fadeout tail followed by silence
    outdata = np.full(100, -1, dtype=np.int16)
    n = buf.read_into(outdata)
    assert n == 48
    assert np.all(outdata[48:] == 0)

def test_micro_fadeout_wrap_around_boundary():
    buf = AudioRingBuffer(capacity_samples=1000, sample_rate=24000)
    # Position read pointer right near the wrap boundary
    buf.write(np.ones(980, dtype=np.int16) * 100)
    buf.read(980)
    assert buf.read_pos == 980
    
    # Write 100 samples wrapping around 980..1000 and 0..80
    buf.write(np.full(100, 10000, dtype=np.int16))
    # Drain spans wrap boundary: 980..1000 (20) and 0..28 (28)
    buf.drain()
    assert buf.fadeout_tail is not None
    assert len(buf.fadeout_tail) == 48
    assert abs(buf.fadeout_tail[0] - 10000) <= 10
    assert abs(buf.fadeout_tail[-1]) <= 1

def test_micro_fadeout_chunked_reads():
    buf = AudioRingBuffer(capacity_samples=1000, sample_rate=24000)
    buf.write(np.full(100, 10000, dtype=np.int16))
    buf.drain()
    
    out1 = np.empty(20, dtype=np.int16)
    n1 = buf.read_into(out1)
    assert n1 == 20
    
    out2 = np.empty(20, dtype=np.int16)
    n2 = buf.read_into(out2)
    assert n2 == 20
    
    out3 = np.empty(20, dtype=np.int16)
    n3 = buf.read_into(out3)
    assert n3 == 8  # 48 - 40 = 8 remaining
    assert np.all(out3[8:] == 0)  # Silence padded

def test_available_write_and_capacity_validation():
    # Validation
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity_samples=0)
    with pytest.raises(ValueError):
        AudioRingBuffer(capacity_samples=-100)
    with pytest.raises(ValueError):
        AudioRingBuffer(sample_rate=0)
        
    buf = AudioRingBuffer(capacity_samples=1000, sample_rate=16000)
    assert buf.available_write == 1000
    
    buf.write(np.ones(400, dtype=np.int16))
    assert buf.available_write == 600
    assert buf.available_read() == 400
    
    buf.read(200)
    assert buf.available_write == 800
    assert buf.available_read() == 200
    
    buf.drain()
    assert buf.available_write == 1000
    assert buf.available_read() == 0

def test_monotonic_playhead_tracking():
    buf = AudioRingBuffer(capacity_samples=16000, sample_rate=16000)
    assert buf.total_samples_played == 0
    assert buf.playhead_seconds == 0.0
    assert buf.get_playhead_seconds() == 0.0
    
    buf.write(np.ones(1600, dtype=np.int16))
    buf.read(800)
    assert buf.total_samples_played == 800
    assert buf.playhead_seconds == 800 / 16000
    
    outdata = np.zeros(400, dtype=np.int16)
    buf.read_into(outdata)
    assert buf.total_samples_played == 1200
    assert buf.playhead_seconds == 1200 / 16000
    
    # Drain does not decrease playhead
    buf.drain()
    assert buf.total_samples_played >= 1200

def test_concurrent_threads_safety():
    buf = AudioRingBuffer(capacity_samples=4000, sample_rate=16000)
    stop_event = threading.Event()
    errors = []
    
    def writer():
        while not stop_event.is_set():
            try:
                data = np.ones(64, dtype=np.int16) * 42
                buf.write(data, generation=buf.generation)
                time.sleep(0.0005)
            except Exception as e:
                errors.append(f"Writer error: {e}")
                
    def reader_into():
        out = np.empty(128, dtype=np.int16)
        while not stop_event.is_set():
            try:
                buf.read_into(out)
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader_into error: {e}")
                
    def reader_regular():
        while not stop_event.is_set():
            try:
                buf.read(64)
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader error: {e}")
                
    def drainer():
        while not stop_event.is_set():
            try:
                time.sleep(0.01)
                buf.drain()
            except Exception as e:
                errors.append(f"Drainer error: {e}")
                
    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=writer),
        threading.Thread(target=reader_into),
        threading.Thread(target=reader_regular),
        threading.Thread(target=drainer),
    ]
    
    for t in threads:
        t.start()
        
    time.sleep(0.25)
    stop_event.set()
    
    for t in threads:
        t.join(timeout=1.0)
        
    assert len(errors) == 0, f"Concurrent thread errors: {errors}"
