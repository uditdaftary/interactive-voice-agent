import numpy as np
import pytest
from voice_agent.vad.energy_vad import VoiceActivityDetector

def test_silence_vs_speech():
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, energy_threshold=300.0)
    
    # 30 ms of pure silence
    silence = np.zeros(480, dtype=np.int16)
    assert vad.process_frame(silence) == False
    
    # 30 ms of 200 Hz tone (voiced speech range)
    t = np.linspace(0, 0.030, 480, endpoint=False)
    tone = (np.sin(2 * np.pi * 200 * t) * 15000).astype(np.int16)
    
    # Send multiple frames to overcome hangover/adaptation
    detected = any(vad.process_frame(tone) for _ in range(5))
    assert detected == True

def test_dc_offset_tolerance():
    """Verify constant hardware DC offset does NOT cause false speech triggers."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    
    # Constant DC bias of +500 in int16 scale (without speech)
    dc_bias_int16 = np.full(480, 500, dtype=np.int16)
    for _ in range(10):
        assert vad.process_frame(dc_bias_int16) == False
    
    # Float32 constant DC bias of +0.3
    dc_bias_float32 = np.full(480, 0.3, dtype=np.float32)
    for _ in range(10):
        assert vad.process_frame(dc_bias_float32) == False
    assert vad.is_speech == False

def test_float32_int16_scaling_parity():
    """Verify float32 [-1.0, 1.0] and int16 [-32768, 32767] produce identical dBFS and detection."""
    vad_int = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    vad_flt = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    
    t = np.linspace(0, 0.030, 480, endpoint=False)
    # Sine wave with peak 16384 (amplitude 0.5)
    tone_int = (np.sin(2 * np.pi * 300 * t) * 16384).astype(np.int16)
    tone_flt = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    
    dbfs_int = vad_int.compute_dbfs(tone_int)
    dbfs_flt = vad_flt.compute_dbfs(tone_flt)
    
    # Parity check: dBFS values should match within 0.1 dB
    assert abs(dbfs_int - dbfs_flt) < 0.1
    
    # Both detectors should transition identically over 3 frames
    res_int = [vad_int.process_frame(tone_int) for _ in range(3)]
    res_flt = [vad_flt.process_frame(tone_flt) for _ in range(3)]
    assert res_int == res_flt
    assert res_int[-1] == True

def test_transient_click_rejection():
    """Single-frame impulse spike (typing/click) must NOT trigger speech onset."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, onset_frames=2)
    
    silence = np.zeros(480, dtype=np.float32)
    # Priming noise floor with silence
    for _ in range(5):
        vad.process_frame(silence)
    assert vad.is_speech == False
    
    # 1 single frame impulse click (full scale)
    click = np.zeros(480, dtype=np.float32)
    click[10:30] = 0.9  # 20-sample high energy transient
    assert vad.process_frame(click) == False
    assert vad.is_speech == False
    
    # Next frame is silence again
    assert vad.process_frame(silence) == False
    assert vad.is_speech == False
    
    # Now feed 2 consecutive active frames -> should trigger on 2nd frame
    t = np.linspace(0, 0.030, 480, endpoint=False)
    tone = (np.sin(2 * np.pi * 250 * t) * 0.6).astype(np.float32)
    
    frame1 = vad.process_frame(tone)
    assert frame1 == False  # 1 frame: onset persistence not reached
    assert vad.is_speech == False
    
    frame2 = vad.process_frame(tone)
    assert frame2 == True   # 2 frames: onset persistence satisfied!
    assert vad.is_speech == True

def test_continuous_speech_and_hangover_bridging():
    """Continuous speech, pause bridging (6 frames = 180ms), and turn completion (16 frames = 480ms)."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        hangover_frames=6,
        turn_completion_frames=16,
        onset_frames=2
    )
    
    t = np.linspace(0, 0.030, 480, endpoint=False)
    tone = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    silence = np.zeros(480, dtype=np.float32)
    
    # Trigger active speech (2 frames)
    vad.process_frame(tone)
    assert vad.process_frame(tone) == True
    assert vad.is_speech == True
    assert vad.is_turn_complete == False
    
    # 4 frames of silence (120ms pause between words) -> must be bridged!
    for _ in range(4):
        assert vad.process_frame(silence) == True
        assert vad.is_speech == True
        assert vad.is_turn_complete == False
        
    # Speech resumes
    assert vad.process_frame(tone) == True
    
    # Now user stops speaking: 6 hangover frames keep is_speech True
    for _ in range(6):
        assert vad.process_frame(silence) == True
        assert vad.is_turn_complete == False
        
    # Frame 7 of silence: pause bridging expires
    assert vad.process_frame(silence) == False
    assert vad.is_speech == False
    
    # Feed remaining silence frames until turn completion (16 frames total silence)
    # We already fed 7 silence frames. Need 9 more to reach 16 silence frames.
    for _ in range(8):
        vad.process_frame(silence)
        assert vad.is_turn_complete == False
        
    vad.process_frame(silence)  # 16th silence frame
    assert vad.is_turn_complete == True

def test_variable_chunk_streaming():
    """Streaming variable chunk sizes (10-sample and 1024-sample slices) through 30ms FIFO."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, onset_frames=2)
    t = np.linspace(0, 0.120, 1920, endpoint=False)  # 4 frames (120ms) of voiced tone
    long_tone = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    
    # Feed in tiny 10-sample slices
    chunk_size = 10
    detected_history = []
    for i in range(0, len(long_tone), chunk_size):
        slice_10 = long_tone[i : i + chunk_size]
        detected_history.append(vad.process_frame(slice_10))
    
    # Prior to 48 chunks (480 samples = 1st frame), is_speech is False
    assert detected_history[47] == False
    # At chunk 96 (960 samples = 2nd frame), onset persistence of 2 frames is reached
    assert detected_history[95] == True
    assert vad.is_speech == True
    
    # Now test with 1024-sample chunks
    vad_large = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, onset_frames=2)
    chunk_1024 = (np.sin(2 * np.pi * 300 * np.linspace(0, 1024/16000, 1024, endpoint=False)) * 0.5).astype(np.float32)
    
    # 1024 samples has 2 full 480-sample frames (960 samples). Onset persistence (2 frames) should be satisfied!
    res = vad_large.process_frame(chunk_1024)
    assert res == True
    assert vad_large.is_speech == True
    # FIFO should retain remaining 64 samples
    assert len(vad_large._fifo) == 64

def test_input_validation():
    """Detector gracefully handles None, empty arrays, NaN, Inf, and multi-channel audio."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30)
    
    # None input
    assert vad.process_frame(None) == False
    
    # Empty array
    assert vad.process_frame(np.array([], dtype=np.float32)) == False
    
    # NaN and Inf inputs
    dirty = np.full(480, np.nan, dtype=np.float32)
    dirty[0] = np.inf
    dirty[1] = -np.inf
    assert vad.process_frame(dirty) == False
    
    # Stereo multi-channel downmix (480, 2)
    stereo = np.zeros((480, 2), dtype=np.float32)
    assert vad.process_frame(stereo) == False
    
    # Active tone stereo
    t = np.linspace(0, 0.030, 480, endpoint=False)
    mono_tone = (np.sin(2 * np.pi * 300 * t) * 0.6).astype(np.float32)
    stereo_tone = np.column_stack([mono_tone, mono_tone])
    res = any(vad.process_frame(stereo_tone) for _ in range(3))
    assert res == True

def test_acoustic_echo_suppression():
    """Verify ducking/masking threshold elevation during playback prevents self-interruption."""
    vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=30, ducking_attenuation_db=15.0)
    
    # Prime noise floor
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(5):
        vad.process_frame(silence)
        
    t = np.linspace(0, 0.030, 480, endpoint=False)
    # Moderate amplitude speaker leakage audio (-35 dBFS approx)
    leakage = (np.sin(2 * np.pi * 400 * t) * 0.02).astype(np.float32)
    
    # During active playback, ducking is applied
    vad.set_playback_active(True)
    assert vad.is_playback_active == True
    
    # Leakage should be suppressed by ducked threshold
    for _ in range(5):
        assert vad.process_frame(leakage) == False
    assert vad.is_speech == False
    
    # Loud human voice (-10 dBFS, amplitude 0.4) barge-in over speaker playback
    loud_human = (np.sin(2 * np.pi * 200 * t) * 0.4).astype(np.float32)
    # 2 frames of loud human voice should overcome ducked threshold and trigger speech
    vad.process_frame(loud_human)
    assert vad.process_frame(loud_human) == True
    assert vad.is_speech == True
    
    # Deactivate playback
    vad.set_playback_active(False)
    assert vad.is_playback_active == False

def test_adaptive_noise_floor():
    """Verify dual-rate EMA adapts ambient noise floor between -90 and -20 dBFS."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        frame_duration_ms=30,
        min_noise_floor_dbfs=-90.0,
        max_noise_floor_dbfs=-20.0,
        initial_noise_floor_dbfs=-60.0
    )
    assert vad.noise_floor_dbfs == -60.0
    
    # Feed quiet room noise (~ -75 dBFS)
    # RMS = 10^(-75/20) ~ 0.000177
    quiet_noise = np.random.normal(0, 0.00018, 480).astype(np.float32)
    for _ in range(20):
        vad.process_frame(quiet_noise)
        
    # Noise floor should have tracked downwards
    assert vad.noise_floor_dbfs < -62.0
    assert vad.noise_floor_dbfs >= -90.0
