"""Unit tests for the signal-processing pipeline (``src.signal_pipeline``).

All tests use synthetic signals with known frequencies so correctness can be asserted
without a webcam or the dataset.
"""

import numpy as np

from src.signal_pipeline import (
    SignalBuffer,
    bandpass_filter,
    chrom_pulse,
    detrend_signal,
    estimate_heart_rate,
    estimate_rate_fft,
)


def _rms(x):
    return np.sqrt(np.mean(np.square(x)))


# --------------------------------------------------------------------------- Phase 2
def test_signal_buffer_shape_and_measured_fs():
    buf = SignalBuffer(maxlen_seconds=30.0)
    for i in range(300):
        buf.append([i, i + 1, i + 2], ts=i / 30.0)

    rgb, fs = buf.get_rgb_window()

    assert rgb.shape == (300, 3)
    assert abs(fs - 30.0) < 0.5


def test_signal_buffer_windowing_by_seconds():
    buf = SignalBuffer(maxlen_seconds=30.0)
    for i in range(900):  # 30 s at 30 fps
        buf.append([1.0, 2.0, 3.0], ts=i / 30.0)

    rgb, fs = buf.get_rgb_window(seconds=10.0)

    # last 10 s at 30 fps -> ~300 samples
    assert 295 <= rgb.shape[0] <= 305
    assert abs(fs - 30.0) < 0.5


def test_signal_buffer_evicts_old_samples():
    buf = SignalBuffer(maxlen_seconds=10.0)
    for i in range(600):  # 20 s at 30 fps
        buf.append([0.0, 0.0, 0.0], ts=i / 30.0)

    # Only the most recent ~10 s should remain.
    assert buf.duration() <= 10.0 + 1e-6
    assert 290 <= len(buf) <= 310


# --------------------------------------------------------------------------- Phase 3
def test_detrend_removes_linear_trend():
    t = np.arange(300)
    x = 5.0 + 0.1 * t + np.sin(2 * np.pi * 0.05 * t)

    d = detrend_signal(x)

    assert abs(d.mean()) < 1e-6
    assert abs(np.polyfit(t, d, 1)[0]) < 1e-3  # residual slope ~ 0


def test_bandpass_passes_hr_band_rejects_out_of_band():
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    in_band = np.sin(2 * np.pi * 1.2 * t)   # 72 bpm, inside 0.75-4 Hz
    low_oob = np.sin(2 * np.pi * 0.2 * t)   # 12 bpm, below band
    high_oob = np.sin(2 * np.pi * 6.0 * t)  # 360 bpm, above band

    assert _rms(bandpass_filter(in_band, fs)) > 0.7 * _rms(in_band)
    assert _rms(bandpass_filter(low_oob, fs)) < 0.2 * _rms(low_oob)
    assert _rms(bandpass_filter(high_oob, fs)) < 0.2 * _rms(high_oob)


def test_estimate_rate_fft_recovers_known_frequency():
    fs = 30.0
    t = np.arange(0, 30, 1 / fs)
    x = np.sin(2 * np.pi * 1.2 * t)  # exactly 72 bpm

    freq, snr = estimate_rate_fft(x, fs, band=(0.75, 4.0))

    assert abs(freq * 60 - 72) < 1.0
    assert snr > 3.0  # clean sine -> sharp, high-SNR peak


def test_estimate_rate_fft_subbin_via_parabolic_interpolation():
    fs = 30.0
    t = np.arange(0, 30, 1 / fs)
    x = np.sin(2 * np.pi * (73 / 60) * t)  # 73 bpm: not aligned to an FFT bin

    freq, _ = estimate_rate_fft(x, fs, band=(0.75, 4.0))

    assert abs(freq * 60 - 73) < 2.0


def test_chrom_pipeline_recovers_heart_rate_from_synthetic_rgb():
    rng = np.random.default_rng(0)
    fs = 30.0
    n = int(30 * fs)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
    # Green carries the strongest pulse (as in real rPPG); modest sensor noise.
    g = 180.0 + 3.6 * pulse + 0.5 * rng.standard_normal(n)
    r = 200.0 + 1.6 * pulse + 0.5 * rng.standard_normal(n)
    b = 160.0 + 0.4 * pulse + 0.5 * rng.standard_normal(n)
    rgb = np.stack([r, g, b], axis=1)

    bpm, snr = estimate_heart_rate(rgb, fs)

    assert abs(bpm - 72) < 2.0
    assert snr > 3.0


def test_chrom_pulse_output_shape_and_bandlimited():
    fs = 30.0
    n = int(20 * fs)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack([200 + 2 * pulse, 180 + 4 * pulse, 160 + pulse], axis=1)

    s = chrom_pulse(rgb, fs)

    assert s.shape == (n,)
    # dominant frequency of the CHROM pulse is the heart-rate tone
    freq, _ = estimate_rate_fft(s, fs, band=(0.75, 4.0))
    assert abs(freq * 60 - 72) < 2.0
