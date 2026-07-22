"""Unit tests for the signal-quality metrics (``src.quality``)."""

import numpy as np

from src.quality import assess_quality, motion_score, spectral_snr


def test_spectral_snr_high_for_clean_sine():
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    x = np.sin(2 * np.pi * 1.2 * t)

    assert spectral_snr(x, fs, band=(0.75, 4.0)) > 3.0


def test_spectral_snr_low_for_white_noise():
    rng = np.random.default_rng(1)
    fs = 30.0
    x = rng.standard_normal(int(20 * fs))

    assert spectral_snr(x, fs, band=(0.75, 4.0)) < 1.0


def test_motion_score_zero_for_static_centroids():
    centroids = np.tile([320.0, 240.0], (100, 1))

    assert motion_score(centroids) < 1e-6


def test_motion_score_detects_jitter():
    rng = np.random.default_rng(2)
    centroids = np.array([320.0, 240.0]) + 10.0 * rng.standard_normal((100, 2))

    assert motion_score(centroids) > 5.0


def test_assess_quality_good_for_clean_static():
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    pulse = np.sin(2 * np.pi * 1.2 * t)
    centroids = np.tile([320.0, 240.0], (len(t), 1))

    result = assess_quality(pulse, fs, centroids, band=(0.75, 4.0))

    assert result.level == "good"
    assert result.snr > 3.0


def test_assess_quality_poor_for_noise():
    rng = np.random.default_rng(3)
    fs = 30.0
    pulse = rng.standard_normal(int(20 * fs))
    centroids = np.tile([320.0, 240.0], (len(pulse), 1))

    result = assess_quality(pulse, fs, centroids, band=(0.75, 4.0))

    assert result.level == "poor"


def test_assess_quality_poor_for_motion():
    rng = np.random.default_rng(4)
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    pulse = np.sin(2 * np.pi * 1.2 * t)  # clean signal ...
    centroids = np.array([320.0, 240.0]) + 15.0 * rng.standard_normal((len(t), 2))  # ... but big motion

    result = assess_quality(pulse, fs, centroids, band=(0.75, 4.0))

    assert result.level == "poor"
    assert any("motion" in r for r in result.reasons)
