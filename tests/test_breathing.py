"""Unit tests for low-frequency breathing-rate extraction (``src.breathing``)."""

import numpy as np

from src.breathing import estimate_breathing_rate


def test_recovers_known_breathing_rate():
    fs = 30.0
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz = 15 breaths/min

    brpm, snr = estimate_breathing_rate(x, fs)

    assert abs(brpm - 15.0) < 1.0
    assert snr > 3.0


def test_rejects_heart_rate_band_component():
    fs = 30.0
    t = np.arange(0, 60, 1 / fs)
    # Breathing at 15 br/min plus a much stronger heart-rate tone at 72 bpm.
    x = np.sin(2 * np.pi * 0.25 * t) + 3.0 * np.sin(2 * np.pi * 1.2 * t)

    brpm, _ = estimate_breathing_rate(x, fs)

    assert abs(brpm - 15.0) < 1.5  # HR tone is filtered out; RR peak survives


def test_recovers_slow_breathing():
    fs = 30.0
    t = np.arange(0, 90, 1 / fs)
    x = np.sin(2 * np.pi * (9 / 60) * t)  # 9 breaths/min (bradypnea-ish)

    brpm, _ = estimate_breathing_rate(x, fs)

    assert abs(brpm - 9.0) < 1.0
