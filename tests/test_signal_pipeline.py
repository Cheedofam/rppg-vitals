"""Unit tests for the signal-processing pipeline (``src.signal_pipeline``).

All tests use synthetic signals with known frequencies so correctness can be asserted
without a webcam or the dataset.
"""

import numpy as np

from src.signal_pipeline import SignalBuffer


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
