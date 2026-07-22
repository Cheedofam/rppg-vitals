"""Signal-processing pipeline: rolling RGB buffer, filtering, CHROM, and FFT extraction.

This module is split into two parts:
  * ``SignalBuffer`` -- a time-stamped rolling buffer of per-frame ROI mean RGB values
    (Phase 2). It reports the *measured* sampling rate from the timestamps, because a
    webcam's real frame rate drifts around its nominal value.
  * The pure DSP functions (Phase 3) -- detrending, Butterworth band-pass filtering, the
    CHROM combination, and FFT-based rate estimation.

All DSP functions are pure (NumPy/SciPy only) and unit tested with synthetic signals.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SignalBuffer:
    """Rolling, time-stamped buffer of per-frame ROI mean-RGB samples.

    Parameters
    ----------
    maxlen_seconds : keep only the most recent ``maxlen_seconds`` of samples.
    fs_nominal : nominal frame rate, used only as a fallback before enough samples
        exist to measure the true rate.
    """

    def __init__(self, maxlen_seconds: float = 30.0, fs_nominal: float = 30.0):
        self.maxlen_seconds = float(maxlen_seconds)
        self.fs_nominal = float(fs_nominal)
        self._rgb: deque[np.ndarray] = deque()
        self._ts: deque[float] = deque()

    def append(self, rgb, ts: float) -> None:
        """Append one ``[R, G, B]`` sample at timestamp ``ts`` (seconds); evict stale ones."""
        self._rgb.append(np.asarray(rgb, dtype=np.float64))
        self._ts.append(float(ts))
        while self._ts and (self._ts[-1] - self._ts[0]) > self.maxlen_seconds:
            self._ts.popleft()
            self._rgb.popleft()

    def __len__(self) -> int:
        return len(self._ts)

    def duration(self) -> float:
        """Seconds spanned by the buffered samples."""
        if len(self._ts) < 2:
            return 0.0
        return self._ts[-1] - self._ts[0]

    def get_rgb_window(self, seconds: float | None = None) -> tuple[np.ndarray, float]:
        """Return ``(rgb[N, 3], fs_measured)`` for the most recent ``seconds`` of data.

        If ``seconds`` is ``None`` the whole buffer is returned. ``fs_measured`` is the
        mean sampling rate estimated from the timestamps (``(N - 1) / (t[-1] - t[0])``).
        """
        if len(self._ts) < 2:
            return np.empty((0, 3), dtype=np.float64), self.fs_nominal
        ts = np.asarray(self._ts, dtype=np.float64)
        rgb = np.vstack(self._rgb)
        if seconds is not None:
            keep = ts >= (ts[-1] - seconds)
            ts, rgb = ts[keep], rgb[keep]
        span = ts[-1] - ts[0]
        fs = (len(ts) - 1) / span if span > 0 else self.fs_nominal
        return rgb, float(fs)
