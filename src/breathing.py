"""Breathing-rate estimation from the low-frequency component of the ROI signal.

Breathing modulates skin colour and ROI position slowly (0.1-0.5 Hz, i.e. 6-30
breaths/min) -- well below the heart-rate band. The same FFT estimator used for heart
rate is reused here on a low-frequency band-passed signal, keeping the two paths
consistent and DRY. The input is typically the ROI mean green channel (or mean
luminance) over a rolling window; any 1-D signal carrying the respiratory modulation
works.
"""

from __future__ import annotations

import numpy as np

from src.signal_pipeline import RR_BAND, bandpass_filter, estimate_rate_fft


def estimate_breathing_rate(signal, fs: float,
                            low: float = RR_BAND[0],
                            high: float = RR_BAND[1]) -> tuple[float, float]:
    """Estimate breathing rate (breaths/min) and signal quality from a 1-D signal.

    A low-order Butterworth band-pass isolates the respiratory band before the FFT peak
    search, so heart-rate and DC/drift components do not contaminate the estimate.

    Returns ``(breaths_per_minute, snr)``.
    """
    x = np.asarray(signal, dtype=np.float64)
    # Order 2: the respiratory band sits at very low normalised frequency, where a
    # lower filter order stays numerically well-conditioned with sosfiltfilt.
    xf = bandpass_filter(x, fs, low, high, order=2)
    freq, snr = estimate_rate_fft(xf, fs, (low, high))
    return freq * 60.0, snr
