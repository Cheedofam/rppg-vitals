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
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, detrend, sosfiltfilt

# Physiological analysis bands.
HR_BAND = (0.75, 4.0)   # 45-240 bpm
RR_BAND = (0.1, 0.5)    # 6-30 breaths/min


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


# --------------------------------------------------------------------------- Phase 3
def detrend_signal(x) -> np.ndarray:
    """Remove the linear trend (and mean) from ``x`` -- cuts illumination drift."""
    return detrend(np.asarray(x, dtype=np.float64), type="linear")


def bandpass_filter(x, fs: float, low: float = HR_BAND[0], high: float = HR_BAND[1],
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter.

    Uses second-order-section form with ``sosfiltfilt`` for numerical stability at the
    low normalised frequencies involved (and to avoid phase distortion of the waveform).
    """
    x = np.asarray(x, dtype=np.float64)
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def chrom_pulse(rgb, fs: float, low: float = HR_BAND[0], high: float = HR_BAND[1]) -> np.ndarray:
    """Combine RGB into a single pulse signal via the CHROM method.

    CHROM (de Haan & Jeanne, 2013) builds two chrominance projections that cancel the
    specular/motion component under a standardised skin-tone assumption, band-passes
    them, and combines them with a ratio ``alpha`` that equalises their amplitudes::

        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn
        S  = bandpass(Xs) - alpha * bandpass(Ys),  alpha = std(Xf) / std(Yf)

    where ``Rn, Gn, Bn`` are each channel divided by its temporal mean.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    eps = 1e-9
    rn = r / (np.mean(r) + eps)
    gn = g / (np.mean(g) + eps)
    bn = b / (np.mean(b) + eps)

    xs = 3.0 * rn - 2.0 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    xf = bandpass_filter(xs, fs, low, high)
    yf = bandpass_filter(ys, fs, low, high)

    sy = np.std(yf)
    alpha = (np.std(xf) / sy) if sy > eps else 0.0
    return xf - alpha * yf


def _parabolic_peak(spectrum: np.ndarray, k: int, freqs: np.ndarray) -> float:
    """Sub-bin peak frequency via parabolic interpolation around bin ``k``."""
    if k <= 0 or k >= len(spectrum) - 1:
        return float(freqs[k])
    a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df)


def estimate_rate_fft(x, fs: float, band: tuple[float, float],
                      peak_bw: float = 0.2) -> tuple[float, float]:
    """Estimate the dominant frequency of ``x`` within ``band`` and its spectral SNR.

    A Hann window suppresses leakage; the spectrum is zero-padded to at least 4x length
    so parabolic interpolation locates the peak to sub-bin (<1 bpm) accuracy. The
    returned quality is the ratio of spectral power within ``peak_bw`` Hz of the peak to
    the remaining in-band power (large for a clean, sharp pulse; small for noise).

    Returns ``(freq_hz, snr)``.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return 0.0, 0.0
    nfft = int(2 ** np.ceil(np.log2(n * 4)))
    spectrum = np.abs(rfft(x * np.hanning(n), nfft))
    freqs = rfftfreq(nfft, 1.0 / fs)

    lo, hi = band
    in_band = np.where((freqs >= lo) & (freqs <= hi))[0]
    if in_band.size == 0:
        return 0.0, 0.0
    k = int(in_band[np.argmax(spectrum[in_band])])
    freq = _parabolic_peak(spectrum, k, freqs)

    power = spectrum ** 2
    near = (freqs >= freq - peak_bw) & (freqs <= freq + peak_bw)
    band_mask = (freqs >= lo) & (freqs <= hi)
    near_power = power[near & band_mask].sum()
    rest_power = power[band_mask].sum() - near_power
    snr = float(near_power / rest_power) if rest_power > 1e-12 else float("inf")
    return freq, snr


def estimate_heart_rate(rgb, fs: float) -> tuple[float, float]:
    """Estimate heart rate (bpm) and signal quality from a window of ROI mean RGB."""
    pulse = chrom_pulse(rgb, fs, *HR_BAND)
    freq, snr = estimate_rate_fft(pulse, fs, HR_BAND)
    return freq * 60.0, snr
