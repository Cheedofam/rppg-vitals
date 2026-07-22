"""Signal-quality assessment for rPPG estimates.

Two failure modes dominate webcam rPPG and are flagged here:
  * **Poor signal** -- a weak or absent spectral peak (low SNR), caused by dim/uneven
    lighting or a badly placed ROI.
  * **Motion artifact** -- large frame-to-frame movement of the ROI, which injects
    broadband noise and shifts the sampled skin region.

Validation on the MCD-rPPG dataset showed rPPG heart-rate estimates agree closely with
contact PPG when the spectral SNR is high, and scatter when it is low -- so the SNR-based
gate below is what makes reported estimates trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.signal_pipeline import HR_BAND, estimate_rate_fft

# Heuristic thresholds (tuned against MCD-rPPG observations).
SNR_GOOD = 1.5     # >= this: clean, trustworthy peak
SNR_FAIR = 0.7     # >= this: usable but noisy; below: unreliable
MOTION_PX = 5.0    # mean ROI-centroid shift (px/frame) above this flags motion


@dataclass
class QualityResult:
    """Outcome of a quality assessment for one analysis window."""

    level: str                    # "good" | "fair" | "poor"
    snr: float                    # spectral SNR of the pulse signal
    motion: float                 # mean ROI-centroid displacement (px/frame)
    reasons: list[str] = field(default_factory=list)


def spectral_snr(x, fs: float, band: tuple[float, float] = HR_BAND) -> float:
    """Spectral SNR of ``x`` within ``band`` (power near the dominant peak vs the rest)."""
    _, snr = estimate_rate_fft(x, fs, band)
    return snr


def motion_score(centroids) -> float:
    """Mean Euclidean frame-to-frame displacement of the ROI centroid (pixels/frame)."""
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2 or len(c) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1)))


def assess_quality(pulse, fs: float, centroids, band: tuple[float, float] = HR_BAND,
                   snr_good: float = SNR_GOOD, snr_fair: float = SNR_FAIR,
                   motion_px: float = MOTION_PX) -> QualityResult:
    """Classify signal quality as good / fair / poor from SNR and ROI motion."""
    snr = spectral_snr(pulse, fs, band)
    motion = motion_score(centroids)

    reasons: list[str] = []
    level = "good"
    if snr < snr_fair:
        level = "poor"
        reasons.append(f"low signal SNR ({snr:.2f})")
    elif snr < snr_good:
        level = "fair"
        reasons.append(f"moderate signal SNR ({snr:.2f})")

    if motion > motion_px:
        level = "poor"
        reasons.append(f"motion artifact ({motion:.1f} px/frame)")

    if not reasons:
        reasons.append("clean signal, low motion")
    return QualityResult(level=level, snr=snr, motion=motion, reasons=reasons)
