"""Live rPPG dashboard: camera feed + ROI overlay, waveform, HR/RR, quality badge.

The module is split into an engine and a UI:

  * ``VitalsMonitor`` -- the engine. It pulls frames from a ``WebcamCapture`` (webcam or
    video file), extracts the ROI, buffers the signal, and produces a ``VitalsUpdate``
    per frame with the annotated frame, smoothed HR/RR, quality, and pulse waveform.
    It has no Streamlit dependency, so it can be run and tested headlessly.

  * ``run_streamlit`` -- a thin Streamlit rendering loop over ``VitalsMonitor``. It runs
    when the file is executed via ``streamlit run src/dashboard.py`` (see ``run.py``).

Reported HR is an SNR-weighted median over the last few windows, so a single noisy
window does not make the displayed number jump.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.breathing import estimate_breathing_rate
from src.capture import WebcamCapture
from src.face_roi import FaceROIExtractor, draw_roi
from src.quality import QualityResult, assess_quality
from src.signal_pipeline import HR_BAND, SignalBuffer, chrom_pulse, estimate_heart_rate


@dataclass
class VitalsUpdate:
    """Everything the UI needs to render one frame."""

    frame_bgr: np.ndarray
    timestamp: float
    face_found: bool
    heart_rate: float | None = None
    breathing_rate: float | None = None
    quality: QualityResult | None = None
    pulse_waveform: np.ndarray | None = None


class VitalsMonitor:
    """Streaming rPPG engine producing one :class:`VitalsUpdate` per processed frame."""

    def __init__(self, source: int | str = 0, hr_window: float = 10.0,
                 rr_window: float = 30.0, smooth: int = 8):
        self.hr_window = hr_window
        self.rr_window = rr_window
        self._maxlen = max(hr_window, rr_window) + 2.0
        self.cap = WebcamCapture(source)
        self.ext = FaceROIExtractor()
        self.buf = SignalBuffer(maxlen_seconds=self._maxlen)
        self._cent: deque[tuple[float, float, float]] = deque()
        self._hr_hist: deque[tuple[float, float]] = deque(maxlen=smooth)

    def _smoothed_hr(self) -> float:
        """SNR-weighted median of the recent heart-rate estimates."""
        hrs = np.array([h for h, _ in self._hr_hist])
        w = np.array([max(s, 1e-3) for _, s in self._hr_hist])
        order = np.argsort(hrs)
        hrs, w = hrs[order], w[order]
        cutoff = w.sum() / 2.0
        return float(hrs[np.searchsorted(np.cumsum(w), cutoff)])

    def step(self) -> VitalsUpdate | None:
        """Process the next frame; return ``None`` when the source is exhausted."""
        ok, frame, ts = self.cap.read()
        if not ok:
            return None

        roi = self.ext.extract(frame)
        face = roi is not None
        annotated = draw_roi(frame, roi) if face else frame

        if face:
            self.buf.append(roi.rgb_mean, ts)
            self._cent.append((ts, roi.centroid[0], roi.centroid[1]))
            while self._cent and (ts - self._cent[0][0]) > self._maxlen:
                self._cent.popleft()

        update = VitalsUpdate(frame_bgr=annotated, timestamp=ts, face_found=face)

        rgb, fs = self.buf.get_rgb_window(self.hr_window)
        if self.buf.duration() >= self.hr_window * 0.9 and len(rgb) > fs * 4:
            pulse = chrom_pulse(rgb, fs)
            hr_raw, snr = estimate_heart_rate(rgb, fs)
            self._hr_hist.append((hr_raw, snr))

            cents = np.array([[cx, cy] for (tt, cx, cy) in self._cent
                              if tt >= ts - self.hr_window])
            update.quality = assess_quality(pulse, fs, cents, HR_BAND)
            update.heart_rate = self._smoothed_hr()
            update.pulse_waveform = pulse

            rgb_rr, fs_rr = self.buf.get_rgb_window(self.rr_window)
            if self.buf.duration() >= self.rr_window * 0.5:
                rr, _ = estimate_breathing_rate(rgb_rr[:, 1], fs_rr)
                update.breathing_rate = rr

        return update

    def close(self) -> None:
        self.cap.release()
        self.ext.close()


# --------------------------------------------------------------------------- Streamlit UI
_QUALITY_COLOR = {"good": "#1a9850", "fair": "#f9a825", "poor": "#d73027"}


def run_streamlit() -> None:
    """Streamlit rendering loop. Executed by ``streamlit run src/dashboard.py``."""
    import cv2
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Contactless Vitals (rPPG)", layout="wide")
    st.title("Contactless Vital Signs Monitor")
    st.caption("Heart rate & breathing rate from webcam video via remote photoplethysmography (CHROM).")

    st.sidebar.header("Source")
    kind = st.sidebar.radio("Input", ["Webcam", "Video file"])
    if kind == "Webcam":
        source: int | str = int(st.sidebar.number_input("Camera index", 0, 4, 0, step=1))
    else:
        source = st.sidebar.text_input(
            "Video path", "data/mcd/video/1024_FullHDwebcam_before.avi")

    if "running" not in st.session_state:
        st.session_state.running = False
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Start", use_container_width=True):
        st.session_state.running = True
    if col_b.button("Stop", use_container_width=True):
        st.session_state.running = False

    frame_slot = st.empty()
    metric_slot = st.empty()
    wave_slot = st.empty()

    if not st.session_state.running:
        frame_slot.info("Choose a source and press **Start**.")
        return

    try:
        monitor = VitalsMonitor(source)
    except RuntimeError as exc:
        st.error(str(exc))
        st.session_state.running = False
        return

    try:
        while st.session_state.running:
            update = monitor.step()
            if update is None:
                st.session_state.running = False
                st.info("End of video source.")
                break

            frame_slot.image(cv2.cvtColor(update.frame_bgr, cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)

            with metric_slot.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Heart rate (bpm)",
                          f"{update.heart_rate:.0f}" if update.heart_rate else "--")
                c2.metric("Breathing (br/min)",
                          f"{update.breathing_rate:.0f}" if update.breathing_rate else "--")
                if update.quality is not None:
                    color = _QUALITY_COLOR[update.quality.level]
                    c3.markdown(
                        f"**Signal quality**<br>"
                        f"<span style='color:{color};font-size:1.6rem;font-weight:700'>"
                        f"{update.quality.level.upper()}</span><br>"
                        f"<small>{'; '.join(update.quality.reasons)}</small>",
                        unsafe_allow_html=True)
                else:
                    c3.metric("Signal quality", "warming up")

            if update.pulse_waveform is not None:
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(update.pulse_waveform, color="#c1121f", lw=1.0)
                ax.set_title("CHROM pulse waveform")
                ax.set_xticks([])
                ax.set_yticks([])
                wave_slot.pyplot(fig)
                plt.close(fig)
    finally:
        monitor.close()


if __name__ == "__main__":
    run_streamlit()
