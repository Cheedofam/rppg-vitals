"""Webcam / video-file frame capture.

``WebcamCapture`` wraps ``cv2.VideoCapture`` and yields ``(ok, frame_bgr, timestamp)``
tuples. It accepts either a camera index (live webcam) or a path to a video file, so the
whole pipeline can be developed and validated against recorded datasets without a camera
attached, then run live unchanged.

Timestamps are seconds since the first frame:
  * video file  -> ``frame_index / container_fps`` (deterministic, matches ground truth)
  * live webcam -> wall-clock elapsed time (captures the true, possibly variable, rate)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class WebcamCapture:
    """Frame source backed by a webcam index or a video file path."""

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.is_file = not isinstance(source, int)
        self.cap = cv2.VideoCapture(str(source) if self.is_file else source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open capture source {source!r}. "
                "For a webcam try indices 0-2 and check camera privacy settings; "
                "for a file check the path exists and the codec is supported."
            )
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._frame_idx = -1
        self._t0: float | None = None

    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        """Return ``(ok, frame_bgr, timestamp_seconds)``; ``(False, None, None)`` at end."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        self._frame_idx += 1
        if self.is_file:
            ts = self._frame_idx / self.fps
        else:
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            ts = now - self._t0
        return True, frame, ts

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "WebcamCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
