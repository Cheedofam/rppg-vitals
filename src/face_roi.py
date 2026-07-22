"""Face landmark detection and skin-ROI extraction (MediaPipe Face Mesh).

The pulse signal is drawn from well-perfused, relatively flat skin regions -- the
forehead and both cheeks -- while eyes, eyebrows, nostrils and mouth are avoided. Each
region is built as the convex hull of a set of MediaPipe Face Mesh landmarks, so exact
landmark ordering does not matter and the polygon is always simple (non-self-intersecting).

``mean_rgb_in_polygon`` is a pure, dependency-light helper (NumPy + OpenCV only) and is
unit tested with synthetic images. MediaPipe is imported lazily inside ``FaceROIExtractor``
so importing this module for the pure helper does not pay the MediaPipe import cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Face Mesh landmark index sets (canonical 468-point topology, shared by the Tasks
# FaceLandmarker). Convex hulls of these define the ROIs; tuned against the live overlay
# so patches sit on skin and avoid eyes/mouth.
FOREHEAD_IDS = [10, 67, 69, 104, 108, 151, 337, 299, 333, 297, 338, 109]
LEFT_CHEEK_IDS = [116, 117, 118, 101, 205, 187, 123, 147]
RIGHT_CHEEK_IDS = [345, 346, 347, 330, 425, 411, 352, 376]

# MediaPipe Tasks face-landmarker model (auto-downloaded on first use).
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_face_landmarker_model(path: Path | None = None) -> str:
    """Return the local path to the face-landmarker model, downloading it if missing."""
    path = Path(path) if path is not None else _MODEL_PATH
    if not path.exists() or path.stat().st_size == 0:
        import requests

        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_MODEL_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return str(path)


@dataclass
class ROIResult:
    """Result of ROI extraction for a single frame."""

    rgb_mean: np.ndarray          # [R, G, B] mean over the pooled ROI pixels
    regions: list[np.ndarray]     # list of (K, 2) int polygons for drawing
    centroid: tuple[float, float]  # mean landmark position of all ROI points (x, y)
    ok: bool


def mean_rgb_in_polygon(frame_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Mean colour of the pixels inside ``polygon``, returned as ``[R, G, B]`` floats.

    Parameters
    ----------
    frame_bgr : (H, W, 3) uint8 array in OpenCV BGR order.
    polygon   : (K, 2) array of (x, y) vertices.
    """
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return np.array([r, g, b], dtype=np.float64)


def _mean_rgb_in_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return np.array([r, g, b], dtype=np.float64)


class FaceROIExtractor:
    """Detect a face and extract the mean RGB of its forehead + cheek skin regions."""

    def __init__(self, min_detection_confidence: float = 0.5,
                 model_path: str | None = None):
        # Lazy imports: keep the pure ``mean_rgb_in_polygon`` helper importable (and the
        # unit tests fast) without paying the MediaPipe import cost.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model = model_path or ensure_face_landmarker_model()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _landmark_polygon(self, landmarks, ids, w: int, h: int) -> np.ndarray:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids],
                       dtype=np.float32)
        hull = cv2.convexHull(pts)
        return hull.reshape(-1, 2)

    def extract(self, frame_bgr: np.ndarray) -> ROIResult | None:
        """Return ROI mean RGB + polygons for ``frame_bgr``, or ``None`` if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]

        regions = [self._landmark_polygon(lm, ids, w, h)
                   for ids in (FOREHEAD_IDS, LEFT_CHEEK_IDS, RIGHT_CHEEK_IDS)]

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in regions:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        rgb_mean = _mean_rgb_in_mask(frame_bgr, mask)

        all_ids = FOREHEAD_IDS + LEFT_CHEEK_IDS + RIGHT_CHEEK_IDS
        cx = float(np.mean([lm[i].x for i in all_ids]) * w)
        cy = float(np.mean([lm[i].y for i in all_ids]) * h)

        return ROIResult(rgb_mean=rgb_mean, regions=regions, centroid=(cx, cy), ok=True)

    def close(self) -> None:
        self._landmarker.close()


def draw_roi(frame_bgr: np.ndarray, roi: ROIResult) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with the ROI polygons and centroid overlaid."""
    out = frame_bgr.copy()
    for poly in roi.regions:
        cv2.polylines(out, [poly.astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=2)
    cx, cy = int(roi.centroid[0]), int(roi.centroid[1])
    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out


def _main() -> int:
    """Live/offline sanity check: draw the ROI overlay for a webcam or a video frame."""
    import argparse

    from src.capture import WebcamCapture

    parser = argparse.ArgumentParser(description="ROI overlay sanity check.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, help="Path to a video file.")
    src.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    parser.add_argument("--frame", type=int, default=150,
                        help="For --video: frame index to sample (default 150).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the overlay image here instead of showing a window.")
    args = parser.parse_args()

    cap = WebcamCapture(source=args.video if args.video else args.camera)
    extractor = FaceROIExtractor()

    if args.video and args.save:
        frame = None
        for _ in range(args.frame + 1):
            ok, frame, _ = cap.read()
            if not ok:
                break
        cap.release()
        if frame is None:
            print("Could not read the requested frame.")
            return 1
        roi = extractor.extract(frame)
        if roi is None:
            print("No face detected in the sampled frame.")
            return 1
        cv2.imwrite(args.save, draw_roi(frame, roi))
        print(f"Saved overlay to {args.save}; ROI mean RGB = {roi.rgb_mean.round(1)}")
        return 0

    print("Press 'q' to quit.")
    while True:
        ok, frame, _ = cap.read()
        if not ok:
            break
        roi = extractor.extract(frame)
        shown = draw_roi(frame, roi) if roi else frame
        cv2.imshow("ROI sanity check", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
