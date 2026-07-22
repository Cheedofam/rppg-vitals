"""Unit tests for the pure ROI colour-averaging helper in ``src.face_roi``.

These tests use synthetic images only, so they run without a webcam, without the
dataset, and without loading MediaPipe.
"""

import numpy as np

from src.face_roi import mean_rgb_in_polygon


def test_mean_rgb_uniform_image():
    """A solid-colour image returns that colour regardless of the polygon."""
    # OpenCV frames are BGR; fill with B=10, G=20, R=30.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    polygon = np.array([[10, 10], [80, 10], [80, 80], [10, 80]])

    rgb = mean_rgb_in_polygon(frame, polygon)

    assert np.allclose(rgb, [30, 20, 10], atol=1e-6)  # returned as R, G, B


def test_mean_rgb_selects_polygon_region():
    """Only pixels inside the polygon contribute to the mean."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :50] = (0, 0, 255)   # left half red   (BGR)
    frame[:, 50:] = (255, 0, 0)   # right half blue (BGR)
    left_polygon = np.array([[5, 5], [45, 5], [45, 95], [5, 95]])

    rgb = mean_rgb_in_polygon(frame, left_polygon)

    assert rgb[0] > 250 and rgb[2] < 5  # essentially pure red


def test_mean_rgb_returns_float_rgb_triplet():
    frame = np.full((20, 20, 3), 128, dtype=np.uint8)
    polygon = np.array([[0, 0], [19, 0], [19, 19], [0, 19]])

    rgb = mean_rgb_in_polygon(frame, polygon)

    assert rgb.shape == (3,)
    assert rgb.dtype == np.float64
