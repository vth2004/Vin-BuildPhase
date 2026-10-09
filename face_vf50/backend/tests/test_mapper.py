"""Unit tests for MediaPipe to VinFast VF-50 landmark mapping."""

import unittest
import numpy as np
from face_vf50.backend.app.model.mapper import (
    resample_polyline,
    map_mediapipe_to_vf50,
)
from face_vf50.backend.app.model.mediapipe_face import get_detector


class TestMapper(unittest.TestCase):

    def test_resample_polyline_basic(self):
        # 3 points forming a horizontal line of length 10
        line = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        res = resample_polyline(line, fractions)
        self.assertEqual(res.shape, (5, 2))
        np.testing.assert_allclose(res[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0], atol=1e-5)
        np.testing.assert_allclose(res[:, 1], [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_resample_polyline_corner_cases(self):
        # Single point
        pt = np.array([[5.0, 5.0]], dtype=np.float32)
        res = resample_polyline(pt, [0.0, 0.5, 1.0])
        self.assertEqual(res.shape, (3, 2))
        np.testing.assert_allclose(res, [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])

        # Degenerate polyline (same points)
        deg = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        res_deg = resample_polyline(deg, [0.0, 1.0])
        self.assertEqual(res_deg.shape, (2, 2))
        np.testing.assert_allclose(res_deg, [[2.0, 2.0], [2.0, 2.0]])

    def test_map_mediapipe_to_vf50_shape_and_range(self):
        # Create synthetic 478 landmarks in normalized [0, 1] range
        # Arrange mock face landmarks so that viewer's left has x ~ 0.35, right x ~ 0.65
        mock_478 = np.zeros((478, 2), dtype=np.float32)
        # Left eyebrow / eye / mouth around x=0.35
        mock_478[:200, 0] = np.linspace(0.3, 0.45, 200)
        mock_478[:200, 1] = np.linspace(0.2, 0.5, 200)
        # Right eyebrow / eye / mouth around x=0.65
        mock_478[200:, 0] = np.linspace(0.55, 0.7, 278)
        mock_478[200:, 1] = np.linspace(0.2, 0.5, 278)

        # Specifically set nose bridge midline
        mock_478[168] = [0.5, 0.3]   # glabella
        mock_478[6]   = [0.5, 0.35]  # upper bridge
        mock_478[197] = [0.5, 0.40]  # mid bridge
        mock_478[195] = [0.5, 0.45]  # foot of bridge

        w, h = 1280.0, 720.0
        vf50 = map_mediapipe_to_vf50(mock_478, image_width=w, image_height=h, normalized=True)

        # Must return exactly (50, 2)
        self.assertEqual(vf50.shape, (50, 2))
        # Coordinates must be within image bounds
        self.assertTrue(np.all(vf50[:, 0] >= 0) and np.all(vf50[:, 0] <= w))
        self.assertTrue(np.all(vf50[:, 1] >= 0) and np.all(vf50[:, 1] <= h))

        # Nose bridge: point 10 (top) must have smaller y than point 13 (bottom)
        self.assertLess(vf50[10, 1], vf50[13, 1])

    def test_detector_initialization(self):
        detector = get_detector()
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.detector)


if __name__ == "__main__":
    unittest.main()
