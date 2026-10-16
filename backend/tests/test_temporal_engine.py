from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.shared.temporal.engine import ObservationFrame, estimate_keypoint, estimate_track, normalization_transform, validate_window_size


def _pose_frame(frame_index: int, normalized_x: float, *, timestamp_us: int | None = None,
                include_anchors: bool = True, point_x_override: float | None = None,
                tracking_status: str = "confirmed") -> ObservationFrame:
    origin_x = 100 + 3 * frame_index
    origin_y = 200 + frame_index
    scale = 50.0
    points = {1: {"id": 1, "x": point_x_override if point_x_override is not None else origin_x + normalized_x * scale,
                  "y": origin_y + 0.2 * scale, "conf": 0.9}}
    if include_anchors:
        points.update({
            12: {"id": 12, "x": origin_x - 10, "y": origin_y},
            13: {"id": 13, "x": origin_x + 10, "y": origin_y},
            6: {"id": 6, "x": origin_x - 10, "y": origin_y - scale},
            7: {"id": 7, "x": origin_x + 10, "y": origin_y - scale},
        })
    return ObservationFrame(frame_index, points, (origin_x - 25, origin_y - 60, origin_x + 25, origin_y + 20),
                            timestamp_us, "manifest" if timestamp_us is not None else "unavailable",
                            640, 480, 0.9, tracking_status)


def _face_frame(frame_index: int, normalized_x: float) -> ObservationFrame:
    center_x, center_y, iod = 200 + frame_index * 2, 100.0, 40.0
    points = {1: {"id": 1, "x": center_x + normalized_x * iod, "y": center_y}}
    for point_id in range(14, 22):
        points[point_id] = {"id": point_id, "x": center_x - iod / 2, "y": center_y}
    for point_id in range(22, 30):
        points[point_id] = {"id": point_id, "x": center_x + iod / 2, "y": center_y}
    return ObservationFrame(frame_index, points, (center_x - 50, 50, center_x + 50, 150), image_width=640, image_height=480)


class TemporalEngineTests(unittest.TestCase):
    def test_window_validation(self) -> None:
        for value in (5, 15, 29):
            validate_window_size(value)
        for value in (3, 4, 6, 30, True):
            with self.assertRaises(ValueError):
                validate_window_size(value)
        for value in (5, 29):
            frames = [_pose_frame(i, 0.2) for i in range(value)]
            self.assertEqual("available", estimate_keypoint(frames, value // 2, 1, feature="pose",
                                                             window_size=value).estimate_status)

    def test_stationary_and_linear_motion_are_recovered_leave_one_out(self) -> None:
        stationary = [_pose_frame(i, 0.4, point_x_override=999 if i == 7 else None) for i in range(15)]
        estimate = estimate_keypoint(stationary, 7, 1, feature="pose", window_size=15)
        expected = 100 + 3 * 7 + 0.4 * 50
        self.assertEqual("available", estimate.estimate_status)
        self.assertAlmostEqual(expected, estimate.coordinate[0], places=4)
        self.assertEqual(2, estimate.polynomial_degree)

        linear = [_pose_frame(i, 0.1 + 0.03 * i) for i in range(15)]
        estimate = estimate_keypoint(linear, 7, 1, feature="pose", window_size=15)
        self.assertAlmostEqual(100 + 21 + (0.1 + 0.21) * 50, estimate.coordinate[0], places=3)

    def test_quadratic_turn_and_outlier_use_robust_fit(self) -> None:
        frames = [_pose_frame(i, 0.2 + 0.015 * (i - 7) ** 2,
                              point_x_override=800 if i == 5 else None) for i in range(15)]
        estimate = estimate_keypoint(frames, 7, 1, feature="pose", window_size=15)
        self.assertEqual("available", estimate.estimate_status)
        self.assertLess(abs(estimate.coordinate[0] - (121 + 0.2 * 50)), 3.0)

    def test_extrapolation_requires_window_seven_and_three_same_side(self) -> None:
        frames = [_pose_frame(i, 0.1 + i * 0.02) for i in range(8)]
        small = estimate_keypoint(frames, 0, 1, feature="pose", window_size=5)
        self.assertEqual("window_too_small", small.unavailable_reason)
        available = estimate_keypoint(frames, 0, 1, feature="pose", window_size=7)
        self.assertEqual("extrapolation", available.method)
        self.assertEqual(3, available.support_count)

    def test_gap_uses_original_frame_delta_and_time(self) -> None:
        indices = [0, 1, 2, 10, 11, 12, 13]
        frames = [_pose_frame(index, 0.2) for index in indices]
        estimate = estimate_keypoint(frames, 3, 1, feature="pose", window_size=7)
        self.assertEqual("gap_too_long", estimate.unavailable_reason)

        timed = [_pose_frame(i, 0.2, timestamp_us=i * 300_000) for i in range(7)]
        estimate = estimate_keypoint(timed, 3, 1, feature="pose", window_size=7)
        self.assertEqual("gap_too_long", estimate.unavailable_reason)
        self.assertEqual(600_000, estimate.gap_us)

    def test_invalid_timestamp_falls_back_to_frame_axis(self) -> None:
        frames = [_pose_frame(i, 0.2, timestamp_us=(5 - i) * 10_000) for i in range(7)]
        estimate = estimate_keypoint(frames, 3, 1, feature="pose", window_size=7)
        self.assertEqual("available", estimate.estimate_status)
        self.assertEqual("frame_index", estimate.time_source)

        unconfirmed = [_pose_frame(i, 0.2, timestamp_us=i * 10_000) for i in range(7)]
        unconfirmed = [ObservationFrame(frame.frame_index, frame.keypoints, frame.bbox, frame.timestamp_us,
                                        "unavailable", frame.image_width, frame.image_height)
                       for frame in unconfirmed]
        estimate = estimate_keypoint(unconfirmed, 3, 1, feature="pose", window_size=7)
        self.assertEqual("frame_index", estimate.time_source)

    def test_normalization_source_is_not_mixed_and_target_transform_is_required(self) -> None:
        frames = [_pose_frame(i, 0.2, include_anchors=i != 2) for i in range(7)]
        estimate = estimate_keypoint(frames, 3, 1, feature="pose", window_size=7)
        self.assertEqual("available", estimate.estimate_status)
        self.assertEqual("torso", estimate.normalization_source)
        no_transform = ObservationFrame(3, {1: {"id": 1, "x": 1, "y": 1}})
        broken = frames[:3] + [no_transform] + frames[4:]
        estimate = estimate_keypoint(broken, 3, 1, feature="pose", window_size=7)
        self.assertEqual("normalization_unavailable", estimate.unavailable_reason)

    def test_tracking_uncertain_and_face_iod_normalization(self) -> None:
        frames = [_pose_frame(i, 0.2, tracking_status="tracking_uncertain" if i == 3 else "confirmed") for i in range(7)]
        self.assertEqual("tracking_uncertain", estimate_keypoint(frames, 3, 1, feature="pose", window_size=7).unavailable_reason)
        face = [_face_frame(i, 0.3) for i in range(7)]
        self.assertEqual("iod", normalization_transform(face[3], "face").source)
        estimate = estimate_keypoint(face, 3, 1, feature="face", window_size=7)
        self.assertTrue(math.isclose(estimate.coordinate[0], 206 + 0.3 * 40, abs_tol=1e-5))

    def test_batch_returns_result_for_every_requested_point(self) -> None:
        frames = [_pose_frame(i, 0.2) for i in range(7)]
        results = estimate_track(frames, [1, 99], feature="pose", window_size=7)
        self.assertEqual(14, len(results))
        self.assertEqual("available", results[(3, 1)].estimate_status)
        self.assertEqual("low_support", results[(3, 99)].unavailable_reason)


if __name__ == "__main__":
    unittest.main()
