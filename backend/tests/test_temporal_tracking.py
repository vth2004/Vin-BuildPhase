from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.features.face_landmark_vf50.services.run import _match_faces
from app.features.human_pose.services.observations import build_pose_observations
from app.shared.temporal.segmentation import is_scene_cut, luminance_histogram
from app.shared.temporal.tracking import DetectionTracker, assign_one_to_one


def _person(x: float, *, point_x: float | None = None) -> dict:
    px = x + 5 if point_x is None else point_x
    return {"bbox": (x, 0, x + 10, 20), "score": 0.9,
            "keypoints": {1: {"id": 1, "x": px, "y": 5, "conf": 0.8}}}


def _face(x: float) -> dict[int, dict]:
    return {index: {"id": index, "x": x + index % 2, "y": 10 + index // 2, "state": "visible"}
            for index in range(50)}


class TemporalTrackingTests(unittest.TestCase):
    def test_pose_observation_is_detection_first_with_provenance(self) -> None:
        predictions = {
            "yolo26s-pose": [_person(0), _person(100)],
            "rtmpose-m": [_person(101, point_x=108), _person(1, point_x=8)],
        }
        observations = build_pose_observations(predictions)
        self.assertEqual(2, len(observations))
        self.assertEqual([2, 2], [item["supporting_model_count"] for item in observations])
        self.assertEqual(["ensemble", "ensemble"], [item["observation_source"] for item in observations])
        self.assertGreater(observations[0]["keypoints"][1]["x"], 5)
        self.assertLessEqual(observations[0]["keypoints"][1]["x"], 8)
        self.assertEqual(1, observations[0]["model_matches"]["rtmpose-m"]["bbox"][0])
        self.assertEqual(["rtmpose-m", "yolo26s-pose"], observations[0]["supporting_models"])

    def test_primary_missing_ends_observation_stream(self) -> None:
        tracker = DetectionTracker()
        first = tracker.update(build_pose_observations({"yolo26s-pose": [_person(0)]}))[0]
        observations = build_pose_observations({"yolo26s-pose": [], "rtmpose-m": [_person(0)]})
        self.assertEqual([], observations)
        tracker.update(observations)
        returned = tracker.update(build_pose_observations({"yolo26s-pose": [_person(0)]}))[0]
        self.assertNotEqual(first["internal_track_id"], returned["internal_track_id"])

    def test_assignment_is_one_to_one_and_rejects_ambiguity(self) -> None:
        matched = assign_one_to_one([[0.1, 0.9], [0.8, 0.1]], 0.5)
        self.assertEqual({(0, 0), (1, 1)}, {(item.left_index, item.right_index) for item in matched.matches})
        ambiguous = assign_one_to_one([[0.10, 0.11], [0.11, 0.10]], 0.5, ambiguity_margin=0.05)
        self.assertEqual([], ambiguous.matches)
        self.assertEqual([0, 1], ambiguous.uncertain_left)

    def test_tracker_survives_detection_order_change_and_splits_on_gap(self) -> None:
        tracker = DetectionTracker()
        first = tracker.update([_person(0), _person(100)])
        second = tracker.update([_person(102), _person(2)])
        self.assertEqual(first[0]["internal_track_id"], second[1]["internal_track_id"])
        self.assertEqual(first[1]["internal_track_id"], second[0]["internal_track_id"])
        after_gap = tracker.update([_person(4)], discontinuity=True)
        self.assertNotEqual(first[0]["internal_track_id"], after_gap[0]["internal_track_id"])
        self.assertEqual(1, after_gap[0]["segment_id"])

    def test_tracker_does_not_reidentify_after_all_detections_disappear(self) -> None:
        tracker = DetectionTracker()
        first = tracker.update([_person(0)])[0]
        self.assertEqual([], tracker.update([]))
        returned = tracker.update([_person(0)])[0]
        self.assertNotEqual(first["internal_track_id"], returned["internal_track_id"])
        self.assertGreater(returned["segment_id"], first["segment_id"])

    def test_face_matcher_keeps_ambiguous_items_unmatched(self) -> None:
        annotations = [{"id": "a", "keypoints": _face(10)}, {"id": "b", "keypoints": _face(12)}]
        matches = _match_faces(annotations, [_face(11), _face(11.5)])
        self.assertEqual(0, sum(1 for annotation, model in matches if annotation and model))
        self.assertEqual(4, len(matches))

    def test_scene_cut_histogram(self) -> None:
        dark = luminance_histogram(np.zeros((20, 20), dtype=np.uint8))
        bright = luminance_histogram(np.full((20, 20), 255, dtype=np.uint8))
        self.assertTrue(is_scene_cut(dark, bright))
        self.assertFalse(is_scene_cut(dark, dark))


if __name__ == "__main__":
    unittest.main()
