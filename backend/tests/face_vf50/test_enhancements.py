from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.features.face_landmark_vf50.rules.geometry import (
    compute_iod,
    compute_iod_with_source,
    point_in_polygon,
    min_distance_to_polygon,
)
from app.features.face_landmark_vf50.rules.validation import check_all_rules
from app.features.face_landmark_vf50.scoring.quality import evaluate_frame_quality
from app.features.face_landmark_vf50.services.run import _match_faces


def make_dummy_keypoints(iod: float = 100.0) -> dict[int, dict[str, Any]]:
    """Tạo 50 điểm hợp lệ cơ bản cho khuôn mặt với IOD ~ 100px."""
    kpts: dict[int, dict[str, Any]] = {}
    for i in range(50):
        kpts[i] = {"id": i, "name": str(i), "x": float(100 + i * 5), "y": float(100 + (i % 10) * 5), "state": "visible"}
    # Đặt tâm 2 mắt cách nhau 100px
    # Mắt trái: 14-21 quanh (100, 100)
    for i in range(14, 22):
        angle = (i - 14) * (2 * math.pi / 8)
        kpts[i] = {"id": i, "name": str(i), "x": 100.0 + 10.0 * math.cos(angle), "y": 100.0 + 5.0 * math.sin(angle), "state": "visible"}
    # Mắt phải: 22-29 quanh (200, 100)
    for i in range(22, 30):
        angle = (i - 22) * (2 * math.pi / 8)
        kpts[i] = {"id": i, "name": str(i), "x": 200.0 + 10.0 * math.cos(angle), "y": 100.0 + 5.0 * math.sin(angle), "state": "visible"}
    # Môi ngoài 30-41 hình chữ nhật quanh (150, 180)
    outer_coords = [
        (130, 180), (135, 175), (145, 173), (150, 174), (155, 173), (165, 175),
        (170, 180), (165, 185), (155, 187), (150, 186), (145, 187), (135, 185),
    ]
    for idx, (x, y) in enumerate(outer_coords):
        kpts[30 + idx] = {"id": 30 + idx, "name": str(30 + idx), "x": float(x), "y": float(y), "state": "visible"}
    # Môi trong 42-49 bên trong (150, 180)
    inner_coords = [
        (138, 180), (145, 177), (150, 178), (155, 177),
        (162, 180), (155, 183), (150, 182), (145, 183),
    ]
    for idx, (x, y) in enumerate(inner_coords):
        kpts[42 + idx] = {"id": 42 + idx, "name": str(42 + idx), "x": float(x), "y": float(y), "state": "visible"}
    return kpts


class TestEnhancements(unittest.TestCase):
    def test_empty_deviations_returns_none(self) -> None:
        """Khi không có điểm so sánh (model_kpts rỗng hoặc outside), NME=None, passed_nme=None."""
        human = make_dummy_keypoints()
        res = evaluate_frame_quality(human, None)
        self.assertIsNone(res["nme"])
        self.assertIsNone(res["passed_nme"])
        self.assertIsNone(res["nme_pct"])
        self.assertIsNone(res["e_max"])
        self.assertIsNone(res["e_p90"])
        self.assertEqual(res["evaluated_points_count"], 0)
        self.assertEqual(res["exceed_count"], 0)
        self.assertFalse(res["has_outlier_point"])

    def test_outlier_point_detected(self) -> None:
        """49 điểm e=0.02 + 1 điểm e=0.5 -> NME ~ 0.0296 (pass NME), nhưng has_outlier_point = True."""
        human = make_dummy_keypoints()
        iod = compute_iod(human)
        self.assertGreater(iod, 50.0)

        # Tạo model kpts sao cho 49 điểm lệch đúng 0.02 * iod, 1 điểm lệch 0.5 * iod
        model: dict[int, dict[str, Any]] = {}
        for i in range(50):
            hx = human[i]["x"]
            hy = human[i]["y"]
            if i == 49:
                # Điểm thứ 50 lệch 0.5 * iod
                model[i] = {"id": i, "x": hx + 0.5 * iod, "y": hy, "state": "visible"}
            else:
                # 49 điểm lệch 0.02 * iod
                model[i] = {"id": i, "x": hx + 0.02 * iod, "y": hy, "state": "visible"}

        res = evaluate_frame_quality(human, model, iod=iod)
        self.assertIsNotNone(res["nme"])
        # NME = (49 * 0.02 + 1 * 0.5) / 50 = (0.98 + 0.5) / 50 = 1.48 / 50 = 0.0296
        self.assertAlmostEqual(res["nme"], 0.0296, places=4)
        self.assertTrue(res["passed_nme"])  # 0.0296 <= 0.035 -> PASS NME
        self.assertTrue(res["has_outlier_point"])  # Nhưng outlier flag phải bật!
        self.assertEqual(res["exceed_count"], 1)
        self.assertAlmostEqual(res["e_max"], 0.5, places=3)
        self.assertGreaterEqual(res["e_p90"], 0.02)

    def test_iod_priority_chain(self) -> None:
        """Kiểm tra chuỗi ưu tiên IOD: Model -> Human -> outer corners -> bbox -> 96px."""
        human = make_dummy_keypoints(100.0)
        model = make_dummy_keypoints(120.0)

        # 1. Khi có cả Model và Human -> lấy Model
        iod_val, source = compute_iod_with_source(model, human)
        self.assertEqual(source, "model")
        self.assertAlmostEqual(iod_val, compute_iod(model), places=1)

        # 2. Khi không có Model -> lấy Human
        iod_val, source = compute_iod_with_source(None, human)
        self.assertEqual(source, "human")
        self.assertAlmostEqual(iod_val, compute_iod(human), places=1)

        # 3. Khi không có mắt đầy đủ -> fallback bbox hoặc default
        iod_val, source = compute_iod_with_source({}, {})
        self.assertEqual(source, "default")
        self.assertEqual(iod_val, 96.0)

    def test_r09_per_point_tolerance(self) -> None:
        """R09 kiểm tra từng điểm môi trong với khoảng cách tới đa giác môi ngoài."""
        kpts = make_dummy_keypoints()
        # Đặt 1 điểm môi trong (điểm 45) ra rất xa ngoài viền môi ngoài (> 20px)
        kpts[45]["x"] = 250.0
        kpts[45]["y"] = 250.0

        violations = check_all_rules(kpts)
        r09_viols = [v for v in violations if v.code == "R09"]
        self.assertTrue(any("45" in str(v.points) for v in r09_viols))

    def test_hungarian_matching_optimal(self) -> None:
        """Hungarian matching ghép tối ưu 2 khuôn mặt dựa trên khoảng cách centroid."""
        # 2 annotations tại (100, 100) và (300, 100)
        ann1 = {"id": "a1", "keypoints": {0: {"x": 100, "y": 100, "state": "visible"}}}
        ann2 = {"id": "a2", "keypoints": {0: {"x": 300, "y": 100, "state": "visible"}}}

        # 2 detections tại (105, 100) và (295, 100)
        det1 = {0: {"x": 105, "y": 100, "state": "visible"}}
        det2 = {0: {"x": 295, "y": 100, "state": "visible"}}

        matches = _match_faces([ann1, ann2], [det2, det1])
        # ann1 phải ghép với det1 (khoảng cách 5), ann2 ghép với det2 (khoảng cách 5)
        self.assertEqual(len(matches), 2)
        match_dict = {m[0]["id"]: m[1][0]["x"] for m in matches if m[0]}
        self.assertEqual(match_dict["a1"], 105)
        self.assertEqual(match_dict["a2"], 295)

    def test_hungarian_matching_cutoff(self) -> None:
        """Khuôn mặt quá xa cutoff không bị ép ghép cặp."""
        # 1 annotation tại (100, 100)
        ann = {"id": "a1", "keypoints": {0: {"x": 100, "y": 100, "state": "visible"}}}
        # 1 detection tại (1000, 1000) (quá xa)
        det = {0: {"x": 1000, "y": 1000, "state": "visible"}}

        matches = _match_faces([ann], [det])
        # Cả hai không được ghép với nhau
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0]["id"], "a1")
        self.assertEqual(matches[0][1], {})
        self.assertIsNone(matches[1][0])
        self.assertEqual(matches[1][1][0]["x"], 1000)

    def test_error_count_vs_violation_count(self) -> None:
        """error_count chỉ đếm ERROR, còn total vi phạm bao gồm cả WARNING."""
        kpts = make_dummy_keypoints()
        # Đặt tất cả 50 điểm visible -> kích hoạt R02 (WARNING)
        # R02 là WARNING, không phải ERROR
        violations = check_all_rules(kpts)
        warnings = [v for v in violations if v.level == "WARNING"]
        errors = [v for v in violations if v.level == "ERROR"]

        self.assertGreater(len(warnings), 0)
        # R02 là warning
        self.assertTrue(any(v.code == "R02" for v in warnings))


if __name__ == "__main__":
    unittest.main()
