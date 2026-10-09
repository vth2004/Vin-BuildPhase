from __future__ import annotations

import unittest

try:
    from face_vf50.backend.app.qa.rules import check_all_rules
except ModuleNotFoundError:
    from app.qa.rules import check_all_rules


def create_synthetic_valid_face() -> dict[int, dict]:
    """Tạo bộ 50 điểm khuôn mặt hợp lệ thỏa mãn toàn bộ các luật."""
    kpts = {}

    # longmaytrai (0-4): x tăng dần từ 480 đến 540, y ~ 260
    for i, x in enumerate([480.0, 495.0, 510.0, 525.0, 540.0]):
        kpts[i] = {"id": i, "x": x, "y": 260.0, "state": "visible"}

    # longmayphai (5-9): x tăng dần từ 560 đến 620, y ~ 260
    for i, x in enumerate([560.0, 575.0, 590.0, 605.0, 620.0], start=5):
        kpts[i] = {"id": i, "x": x, "y": 260.0, "state": "visible"}

    # songmui (10-13): x ~ 550, y từ 280 xuống 370 (3 đoạn đều nhau: 30px mỗi đoạn)
    for i, y in enumerate([280.0, 310.0, 340.0, 370.0], start=10):
        kpts[i] = {"id": i, "x": 550.0, "y": y, "state": "visible"}

    # mattrai (14-21): khoé ngoài 14 (x=470, y=300), khoé trong 18 (x=530, y=300)
    # mí trên (15, 16, 17) có y nhỏ hơn mí dưới (21, 20, 19)
    kpts[14] = {"id": 14, "x": 470.0, "y": 300.0, "state": "visible"}
    kpts[15] = {"id": 15, "x": 485.0, "y": 290.0, "state": "visible"}
    kpts[16] = {"id": 16, "x": 500.0, "y": 285.0, "state": "visible"}
    kpts[17] = {"id": 17, "x": 515.0, "y": 290.0, "state": "visible"}
    kpts[18] = {"id": 18, "x": 530.0, "y": 300.0, "state": "visible"}
    kpts[19] = {"id": 19, "x": 515.0, "y": 310.0, "state": "visible"}
    kpts[20] = {"id": 20, "x": 500.0, "y": 315.0, "state": "visible"}
    kpts[21] = {"id": 21, "x": 485.0, "y": 310.0, "state": "visible"}

    # matphai (22-29): khoé trong 22 (x=570, y=300), khoé ngoài 26 (x=630, y=300)
    kpts[22] = {"id": 22, "x": 570.0, "y": 300.0, "state": "visible"}
    kpts[23] = {"id": 23, "x": 585.0, "y": 290.0, "state": "visible"}
    kpts[24] = {"id": 24, "x": 600.0, "y": 285.0, "state": "visible"}
    kpts[25] = {"id": 25, "x": 615.0, "y": 290.0, "state": "visible"}
    kpts[26] = {"id": 26, "x": 630.0, "y": 300.0, "state": "visible"}
    kpts[27] = {"id": 27, "x": 615.0, "y": 310.0, "state": "visible"}
    kpts[28] = {"id": 28, "x": 600.0, "y": 315.0, "state": "visible"}
    kpts[29] = {"id": 29, "x": 585.0, "y": 310.0, "state": "visible"}

    # moingoai (30-41): khoé trái 30 (x=500, y=420), khoé phải 36 (x=600, y=420)
    kpts[30] = {"id": 30, "x": 500.0, "y": 420.0, "state": "visible"}
    kpts[31] = {"id": 31, "x": 520.0, "y": 405.0, "state": "visible"}
    kpts[32] = {"id": 32, "x": 540.0, "y": 400.0, "state": "visible"}
    kpts[33] = {"id": 33, "x": 550.0, "y": 400.0, "state": "visible"}
    kpts[34] = {"id": 34, "x": 560.0, "y": 400.0, "state": "visible"}
    kpts[35] = {"id": 35, "x": 580.0, "y": 405.0, "state": "visible"}
    kpts[36] = {"id": 36, "x": 600.0, "y": 420.0, "state": "visible"}
    kpts[37] = {"id": 37, "x": 580.0, "y": 435.0, "state": "visible"}
    kpts[38] = {"id": 38, "x": 560.0, "y": 440.0, "state": "visible"}
    kpts[39] = {"id": 39, "x": 550.0, "y": 440.0, "state": "visible"}
    kpts[40] = {"id": 40, "x": 540.0, "y": 440.0, "state": "visible"}
    kpts[41] = {"id": 41, "x": 520.0, "y": 435.0, "state": "visible"}

    # moitrong (42-49): nằm gọn bên trong moingoai
    kpts[42] = {"id": 42, "x": 515.0, "y": 420.0, "state": "visible"}
    kpts[43] = {"id": 43, "x": 535.0, "y": 415.0, "state": "visible"}
    kpts[44] = {"id": 44, "x": 550.0, "y": 415.0, "state": "visible"}
    kpts[45] = {"id": 45, "x": 565.0, "y": 415.0, "state": "visible"}
    kpts[46] = {"id": 46, "x": 585.0, "y": 420.0, "state": "visible"}
    kpts[47] = {"id": 47, "x": 565.0, "y": 425.0, "state": "visible"}
    kpts[48] = {"id": 48, "x": 550.0, "y": 425.0, "state": "visible"}
    kpts[49] = {"id": 49, "x": 535.0, "y": 425.0, "state": "visible"}

    # Đặt 1 điểm occluded để không kích hoạt cảnh báo R02 (toàn bộ là visible)
    kpts[0]["state"] = "occluded"

    return kpts


class TestFaceRules(unittest.TestCase):
    def test_valid_face_no_errors(self):
        face = create_synthetic_valid_face()
        violations = check_all_rules(face)
        errors = [v for v in violations if v.level == "ERROR"]
        self.assertEqual(len(errors), 0, f"Expected 0 errors on valid face, got: {errors}")

    def test_r01_missing_points(self):
        face = create_synthetic_valid_face()
        del face[0]
        del face[49]
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R01", codes)

    def test_r02_all_visible_warning(self):
        face = create_synthetic_valid_face()
        for k in face.values():
            k["state"] = "visible"
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R02", codes)

    def test_r03_left_right_swap(self):
        face = create_synthetic_valid_face()
        # Đổi tọa độ x giữa lông mày trái và lông mày phải
        for i in range(5):
            face[i]["x"] = 700.0 + i * 10
            face[i + 5]["x"] = 300.0 + i * 10
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R03", codes)

    def test_r06_eyelid_inverted(self):
        face = create_synthetic_valid_face()
        # Đảo điểm mí trên 16 xuống dưới điểm mí dưới 20
        face[16]["y"] = 330.0  # Thấp hơn 20 (y=315)
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R06", codes)

    def test_r09_inner_lip_outside(self):
        face = create_synthetic_valid_face()
        # Kéo khóe trong trái 42 ra xa hơn khóe ngoài trái 30 (x42 < x30)
        face[42]["x"] = 490.0  # nhỏ hơn 30 (x=500)
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R09", codes)

    def test_r11_out_of_bounds(self):
        face = create_synthetic_valid_face()
        face[10]["x"] = -15.0  # Ngoài khung
        violations = check_all_rules(face)
        codes = [v.code for v in violations]
        self.assertIn("R11", codes)


if __name__ == "__main__":
    unittest.main()
