from __future__ import annotations

import math
from typing import Any


def euclidean_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Khoảng cách Euclid giữa 2 điểm (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Tính trọng tâm (x_mean, y_mean) của danh sách các điểm."""
    if not points:
        return (0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def compute_iod(keypoints_by_id: dict[int, dict[str, Any]]) -> float:
    """
    Tính khoảng cách giữa tâm 2 mắt (Interocular Distance - IOD).
    Theo Mục 8 GL: IOD là khoảng cách giữa tâm 2 mắt, trung vị khoảng 96 px trong bộ ảnh 1280x720.
    Tâm mắt trái = trọng tâm của 8 điểm mắt trái (14 đến 21).
    Tâm mắt phải = trọng tâm của 8 điểm mắt phải (22 đến 29).
    """
    left_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(14, 22)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("y") is not None
    ]
    right_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(22, 30)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("y") is not None
    ]

    from ..spec import get_geometry_spec
    geom = get_geometry_spec().get("iod", {})
    min_eye_pts = geom.get("min_eye_points", 4)
    min_center_dist = geom.get("min_eye_center_dist", 10.0)

    if len(left_eye_pts) >= min_eye_pts and len(right_eye_pts) >= min_eye_pts:
        c_left = compute_centroid(left_eye_pts)
        c_right = compute_centroid(right_eye_pts)
        iod = euclidean_dist(c_left, c_right)
        if iod > min_center_dist:
            return iod

    # Fallback nếu một trong 2 mắt bị che: dùng khoảng cách giữa 2 khóe ngoài 14 và 26
    p14 = keypoints_by_id.get(14)
    p26 = keypoints_by_id.get(26)
    min_outer_dist = geom.get("outer_corners_min_dist", 5.0)
    outer_ratio = geom.get("outer_corners_ratio", 0.65)
    if p14 and p26 and p14.get("x") is not None and p26.get("x") is not None:
        outer_dist = euclidean_dist((float(p14["x"]), float(p14["y"])), (float(p26["x"]), float(p26["y"])))
        if outer_dist > min_outer_dist:
            return outer_dist * outer_ratio

    # Fallback thích ứng dựa trên bounding box thực tế của toàn bộ khuôn mặt:
    all_valid_pts = [
        (float(kp["x"]), float(kp["y"]))
        for kp in keypoints_by_id.values()
        if kp.get("state") != "outside" and kp.get("x") is not None and kp.get("y") is not None
    ]
    min_face_pts = geom.get("face_width_min_points", 8)
    min_face_w = geom.get("face_width_min_px", 10.0)
    min_result_px = geom.get("face_width_min_result_px", 12.0)
    width_ratio = geom.get("face_width_ratio", 0.42)
    if len(all_valid_pts) >= min_face_pts:
        min_x = min(p[0] for p in all_valid_pts)
        max_x = max(p[0] for p in all_valid_pts)
        face_width = max_x - min_x
        if face_width > min_face_w:
            return max(min_result_px, face_width * width_ratio)

    return float(geom.get("default_iod_px", 96.0))


def compute_iod_with_source(
    model_kpts: dict[int, dict[str, Any]] | None,
    human_kpts: dict[int, dict[str, Any]] | None = None,
) -> tuple[float, str]:
    """
    Tính IOD MỘT LẦN khi chạy run theo chuỗi dự phòng:
    1. 'model': từ model_kpts (tâm mắt hoặc khóe ngoài hoặc bbox)
    2. 'human': từ human_kpts (tâm mắt hoặc khóe ngoài hoặc bbox)
    3. 'bbox': từ bounding box của model hoặc human
    4. 'default': 96.0 px (hoặc cấu hình)
    Trả về: (iod_value, source_name)
    """
    from ..spec import get_geometry_spec
    geom = get_geometry_spec().get("iod", {})
    min_eye_pts = geom.get("min_eye_points", 4)
    min_center_dist = geom.get("min_eye_center_dist", 10.0)
    min_outer_dist = geom.get("outer_corners_min_dist", 5.0)
    outer_ratio = geom.get("outer_corners_ratio", 0.65)
    min_face_pts = geom.get("face_width_min_points", 8)
    min_face_w = geom.get("face_width_min_px", 10.0)
    min_result_px = geom.get("face_width_min_result_px", 12.0)
    width_ratio = geom.get("face_width_ratio", 0.42)
    default_iod = float(geom.get("default_iod_px", 96.0))

    def _eye_pts(kpts: dict[int, dict[str, Any]] | None, start: int, end: int) -> list[tuple[float, float]]:
        if not kpts:
            return []
        return [
            (float(kpts[i]["x"]), float(kpts[i]["y"]))
            for i in range(start, end)
            if i in kpts and kpts[i].get("x") is not None and kpts[i].get("y") is not None
        ]

    # 1. Tâm mắt từ Model
    if model_kpts:
        l_m = _eye_pts(model_kpts, 14, 22)
        r_m = _eye_pts(model_kpts, 22, 30)
        if len(l_m) >= min_eye_pts and len(r_m) >= min_eye_pts:
            iod_m = euclidean_dist(compute_centroid(l_m), compute_centroid(r_m))
            if iod_m > min_center_dist:
                return (iod_m, "model")

    # 2. Tâm mắt từ Human
    if human_kpts:
        l_h = _eye_pts(human_kpts, 14, 22)
        r_h = _eye_pts(human_kpts, 22, 30)
        if len(l_h) >= min_eye_pts and len(r_h) >= min_eye_pts:
            iod_h = euclidean_dist(compute_centroid(l_h), compute_centroid(r_h))
            if iod_h > min_center_dist:
                return (iod_h, "human")

    # 3. Khóe ngoài từ Model rồi Human
    for kpts, src in ((model_kpts, "model"), (human_kpts, "human")):
        if kpts:
            p14 = kpts.get(14)
            p26 = kpts.get(26)
            if p14 and p26 and p14.get("x") is not None and p26.get("x") is not None:
                d = euclidean_dist((float(p14["x"]), float(p14["y"])), (float(p26["x"]), float(p26["y"])))
                if d > min_outer_dist:
                    return (d * outer_ratio, src)

    # 4. Bounding box từ Model rồi Human
    for kpts in (model_kpts, human_kpts):
        if kpts:
            valid = [
                (float(kp["x"]), float(kp["y"]))
                for kp in kpts.values()
                if kp.get("state") != "outside" and kp.get("x") is not None and kp.get("y") is not None
            ]
            if len(valid) >= min_face_pts:
                fw = max(p[0] for p in valid) - min(p[0] for p in valid)
                if fw > min_face_w:
                    return (max(min_result_px, fw * width_ratio), "bbox")

    # 5. Default
    return (default_iod, "default")


def point_to_segment_distance(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    """Khoảng cách Euclid ngắn nhất từ điểm point tới đoạn thẳng seg_start-seg_end."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return euclidean_dist(point, seg_start)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (x1 + t * dx, y1 + t * dy)
    return euclidean_dist(point, proj)


def min_distance_to_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> float:
    """Khoảng cách ngắn nhất từ điểm point tới đường bao của đa giác polygon."""
    n = len(polygon)
    if n < 2:
        return 0.0
    return min(
        point_to_segment_distance(point, polygon[i], polygon[(i + 1) % n])
        for i in range(n)
    )


def compute_face_bbox(keypoints_by_id: dict[int, dict[str, Any]]) -> tuple[float, float, float, float]:
    """Tính bounding box (min_x, min_y, width, height) của các điểm hợp lệ."""
    valid_pts = [
        (float(kp["x"]), float(kp["y"]))
        for kp in keypoints_by_id.values()
        if kp.get("state") != "outside" and kp.get("x") is not None and kp.get("y") is not None
    ]
    if not valid_pts:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(p[0] for p in valid_pts)
    max_x = max(p[0] for p in valid_pts)
    min_y = min(p[1] for p in valid_pts)
    max_y = max(p[1] for p in valid_pts)
    return (min_x, min_y, max_x - min_x, max_y - min_y)



def compute_head_tilt_angle_rad(keypoints_by_id: dict[int, dict[str, Any]]) -> float:
    """
    Tính góc nghiêng của đầu (radian) theo đường nối 2 mắt.
    Theo Mục 6.6 GL: Góc nghiêng đường nối hai mắt dao động từ -21 độ đến +5 độ.
    """
    left_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(14, 22)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("y") is not None
    ]
    right_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(22, 30)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("y") is not None
    ]

    if len(left_eye_pts) >= 4 and len(right_eye_pts) >= 4:
        c_left = compute_centroid(left_eye_pts)
        c_right = compute_centroid(right_eye_pts)
        dx = c_right[0] - c_left[0]
        dy = c_right[1] - c_left[1]
        return math.atan2(dy, dx)

    return 0.0


def rotate_point(pt: tuple[float, float], center: tuple[float, float], angle_rad: float) -> tuple[float, float]:
    """Xoay điểm pt quanh center một góc angle_rad."""
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    rx = dx * cos_a - dy * sin_a + center[0]
    ry = dx * sin_a + dy * cos_a + center[1]
    return (rx, ry)


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Kiểm tra xem 2 đoạn thẳng p1-p2 và p3-p4 có cắt nhau hay không."""
    def ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Nếu chung điểm đầu/cuối thì không coi là tự cắt
    if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
        return False

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def has_self_intersection(contour_pts: list[tuple[float, float]]) -> bool:
    """Kiểm tra đa giác kín có bị tự cắt (ví dụ tạo thành hình chữ X) hay không."""
    n = len(contour_pts)
    if n < 4:
        return False

    edges = [(contour_pts[i], contour_pts[(i + 1) % n]) for i in range(n)]
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            # Không xét 2 cạnh kề nhau (vì chúng có chung đỉnh)
            if abs(i - j) == 1 or abs(i - j) == n - 1:
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray casting kiểm tra điểm point có nằm bên trong polygon hay không."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
