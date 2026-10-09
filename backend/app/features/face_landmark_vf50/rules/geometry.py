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

    if len(left_eye_pts) >= 4 and len(right_eye_pts) >= 4:
        c_left = compute_centroid(left_eye_pts)
        c_right = compute_centroid(right_eye_pts)
        iod = euclidean_dist(c_left, c_right)
        if iod > 10.0:
            return iod

    # Fallback nếu một trong 2 mắt bị che: dùng khoảng cách giữa 2 khóe ngoài 14 và 26
    p14 = keypoints_by_id.get(14)
    p26 = keypoints_by_id.get(26)
    if p14 and p26 and p14.get("x") is not None and p26.get("x") is not None:
        outer_dist = euclidean_dist((float(p14["x"]), float(p14["y"])), (float(p26["x"]), float(p26["y"])))
        if outer_dist > 5.0:
            return outer_dist * 0.65

    # Fallback thích ứng dựa trên bounding box thực tế của toàn bộ khuôn mặt:
    # Tránh sai số nghiêm trọng khi mặt ở xa hoặc ảnh có độ phân giải nhỏ
    all_valid_pts = [
        (float(kp["x"]), float(kp["y"]))
        for kp in keypoints_by_id.values()
        if kp.get("state") != "outside" and kp.get("x") is not None and kp.get("y") is not None
    ]
    if len(all_valid_pts) >= 8:
        min_x = min(p[0] for p in all_valid_pts)
        max_x = max(p[0] for p in all_valid_pts)
        face_width = max_x - min_x
        if face_width > 10.0:
            return max(12.0, face_width * 0.42)

    return 96.0  # Mặc định theo phân vị trung vị của GL


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
