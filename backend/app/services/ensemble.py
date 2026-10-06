from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

# =========================================================================
# 1. HẰNG SỐ & SIÊU THAM SỐ (TỐI ƯU HÓA TRÊN DEV SET COCO VAL2017)
# =========================================================================
LAMBDA: float = 0.5   # Hệ số điều hòa dung sai bất đồng trong mẫu số e_i
TAU: float = 0.08      # Ngưỡng dung sai bất đồng trong độ tin cậy R_i
ALPHA: float = 0.5    # Số mũ điều hòa độ tin cậy trong điểm xếp hạng score = e * R^alpha

# Trọng số nghịch bình phương MPE fit trên Dev: w_k = 1 / MPE_k^2
DEFAULT_MODEL_WEIGHTS: dict[str, float] = {
    "yolo26s-pose": 2026.5,  # MPE_dev = 0.0222 -> w = 1.000 (chuẩn hóa)
    "rtmpose-m": 2854.5,     # MPE_dev = 0.0187 -> w = 1.409
    "yolov8m-pose": 1827.1,  # MPE_dev = 0.0234 -> w = 0.902
}

# Các cặp khớp đối xứng trái - phải trong vf_humanpose17_v1: (Right_ID, Left_ID)
SYMMETRIC_PAIRS: list[tuple[int, int]] = [
    (2, 3),    # r_eye, l_eye
    (4, 5),    # r_ear, l_ear
    (6, 7),    # r_shoulder, l_shoulder
    (8, 9),    # r_elbow, l_elbow
    (10, 11),  # r_wrist, l_wrist
    (12, 13),  # r_hip, l_hip
    (14, 15),  # r_knee, l_knee
    (16, 17),  # r_ankle, l_ankle
]

PAIR_LOOKUP: dict[int, int] = {}
for r_id, l_id in SYMMETRIC_PAIRS:
    PAIR_LOOKUP[r_id] = l_id
    PAIR_LOOKUP[l_id] = r_id

# Hệ số dung sai giải phẫu chuẩn COCO (OKS per-keypoint standard deviations sigma_i)
# id 1..17 theo chuẩn VinFast vf_humanpose17_v1
COCO_KEYPOINT_SIGMAS: dict[int, float] = {
    1: 0.026,   # nose
    2: 0.025,   # r_eye
    3: 0.025,   # l_eye
    4: 0.035,   # r_ear
    5: 0.035,   # l_ear
    6: 0.079,   # r_shoulder
    7: 0.079,   # l_shoulder
    8: 0.072,   # r_elbow
    9: 0.072,   # l_elbow
    10: 0.062,  # r_wrist
    11: 0.062,  # l_wrist
    12: 0.107,  # r_hip
    13: 0.107,  # l_hip
    14: 0.087,  # r_knee
    15: 0.087,  # l_knee
    16: 0.089,  # r_ankle
    17: 0.089,  # l_ankle
}
DEFAULT_KEYPOINT_SIGMA: float = 0.070


# =========================================================================
# 2. BỘ NHỚ ĐỆM DỰ ĐOÁN (IN-MEMORY PREDICTION CACHE)
# =========================================================================
# Cache theo (image_path_str, model_name, mtime) để tránh dự đoán lặp lại
_PREDICTION_CACHE: dict[tuple[str, str, float], list[dict[str, Any]]] = {}


def get_cached_prediction(
    image_path: Path,
    model_name: str,
    predict_fn: Callable[[Path], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Lấy kết quả dự đoán từ bộ nhớ đệm hoặc tính mới."""
    try:
        mtime = image_path.stat().st_mtime
    except Exception:
        mtime = 0.0

    key = (str(image_path.resolve()), model_name, mtime)
    if key in _PREDICTION_CACHE:
        return _PREDICTION_CACHE[key]

    preds = predict_fn(image_path)
    _PREDICTION_CACHE[key] = preds
    return preds


def clear_prediction_cache() -> None:
    """Xóa sạch bộ nhớ đệm."""
    _PREDICTION_CACHE.clear()


# =========================================================================
# 3. HÌNH HỌC & GHÉP NGƯỜI (PERSON MATCHING & IOU CONSENSUS)
# =========================================================================
def compute_person_scale(
    bbox: tuple[float, float, float, float] | list[float] | None,
    image_w: int = 1000,
    image_h: int = 1000,
) -> float:
    """Tính scale chuẩn hóa dựa trên diện tích căn bậc 2 của bbox s = sqrt(W*H)."""
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        w = max(10.0, x2 - x1)
        h = max(10.0, y2 - y1)
        return math.sqrt(w * h)
    return math.sqrt(image_w * image_h) * 0.3


def compute_bbox_iou(
    box1: tuple[float, float, float, float] | list[float] | None,
    box2: tuple[float, float, float, float] | list[float] | None,
) -> float:
    """Tính chỉ số IoU giữa 2 bounding box (x1, y1, x2, y2)."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def match_person_for_model(
    human_points: list[dict],
    predicted_persons: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Ghép người gán với người do một model phát hiện dựa vào khoảng cách tâm."""
    if not predicted_persons:
        return None
    if len(predicted_persons) == 1:
        return predicted_persons[0]

    valid_coords = [(pt["x"], pt["y"]) for pt in human_points if pt.get("x") is not None and pt.get("y") is not None]
    if not valid_coords:
        return predicted_persons[0]

    center_hx = sum(c[0] for c in valid_coords) / len(valid_coords)
    center_hy = sum(c[1] for c in valid_coords) / len(valid_coords)

    best_match = None
    min_dist = float("inf")

    for p in predicted_persons:
        if p.get("bbox"):
            x1, y1, x2, y2 = p["bbox"][:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        else:
            kpts = [
                (kp["x"], kp["y"])
                for kp in p.get("keypoints", {}).values()
                if kp.get("x") is not None and kp.get("y") is not None
            ]
            if not kpts:
                continue
            cx = sum(k[0] for k in kpts) / len(kpts)
            cy = sum(k[1] for k in kpts) / len(kpts)

        dist = math.hypot(center_hx - cx, center_hy - cy)
        if dist < min_dist:
            min_dist = dist
            best_match = p

    return best_match or predicted_persons[0]


def filter_consensus_models(
    model_matches: dict[str, dict[str, Any] | None],
    primary_model: str = "yolo26s-pose",
    iou_threshold: float = 0.30,
) -> tuple[dict[str, dict[str, Any]], int]:
    """
    Kiểm tra đồng thuận Bounding Box giữa các model để lọc trường hợp bắt nhầm người khác (ID switch).
    
    Trả về:
        (valid_matches: dict[str, dict], K_eff: int)
    """
    # Lọc các model có dự đoán
    active = {m: match for m, match in model_matches.items() if match is not None}
    if not active:
        return {}, 0

    if len(active) == 1:
        return active, 1

    # Lấy bbox của primary model làm chuẩn đối chiếu (hoặc model đầu tiên nếu primary vắng mặt)
    ref_model = primary_model if primary_model in active else next(iter(active.keys()))
    ref_box = active[ref_model].get("bbox")

    valid_matches: dict[str, dict[str, Any]] = {ref_model: active[ref_model]}

    for m, match in active.items():
        if m == ref_model:
            continue
        cur_box = match.get("bbox")
        if ref_box and cur_box:
            iou = compute_bbox_iou(ref_box, cur_box)
            if iou >= iou_threshold:
                valid_matches[m] = match
            else:
                # Bbox không đồng thuận -> model m có thể đã bắt nhầm người khác trong ảnh đông người
                pass
        else:
            # Nếu một trong 2 không có bbox, vẫn giữ lại
            valid_matches[m] = match

    return valid_matches, len(valid_matches)


# =========================================================================
# 4. CÔNG THỨC TOÁN ENSEMBLE (p*, delta, e, R, Score)
# =========================================================================
def compute_reference_point(
    observations: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """
    Tính vị trí tham chiếu trọng số p* = (px, py).
    Mỗi phần tử trong observations là (x, y, conf, weight).
    """
    if not observations:
        return (0.0, 0.0)

    sum_weight = sum(w * conf for _, _, conf, w in observations)
    if sum_weight <= 1e-9:
        # Nếu tất cả conf = 0, chia đều
        sum_w_raw = sum(w for _, _, _, w in observations)
        if sum_w_raw <= 1e-9:
            return (observations[0][0], observations[0][1])
        px = sum(w * x for x, _, _, w in observations) / sum_w_raw
        py = sum(w * y for _, y, _, w in observations) / sum_w_raw
        return (px, py)

    px = sum(w * conf * x for x, _, conf, w in observations) / sum_weight
    py = sum(w * conf * y for _, y, conf, w in observations) / sum_weight
    return (px, py)


def compute_disagreement(
    observations: list[tuple[float, float, float, float]],
    p_star: tuple[float, float],
    scale: float,
) -> float:
    """
    Tính độ phân tán không gian chuẩn hóa giữa các model (delta).
    delta_i = (1 / s) * sqrt( sum(w * c * ||p_k - p*||^2) / sum(w * c) )
    """
    if len(observations) <= 1 or scale <= 1e-6:
        return 0.0

    px, py = p_star
    sum_weight = sum(w * conf for _, _, conf, w in observations)
    if sum_weight <= 1e-9:
        return 0.0

    disp = sum(
        w * conf * ((x - px) ** 2 + (y - py) ** 2)
        for x, y, conf, w in observations
    ) / sum_weight

    return math.sqrt(max(0.0, disp)) / scale


def compute_oks_suspicion(
    dist_to_pstar: float,
    scale: float,
    delta: float,
    sigma_i: float,
    lam: float = LAMBDA,
) -> float:
    """
    Tính sai số nghi ngờ chuẩn hóa OKS e_i:
    e_i = 1 - exp( - (d/s)^2 / ( 2 * ( (2*sigma_i)^2 + lambda * delta^2 ) ) )
    """
    if scale <= 1e-6:
        return 0.0

    d_norm = dist_to_pstar / scale
    k_i = 2.0 * sigma_i
    denom = 2.0 * (k_i ** 2 + lam * (delta ** 2))

    if denom <= 1e-9:
        return 1.0

    return 1.0 - math.exp(- (d_norm ** 2) / denom)


def compute_reliability(
    c_bar: float,
    delta: float,
    tau: float = TAU,
    rho: float = 1.0,
) -> float:
    """
    Tính độ tin cậy của đánh giá R_i:
    R_i = rho(K_eff) * c_bar * exp( - delta^2 / (2 * tau^2) )
    """
    if tau <= 1e-6:
        return 0.0

    exp_term = math.exp(- (delta ** 2) / (2.0 * (tau ** 2)))
    r_val = rho * c_bar * exp_term
    return min(1.0, max(0.0, r_val))


def compute_ranking_score(
    e: float,
    r: float,
    alpha: float = ALPHA,
) -> float:
    """
    Tính điểm xếp hạng tổng hợp score_i = e_i * (R_i ^ alpha).
    """
    r_bounded = min(1.0, max(0.0, r))
    score = e * (r_bounded ** alpha)
    return min(1.0, max(0.0, score))


def compute_rho(K_eff: int, K_target: int) -> float:
    """
    Tính hệ số phạt độ tin cậy khi số model thực tế K_eff ít hơn số model mục tiêu K_target.
    Ví dụ: K_target = 2, K_eff = 1 -> rho = 1/2 = 0.5.
    """
    if K_target <= 1:
        return 1.0
    return min(1.0, max(0.1, K_eff / K_target))
