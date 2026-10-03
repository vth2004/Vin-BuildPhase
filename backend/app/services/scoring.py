from __future__ import annotations

import json
import math
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

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

# Tạo tra cứu cặp đối xứng: point_id -> counterpart_id
PAIR_LOOKUP: dict[int, int] = {}
for r_id, l_id in SYMMETRIC_PAIRS:
    PAIR_LOOKUP[r_id] = l_id
    PAIR_LOOKUP[l_id] = r_id


def compute_person_scale(bbox: tuple[float, float, float, float] | None, image_w: int = 1000, image_h: int = 1000) -> float:
    """Tính scale chuẩn hóa dựa trên diện tích hoặc đường chéo bbox."""
    if bbox:
        x1, y1, x2, y2 = bbox
        w = max(10.0, x2 - x1)
        h = max(10.0, y2 - y1)
        return math.sqrt(w * h)
    return math.sqrt(image_w * image_h) * 0.3


def match_person(human_points: list[dict], predicted_persons: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ghép người gán với người model tìm thấy (dựa vào tâm cụm điểm keypoint)."""
    if not predicted_persons:
        return None
    if len(predicted_persons) == 1:
        return predicted_persons[0]

    # Tính tâm của các điểm người gán (chỉ xét điểm có tọa độ)
    valid_coords = [(pt["x"], pt["y"]) for pt in human_points if pt.get("x") is not None and pt.get("y") is not None]
    if not valid_coords:
        return predicted_persons[0]

    center_hx = sum(c[0] for c in valid_coords) / len(valid_coords)
    center_hy = sum(c[1] for c in valid_coords) / len(valid_coords)

    best_match = None
    min_dist = float("inf")

    for p in predicted_persons:
        if p.get("bbox"):
            x1, y1, x2, y2 = p["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = math.hypot(center_hx - cx, center_hy - cy)
            if dist < min_dist:
                min_dist = dist
                best_match = p

    return best_match or predicted_persons[0]


def run_scoring(run_id: str, db_factory: Callable[[], sqlite3.Connection], schema_loader: Callable[[str], dict]) -> None:
    """
    Thuật toán chấm điểm nghi ngờ thật cho run:
    1. Chạy YOLO-pose dự đoán lại keypoints
    2. Bỏ qua các điểm 'outside' (không so tọa độ)
    3. Tính khoảng cách Euclid chuẩn hóa theo scale của người
    4. Nhân với độ tin cậy của Model
    5. Xử lý 'occluded' (hạ mức ưu tiên, nhãn hard_case)
    6. Bắt lỗi hoán đổi trái/phải (swap_error)
    """
    from app.adapters.yolo_pose import YoloPoseAdapter

    with db_factory() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],)).fetchone()
        conn.execute("UPDATE runs SET status = 'running', progress = 5 WHERE id = ?", (run_id,))

        schema = schema_loader(dataset["schema_id"])
        image_rows = conn.execute("""
            SELECT i.id, i.file_name, i.width, i.height, a.keypoints_json
            FROM images i
            JOIN annotations a ON a.image_id = i.id
            WHERE i.dataset_id = ?
            ORDER BY i.file_name
        """, (dataset["id"],)).fetchall()

    storage_root = Path(dataset["storage_path"]).resolve()
    keypoint_meta = {kp["id"]: kp for kp in schema["keypoints"]}

    # Khởi tạo adapter (hiện tại hỗ trợ vf_humanpose17_v1)
    adapter = YoloPoseAdapter(model_name="yolo26s-pose.pt")
    total_images = len(image_rows)

    warnings_to_insert = []

    for idx, row in enumerate(image_rows):
        file_name = row["file_name"]
        image_path = storage_root / file_name
        if not image_path.is_file():
            continue

        try:
            human_points = json.loads(row["keypoints_json"])
        except Exception:
            continue

        human_dict = {pt["id"]: pt for pt in human_points if isinstance(pt, dict) and "id" in pt}

        # Model dự đoán
        try:
            preds = adapter.predict(image_path)
        except Exception as e:
            preds = []

        matched = match_person(human_points, preds)
        if not matched:
            continue

        model_kpts = matched.get("keypoints", {})
        scale = compute_person_scale(matched.get("bbox"), row["width"] or 1000, row["height"] or 1000)

        for pt_id, h_pt in human_dict.items():
            state = h_pt.get("state")
            hx = h_pt.get("x")
            hy = h_pt.get("y")

            # QUY TẮC 1: Nếu 'outside' hoặc tọa độ null -> BỎ QUA KHÔNG SO SÁNH
            if state == "outside" or hx is None or hy is None:
                continue

            m_pt = model_kpts.get(pt_id)
            if not m_pt or m_pt.get("x") is None or m_pt.get("y") is None:
                continue

            mx = float(m_pt["x"])
            my = float(m_pt["y"])
            conf = float(m_pt.get("conf", 1.0))

            dist = math.hypot(hx - mx, hy - my)
            dist_norm = dist / scale

            # Độ nghi ngờ thô (chuẩn hóa tỷ lệ lệch nhân độ tin cậy)
            raw_suspicion = min(1.0, dist_norm * 4.5) * conf

            warning_type = "suspected_error"
            suspicion = raw_suspicion

            # QUY TẮC 2: Kiểm tra hoán đổi Trái / Phải (Left-Right Swap)
            if pt_id in PAIR_LOOKUP:
                other_id = PAIR_LOOKUP[pt_id]
                other_h_pt = human_dict.get(other_id)
                other_m_pt = model_kpts.get(other_id)

                if (other_h_pt and other_h_pt.get("x") is not None and 
                    other_m_pt and other_m_pt.get("x") is not None):
                    o_hx, o_hy = other_h_pt["x"], other_h_pt["y"]
                    o_mx, o_my = other_m_pt["x"], other_m_pt["y"]

                    orig_dist = dist + math.hypot(o_hx - o_mx, o_hy - o_my)
                    swap_dist = math.hypot(hx - o_mx, hy - o_my) + math.hypot(o_hx - mx, o_hy - my)

                    # Nếu hoán đổi làm khoảng cách giảm đi rõ rệt
                    if swap_dist < orig_dist * 0.6 and orig_dist > scale * 0.15:
                        warning_type = "swap_error"
                        suspicion = max(suspicion, 0.88)

            # QUY TẮC 3: Xử lý điểm bị che khuất ('occluded')
            if state == "occluded" and warning_type != "swap_error":
                warning_type = "hard_case"
                suspicion = suspicion * 0.5  # Hạ mức ưu tiên

            # Chỉ lưu cảnh báo nếu độ nghi ngờ đủ đáng chú ý (>= 0.20)
            if suspicion >= 0.20:
                kp_name = keypoint_meta.get(pt_id, {}).get("name", f"point_{pt_id}")
                warnings_to_insert.append((
                    str(uuid.uuid4()),
                    run_id,
                    file_name,
                    kp_name,
                    warning_type,
                    round(suspicion, 3),
                    round(hx, 1),
                    round(hy, 1),
                    round(mx, 1),
                    round(my, 1),
                    None,
                ))

        # Cập nhật tiến độ định kỳ
        current_progress = int(10 + ((idx + 1) / max(total_images, 1)) * 85)
        if idx % 5 == 0 or idx == total_images - 1:
            with db_factory() as conn:
                conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (current_progress, run_id))

    # Ghi toàn bộ cảnh báo vào DB
    # Sắp xếp theo độ nghi ngờ giảm dần
    warnings_to_insert.sort(key=lambda w: w[5], reverse=True)

    with db_factory() as conn:
        for w in warnings_to_insert:
            conn.execute(
                "INSERT INTO warnings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                w,
            )
        conn.execute("UPDATE runs SET status = 'completed', progress = 100 WHERE id = ?", (run_id,))
