from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

from app.adapters.yolo_pose import YoloPoseAdapter
from app.services.ensemble import (
    ALPHA,
    COCO_KEYPOINT_SIGMAS,
    DEFAULT_KEYPOINT_SIGMA,
    DEFAULT_MODEL_WEIGHTS,
    LAMBDA,
    PAIR_LOOKUP,
    TAU,
    compute_disagreement,
    compute_oks_suspicion,
    compute_person_scale,
    compute_ranking_score,
    compute_reference_point,
    compute_reliability,
    compute_rho,
    filter_consensus_models,
    get_cached_prediction,
    match_person_for_model,
)

logger = logging.getLogger(__name__)


def setup_predictors(
    target_mode: str,
) -> tuple[dict[str, Any], str]:
    """
    Khởi tạo các mô hình dự đoán theo chế độ cấu hình.
    
    Hỗ trợ:
      - 'K2': yolo26s-pose + rtmpose-m (mặc định)
      - 'K1': yolo26s-pose (chế độ nhanh)
      - 'K3': yolo26s-pose + rtmpose-m + yolov8m-pose (tùy chọn)
      
    Nếu RTMPose gặp lỗi, tự động fallback về 'K1_fallback'.
    """
    predictors: dict[str, Any] = {}
    actual_mode = target_mode

    # 1. Luôn nạp primary model: yolo26s-pose
    try:
        predictors["yolo26s-pose"] = YoloPoseAdapter(model_name="yolo26s-pose.pt")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo YOLO26s: {e}")
        raise e

    if target_mode == "K1":
        return predictors, "K1"

    # 2. Nạp RTMPose cho K=2 hoặc K=3
    if target_mode in ("K2", "K3"):
        try:
            from app.adapters.rtm_pose import RtmPosePredictor
            predictors["rtmpose-m"] = RtmPosePredictor(mode="balanced")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo RTMPose ({e}). Tự động fallback về K=1 (K1_fallback).")
            actual_mode = "K1_fallback"

    # 3. Nạp YOLOv8m nếu cấu hình K=3
    if target_mode == "K3" and actual_mode != "K1_fallback":
        try:
            from app.adapters.yolo8m_pose import Yolo8mPosePredictor
            predictors["yolov8m-pose"] = Yolo8mPosePredictor(model_name="yolov8m-pose.pt")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo YOLOv8m ({e}). Tiếp tục với các model khả dụng.")

    return predictors, actual_mode


def run_scoring(
    run_id: str,
    db_factory: Callable[[], sqlite3.Connection],
    schema_loader: Callable[[str], dict],
    mode: str | None = None,
) -> None:
    """
    Thuật toán chấm điểm nghi ngờ Ensemble cho run:
    1. Lấy cấu hình chế độ (mặc định K=2: yolo26s + rtmpose-m; K=1: nhanh; K=3: tùy chọn)
    2. Chạy tuần tự các model (sử dụng in-memory cache)
    3. Ghép người theo tâm nhãn & lọc bounding box không đồng thuận (IoU >= 0.30)
    4. Tính vị trí tham chiếu p*, độ bất đồng delta, sai số OKS e, độ tin cậy R và ranking score = e * R^alpha
    5. Phát hiện lỗi hoán đổi trái/phải (swap_error) và xử lý điểm bị che khuất (occluded)
    6. Lưu cảnh báo kèm R, delta vào DB và cập nhật chế độ đã dùng vào run.
    """
    # Xác định chế độ mục tiêu
    target_mode = mode or os.environ.get("QA_ENSEMBLE_MODE", "K2").upper()
    if target_mode not in ("K1", "K2", "K3"):
        target_mode = "K2"

    predictors, actual_mode = setup_predictors(target_mode)
    K_target = 2 if target_mode == "K2" else (3 if target_mode == "K3" else 1)

    with db_factory() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],)).fetchone()
        conn.execute(
            "UPDATE runs SET status = 'running', progress = 5, mode = ? WHERE id = ?",
            (actual_mode, run_id),
        )

        schema = schema_loader(dataset["schema_id"])
        image_rows = conn.execute("""
            SELECT i.id, i.file_name, i.width, i.height, a.id AS annotation_id, a.keypoints_json
            FROM images i
            JOIN annotations a ON a.image_id = i.id
            WHERE i.dataset_id = ?
            ORDER BY i.file_name
        """, (dataset["id"],)).fetchall()

    storage_root = Path(dataset["storage_path"]).resolve()
    keypoint_meta = {kp["id"]: kp for kp in schema["keypoints"]}
    total_images = len(image_rows)

    warnings_to_insert = []

    for idx, row in enumerate(image_rows):
        file_name = row["file_name"]
        annot_id = row["annotation_id"]
        image_path = storage_root / file_name
        if not image_path.is_file():
            continue

        try:
            human_points = json.loads(row["keypoints_json"])
        except Exception:
            continue

        human_dict = {pt["id"]: pt for pt in human_points if isinstance(pt, dict) and "id" in pt}

        # 1. Dự đoán tuần tự từ các model khả dụng
        model_matches: dict[str, dict[str, Any] | None] = {}
        for m_name, predictor in predictors.items():
            try:
                preds = get_cached_prediction(image_path, m_name, predictor.predict)
                matched = match_person_for_model(human_points, preds)
                model_matches[m_name] = matched
            except Exception as e:
                logger.warning(f"Lỗi dự đoán model {m_name} trên ảnh {file_name}: {e}")
                model_matches[m_name] = None

        # 2. Lọc đồng thuận Bounding Box (loại trừ model bắt nhầm người khác)
        valid_matches, K_eff = filter_consensus_models(
            model_matches,
            primary_model="yolo26s-pose",
            iou_threshold=0.30,
        )

        if not valid_matches:
            continue

        # Lấy scale từ match chuẩn
        ref_match = valid_matches.get("yolo26s-pose") or next(iter(valid_matches.values()))
        scale = compute_person_scale(ref_match.get("bbox"), row["width"] or 1000, row["height"] or 1000)
        rho_val = compute_rho(K_eff, K_target)

        # 3. Duyệt từng keypoint người gán
        for pt_id, h_pt in human_dict.items():
            state = h_pt.get("state")
            hx = h_pt.get("x")
            hy = h_pt.get("y")

            # Bỏ qua nếu 'outside' hoặc tọa độ rỗng
            if state == "outside" or hx is None or hy is None:
                continue

            # Thu thập quan sát từ các model hợp lệ: (x, y, conf, weight)
            observations: list[tuple[float, float, float, float]] = []
            for m_name, match_data in valid_matches.items():
                m_kpts = match_data.get("keypoints", {})
                pt_obs = m_kpts.get(pt_id)
                if pt_obs and pt_obs.get("x") is not None and pt_obs.get("y") is not None:
                    px_val = float(pt_obs["x"])
                    py_val = float(pt_obs["y"])
                    c_val = float(pt_obs.get("conf", 1.0))
                    w_val = DEFAULT_MODEL_WEIGHTS.get(m_name, 1000.0)
                    observations.append((px_val, py_val, c_val, w_val))

            if not observations:
                continue

            # 4. Tính toán các chỉ số toán học Ensemble
            p_star = compute_reference_point(observations)
            delta_val = compute_disagreement(observations, p_star, scale)

            dist = math.hypot(hx - p_star[0], hy - p_star[1])
            sigma_i = COCO_KEYPOINT_SIGMAS.get(pt_id, DEFAULT_KEYPOINT_SIGMA)

            # Tính sai số OKS e_i
            # Khi K=1 hoặc delta=0, mẫu số thuần túy là 2 * (2*sigma_i)^2
            oks_e = compute_oks_suspicion(dist, scale, delta_val, sigma_i, lam=LAMBDA)

            # Tính độ tin cậy R_i
            sum_weights = sum(w for _, _, _, w in observations)
            c_bar = (
                sum(w * conf for _, _, conf, w in observations) / sum_weights
                if sum_weights > 0
                else 1.0
            )
            rel_R = compute_reliability(c_bar, delta_val, tau=TAU, rho=rho_val)

            # Tính điểm xếp hạng: score = e * R^alpha (ở K=1: score = e * conf^alpha)
            suspicion = compute_ranking_score(oks_e, rel_R, alpha=ALPHA)
            warning_type = "suspected_error"

            # 5. Kiểm tra hoán đổi Trái / Phải (Left-Right Swap)
            if pt_id in PAIR_LOOKUP:
                other_id = PAIR_LOOKUP[pt_id]
                other_h_pt = human_dict.get(other_id)

                if other_h_pt and other_h_pt.get("x") is not None and other_h_pt.get("y") is not None:
                    o_hx, o_hy = other_h_pt["x"], other_h_pt["y"]

                    # Lấy p* của khớp đối xứng nếu có
                    other_obs = []
                    for m_name, match_data in valid_matches.items():
                        o_kpts = match_data.get("keypoints", {})
                        o_pt_obs = o_kpts.get(other_id)
                        if o_pt_obs and o_pt_obs.get("x") is not None and o_pt_obs.get("y") is not None:
                            other_obs.append((
                                float(o_pt_obs["x"]),
                                float(o_pt_obs["y"]),
                                float(o_pt_obs.get("conf", 1.0)),
                                DEFAULT_MODEL_WEIGHTS.get(m_name, 1000.0),
                            ))

                    if other_obs:
                        other_p_star = compute_reference_point(other_obs)
                        orig_dist = dist + math.hypot(o_hx - other_p_star[0], o_hy - other_p_star[1])
                        swap_dist = (
                            math.hypot(hx - other_p_star[0], hy - other_p_star[1]) +
                            math.hypot(o_hx - p_star[0], o_hy - p_star[1])
                        )

                        if swap_dist < orig_dist * 0.6 and orig_dist > scale * 0.15:
                            warning_type = "swap_error"
                            suspicion = max(suspicion, 0.88)

            # 6. Xử lý điểm bị che khuất ('occluded')
            if state == "occluded" and warning_type != "swap_error":
                warning_type = "hard_case"
                suspicion = suspicion * 0.5  # Hạ mức ưu tiên

            # 7. Lưu cảnh báo nếu độ nghi ngờ >= 0.20
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
                    round(p_star[0], 1),
                    round(p_star[1], 1),
                    None,
                    annot_id,
                    round(rel_R, 3),
                    round(delta_val, 4),
                ))

        # Cập nhật tiến độ định kỳ
        current_progress = int(10 + ((idx + 1) / max(total_images, 1)) * 85)
        if idx % 5 == 0 or idx == total_images - 1:
            with db_factory() as conn:
                conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (current_progress, run_id))

    # Sắp xếp cảnh báo theo độ nghi ngờ giảm dần
    warnings_to_insert.sort(key=lambda w: w[5], reverse=True)

    with db_factory() as conn:
        for w in warnings_to_insert:
            conn.execute(
                """INSERT INTO warnings (
                    id, run_id, image_name, keypoint, warning_type, suspicion,
                    human_x, human_y, suggested_x, suggested_y, review, annotation_id,
                    reliability, delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                w,
            )
        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 100, mode = ? WHERE id = ?",
            (actual_mode, run_id),
        )
