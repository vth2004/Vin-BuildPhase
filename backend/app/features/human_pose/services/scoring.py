from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.core.database import resolve_data_path

from .observations import build_pose_observations
from app.shared.ai_drafts import apply_temporal_prediction_qc
from app.shared.temporal.tracking import DetectionTracker, assign_one_to_one, bbox_iou, item_center, item_scale
from app.shared.temporal.segmentation import is_scene_cut, luminance_histogram
from app.shared.temporal.service import process_temporal_run

from ..adapters.yolo_pose import YoloPoseAdapter
from .ensemble import (
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


def _annotation_item(points: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [(float(point["x"]), float(point["y"])) for point in points
             if point.get("x") is not None and point.get("y") is not None and point.get("state") != "outside"]
    bbox = (min(x for x, _ in valid), min(y for _, y in valid),
            max(x for x, _ in valid), max(y for _, y in valid)) if valid else None
    return {"bbox": bbox, "keypoints": {point["id"]: point for point in points if "id" in point}}


def _match_pose_annotations(rows: list[Any], observations: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    annotation_items: list[dict[str, Any]] = []
    valid_rows: list[Any] = []
    for row in rows:
        try:
            points = json.loads(row["keypoints_json"])
        except Exception:
            continue
        annotation_items.append(_annotation_item(points))
        valid_rows.append(row)
    costs: list[list[float]] = []
    for annotation in annotation_items:
        row_costs = []
        for observation in observations:
            ann_center, obs_center = item_center(annotation), item_center(observation)
            center_cost = math.dist(ann_center, obs_center) / max(item_scale(annotation), item_scale(observation)) if ann_center and obs_center else 1e6
            iou_cost = 1.0 - bbox_iou(annotation.get("bbox"), observation.get("bbox"))
            row_costs.append(0.65 * center_cost + 0.35 * iou_cost)
        costs.append(row_costs)
    assignment = assign_one_to_one(costs, 1.25, ambiguity_margin=0.08) if costs else None
    matched = {valid_rows[item.left_index]["annotation_id"]: observations[item.right_index]
               for item in assignment.matches} if assignment else {}
    uncertain = {valid_rows[index]["annotation_id"] for index in assignment.uncertain_left} if assignment else set()
    return matched, uncertain


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
            from ..adapters.rtm_pose import RtmPosePredictor
            predictors["rtmpose-m"] = RtmPosePredictor(mode="balanced")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo RTMPose ({e}). Tự động fallback về K=1 (K1_fallback).")
            actual_mode = "K1_fallback"

    # 3. Nạp YOLOv8m nếu cấu hình K=3
    if target_mode == "K3" and actual_mode != "K1_fallback":
        try:
            from ..adapters.yolo8m_pose import Yolo8mPosePredictor
            predictors["yolov8m-pose"] = Yolo8mPosePredictor(model_name="yolov8m-pose.pt")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo YOLOv8m ({e}). Tiếp tục với các model khả dụng.")

    return predictors, actual_mode


def _run_unlabeled_pose(
    run_id: str,
    dataset: Any,
    run: Any,
    predictors: dict[str, Any],
    db_factory: Callable[[], sqlite3.Connection],
) -> None:
    with db_factory() as conn:
        images = conn.execute(
            """SELECT id, file_name, width, height, sequence_id, original_frame_index
                 FROM images WHERE dataset_id = ?
                ORDER BY COALESCE(sequence_id, ''), COALESCE(original_frame_index, 2147483647), file_name""",
            (dataset["id"],),
        ).fetchall()
        conn.execute("DELETE FROM ai_draft_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM temporal_observations WHERE run_id = ?", (run_id,))
    root = resolve_data_path(dataset["storage_path"])
    trackers: dict[str, DetectionTracker] = {}
    previous: dict[str, tuple[int | None, int | None, int | None]] = {}
    for position, image in enumerate(images):
        path = (root / image["file_name"]).resolve()
        predictions: dict[str, list[dict[str, Any]]] = {}
        for model_name, predictor in predictors.items():
            try:
                predictions[model_name] = get_cached_prediction(path, model_name, predictor.predict)
            except Exception as error:
                logger.warning("Pose AI-QC prediction failed for %s on %s: %s", model_name, image["file_name"], error)
                predictions[model_name] = []
        observations = build_pose_observations(predictions)
        sequence_key = image["sequence_id"] or f"legacy:{dataset['id']}"
        prior = previous.get(sequence_key)
        frame_index = image["original_frame_index"]
        discontinuity = bool(prior and (
            frame_index is None or prior[0] is None or frame_index - prior[0] != 1
            or prior[1] != image["width"] or prior[2] != image["height"]
        ))
        tracked = trackers.setdefault(sequence_key, DetectionTracker()).update(observations, discontinuity=discontinuity)
        previous[sequence_key] = (frame_index, image["width"], image["height"])
        now = int(time.time())
        with db_factory() as conn:
            for instance_index, item in enumerate(tracked):
                points = [{
                    "id": int(point_id), "name": str(point_id),
                    "x": point.get("x"), "y": point.get("y"), "state": "visible",
                    "confidence": point.get("conf"), "visibility_confirmed": False,
                } for point_id, point in sorted(item["keypoints"].items())]
                flags = []
                if item.get("tracking_uncertain"):
                    flags.append("tracking_uncertain")
                if float(item.get("ensemble_disagreement", 0.0)) >= 0.15:
                    flags.append("prediction_instability")
                provenance = {
                    "label": "AI draft", "observation_source": item["observation_source"],
                    "supporting_models": item["supporting_models"],
                    "supporting_model_count": item["supporting_model_count"],
                    "algorithm_version": "pose-ensemble-ai-draft-v1",
                }
                conn.execute(
                    """INSERT INTO ai_draft_results (
                         id, run_id, image_id, schema_id, instance_index, internal_track_id,
                         keypoints_json, provenance_json, qc_json, visibility_confirmed, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (str(uuid.uuid4()), run_id, image["id"], dataset["schema_id"], instance_index,
                     item.get("internal_track_id"), json.dumps(points), json.dumps(provenance),
                     json.dumps({"flags": flags, "ensemble_disagreement": item.get("ensemble_disagreement", 0.0)}), now),
                )
                if run["temporal_enabled"]:
                    conn.execute(
                        """INSERT INTO temporal_observations (
                             id, run_id, sequence_id, segment_id, internal_track_id, image_id, detection_index,
                             bbox_json, keypoints_json, observation_source, supporting_model_count,
                             supporting_models_json, ensemble_disagreement, tracking_status, raw_keypoints_json
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), run_id, image["sequence_id"], item["segment_id"], item["internal_track_id"],
                         image["id"], instance_index, json.dumps(item.get("bbox")), json.dumps(item["keypoints"]),
                         item["observation_source"], item["supporting_model_count"], json.dumps(item["supporting_models"]),
                         item["ensemble_disagreement"], "uncertain" if item.get("tracking_uncertain") else "confirmed",
                         json.dumps(item.get("raw_keypoints", {}))),
                    )
            progress = int(5 + ((position + 1) / max(len(images), 1)) * 94)
            conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (progress, run_id))
    if run["temporal_enabled"]:
        apply_temporal_prediction_qc(run_id, db_factory)
    with db_factory() as conn:
        conn.execute("UPDATE runs SET status = 'completed', progress = 100, error_message = NULL WHERE id = ?", (run_id,))


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
        if run["temporal_enabled"]:
            conn.execute("DELETE FROM temporal_observations WHERE run_id = ?", (run_id,))
            conn.execute("UPDATE annotations SET internal_track_id = NULL, tracking_status = 'untracked' WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?)",
                         (dataset["id"],))

        schema = schema_loader(dataset["schema_id"])
        image_rows = conn.execute("""
            SELECT i.id, i.file_name, i.width, i.height, i.sequence_id, i.original_frame_index,
                   a.id AS annotation_id, a.keypoints_json
            FROM images i
            JOIN annotations a ON a.image_id = i.id
            WHERE i.dataset_id = ?
            ORDER BY COALESCE(i.sequence_id, ''), COALESCE(i.original_frame_index, 2147483647), i.file_name, a.id
        """, (dataset["id"],)).fetchall()

    if run["input_mode"] == "ai_qc" and not image_rows:
        _run_unlabeled_pose(run_id, dataset, run, predictors, db_factory)
        return

    storage_root = resolve_data_path(dataset["storage_path"])
    keypoint_meta = {kp["id"]: kp for kp in schema["keypoints"]}
    total_images = len(image_rows)

    warnings_to_insert = []
    rows_by_image: dict[str, list[Any]] = {}
    for image_row in image_rows:
        rows_by_image.setdefault(image_row["id"], []).append(image_row)
    frame_cache: dict[str, tuple[dict[str, dict[str, Any]], set[str]]] = {}
    trackers: dict[str, DetectionTracker] = {}
    previous_frame: dict[str, tuple[int | None, int | None, int | None]] = {}
    previous_histogram: dict[str, Any] = {}

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

        model_matches: dict[str, dict[str, Any] | None]
        if not run["temporal_enabled"]:
            model_matches = {}
            for m_name, predictor in predictors.items():
                try:
                    predictions = get_cached_prediction(image_path, m_name, predictor.predict)
                    model_matches[m_name] = match_person_for_model(human_points, predictions)
                except Exception as error:
                    logger.warning("Lỗi dự đoán model %s trên ảnh %s: %s", m_name, file_name, error)
                    model_matches[m_name] = None
        elif row["id"] not in frame_cache:
            predictions: dict[str, list[dict[str, Any]]] = {}
            for m_name, predictor in predictors.items():
                try:
                    predictions[m_name] = get_cached_prediction(image_path, m_name, predictor.predict)
                except Exception as e:
                    logger.warning(f"Lỗi dự đoán model {m_name} trên ảnh {file_name}: {e}")
                    predictions[m_name] = []
            observations = build_pose_observations(predictions)
            sequence_key = row["sequence_id"] or f"legacy:{dataset['id']}"
            frame_index = row["original_frame_index"]
            prior = previous_frame.get(sequence_key)
            try:
                from PIL import Image
                with Image.open(image_path) as source_image:
                    histogram = luminance_histogram(source_image.convert("L"))
            except Exception:
                histogram = None
            discontinuity = bool(prior and (
                frame_index is None or prior[0] is None or frame_index - prior[0] != 1
                or prior[1] != row["width"] or prior[2] != row["height"]
            )) or bool(histogram is not None and is_scene_cut(previous_histogram.get(sequence_key), histogram))
            tracked = trackers.setdefault(sequence_key, DetectionTracker()).update(observations, discontinuity=discontinuity)
            previous_frame[sequence_key] = (frame_index, row["width"], row["height"])
            if histogram is not None:
                previous_histogram[sequence_key] = histogram
            with db_factory() as conn:
                for detection_index, item in enumerate(tracked):
                    conn.execute(
                        """INSERT INTO temporal_observations (
                             id, run_id, sequence_id, segment_id, internal_track_id, image_id, detection_index,
                             bbox_json, keypoints_json, observation_source, supporting_model_count,
                             supporting_models_json, ensemble_disagreement, tracking_status, raw_keypoints_json
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), run_id, row["sequence_id"], item["segment_id"], item["internal_track_id"],
                         row["id"], detection_index, json.dumps(item.get("bbox")), json.dumps(item["keypoints"]),
                         item["observation_source"], item["supporting_model_count"], json.dumps(item["supporting_models"]),
                         item["ensemble_disagreement"], "uncertain" if item.get("tracking_uncertain") else "confirmed",
                         json.dumps(item.get("raw_keypoints", {}))),
                    )
            matched_by_annotation, uncertain = _match_pose_annotations(rows_by_image[row["id"]], tracked)
            frame_cache[row["id"]] = (matched_by_annotation, uncertain)
            with db_factory() as conn:
                for annotation_id, observation in matched_by_annotation.items():
                    status = "tracking_uncertain" if observation.get("tracking_uncertain") else "confirmed"
                    conn.execute("UPDATE annotations SET internal_track_id = ?, tracking_status = ? WHERE id = ?",
                                 (observation["internal_track_id"], status, annotation_id))
                for annotation_id in uncertain:
                    conn.execute("UPDATE annotations SET tracking_status = 'tracking_uncertain' WHERE id = ?", (annotation_id,))

        if run["temporal_enabled"]:
            matched_by_annotation, uncertain = frame_cache[row["id"]]
            observation = matched_by_annotation.get(annot_id)
            if observation is None:
                if annot_id in uncertain:
                    logger.info("Pose annotation %s is tracking_uncertain on %s", annot_id, file_name)
                continue
            model_matches = observation["model_matches"]

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
    process_temporal_run(run_id, "pose", list(range(1, 18)), db_factory)
    with db_factory() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 100, mode = ? WHERE id = ?",
            (actual_mode, run_id),
        )
