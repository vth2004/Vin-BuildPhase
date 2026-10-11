from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.database import db

from ..rules.geometry import compute_iod, compute_iod_with_source
from ..rules.validation import RuleViolation, check_all_rules
from ..scoring.quality import evaluate_frame_quality
from ..spec import get_hungarian_matching_spec


def _point_mapping(raw: str | None) -> dict[int, dict[str, Any]]:
    if not raw:
        return {}
    points = json.loads(raw)
    return {int(point["id"]): point for point in points if isinstance(point, dict) and "id" in point}


def _model_mapping(coords: Any) -> dict[int, dict[str, Any]]:
    return {
        point_id: {"id": point_id, "name": str(point_id), "x": float(point[0]), "y": float(point[1]), "state": "visible"}
        for point_id, point in enumerate(coords)
    }


def _centroid(points: dict[int, dict[str, Any]]) -> tuple[float, float] | None:
    visible = [
        (float(point["x"]), float(point["y"]))
        for point in points.values()
        if point.get("state") != "outside" and point.get("x") is not None and point.get("y") is not None
    ]
    if not visible:
        return None
    return sum(point[0] for point in visible) / len(visible), sum(point[1] for point in visible) / len(visible)


def _merge_split_skeleton_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover one VF50 face from seven legacy skeleton-only annotation rows.

    Earlier imports could persist each VF50 skeleton as a separate annotation.
    Those rows have disjoint populated point IDs (brow, eye, lip, ...), whereas
    real multiple face annotations overlap on the same VF50 IDs. Merge only the
    provably disjoint case in memory; the source annotation rows remain intact.
    """
    if len(annotations) < 2:
        return annotations

    populated_sets = [
        {
            point_id for point_id, point in annotation["keypoints"].items()
            if point.get("state") != "outside" and point.get("x") is not None and point.get("y") is not None
        }
        for annotation in annotations
    ]
    non_empty = [point_ids for point_ids in populated_sets if point_ids]
    if len(non_empty) < 2:
        return annotations
    union = set().union(*non_empty)
    if len(union) <= max(len(point_ids) for point_ids in non_empty) or sum(map(len, non_empty)) != len(union):
        return annotations

    merged_points: dict[int, dict[str, Any]] = {}
    for point_id in range(50):
        candidates = [annotation["keypoints"].get(point_id) for annotation in annotations]
        populated = next((point for point in candidates if point and point_id in union and point.get("state") != "outside"), None)
        merged_points[point_id] = dict(populated or next((point for point in candidates if point), {
            "id": point_id, "name": str(point_id), "x": None, "y": None, "state": "outside",
        }))
    return [{"id": annotations[0]["id"], "keypoints": merged_points}]


def _match_faces(
    annotations: list[dict[str, Any]],
    detections: list[dict[int, dict[str, Any]]],
) -> list[tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]]:
    # 1-to-1 face case fast-path
    if len(annotations) == 1 and len(detections) == 1:
        ann = annotations[0]
        det = detections[0]
        c_ann = _centroid(ann["keypoints"])
        c_det = _centroid(det)
        if c_ann and c_det:
            det_iod = compute_iod(det)
            spec = get_hungarian_matching_spec()
            multiplier = spec.get("centroid_distance_cutoff_iod_multiplier", 2.0)
            fallback_cut = spec.get("max_distance_px_fallback", 150.0)
            cutoff = multiplier * det_iod if det_iod > 10.0 else fallback_cut
            dist = math.dist(c_ann, c_det)
            if dist <= cutoff:
                return [(ann, det)]
            else:
                return [(ann, {}), (None, det)]
        return [(ann, det)]

    if not annotations:
        return [(None, d) for d in detections]
    if not detections:
        return [(a, {}) for a in annotations]

    # Global optimal Hungarian matching
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    spec = get_hungarian_matching_spec()
    multiplier = spec.get("centroid_distance_cutoff_iod_multiplier", 2.0)
    fallback_cut = spec.get("max_distance_px_fallback", 150.0)

    num_ann = len(annotations)
    num_det = len(detections)
    cost = np.zeros((num_ann, num_det), dtype=np.float64)

    ann_centroids = [_centroid(a["keypoints"]) for a in annotations]
    det_centroids = [_centroid(d) for d in detections]
    det_cutoffs = []
    for d in detections:
        d_iod = compute_iod(d)
        det_cutoffs.append(multiplier * d_iod if d_iod > 10.0 else fallback_cut)

    for i in range(num_ann):
        for j in range(num_det):
            c_a = ann_centroids[i]
            c_d = det_centroids[j]
            if c_a and c_d:
                cost[i, j] = math.dist(c_a, c_d)
            else:
                cost[i, j] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_ann = set()
    matched_det = set()
    matches: list[tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]] = []

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= det_cutoffs[c]:
            matches.append((annotations[r], detections[c]))
            matched_ann.add(r)
            matched_det.add(c)

    for i in range(num_ann):
        if i not in matched_ann:
            matches.append((annotations[i], {}))

    for j in range(num_det):
        if j not in matched_det:
            matches.append((None, detections[j]))

    return matches


def _violation_dict(violation: RuleViolation) -> dict[str, Any]:
    return {
        "code": violation.code,
        "rule_name": violation.rule_name,
        "level": violation.level,
        "severity": "critical" if violation.level == "ERROR" else "major",
        "message": violation.message,
        "points": violation.points,
        "gl_section": violation.gl_section,
    }


def execute_face_run(run_id: str) -> None:
    # Import optional heavy dependencies only when a Face run is requested.
    import cv2

    from ..adapters.mediapipe_face import get_detector

    detector = get_detector()
    with db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],)).fetchone()
        if not dataset:
            raise RuntimeError("Dataset not found for Face run")
        images = conn.execute(
            """SELECT id, file_name, width, height, frame_index
               FROM images WHERE dataset_id = ? ORDER BY COALESCE(frame_index, 2147483647), file_name""",
            (dataset["id"],),
        ).fetchall()
        conn.execute("DELETE FROM face_frame_results WHERE run_id = ?", (run_id,))
        conn.execute(
            "UPDATE runs SET status = 'running', progress = 1, error_message = NULL WHERE id = ?",
            (run_id,),
        )

    root = Path(dataset["storage_path"]).resolve()
    has_real_frame_index = any(img["frame_index"] is not None for img in images)
    prev_image_frame_index: int | None = None
    prev_frame_faces: list[dict[str, Any]] = []

    total = len(images)
    for image_position, image in enumerate(images):
        image_path = (root / image["file_name"]).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"Image not found: {image['file_name']}")
        pixels = cv2.imread(str(image_path))
        if pixels is None:
            raise RuntimeError(f"Cannot decode image: {image['file_name']}")
        height, width = pixels.shape[:2]
        cur_frame_index = image["frame_index"]
        is_consecutive = (
            has_real_frame_index
            and cur_frame_index is not None
            and prev_image_frame_index is not None
            and cur_frame_index - prev_image_frame_index == 1
        )

        with db() as conn:
            annotation_rows = conn.execute(
                "SELECT id, keypoints_json FROM annotations WHERE image_id = ? ORDER BY id",
                (image["id"],),
            ).fetchall()
            conn.execute(
                "UPDATE images SET width = COALESCE(width, ?), height = COALESCE(height, ?), frame_index = COALESCE(frame_index, ?) WHERE id = ?",
                (width, height, image_position, image["id"]),
            )
        annotations = [
            {"id": row["id"], "keypoints": _point_mapping(row["keypoints_json"])}
            for row in annotation_rows
        ]
        annotations = _merge_split_skeleton_annotations(annotations)
        detections = [_model_mapping(face[0]) for face in detector.detect_all_faces(pixels)]
        matches = _match_faces(annotations, detections)
        if not matches:
            matches = [(annotation, {}) for annotation in annotations] or [(None, {})]

        current_faces_for_prev: list[dict[str, Any]] = []
        for face_index, (annotation, model) in enumerate(matches):
            human = annotation["keypoints"] if annotation else {}
            active = human or model
            iod_val, iod_source = compute_iod_with_source(model, human)

            prev_face_kpts = None
            if active and is_consecutive and prev_frame_faces:
                cur_c = _centroid(active)
                if cur_c:
                    closest_prev = min(
                        prev_frame_faces,
                        key=lambda pf: math.dist(cur_c, pf["centroid"]) if pf.get("centroid") else float("inf"),
                    )
                    if closest_prev.get("centroid"):
                        c_dist = math.dist(cur_c, closest_prev["centroid"])
                        if c_dist <= 1.5 * iod_val:
                            prev_face_kpts = closest_prev["keypoints"]

            raw_violations = check_all_rules(
                active,
                prev_keypoints_by_id=prev_face_kpts,
                model_keypoints_by_id=model if human else None,
                image_width=width,
                image_height=height,
                iod=iod_val,
            ) if active else []
            violations = [_violation_dict(item) for item in raw_violations]
            quality = evaluate_frame_quality(human or model, model if human else None, iod=iod_val)
            severity = sum(3.0 if item["level"] == "ERROR" else 2.0 for item in violations)
            error_count = sum(1 for item in violations if item["level"] == "ERROR")
            violation_count = len(violations)
            result_kind = "annotated_face" if annotation else "unlabeled_face"
            now = int(time.time())

            if result_kind == "unlabeled_face":
                nme_val = None
                quality["nme"] = None
                quality["nme_pct"] = None
                quality["passed_nme"] = None
            else:
                nme_val = quality["nme"]

            with db() as conn:
                conn.execute(
                    """INSERT INTO face_frame_results (
                         id, run_id, image_id, annotation_id, face_index, frame_index,
                         human_keypoints_json, model_keypoints_json, iod, nme, quality_json,
                         violations_json, severity, error_count, violation_count,
                         result_kind, iod_source, detector_confidence, status,
                         case_type, ai_reliability, case_label, case_description,
                         created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), run_id, image["id"], annotation["id"] if annotation else None,
                        face_index, image["frame_index"] if image["frame_index"] is not None else image_position,
                        json.dumps(human), json.dumps(model), float(quality["iod"]), nme_val,
                        json.dumps(quality, ensure_ascii=False),
                        json.dumps(violations, ensure_ascii=False), severity, error_count, violation_count,
                        result_kind, iod_source, None, "pending",
                        quality["case_type"], float(quality["ai_reliability"]), quality["case_label"],
                        quality["case_description"], now, now,
                    ),
                )
            if active:
                current_faces_for_prev.append({
                    "centroid": _centroid(active),
                    "keypoints": active,
                    "iod": iod_val,
                })

        prev_image_frame_index = cur_frame_index
        prev_frame_faces = current_faces_for_prev

        progress = int(5 + ((image_position + 1) / max(total, 1)) * 94)
        with db() as conn:
            conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (progress, run_id))

    with db() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 100, error_message = NULL WHERE id = ?",
            (run_id,),
        )
