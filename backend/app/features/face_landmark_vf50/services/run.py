from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.database import db
from app.shared.temporal.tracking import DetectionTracker, assign_one_to_one
from app.shared.temporal.segmentation import is_scene_cut, luminance_histogram
from app.shared.temporal.service import process_temporal_run
from app.shared.ai_drafts import apply_temporal_prediction_qc

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


def _detection_item(points: dict[int, dict[str, Any]]) -> dict[str, Any]:
    visible = [(float(p["x"]), float(p["y"])) for p in points.values()
               if p.get("x") is not None and p.get("y") is not None]
    bbox = (min(x for x, _ in visible), min(y for _, y in visible),
            max(x for x, _ in visible), max(y for _, y in visible)) if visible else None
    return {"bbox": bbox, "keypoints": points}


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
    *,
    ambiguity_enabled: bool = True,
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

    spec = get_hungarian_matching_spec()
    multiplier = spec.get("centroid_distance_cutoff_iod_multiplier", 2.0)
    fallback_cut = spec.get("max_distance_px_fallback", 150.0)

    num_ann = len(annotations)
    num_det = len(detections)
    ann_centroids = [_centroid(a["keypoints"]) for a in annotations]
    det_centroids = [_centroid(d) for d in detections]
    det_cutoffs = []
    for d in detections:
        d_iod = compute_iod(d)
        det_cutoffs.append(multiplier * d_iod if d_iod > 10.0 else fallback_cut)

    raw_cost = [[
        math.dist(ann_centroids[i], det_centroids[j])
        if ann_centroids[i] and det_centroids[j] else 1e6
        for j in range(num_det)
    ] for i in range(num_ann)]
    if ambiguity_enabled:
        cost = [[value / max(det_cutoffs[j], 1e-6) for j, value in enumerate(row)] for row in raw_cost]
        assignment = assign_one_to_one(cost, 1.0, ambiguity_margin=float(spec.get("ambiguity_margin", 0.08)))
    else:
        assignment = assign_one_to_one(raw_cost, det_cutoffs, ambiguity_margin=-1.0)

    matched_ann = set()
    matched_det = set()
    matches: list[tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]] = []

    for match in assignment.matches:
        matches.append((annotations[match.left_index], detections[match.right_index]))
        matched_ann.add(match.left_index)
        matched_det.add(match.right_index)

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
    import numpy as np

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
            """SELECT id, file_name, width, height, frame_index, sequence_id,
                      original_frame_index
               FROM images WHERE dataset_id = ?
               ORDER BY COALESCE(sequence_id, ''), COALESCE(original_frame_index, frame_index, 2147483647), file_name""",
            (dataset["id"],),
        ).fetchall()
        conn.execute("DELETE FROM face_frame_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM ai_draft_results WHERE run_id = ?", (run_id,))
        if run["temporal_enabled"]:
            conn.execute("DELETE FROM temporal_observations WHERE run_id = ?", (run_id,))
            conn.execute("UPDATE annotations SET internal_track_id = NULL, tracking_status = 'untracked' WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?)",
                         (dataset["id"],))
        conn.execute(
            "UPDATE runs SET status = 'running', progress = 1, error_message = NULL WHERE id = ?",
            (run_id,),
        )

    root = Path(dataset["storage_path"]).resolve()
    trackers: dict[str, DetectionTracker] = {}
    previous_frame: dict[str, tuple[int | None, int, int]] = {}
    previous_keypoints: dict[str, dict[int, dict[str, Any]]] = {}
    previous_histogram: dict[str, Any] = {}
    legacy_prev_index: int | None = None
    legacy_prev_faces: list[dict[str, Any]] = []
    has_real_frame_index = any(image["frame_index"] is not None for image in images)

    total = len(images)
    for image_position, image in enumerate(images):
        image_path = (root / image["file_name"]).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"Image not found: {image['file_name']}")
        # ``cv2.imread`` cannot reliably open Unicode paths on Windows.  Read
        # through Python's Unicode-aware file API then let OpenCV decode bytes.
        pixels = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise RuntimeError(f"Cannot decode image: {image['file_name']}")
        height, width = pixels.shape[:2]
        sequence_key = image["sequence_id"] or f"legacy:{dataset['id']}"
        cur_frame_index = image["original_frame_index"] if image["original_frame_index"] is not None else image["frame_index"]
        previous = previous_frame.get(sequence_key)
        histogram = luminance_histogram(pixels)
        discontinuity = bool(previous and (
            cur_frame_index is None or previous[0] is None or cur_frame_index - previous[0] != 1
            or previous[1] != width or previous[2] != height
        )) or is_scene_cut(previous_histogram.get(sequence_key), histogram)

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
        detection_tracks: dict[int, dict[str, Any]] = {}
        tracking_enabled = bool(run["temporal_enabled"]) or not annotations
        if tracking_enabled:
            tracker = trackers.setdefault(sequence_key, DetectionTracker())
            tracked = tracker.update([_detection_item(detection) for detection in detections], discontinuity=discontinuity)
            detection_tracks = {id(detection): tracked[index] for index, detection in enumerate(detections)}
            if run["temporal_enabled"]:
                with db() as conn:
                    for detection_index, item in enumerate(tracked):
                        conn.execute(
                            """INSERT INTO temporal_observations (
                                 id, run_id, sequence_id, segment_id, internal_track_id, image_id, detection_index,
                                 bbox_json, keypoints_json, observation_source, supporting_model_count,
                                 supporting_models_json, ensemble_disagreement, tracking_status, raw_keypoints_json
                               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'primary', 1, '["mediapipe"]', 0.0, ?, ?)""",
                            (str(uuid.uuid4()), run_id, image["sequence_id"], item["segment_id"], item["internal_track_id"],
                             image["id"], detection_index, json.dumps(item.get("bbox")), json.dumps(item["keypoints"]),
                             "uncertain" if item.get("tracking_uncertain") else "confirmed", json.dumps(item["keypoints"])),
                        )
        matches = _match_faces(annotations, detections, ambiguity_enabled=bool(run["temporal_enabled"]))
        if not matches:
            matches = [(annotation, {}) for annotation in annotations] or [(None, {})]

        legacy_current_faces: list[dict[str, Any]] = []
        for face_index, (annotation, model) in enumerate(matches):
            human = annotation["keypoints"] if annotation else {}
            active = human or model
            iod_val, iod_source = compute_iod_with_source(model, human)
            track = detection_tracks.get(id(model)) if model else None
            track_id = track.get("internal_track_id") if track else None
            if run["temporal_enabled"]:
                prev_face_kpts = previous_keypoints.get(track_id) if track_id and not discontinuity else None
            else:
                prev_face_kpts = None
                is_consecutive = has_real_frame_index and image["frame_index"] is not None and legacy_prev_index is not None and image["frame_index"] - legacy_prev_index == 1
                current_center = _centroid(active)
                if active and is_consecutive and legacy_prev_faces and current_center:
                    closest = min(legacy_prev_faces, key=lambda item: math.dist(current_center, item["centroid"]))
                    if math.dist(current_center, closest["centroid"]) <= 1.5 * iod_val:
                        prev_face_kpts = closest["keypoints"]
            if run["temporal_enabled"] and annotation and track_id:
                with db() as conn:
                    status = "tracking_uncertain" if track.get("tracking_uncertain") else "confirmed"
                    conn.execute("UPDATE annotations SET internal_track_id = ?, tracking_status = ? WHERE id = ?",
                                 (track_id, status, annotation["id"]))
            elif run["temporal_enabled"] and annotation:
                with db() as conn:
                    conn.execute("UPDATE annotations SET tracking_status = 'tracking_uncertain' WHERE id = ?", (annotation["id"],))

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
                if result_kind == "unlabeled_face" and model:
                    draft_points = []
                    for point_id in range(50):
                        point = dict(model.get(point_id, {
                            "id": point_id, "name": str(point_id), "x": None, "y": None, "state": "outside",
                        }))
                        point["visibility_confirmed"] = False
                        draft_points.append(point)
                    conn.execute(
                        """INSERT INTO ai_draft_results (
                             id, run_id, image_id, schema_id, instance_index, internal_track_id,
                             keypoints_json, provenance_json, qc_json, visibility_confirmed, created_at
                           ) VALUES (?, ?, ?, 'vf_face_landmark50_v1', ?, ?, ?, ?, ?, 0, ?)""",
                        (str(uuid.uuid4()), run_id, image["id"], face_index, track_id,
                         json.dumps(draft_points), json.dumps({
                             "label": "AI draft", "model": "mediapipe", "confidence_available": False,
                             "algorithm_version": "mediapipe-vf50-ai-draft-v1",
                         }), json.dumps({"flags": [item["code"] for item in violations],
                                        "geometry_violations": violations}), now),
                    )
            if track_id and active:
                previous_keypoints[track_id] = active
            if active and _centroid(active):
                legacy_current_faces.append({"centroid": _centroid(active), "keypoints": active})

        previous_frame[sequence_key] = (cur_frame_index, width, height)
        previous_histogram[sequence_key] = histogram
        legacy_prev_index = image["frame_index"]
        legacy_prev_faces = legacy_current_faces

        progress = int(5 + ((image_position + 1) / max(total, 1)) * 94)
        with db() as conn:
            conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (progress, run_id))

    process_temporal_run(run_id, "face", list(range(50)), db)
    if run["input_mode"] == "ai_qc" and run["temporal_enabled"]:
        apply_temporal_prediction_qc(run_id, db)
    with db() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 100, error_message = NULL WHERE id = ?",
            (run_id,),
        )
