from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.database import db

from ..rules.validation import RuleViolation, check_all_rules
from ..scoring.quality import evaluate_frame_quality


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


def _match_faces(
    annotations: list[dict[str, Any]],
    detections: list[dict[int, dict[str, Any]]],
) -> list[tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]]:
    remaining = set(range(len(detections)))
    matches: list[tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]] = []
    for annotation in annotations:
        human = annotation["keypoints"]
        human_center = _centroid(human)
        selected: int | None = None
        if remaining and human_center:
            selected = min(
                remaining,
                key=lambda index: math.dist(human_center, _centroid(detections[index]) or human_center),
            )
        elif remaining:
            selected = min(remaining)
        model = detections[selected] if selected is not None else {}
        if selected is not None:
            remaining.remove(selected)
        matches.append((annotation, model))
    for index in sorted(remaining):
        matches.append((None, detections[index]))
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
    previous_by_face: dict[int, dict[int, dict[str, Any]]] = {}
    total = len(images)
    for image_position, image in enumerate(images):
        image_path = (root / image["file_name"]).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"Image not found: {image['file_name']}")
        pixels = cv2.imread(str(image_path))
        if pixels is None:
            raise RuntimeError(f"Cannot decode image: {image['file_name']}")
        height, width = pixels.shape[:2]
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
        detections = [_model_mapping(face[0]) for face in detector.detect_all_faces(pixels)]
        matches = _match_faces(annotations, detections)
        if not matches:
            matches = [(annotation, {}) for annotation in annotations] or [(None, {})]

        for face_index, (annotation, model) in enumerate(matches):
            human = annotation["keypoints"] if annotation else {}
            active = human or model
            raw_violations = check_all_rules(
                active,
                prev_keypoints_by_id=previous_by_face.get(face_index),
                model_keypoints_by_id=model if human else None,
                image_width=width,
                image_height=height,
            ) if active else []
            violations = [_violation_dict(item) for item in raw_violations]
            quality = evaluate_frame_quality(human or model, model if human else None)
            severity = sum(3.0 if item["level"] == "ERROR" else 2.0 for item in violations)
            now = int(time.time())
            with db() as conn:
                conn.execute(
                    """INSERT INTO face_frame_results (
                         id, run_id, image_id, annotation_id, face_index, frame_index,
                         human_keypoints_json, model_keypoints_json, iod, nme, quality_json,
                         violations_json, severity, error_count, status,
                         case_type, ai_reliability, case_label, case_description,
                         created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), run_id, image["id"], annotation["id"] if annotation else None,
                        face_index, image["frame_index"] if image["frame_index"] is not None else image_position,
                        json.dumps(human), json.dumps(model), float(quality["iod"]), float(quality["nme"]),
                        json.dumps(quality, ensure_ascii=False),
                        json.dumps(violations, ensure_ascii=False), severity, len(violations), "pending",
                        quality["case_type"], float(quality["ai_reliability"]), quality["case_label"],
                        quality["case_description"], now, now,
                    ),
                )
            if active:
                previous_by_face[face_index] = active

        progress = int(5 + ((image_position + 1) / max(total, 1)) * 94)
        with db() as conn:
            conn.execute("UPDATE runs SET progress = ? WHERE id = ?", (progress, run_id))

    with db() as conn:
        conn.execute(
            "UPDATE runs SET status = 'completed', progress = 100, error_message = NULL WHERE id = ?",
            (run_id,),
        )
