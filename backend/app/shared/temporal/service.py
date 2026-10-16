from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections import defaultdict
from typing import Any, Callable

from .engine import ObservationFrame, estimate_keypoint, normalization_transform


def _mapping(raw: str) -> dict[int, dict[str, Any]]:
    value = json.loads(raw or "{}")
    if isinstance(value, list):
        return {int(point["id"]): point for point in value if isinstance(point, dict) and "id" in point}
    return {int(key): point for key, point in value.items()}


def _annotation_jitter(frames: list[ObservationFrame], annotations: list[dict[int, dict[str, Any]] | None],
                       position: int, point_id: int, feature: str) -> float | None:
    if position == 0 or position >= len(frames) - 1:
        return None
    normalized = []
    for index in (position - 1, position, position + 1):
        if frames[index].tracking_status != "confirmed":
            return None
        transform = normalization_transform(frames[index], feature)
        point = (annotations[index] or {}).get(point_id)
        if not transform or not point or point.get("x") is None or point.get("y") is None:
            return None
        normalized.append(((float(point["x"]) - transform.origin_x) / transform.scale,
                           (float(point["y"]) - transform.origin_y) / transform.scale,
                           transform.source))
    if len({item[2] for item in normalized}) != 1:
        return None
    if frames[position].frame_index - frames[position - 1].frame_index != 1 or frames[position + 1].frame_index - frames[position].frame_index != 1:
        return None
    return math.hypot(normalized[2][0] - 2 * normalized[1][0] + normalized[0][0],
                      normalized[2][1] - 2 * normalized[1][1] + normalized[0][1])


def process_temporal_run(run_id: str, feature: str, point_ids: list[int],
                         db_factory: Callable[[], sqlite3.Connection]) -> None:
    with db_factory() as conn:
        run = conn.execute("SELECT temporal_enabled, temporal_window_size FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run or not run["temporal_enabled"]:
            return
        rows = conn.execute(
            """SELECT o.*, i.original_frame_index, i.timestamp_us, i.timestamp_source, i.width, i.height,
                      a.id AS matched_annotation_id, a.keypoints_json AS annotation_keypoints
               FROM temporal_observations o
               JOIN images i ON i.id = o.image_id
               LEFT JOIN annotations a ON a.image_id = o.image_id AND a.internal_track_id = o.internal_track_id
               WHERE o.run_id = ?
               ORDER BY o.sequence_id, o.segment_id, o.internal_track_id, i.original_frame_index""",
            (run_id,),
        ).fetchall()
        conn.execute("DELETE FROM temporal_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM warnings WHERE run_id = ? AND warning_type = 'temporal_inconsistency'", (run_id,))
        face_r13: set[tuple[str, int]] = set()
        if feature == "face":
            for face_row in conn.execute("SELECT image_id, violations_json FROM face_frame_results WHERE run_id = ?", (run_id,)):
                for violation in json.loads(face_row["violations_json"] or "[]"):
                    if violation.get("code") == "R13":
                        face_r13.update((face_row["image_id"], int(point_id)) for point_id in violation.get("points", []))

    groups: dict[tuple[Any, ...], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[(row["sequence_id"], row["segment_id"], row["internal_track_id"])].append(row)

    for (_, _, _), track_rows in groups.items():
        frames = []
        annotations = []
        for row in track_rows:
            keypoints = _mapping(row["keypoints_json"])
            confidence = [float(point.get("conf", 1.0)) for point in keypoints.values()]
            quality = (sum(confidence) / len(confidence) if confidence else 0.5) * math.exp(-float(row["ensemble_disagreement"] or 0.0))
            frames.append(ObservationFrame(
                int(row["original_frame_index"]), keypoints, json.loads(row["bbox_json"] or "null"),
                row["timestamp_us"], row["timestamp_source"], row["width"], row["height"], quality,
                "confirmed" if row["tracking_status"] == "confirmed" else "tracking_uncertain",
            ))
            annotations.append(_mapping(row["annotation_keypoints"]) if row["annotation_keypoints"] else None)

        inserts = []
        warnings = []
        for position, (row, frame, annotation) in enumerate(zip(track_rows, frames, annotations)):
            transform = normalization_transform(frame, feature)
            raw_keypoints = _mapping(row["raw_keypoints_json"]) if row["raw_keypoints_json"] else frame.keypoints
            for point_id in point_ids:
                estimate = estimate_keypoint(frames, position, point_id, feature=feature,
                                             window_size=int(run["temporal_window_size"]))
                human = (annotation or {}).get(point_id)
                raw = raw_keypoints.get(point_id)
                score = None
                evidence = None
                if estimate.coordinate and human and human.get("x") is not None and human.get("y") is not None and transform:
                    distance = math.dist((float(human["x"]), float(human["y"])), estimate.coordinate) / transform.scale
                    score = min(1.0, distance / 0.25) * estimate.temporal_quality
                    if score >= 0.60 and estimate.temporal_quality >= 0.35:
                        evidence = "model_trajectory"
                jitter = _annotation_jitter(frames, annotations, position, point_id, feature)
                if evidence is None and jitter is not None and jitter >= 0.30:
                    evidence = "annotation_jitter"
                    score = max(score or 0.0, min(1.0, jitter / 0.6))
                deduplicated_r13 = feature == "face" and (row["image_id"], point_id) in face_r13
                if deduplicated_r13 and evidence:
                    evidence = "face_r13"
                trajectory = [{"frame_index": support.frame_index,
                               "timestamp_us": support.timestamp_us,
                               "point": support.keypoints.get(point_id)} for support in frames]
                result_id = str(uuid.uuid4())
                inserts.append((result_id, run_id, row["sequence_id"], row["segment_id"], row["internal_track_id"],
                                row["image_id"], row["matched_annotation_id"], frame.frame_index, point_id,
                                raw.get("x") if raw else None, raw.get("y") if raw else None,
                                human.get("x") if human else None, human.get("y") if human else None,
                                estimate.coordinate[0] if estimate.coordinate else None,
                                estimate.coordinate[1] if estimate.coordinate else None,
                                estimate.estimate_status, estimate.method, estimate.unavailable_reason,
                                estimate.temporal_quality, score, estimate.support_count, estimate.residual,
                                estimate.gap_frames, estimate.gap_us, estimate.normalization_source, estimate.time_source,
                                evidence, row["observation_source"], row["supporting_model_count"],
                                json.dumps(trajectory), int(time.time())))
                if evidence and evidence != "face_r13" and human:
                    suggested = estimate.coordinate or (float(human["x"]), float(human["y"]))
                    warnings.append((str(uuid.uuid4()), run_id, row["image_id"], str(point_id), score,
                                     float(human["x"]), float(human["y"]), suggested[0], suggested[1],
                                     row["matched_annotation_id"], evidence))
        with db_factory() as conn:
            conn.executemany(
                """INSERT INTO temporal_results (
                     id, run_id, sequence_id, segment_id, internal_track_id, image_id, annotation_id, frame_index,
                     keypoint_id, raw_x, raw_y, human_x, human_y, temporal_x, temporal_y, estimate_status, method, unavailable_reason,
                     temporal_quality, temporal_score, support_count, residual, gap_frames, gap_us,
                     normalization_source, time_source, evidence, observation_source, supporting_model_count,
                     trajectory_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                inserts,
            )
            for warning in warnings:
                image_name = conn.execute("SELECT file_name FROM images WHERE id = ?", (warning[2],)).fetchone()[0]
                conn.execute(
                    """INSERT INTO warnings (id, run_id, image_name, keypoint, warning_type, suspicion,
                         human_x, human_y, suggested_x, suggested_y, annotation_id, reliability, delta, evidence)
                       VALUES (?, ?, ?, ?, 'temporal_inconsistency', ?, ?, ?, ?, ?, ?, 1.0, 0.0, ?)""",
                    (warning[0], warning[1], image_name, warning[3], *warning[4:9], warning[9], warning[10]),
                )
