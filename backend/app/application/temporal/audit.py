from __future__ import annotations

import json
import sqlite3
from typing import Any
from fastapi import HTTPException
from app.core.database import db
import csv
import io
from .access import _run


def _audit_rows(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with db() as conn:
        run = conn.execute(
            """SELECT r.id AS run_id, r.temporal_window_size, r.temporal_algorithm_version,
                      d.schema_id, d.id AS dataset_id
               FROM runs r JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?""", (run_id,),
        ).fetchone()
        rows = conn.execute(
            """SELECT d.id AS review_id, d.annotation_id, d.keypoint_id, d.source,
                      d.selected_x, d.selected_y, d.temporal_x AS reviewed_temporal_x,
                      d.temporal_y AS reviewed_temporal_y, d.reviewed_at, d.reviewed_project_id,
                      t.id AS temporal_result_id, t.raw_x, t.raw_y, t.human_x, t.human_y,
                      t.temporal_x, t.temporal_y, t.temporal_quality, t.temporal_score,
                      t.method, t.evidence, t.observation_source, t.supporting_model_count,
                      t.sequence_id, t.segment_id, t.internal_track_id, t.frame_index,
                      i.file_name, a.keypoints_json
               FROM temporal_review_decisions d
               JOIN temporal_results t ON t.id = d.temporal_result_id AND t.run_id = d.run_id
               JOIN annotations a ON a.id = d.annotation_id
               JOIN images i ON i.id = a.image_id
               WHERE d.run_id = ? ORDER BY i.file_name, d.annotation_id, d.keypoint_id""", (run_id,),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        points = json.loads(item.pop("keypoints_json") or "[]")
        if isinstance(points, dict):
            point = points.get(str(item["keypoint_id"]))
        else:
            point = next((value for value in points if isinstance(value, dict) and value.get("id") == item["keypoint_id"]), None)
        item["source_state"] = point.get("state") if point else None
        items.append(item)
    return dict(run), items


def export_temporal_audit_json(project_id: str, run_id: str) -> dict[str, Any]:
    _run(project_id, run_id)
    run, items = _audit_rows(run_id)
    return {"format": "landmark-qa/temporal-review-v1", "run": run, "decisions": items}


def export_temporal_audit_csv(project_id: str, run_id: str) -> str:
    _run(project_id, run_id)
    _, items = _audit_rows(run_id)
    columns = [
        "review_id", "temporal_result_id", "annotation_id", "file_name", "sequence_id", "segment_id",
        "internal_track_id", "frame_index", "keypoint_id", "source_state", "source", "human_x", "human_y",
        "raw_x", "raw_y", "temporal_x", "temporal_y", "selected_x", "selected_y", "temporal_quality",
        "temporal_score", "method", "evidence", "observation_source", "supporting_model_count", "reviewed_at",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    return output.getvalue()
