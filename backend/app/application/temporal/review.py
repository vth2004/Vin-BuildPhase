from __future__ import annotations

import json
import sqlite3
from typing import Any
from fastapi import HTTPException
from app.core.database import db
import math
import time
import uuid
from app.schemas.temporal import TemporalReviewRequest
from .access import _run


def _review_target(conn: sqlite3.Connection, run_id: str, result_id: str) -> tuple[sqlite3.Row, dict]:
    row = conn.execute(
        """SELECT t.*, a.keypoints_json, d.schema_id FROM temporal_results t
           JOIN runs r ON r.id = t.run_id
           JOIN datasets d ON d.id = r.dataset_id
           JOIN images i ON i.id = t.image_id AND i.dataset_id = r.dataset_id
           JOIN annotations a ON a.id = t.annotation_id AND a.image_id = i.id
           WHERE t.id = ? AND t.run_id = ?""", (result_id, run_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Reviewable temporal result not found")
    points = json.loads(row["keypoints_json"] or "{}")
    if isinstance(points, list):
        point = next((item for item in points if isinstance(item, dict) and item.get("id") == row["keypoint_id"]), None)
    else:
        point = points.get(str(row["keypoint_id"])) if isinstance(points, dict) else None
    if not isinstance(point, dict):
        raise HTTPException(409, "Annotation keypoint is missing")
    return row, point


def list_temporal_reviews(project_id: str, run_id: str) -> dict[str, Any]:
    _run(project_id, run_id)
    with db() as conn:
        rows = conn.execute(
            """SELECT * FROM temporal_review_decisions WHERE run_id = ?
               ORDER BY reviewed_at DESC, annotation_id, keypoint_id""", (run_id,),
        ).fetchall()
    return {"total": len(rows), "items": [dict(row) for row in rows]}


def set_temporal_review(project_id: str, run_id: str, result_id: str, payload: TemporalReviewRequest) -> dict[str, Any]:
    _run(project_id, run_id)
    with db() as conn:
        row, point = _review_target(conn, run_id, result_id)
        if point.get("state") == "outside":
            raise HTTPException(409, "Outside keypoints cannot be reviewed")
        if payload.source == "temporal":
            if row["estimate_status"] != "available" or row["temporal_x"] is None or row["temporal_y"] is None:
                raise HTTPException(409, "Temporal estimate is unavailable")
            if (row["schema_id"] == "vf_face_landmark50_v1"
                    and (not row["evidence"] or float(row["temporal_quality"] or 0.0) < 0.45)):
                raise HTTPException(409, "Face temporal estimate does not have enough reliable evidence")
            selected = (row["temporal_x"], row["temporal_y"])
        else:
            selected = (point.get("x"), point.get("y"))
        if any(value is None or not isinstance(value, (int, float)) or not math.isfinite(value) for value in selected):
            raise HTTPException(409, "Selected coordinates are invalid")
        now = int(time.time())
        conn.execute(
            """INSERT INTO temporal_review_decisions
                 (id, run_id, annotation_id, keypoint_id, temporal_result_id, source,
                  selected_x, selected_y, temporal_x, temporal_y, reviewed_at, reviewed_project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, annotation_id, keypoint_id) DO UPDATE SET
                 temporal_result_id = excluded.temporal_result_id, source = excluded.source,
                 selected_x = excluded.selected_x, selected_y = excluded.selected_y,
                 temporal_x = excluded.temporal_x, temporal_y = excluded.temporal_y,
                 reviewed_at = excluded.reviewed_at, reviewed_project_id = excluded.reviewed_project_id""",
            (str(uuid.uuid4()), run_id, row["annotation_id"], row["keypoint_id"], result_id,
             payload.source, float(selected[0]), float(selected[1]), row["temporal_x"],
             row["temporal_y"], now, project_id),
        )
        decision = conn.execute(
            """SELECT * FROM temporal_review_decisions
               WHERE run_id = ? AND annotation_id = ? AND keypoint_id = ?""",
            (run_id, row["annotation_id"], row["keypoint_id"]),
        ).fetchone()
    return dict(decision)


def clear_temporal_review(project_id: str, run_id: str, result_id: str) -> None:
    _run(project_id, run_id)
    with db() as conn:
        row, _ = _review_target(conn, run_id, result_id)
        conn.execute(
            """DELETE FROM temporal_review_decisions
               WHERE run_id = ? AND annotation_id = ? AND keypoint_id = ?""",
            (run_id, row["annotation_id"], row["keypoint_id"]),
        )
