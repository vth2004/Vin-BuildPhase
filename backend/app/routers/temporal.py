from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
import csv
import io
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from app.core.database import db
from app.core.security import project_guard

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/temporal", tags=["temporal"])


class TemporalReviewRequest(BaseModel):
    source: Literal["human", "temporal"]


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


@router.get("/reviews")
def list_temporal_reviews(project_id: str, run_id: str,
                          _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    _run(project_id, run_id)
    with db() as conn:
        rows = conn.execute(
            """SELECT * FROM temporal_review_decisions WHERE run_id = ?
               ORDER BY reviewed_at DESC, annotation_id, keypoint_id""", (run_id,),
        ).fetchall()
    return {"total": len(rows), "items": [dict(row) for row in rows]}


@router.put("/results/{result_id}/review")
def set_temporal_review(project_id: str, run_id: str, result_id: str, payload: TemporalReviewRequest,
                        _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
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


@router.delete("/results/{result_id}/review", status_code=204)
def clear_temporal_review(project_id: str, run_id: str, result_id: str,
                          _: sqlite3.Row = Depends(project_guard)) -> None:
    _run(project_id, run_id)
    with db() as conn:
        row, _ = _review_target(conn, run_id, result_id)
        conn.execute(
            """DELETE FROM temporal_review_decisions
               WHERE run_id = ? AND annotation_id = ? AND keypoint_id = ?""",
            (run_id, row["annotation_id"], row["keypoint_id"]),
        )


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


@router.get("/export/audit.json")
def export_temporal_audit_json(project_id: str, run_id: str,
                               _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    _run(project_id, run_id)
    run, items = _audit_rows(run_id)
    return {"format": "landmark-qa/temporal-review-v1", "run": run, "decisions": items}


@router.get("/export/audit.csv")
def export_temporal_audit_csv(project_id: str, run_id: str,
                              _: sqlite3.Row = Depends(project_guard)) -> Response:
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
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="temporal-audit-{run_id[:8]}.csv"'})


def _run(project_id: str, run_id: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    if not row["temporal_enabled"]:
        raise HTTPException(409, "Temporal QA is disabled for this run")
    return row


@router.get("/results")
def list_temporal_results(project_id: str, run_id: str, page: int = Query(1, ge=1),
                          page_size: int = Query(50, ge=1, le=200), sequence_id: str | None = None,
                          track_id: str | None = None, frame_index: int | None = None,
                          keypoint_id: int | None = None, estimate_status: str | None = None,
                          evidence: str | None = None, warning_only: bool = False,
                          _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    _run(project_id, run_id)
    clauses, params = ["t.run_id = ?"], [run_id]
    for column, value in (("sequence_id", sequence_id), ("internal_track_id", track_id),
                          ("frame_index", frame_index), ("keypoint_id", keypoint_id),
                          ("estimate_status", estimate_status), ("evidence", evidence)):
        if value is not None:
            clauses.append(f"t.{column} = ?")
            params.append(value)
    if warning_only:
        clauses.append("t.evidence IS NOT NULL")
    predicate = " AND ".join(clauses)
    with db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM temporal_results t WHERE {predicate}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT t.*, i.file_name FROM temporal_results t JOIN images i ON i.id = t.image_id
                 WHERE {predicate} ORDER BY t.frame_index, t.internal_track_id, t.keypoint_id
                 LIMIT ? OFFSET ?""", [*params, page_size, (page - 1) * page_size]
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item.pop("trajectory_json", None)
        items.append(item)
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/results/{result_id}")
def temporal_result_detail(project_id: str, run_id: str, result_id: str,
                           _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    _run(project_id, run_id)
    with db() as conn:
        row = conn.execute(
            """SELECT t.*, i.file_name FROM temporal_results t JOIN images i ON i.id = t.image_id
               WHERE t.id = ? AND t.run_id = ?""", (result_id, run_id)
        ).fetchone()
    if not row:
        raise HTTPException(404, "Temporal result not found")
    result = dict(row)
    result["trajectory"] = json.loads(result.pop("trajectory_json") or "[]")
    if result["sequence_id"] and result["trajectory"]:
        frame_numbers = [item["frame_index"] for item in result["trajectory"]]
        placeholders = ",".join("?" for _ in frame_numbers)
        with db() as conn:
            names = conn.execute(
                f"SELECT original_frame_index, file_name FROM images WHERE sequence_id = ? AND original_frame_index IN ({placeholders})",
                [result["sequence_id"], *frame_numbers],
            ).fetchall()
        by_frame = {item["original_frame_index"]: item["file_name"] for item in names}
        for item in result["trajectory"]:
            item["file_name"] = by_frame.get(item["frame_index"])
    return result


@router.get("/summary")
def temporal_summary(project_id: str, run_id: str,
                     _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    _run(project_id, run_id)
    with db() as conn:
        totals = conn.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN estimate_status = 'available' THEN 1 ELSE 0 END) AS available,
                      SUM(CASE WHEN evidence IS NOT NULL THEN 1 ELSE 0 END) AS warnings
               FROM temporal_results WHERE run_id = ?""", (run_id,)
        ).fetchone()
        reasons = conn.execute(
            """SELECT unavailable_reason, COUNT(*) AS count FROM temporal_results
               WHERE run_id = ? AND unavailable_reason IS NOT NULL
               GROUP BY unavailable_reason ORDER BY count DESC""", (run_id,)
        ).fetchall()
        sequences = conn.execute(
            """SELECT sequence_id, COUNT(*) AS total,
                      SUM(CASE WHEN estimate_status = 'available' THEN 1 ELSE 0 END) AS available,
                      SUM(CASE WHEN evidence IS NOT NULL THEN 1 ELSE 0 END) AS warnings
               FROM temporal_results WHERE run_id = ? GROUP BY sequence_id""", (run_id,)
        ).fetchall()
        review_counts = conn.execute(
            """SELECT source, COUNT(*) AS count FROM temporal_review_decisions
               WHERE run_id = ? GROUP BY source""", (run_id,),
        ).fetchall()
    total = int(totals["total"] or 0)
    available = int(totals["available"] or 0)
    reviewed = {row["source"]: int(row["count"]) for row in review_counts}
    return {"total": total, "available": available, "coverage": available / total if total else 0.0,
            "warnings": int(totals["warnings"] or 0), "accepted": reviewed.get("temporal", 0),
            "reviewed_temporal": reviewed.get("temporal", 0), "reviewed_human": reviewed.get("human", 0),
            "reviewed_total": sum(reviewed.values()),
            "unavailable_reasons": [dict(row) for row in reasons],
            "sequences": [dict(row) for row in sequences]}
