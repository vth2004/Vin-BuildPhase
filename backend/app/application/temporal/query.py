from __future__ import annotations

import json
import sqlite3
from typing import Any
from fastapi import HTTPException
from app.core.database import db
from .access import _run


def list_temporal_results(project_id: str, run_id: str, page: int, page_size: int, sequence_id: str | None, track_id: str | None, frame_index: int | None, keypoint_id: int | None, estimate_status: str | None, evidence: str | None, warning_only: bool) -> dict[str, Any]:
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


def temporal_result_detail(project_id: str, run_id: str, result_id: str) -> dict[str, Any]:
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


def temporal_summary(project_id: str, run_id: str) -> dict[str, Any]:
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
