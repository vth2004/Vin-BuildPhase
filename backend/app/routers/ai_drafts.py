from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.core.database import db
from app.core.security import project_guard

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/ai-drafts", tags=["ai-drafts"])


def _rows(project_id: str, run_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
    with db() as conn:
        run = conn.execute(
            """SELECT r.*, d.name AS dataset_name, d.schema_id
                 FROM runs r JOIN datasets d ON d.id = r.dataset_id
                WHERE r.id = ? AND r.project_id = ?""", (run_id, project_id),
        ).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        if run["input_mode"] != "ai_qc":
            raise HTTPException(409, "AI drafts are available only for runs without source annotations")
        rows = conn.execute(
            """SELECT d.*, i.file_name, i.width, i.height, i.original_frame_index,
                      i.timestamp_us, i.sequence_id
                 FROM ai_draft_results d JOIN images i ON i.id = d.image_id
                WHERE d.run_id = ?
                ORDER BY COALESCE(i.sequence_id, ''), COALESCE(i.original_frame_index, 2147483647),
                         i.file_name, d.instance_index""", (run_id,),
        ).fetchall()
    return run, rows


def _item(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "image_id": row["image_id"], "file_name": row["file_name"],
        "frame_index": row["original_frame_index"], "timestamp_us": row["timestamp_us"],
        "sequence_id": row["sequence_id"], "instance_index": row["instance_index"],
        "track_id": row["internal_track_id"], "keypoints": json.loads(row["keypoints_json"]),
        "provenance": json.loads(row["provenance_json"]), "qc": json.loads(row["qc_json"]),
        "visibility_confirmed": bool(row["visibility_confirmed"]),
    }


@router.get("")
def list_ai_drafts(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> dict:
    run, rows = _rows(project_id, run_id)
    items = [_item(row) for row in rows]
    return {
        "run_id": run_id, "schema_id": run["schema_id"], "label": "AI draft",
        "visibility_confirmed": False, "count": len(items), "items": items,
    }


@router.get("/export")
def export_ai_drafts(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> JSONResponse:
    run, rows = _rows(project_id, run_id)
    images: dict[str, dict] = {}
    for row in rows:
        image = images.setdefault(row["image_id"], {
            "file_name": row["file_name"], "width": row["width"], "height": row["height"],
            "sequence_id": row["sequence_id"], "frame_index": row["original_frame_index"],
            "timestamp_us": row["timestamp_us"], "annotations": [],
        })
        provenance = json.loads(row["provenance_json"])
        image["annotations"].append({
            "label": "AI draft", "instance_id": f"ai-{row['instance_index']}",
            "track_id": row["internal_track_id"], "visibility_confirmed": False,
            "keypoints": json.loads(row["keypoints_json"]), "provenance": provenance,
            "qc": json.loads(row["qc_json"]),
        })
    payload = {
        "format": "landmark-qa/v1", "schema_id": run["schema_id"],
        "dataset_name": run["dataset_name"], "label": "AI draft",
        "visibility_confirmed": False, "source_run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(), "images": list(images.values()),
    }
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="ai_draft_{run_id}.json"',
    })
