from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from app.core.database import db
from app.core.security import project_guard
from app.schemas.warning import ReviewWarning
from app.services.dataset_service import load_schema

router = APIRouter(prefix="/api/projects/{project_id}", tags=["warnings"])


@router.get("/runs/{run_id}/warnings")
def list_warnings(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> list[dict]:
    with db() as conn:
        run = conn.execute(
            "SELECT id FROM runs WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        rows = conn.execute(
            "SELECT * FROM warnings WHERE run_id = ? ORDER BY suspicion DESC",
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@router.post("/warnings/{warning_id}/review")
def review_warning(
    project_id: str,
    warning_id: str,
    payload: ReviewWarning,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        warning = conn.execute(
            """SELECT w.id FROM warnings w JOIN runs r ON r.id = w.run_id
               WHERE w.id = ? AND r.project_id = ?""",
            (warning_id, project_id),
        ).fetchone()
        if not warning:
            raise HTTPException(404, "Warning not found")
        conn.execute("UPDATE warnings SET review = ? WHERE id = ?", (payload.review, warning_id))
    return {"id": warning_id, "review": payload.review}


@router.get("/runs/{run_id}/export")
def export_dataset(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        run = conn.execute(
            "SELECT dataset_id FROM runs WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        dataset = conn.execute(
            "SELECT annotation_path, schema_id FROM datasets WHERE id = ?",
            (run["dataset_id"],),
        ).fetchone()
        if not dataset or not dataset["annotation_path"] or not Path(dataset["annotation_path"]).is_file():
            raise HTTPException(404, "Annotation file not found")
        reviewed = conn.execute(
            """SELECT image_name, keypoint, suggested_x, suggested_y, review
               FROM warnings
               WHERE run_id = ? AND review = 'use_suggestion'""",
            (run_id,),
        ).fetchall()

    schema = load_schema(dataset["schema_id"])
    kp_name_to_id = {kp["name"]: kp["id"] for kp in schema["keypoints"]}
    updates: dict[str, dict[int, tuple[float, float]]] = {}
    for rw in reviewed:
        img_name = rw["image_name"]
        kp_id = kp_name_to_id.get(rw["keypoint"])
        if kp_id is not None:
            if img_name not in updates:
                updates[img_name] = {}
            updates[img_name][kp_id] = (rw["suggested_x"], rw["suggested_y"])

    raw = Path(dataset["annotation_path"]).read_text(encoding="utf-8")
    payload = json.loads(raw)
    for img in payload.get("images", []):
        fn = img.get("file_name", "").replace("\\", "/")
        if fn in updates:
            for annot in img.get("annotations", []):
                for pt in annot.get("keypoints", []):
                    pt_id = pt.get("id")
                    if pt_id in updates[fn]:
                        new_x, new_y = updates[fn][pt_id]
                        pt["x"] = new_x
                        pt["y"] = new_y
                        if pt.get("state") == "outside":
                            pt["state"] = "visible"

    return payload


@router.get("/runs/{run_id}/images/{image_name:path}/keypoints")
def get_image_keypoints(
    project_id: str,
    run_id: str,
    image_name: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        run = conn.execute(
            "SELECT dataset_id FROM runs WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        dataset = conn.execute(
            "SELECT schema_id FROM datasets WHERE id = ?",
            (run["dataset_id"],),
        ).fetchone()
        row = conn.execute(
            """SELECT a.keypoints_json, i.width, i.height
               FROM images i
               JOIN annotations a ON a.image_id = i.id
               WHERE i.dataset_id = ? AND i.file_name = ?""",
            (run["dataset_id"], image_name),
        ).fetchone()

    if not row:
        raise HTTPException(404, "Image annotations not found")

    schema = load_schema(dataset["schema_id"])
    return {
        "keypoints": json.loads(row["keypoints_json"]),
        "edges": schema.get("edges", []),
        "width": row["width"],
        "height": row["height"],
    }
