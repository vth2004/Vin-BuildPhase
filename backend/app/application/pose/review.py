from __future__ import annotations

import json
from fastapi import HTTPException
from app.core.database import db
from app.schemas.warning import ReviewWarning
from app.shared.datasets.schemas import load_schema


def list_warnings(project_id: str, run_id: str) -> list[dict]:
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


def review_warning(project_id: str, warning_id: str, payload: ReviewWarning) -> dict:
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


def get_warning_keypoints(project_id: str, run_id: str, warning_id: str) -> dict:
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
        warning = conn.execute(
            "SELECT * FROM warnings WHERE id = ? AND run_id = ?",
            (warning_id, run_id),
        ).fetchone()
        if not warning:
            raise HTTPException(404, "Warning not found")

        schema = load_schema(dataset["schema_id"])
        selected_annotation = None

        # 1. Nếu có annotation_id được lưu sẵn
        if "annotation_id" in warning.keys() and warning["annotation_id"]:
            selected_annotation = conn.execute(
                """SELECT a.keypoints_json, i.width, i.height
                   FROM annotations a
                   JOIN images i ON i.id = a.image_id
                   WHERE a.id = ?""",
                (warning["annotation_id"],),
            ).fetchone()

        # 2. Fallback cho dữ liệu cũ: so khớp tọa độ (human_x, human_y) của khớp cảnh báo
        if not selected_annotation:
            candidate_rows = conn.execute(
                """SELECT a.id, a.keypoints_json, i.width, i.height
                   FROM images i
                   JOIN annotations a ON a.image_id = i.id
                   WHERE i.dataset_id = ? AND i.file_name = ?""",
                (run["dataset_id"], warning["image_name"]),
            ).fetchall()

            kp_name_to_id = {kp["name"]: kp["id"] for kp in schema["keypoints"]}
            target_kp_id = kp_name_to_id.get(warning["keypoint"])

            best_row = None
            min_dist = float("inf")

            for cand in candidate_rows:
                kpts = json.loads(cand["keypoints_json"])
                for pt in kpts:
                    if target_kp_id is not None and pt.get("id") == target_kp_id:
                        px, py = pt.get("x"), pt.get("y")
                        if px is not None and py is not None:
                            dist = abs(px - warning["human_x"]) + abs(py - warning["human_y"])
                            if dist < min_dist:
                                min_dist = dist
                                best_row = cand
                    elif pt.get("name") == warning["keypoint"]:
                        px, py = pt.get("x"), pt.get("y")
                        if px is not None and py is not None:
                            dist = abs(px - warning["human_x"]) + abs(py - warning["human_y"])
                            if dist < min_dist:
                                min_dist = dist
                                best_row = cand

            selected_annotation = best_row or (candidate_rows[0] if candidate_rows else None)

    if not selected_annotation:
        raise HTTPException(404, "Image annotations not found for warning")

    return {
        "keypoints": json.loads(selected_annotation["keypoints_json"]),
        "edges": schema.get("edges", []),
        "width": selected_annotation["width"],
        "height": selected_annotation["height"],
    }


def get_image_keypoints(project_id: str, run_id: str, image_name: str) -> dict:
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
