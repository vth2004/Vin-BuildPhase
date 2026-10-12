from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.core.database import db
from app.core.security import project_guard
from app.schemas.warning import ReviewWarning
from app.features.human_pose.services.dataset import load_schema
from app.features.human_pose.adapters.yolo_pose import COCO_TO_VF17

router = APIRouter(prefix="/api/projects/{project_id}", tags=["warnings"])

VF_TO_COCO17 = {vf_id: coco_id for coco_id, (vf_id, _) in COCO_TO_VF17.items()}


def _match_source_image_name(raw_name: object, image_names: set[str]) -> str | None:
    """Match a source COCO filename exactly as HumanPose ingestion does."""
    normalized = str(raw_name or "").replace("\\", "/")
    if normalized in image_names:
        return normalized
    base_name = Path(normalized).name
    if base_name in image_names:
        return base_name
    return next((name for name in image_names if name.endswith(base_name)), None)


def _coco_annotation_map(
    payload: dict,
    dataset_id: str,
) -> tuple[dict[str, list[dict]], dict[str, dict], dict[str, str]]:
    """Map stored annotation IDs to the original COCO annotation objects.

    Dataset ingestion preserves COCO annotation order per image.  The explicit
    map lets new warnings target their source person by annotation_id; its
    image-level fallback is deliberately only used for legacy one-person data.
    """
    with db() as conn:
        rows = conn.execute(
            """SELECT a.id, i.file_name FROM annotations a
               JOIN images i ON i.id = a.image_id
               WHERE i.dataset_id = ? ORDER BY i.file_name, a.rowid""",
            (dataset_id,),
        ).fetchall()
    db_ids_by_file: dict[str, list[str]] = {}
    for row in rows:
        db_ids_by_file.setdefault(row["file_name"], []).append(row["id"])

    image_names = set(db_ids_by_file)
    source_by_file: dict[str, list[dict]] = {}
    for image in payload.get("images", []):
        if not isinstance(image, dict) or "id" not in image:
            continue
        file_name = _match_source_image_name(image.get("file_name"), image_names)
        if not file_name:
            continue
        source_by_file[file_name] = [
            annotation for annotation in payload.get("annotations", [])
            if isinstance(annotation, dict) and annotation.get("image_id") == image["id"]
        ]

    source_by_annotation_id: dict[str, dict] = {}
    unresolved_db_ids: dict[str, str] = {}
    for file_name, db_ids in db_ids_by_file.items():
        source_annotations = source_by_file.get(file_name, [])
        if len(db_ids) != len(source_annotations):
            reason = (
                f"source COCO has {len(source_annotations)} annotation(s), "
                f"but dataset has {len(db_ids)} stored annotation(s) for {file_name}"
            )
            unresolved_db_ids.update({annotation_id: reason for annotation_id in db_ids})
            continue
        source_by_annotation_id.update(zip(db_ids, source_annotations))
    return source_by_file, source_by_annotation_id, unresolved_db_ids


def _export_headers(response: Response, requested: int, applied: int, skipped: list[dict]) -> None:
    response.headers["X-Export-Requested"] = str(requested)
    response.headers["X-Export-Applied"] = str(applied)
    response.headers["X-Export-Skipped"] = str(len(skipped))
    if skipped:
        response.headers["X-Export-Skipped-Details"] = quote(json.dumps(skipped, ensure_ascii=False))


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
    response: Response,
    allow_partial: bool = Query(False),
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
            """SELECT id, image_name, keypoint, suggested_x, suggested_y, review, annotation_id
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
    is_coco = (
        isinstance(payload.get("images"), list)
        and isinstance(payload.get("annotations"), list)
        and isinstance(payload.get("categories"), list)
    )
    if is_coco:
        source_by_file, source_by_annotation_id, unresolved_db_ids = _coco_annotation_map(payload, run["dataset_id"])
        skipped: list[dict] = []
        applied = 0
        for warning in reviewed:
            vf_id = kp_name_to_id.get(warning["keypoint"])
            coco_id = VF_TO_COCO17.get(vf_id) if vf_id is not None else None
            detail = {"warning_id": warning["id"], "image_name": warning["image_name"], "keypoint": warning["keypoint"]}
            if coco_id is None:
                skipped.append({**detail, "reason": "keypoint is not mapped to COCO-17"})
                continue
            annotation = None
            annotation_id = warning["annotation_id"]
            if annotation_id:
                annotation = source_by_annotation_id.get(annotation_id)
                if annotation is None:
                    skipped.append({**detail, "reason": unresolved_db_ids.get(annotation_id, "annotation_id is not mapped to source COCO")})
                    continue
            else:
                candidates = source_by_file.get(warning["image_name"], [])
                if len(candidates) != 1:
                    skipped.append({**detail, "reason": f"legacy warning has no annotation_id and image has {len(candidates)} source annotation(s)"})
                    continue
                annotation = candidates[0]
            keypoints = annotation.get("keypoints")
            keypoint_offset = coco_id * 3
            if not isinstance(keypoints, list) or len(keypoints) < keypoint_offset + 3:
                skipped.append({**detail, "reason": f"source COCO annotation lacks keypoint index {coco_id}"})
                continue
            # Preserve v, num_keypoints, and every unrelated source field verbatim.
            keypoints[keypoint_offset] = warning["suggested_x"]
            keypoints[keypoint_offset + 1] = warning["suggested_y"]
            applied += 1
        if applied != len(reviewed) and not allow_partial:
            raise HTTPException(
                409,
                detail={
                    "message": "COCO export could not apply every use_suggestion review; retry with allow_partial=true to export applicable points.",
                    "requested": len(reviewed),
                    "applied": applied,
                    "skipped": skipped,
                },
            )
        _export_headers(response, len(reviewed), applied, skipped)
        return payload

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

    _export_headers(response, len(reviewed), len(reviewed), [])
    return payload


@router.get("/runs/{run_id}/warnings/{warning_id}/keypoints")
def get_warning_keypoints(
    project_id: str,
    run_id: str,
    warning_id: str,
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
