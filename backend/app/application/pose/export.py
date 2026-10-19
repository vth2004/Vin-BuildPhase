from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from fastapi import HTTPException
from app.core.database import db, resolve_data_path
from app.shared.datasets.schemas import load_schema
from app.features.human_pose.adapters.yolo_pose import COCO_TO_VF17

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


def _landmark_annotation_map(payload: dict, dataset_id: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Map stored annotation IDs to landmark-qa/v1 annotation objects by preserved import order."""
    with db() as conn:
        rows = conn.execute(
            """SELECT a.id, i.file_name FROM annotations a JOIN images i ON i.id = a.image_id
               WHERE i.dataset_id = ? ORDER BY i.file_name, a.rowid""", (dataset_id,),
        ).fetchall()
    db_ids_by_file: dict[str, list[str]] = {}
    for row in rows:
        db_ids_by_file.setdefault(row["file_name"], []).append(row["id"])
    source_images = {
        str(image.get("file_name", "")).replace("\\", "/"): image
        for image in payload.get("images", []) if isinstance(image, dict)
    }
    mapped: dict[str, dict] = {}
    unresolved: dict[str, str] = {}
    for file_name, annotation_ids in db_ids_by_file.items():
        annotations = source_images.get(file_name, {}).get("annotations", [])
        if not isinstance(annotations, list) or len(annotations) != len(annotation_ids):
            reason = f"source landmark-qa annotation count does not match stored annotations for {file_name}"
            unresolved.update({annotation_id: reason for annotation_id in annotation_ids})
            continue
        mapped.update(zip(annotation_ids, annotations))
    return mapped, unresolved


def _export_headers(headers: dict[str, str], requested: int, applied: int, skipped: list[dict]) -> None:
    headers["X-Export-Requested"] = str(requested)
    headers["X-Export-Applied"] = str(applied)
    headers["X-Export-Skipped"] = str(len(skipped))
    if skipped:
        headers["X-Export-Skipped-Details"] = quote(json.dumps(skipped, ensure_ascii=False))


def export_dataset(project_id: str, run_id: str, allow_partial: bool, headers: dict[str, str]) -> dict:
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
        if not dataset or not dataset["annotation_path"] or not resolve_data_path(dataset["annotation_path"]).is_file():
            raise HTTPException(404, "Annotation file not found")
        reviewed = conn.execute(
            """SELECT id, image_name, keypoint, suggested_x, suggested_y, review, annotation_id
               FROM warnings
               WHERE run_id = ? AND review = 'use_suggestion'""",
            (run_id,),
        ).fetchall()
        temporal_reviews = conn.execute(
            """SELECT d.id, d.annotation_id, d.keypoint_id, d.source, d.selected_x, d.selected_y,
                      d.temporal_result_id, d.reviewed_at, i.file_name
               FROM temporal_review_decisions d
               JOIN annotations a ON a.id = d.annotation_id
               JOIN images i ON i.id = a.image_id AND i.dataset_id = ?
               WHERE d.run_id = ? ORDER BY d.reviewed_at, d.id""",
            (run["dataset_id"], run_id),
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

    raw = resolve_data_path(dataset["annotation_path"]).read_text(encoding="utf-8")
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
        for decision in temporal_reviews:
            coco_id = VF_TO_COCO17.get(decision["keypoint_id"])
            detail = {"review_id": decision["id"], "annotation_id": decision["annotation_id"],
                      "keypoint_id": decision["keypoint_id"], "source": decision["source"]}
            if coco_id is None:
                skipped.append({**detail, "reason": "keypoint is not mapped to COCO-17"})
                continue
            annotation = source_by_annotation_id.get(decision["annotation_id"])
            if annotation is None:
                skipped.append({**detail, "reason": unresolved_db_ids.get(decision["annotation_id"], "annotation_id is not mapped to source COCO")})
                continue
            keypoints = annotation.get("keypoints")
            offset = coco_id * 3
            if not isinstance(keypoints, list) or len(keypoints) < offset + 3:
                skipped.append({**detail, "reason": f"source COCO annotation lacks keypoint index {coco_id}"})
                continue
            if keypoints[offset + 2] == 0:
                skipped.append({**detail, "reason": "outside keypoints cannot be replaced"})
                continue
            keypoints[offset] = decision["selected_x"]
            keypoints[offset + 1] = decision["selected_y"]
            applied += 1
        requested = len(reviewed) + len(temporal_reviews)
        if applied != requested and not allow_partial:
            raise HTTPException(
                409,
                detail={
                    "message": "COCO export could not apply every use_suggestion review; retry with allow_partial=true to export applicable points.",
                    "requested": requested,
                    "applied": applied,
                    "skipped": skipped,
                },
            )
        _export_headers(headers, requested, applied, skipped)
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
    source_by_annotation_id, unresolved_db_ids = _landmark_annotation_map(payload, run["dataset_id"])
    skipped = []
    temporal_applied = 0
    for decision in temporal_reviews:
        annotation = source_by_annotation_id.get(decision["annotation_id"])
        detail = {"review_id": decision["id"], "annotation_id": decision["annotation_id"],
                  "keypoint_id": decision["keypoint_id"], "source": decision["source"]}
        if annotation is None:
            skipped.append({**detail, "reason": unresolved_db_ids.get(decision["annotation_id"], "annotation_id is not mapped to source landmark-qa")})
            continue
        point = next((point for point in annotation.get("keypoints", [])
                      if isinstance(point, dict) and point.get("id") == decision["keypoint_id"]), None)
        if point is None:
            skipped.append({**detail, "reason": "source annotation lacks keypoint"})
            continue
        if point.get("state") == "outside":
            skipped.append({**detail, "reason": "outside keypoints cannot be replaced"})
            continue
        point["x"], point["y"] = decision["selected_x"], decision["selected_y"]
        temporal_applied += 1

    requested = len(reviewed) + len(temporal_reviews)
    applied = len(reviewed) + temporal_applied
    if applied != requested and not allow_partial:
        raise HTTPException(409, detail={"message": "Pose export could not apply every review; retry with allow_partial=true.",
                                         "requested": requested, "applied": applied, "skipped": skipped})

    _export_headers(headers, requested, applied, skipped)
    return payload
