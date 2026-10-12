from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import DATA_DIR, IMAGE_EXTENSIONS, MAX_UPLOAD_BYTES
from app.core.database import db
from app.core.security import project_guard
from app.features.face_landmark_vf50 import SCHEMA_ID as FACE_SCHEMA_ID
from app.features.face_landmark_vf50.services.ingestion import parse_face_annotations
from app.features.human_pose.services.dataset import load_schema, validate_annotations
from app.shared.datasets import safe_extract_images, validate_exif_orientation

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


@router.post("", status_code=201)
def upload_dataset(
    project_id: str,
    file: Annotated[UploadFile, File(description="ZIP containing image files")],
    schema_id: Annotated[str, Form()],
    annotation_file: Annotated[UploadFile | None, File(description="CVAT XML, COCO JSON, landmark-qa/v1 or cleaned JSON")] = None,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file")

    schema = load_schema(schema_id)
    dataset_id = str(uuid.uuid4())
    location = DATA_DIR / "projects" / project_id / "datasets" / dataset_id / "images"
    location.mkdir(parents=True, exist_ok=False)

    try:
        images, embedded_annotation = safe_extract_images(file, location)
        if not images:
            raise HTTPException(400, "ZIP contains no supported images")

        raw_annotation = None
        if annotation_file and annotation_file.filename:
            raw_annotation = annotation_file.file.read(MAX_UPLOAD_BYTES + 1)
        elif embedded_annotation:
            raw_annotation = embedded_annotation

        if not raw_annotation and schema_id != FACE_SCHEMA_ID:
            raise HTTPException(400, "No annotation JSON provided or found inside ZIP")
        if raw_annotation and len(raw_annotation) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Annotation JSON exceeds 500 MB limit")

        if schema_id == FACE_SCHEMA_ID:
            validated = parse_face_annotations(raw_annotation, set(images)) if raw_annotation else []
            records_by_name = {record["file_name"]: record for record in validated}
            validated = [
                records_by_name.get(name, {
                    "file_name": name,
                    "width": None,
                    "height": None,
                    "annotations": [],
                })
                for name in images
            ]
        else:
            if raw_annotation and raw_annotation.lstrip().startswith(b"<"):
                raise HTTPException(
                    400,
                    "File annotation XML này thuộc định dạng Face Landmark (CVAT XML). "
                    "Phân hệ Human Pose chỉ hỗ trợ file JSON COCO 17 điểm. Vui lòng chuyển sang phân hệ Face Landmark.",
                )
            try:
                annotation_payload = json.loads(raw_annotation)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HTTPException(400, "Annotation file is not valid JSON") from error
            validated = validate_annotations(annotation_payload, schema, set(images))

        validate_exif_orientation(location, validated)

        annotation_path = None
        if raw_annotation:
            annotation_suffix = ".xml" if raw_annotation.lstrip().startswith(b"<") else ".json"
            annotation_path = location.parent / f"source_annotation{annotation_suffix}"
            annotation_path.write_bytes(raw_annotation)

        with db() as conn:
            conn.execute(
                "INSERT INTO datasets (id, project_id, name, status, image_count, storage_path, schema_id, annotation_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    project_id,
                    Path(file.filename).stem,
                    "ready",
                    len(images),
                    str(location),
                    schema_id,
                    str(annotation_path) if annotation_path else None,
                    int(time.time()),
                ),
            )
            for frame_index, image in enumerate(validated):
                image_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO images (id, dataset_id, file_name, width, height, frame_index) VALUES (?, ?, ?, ?, ?, ?)",
                    (image_id, dataset_id, image["file_name"], image["width"], image["height"], frame_index),
                )
                for annotation in image["annotations"]:
                    conn.execute(
                        "INSERT INTO annotations VALUES (?, ?, ?, ?)",
                        (str(uuid.uuid4()), image_id, annotation["label"], json.dumps(annotation["keypoints"])),
                    )

        return {
            "id": dataset_id,
            "name": Path(file.filename).stem,
            "status": "ready",
            "image_count": len(images),
            "schema_id": schema_id,
        }
    except Exception:
        shutil.rmtree(location.parent, ignore_errors=True)
        raise


@router.get("/{dataset_id}/images/{image_path:path}")
def get_image(
    project_id: str,
    dataset_id: str,
    image_path: str,
    _: sqlite3.Row = Depends(project_guard),
) -> FileResponse:
    with db() as conn:
        dataset = conn.execute(
            "SELECT storage_path FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()

    if not dataset:
        raise HTTPException(404, "Dataset not found")

    root = Path(dataset["storage_path"]).resolve()
    target = (root / image_path).resolve()

    if root not in target.parents or not target.is_file() or target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "Image not found")

    return FileResponse(target)
