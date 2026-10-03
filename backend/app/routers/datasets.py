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
from app.services.dataset_service import load_schema, safe_extract_images, validate_annotations

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


@router.post("", status_code=201)
def upload_dataset(
    project_id: str,
    file: Annotated[UploadFile, File(description="ZIP containing image files")],
    schema_id: Annotated[str, Form()],
    annotation_file: Annotated[UploadFile | None, File(description="landmark-qa/v1 or COCO annotation JSON")] = None,
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

        if not raw_annotation:
            raise HTTPException(400, "No annotation JSON provided or found inside ZIP")
        if len(raw_annotation) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Annotation JSON exceeds 500 MB limit")

        try:
            annotation_payload = json.loads(raw_annotation)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HTTPException(400, "Annotation file is not valid JSON") from error

        validated = validate_annotations(annotation_payload, schema, set(images))

        annotation_path = location.parent / "source_annotation.json"
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
                    str(annotation_path),
                    int(time.time()),
                ),
            )
            for image in validated:
                image_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO images VALUES (?, ?, ?, ?, ?)",
                    (image_id, dataset_id, image["file_name"], image["width"], image["height"]),
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
