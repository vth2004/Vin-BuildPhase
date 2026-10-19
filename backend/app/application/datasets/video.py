from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import MAX_UPLOAD_BYTES
from app.core.database import db, resolve_data_path, store_data_path
from app.shared.datasets.schemas import load_schema
from app.shared.video import VIDEO_EXTENSIONS

def upload_video_dataset(project_id: str, file: UploadFile, schema_id: UploadFile, annotation_file: UploadFile | None, data_dir: Path, max_frames: int, max_decoded_bytes: int) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Upload an MP4, MOV or AVI video")
    load_schema(schema_id)
    dataset_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    dataset_root = data_dir / "projects" / project_id / "datasets" / dataset_id
    images_root = dataset_root / "images"
    media_path = dataset_root / f"source_video{suffix}"
    staging_path = dataset_root / "staging" / "attempt-1"
    dataset_root.mkdir(parents=True, exist_ok=False)
    try:
        uploaded_bytes = 0
        with media_path.open("wb") as target:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds 500 MB limit")
                target.write(chunk)
        if not uploaded_bytes:
            raise HTTPException(400, "Uploaded video is empty")
        raw_annotation = None
        if annotation_file and annotation_file.filename:
            raw_annotation = annotation_file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw_annotation) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Annotation file exceeds 500 MB limit")
            if not raw_annotation:
                raise HTTPException(400, "Annotation file is empty")

        annotation_path = None
        if raw_annotation:
            suffix_annotation = ".xml" if raw_annotation.lstrip().startswith(b"<") else ".json"
            annotation_path = dataset_root / f"source_annotation{suffix_annotation}"
            annotation_path.write_bytes(raw_annotation)

        now = int(time.time())
        with db() as conn:
            conn.execute(
                """INSERT INTO datasets
                     (id, project_id, name, status, image_count, storage_path, schema_id, annotation_path, created_at)
                   VALUES (?, ?, ?, 'processing', 0, ?, ?, ?, ?)""",
                (dataset_id, project_id, Path(file.filename or "video").stem,
                 store_data_path(images_root), schema_id, store_data_path(annotation_path) if annotation_path else None, now),
            )
            conn.execute(
                """INSERT INTO video_import_jobs (
                     id, dataset_id, attempt, status, progress, staging_path,
                     max_frames, max_decoded_bytes, created_at
                   ) VALUES (?, ?, 1, 'pending', 0, ?, ?, ?, ?)""",
                (job_id, dataset_id, store_data_path(staging_path), max_frames, max_decoded_bytes, now),
            )
        return {
            "id": dataset_id, "name": Path(file.filename or "video").stem, "status": "processing",
            "image_count": 0, "schema_id": schema_id, "import_job_id": job_id,
        }
    except Exception:
        shutil.rmtree(dataset_root, ignore_errors=True)
        raise

def get_video_import(project_id: str, dataset_id: str) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id, status, image_count FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        job = conn.execute(
            """SELECT id, attempt, status, progress, error_message, max_frames, max_decoded_bytes,
                      created_at, started_at, finished_at
                 FROM video_import_jobs WHERE dataset_id = ? ORDER BY attempt DESC LIMIT 1""",
            (dataset_id,),
        ).fetchone()
        if not job:
            raise HTTPException(404, "Video import job not found")
    return {"dataset": dict(dataset), "job": dict(job)}

def retry_video_import(project_id: str, dataset_id: str) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id, status, storage_path FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        if dataset["status"] not in {"failed", "interrupted"}:
            raise HTTPException(409, "Only failed or interrupted video imports can be retried")
        previous = conn.execute(
            "SELECT attempt, max_frames, max_decoded_bytes FROM video_import_jobs WHERE dataset_id = ? ORDER BY attempt DESC LIMIT 1",
            (dataset_id,),
        ).fetchone()
        if not previous:
            raise HTTPException(404, "Video import job not found")
        attempt = previous["attempt"] + 1
        job_id = str(uuid.uuid4())
        staging_path = resolve_data_path(dataset["storage_path"]).parent / "staging" / f"attempt-{attempt}"
        conn.execute("UPDATE datasets SET status = 'processing', image_count = 0 WHERE id = ?", (dataset_id,))
        conn.execute(
            """INSERT INTO video_import_jobs (
                 id, dataset_id, attempt, status, progress, staging_path,
                 max_frames, max_decoded_bytes, created_at
               ) VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (job_id, dataset_id, attempt, store_data_path(staging_path), previous["max_frames"],
             previous["max_decoded_bytes"], int(time.time())),
        )
    return {"dataset_id": dataset_id, "import_job_id": job_id, "attempt": attempt, "status": "processing"}
