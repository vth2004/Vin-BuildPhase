from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import (
    DATA_DIR, IMAGE_EXTENSIONS, MAX_DECODED_VIDEO_BYTES, MAX_UPLOAD_BYTES, MAX_VIDEO_FRAMES,
)
from app.core.database import db, resolve_data_path, store_data_path
from app.core.security import project_guard
from app.features.face_landmark_vf50 import SCHEMA_ID as FACE_SCHEMA_ID
from app.features.face_landmark_vf50.services.ingestion import parse_face_annotations
from app.features.human_pose.services.dataset import load_schema, validate_annotations
from app.shared.datasets import safe_extract_dataset, validate_exif_orientation
from app.shared.temporal import infer_sequences, parse_sequence_manifest
from app.shared.video import VIDEO_EXTENSIONS
from app.shared.video_import import execute_video_import

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


class SequenceConfirmation(BaseModel):
    fps: float | None = Field(default=None, gt=0, le=1000)
    sampling_confirmed: bool = False
    confirmation_source: str = Field(default="user_confirmation", min_length=1, max_length=64)


def _sequence_summary(conn: sqlite3.Connection, dataset_id: str) -> dict:
    sequences = conn.execute(
        "SELECT * FROM sequences WHERE dataset_id = ? ORDER BY name", (dataset_id,)
    ).fetchall()
    items = []
    for sequence in sequences:
        frames = conn.execute(
            """SELECT file_name, original_frame_index, timestamp_us, timestamp_source, frame_index_source
                 FROM images WHERE sequence_id = ?
                 ORDER BY original_frame_index, file_name""",
            (sequence["id"],),
        ).fetchall()
        indices = [row["original_frame_index"] for row in frames if row["original_frame_index"] is not None]
        deltas = [b - a for a, b in zip(indices, indices[1:])]
        items.append({
            "id": sequence["id"], "name": sequence["name"], "source": sequence["source"],
            "frame_order_source": sequence["frame_order_source"], "frame_count": len(frames),
            "first_frame": indices[0] if indices else None, "last_frame": indices[-1] if indices else None,
            "frame_deltas": deltas, "confirmed_at": sequence["confirmed_at"],
            "fps": sequence["fps"], "fps_source": sequence["fps_source"],
            "sampling_confirmed": bool(sequence["sampling_confirmed"]),
            "requires_manifest": bool(sequence["requires_manifest"]),
            "video_codec": sequence["video_codec"], "video_width": sequence["video_width"],
            "video_height": sequence["video_height"], "video_frame_count": sequence["video_frame_count"],
            "rotation_degrees": sequence["rotation_degrees"],
            "reason": json.loads(sequence["mapping_snapshot_json"] or "{}").get("reason"),
            "frames": [dict(row) for row in frames],
        })
    return {
        "dataset_id": dataset_id,
        "temporal_ready": bool(items) and all(item["confirmed_at"] and not item["requires_manifest"] for item in items),
        "sequences": items,
    }


@router.post("", status_code=201)
def upload_dataset(
    project_id: str,
    file: Annotated[UploadFile, File(description="ZIP containing image files")],
    schema_id: Annotated[str, Form()],
    annotation_file: Annotated[UploadFile | None, File(description="CVAT XML, COCO JSON, landmark-qa/v1 or cleaned JSON")] = None,
    sequence_manifest_file: Annotated[UploadFile | None, File(description="Optional sequence_manifest.json override")] = None,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file")

    schema = load_schema(schema_id)
    dataset_id = str(uuid.uuid4())
    location = DATA_DIR / "projects" / project_id / "datasets" / dataset_id / "images"
    location.mkdir(parents=True, exist_ok=False)

    try:
        images, embedded_annotation, embedded_manifest = safe_extract_dataset(file, location)
        if not images:
            raise HTTPException(400, "ZIP contains no supported images")

        raw_annotation = None
        if annotation_file and annotation_file.filename:
            raw_annotation = annotation_file.file.read(MAX_UPLOAD_BYTES + 1)
        elif embedded_annotation:
            raw_annotation = embedded_annotation

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
            if not raw_annotation:
                validated = [
                    {"file_name": name, "width": None, "height": None, "annotations": []}
                    for name in images
                ]
            elif raw_annotation.lstrip().startswith(b"<"):
                raise HTTPException(
                    400,
                    "File annotation XML này thuộc định dạng Face Landmark (CVAT XML). "
                    "Phân hệ Human Pose chỉ hỗ trợ file JSON COCO 17 điểm. Vui lòng chuyển sang phân hệ Face Landmark.",
                )
            else:
                try:
                    annotation_payload = json.loads(raw_annotation)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise HTTPException(400, "Annotation file is not valid JSON") from error
                validated = validate_annotations(annotation_payload, schema, set(images))

        validate_exif_orientation(location, validated)

        manifest_raw = embedded_manifest
        if sequence_manifest_file and sequence_manifest_file.filename:
            manifest_raw = sequence_manifest_file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(manifest_raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Sequence manifest exceeds 500 MB limit")
        inference = (
            parse_sequence_manifest(manifest_raw, set(images))
            if manifest_raw else infer_sequences(validated)
        )

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
                    store_data_path(location),
                    schema_id,
                    store_data_path(annotation_path) if annotation_path else None,
                    int(time.time()),
                ),
            )
            mapping_by_name = {mapping.file_name: mapping for mapping in inference.mappings}
            sequence_ids: dict[str, str] = {}
            sequence_names = sorted({mapping.sequence_name for mapping in inference.mappings}) or ["unresolved"]
            trusted = inference.source in {"manifest", "annotation"} and not inference.requires_manifest
            now = int(time.time())
            for sequence_name in sequence_names:
                sequence_id = str(uuid.uuid4())
                sequence_ids[sequence_name] = sequence_id
                sequence_mapping = [
                    {"file_name": m.file_name, "frame_index": m.original_frame_index, "timestamp_us": m.timestamp_us}
                    for m in inference.mappings if m.sequence_name == sequence_name
                ]
                snapshot = {"source": inference.source, "reason": inference.reason, "mapping": sequence_mapping}
                conn.execute(
                    """INSERT INTO sequences (
                         id, dataset_id, name, source, frame_order_source, confirmed_at,
                         confirmed_project_id, confirmation_source, mapping_snapshot_json,
                         requires_manifest, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sequence_id, dataset_id, sequence_name, "zip", inference.source,
                        now if trusted else None, project_id if trusted else None,
                        inference.source if trusted else None, json.dumps(snapshot, ensure_ascii=False),
                        int(inference.requires_manifest), now,
                    ),
                )
            for frame_index, image in enumerate(validated):
                image_id = str(uuid.uuid4())
                mapping = mapping_by_name.get(image["file_name"])
                conn.execute(
                    """INSERT INTO images (
                         id, dataset_id, file_name, width, height, frame_index, sequence_id,
                         original_frame_index, timestamp_us, timestamp_source, frame_index_source
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        image_id, dataset_id, image["file_name"], image["width"], image["height"], frame_index,
                        sequence_ids.get(mapping.sequence_name) if mapping else sequence_ids.get("unresolved"),
                        mapping.original_frame_index if mapping else frame_index,
                        mapping.timestamp_us if mapping else None,
                        inference.source if mapping and mapping.timestamp_us is not None else "unavailable",
                        inference.source if mapping else "import_order",
                    ),
                )
                for annotation in image["annotations"]:
                    conn.execute(
                        """INSERT INTO annotations (
                             id, image_id, label, keypoints_json, source_instance_id, source_track_id
                           ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()), image_id, annotation["label"], json.dumps(annotation["keypoints"]),
                            annotation.get("source_instance_id"), annotation.get("source_track_id"),
                        ),
                    )

        with db() as conn:
            summary = _sequence_summary(conn, dataset_id)
        return {
            "id": dataset_id,
            "name": Path(file.filename).stem,
            "status": "ready",
            "image_count": len(images),
            "schema_id": schema_id,
            "sequence_summary": summary,
        }
    except Exception:
        shutil.rmtree(location.parent, ignore_errors=True)
        raise


@router.post("/video", status_code=202)
def upload_video_dataset(
    project_id: str,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="MP4, MOV or AVI video")],
    schema_id: Annotated[str, Form()],
    annotation_file: Annotated[UploadFile | None, File(
        description="Optional CVAT XML, COCO JSON, landmark-qa/v1 or cleaned JSON mapped by original frame index"
    )] = None,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Upload an MP4, MOV or AVI video")
    load_schema(schema_id)
    dataset_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    dataset_root = DATA_DIR / "projects" / project_id / "datasets" / dataset_id
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
                (job_id, dataset_id, store_data_path(staging_path), MAX_VIDEO_FRAMES, MAX_DECODED_VIDEO_BYTES, now),
            )
        background.add_task(execute_video_import, job_id)
        return {
            "id": dataset_id, "name": Path(file.filename or "video").stem, "status": "processing",
            "image_count": 0, "schema_id": schema_id, "import_job_id": job_id,
        }
    except Exception:
        shutil.rmtree(dataset_root, ignore_errors=True)
        raise


@router.get("/{dataset_id}/import")
def get_video_import(
    project_id: str, dataset_id: str, _: sqlite3.Row = Depends(project_guard),
) -> dict:
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


@router.post("/{dataset_id}/import/retry", status_code=202)
def retry_video_import(
    project_id: str, dataset_id: str, background: BackgroundTasks,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
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
    background.add_task(execute_video_import, job_id)
    return {"dataset_id": dataset_id, "import_job_id": job_id, "attempt": attempt, "status": "processing"}


@router.get("/{dataset_id}/sequences")
def get_sequence_summary(
    project_id: str,
    dataset_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id FROM datasets WHERE id = ? AND project_id = ?", (dataset_id, project_id)
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        return _sequence_summary(conn, dataset_id)


@router.post("/{dataset_id}/sequences/confirm")
def confirm_sequences(
    project_id: str,
    dataset_id: str,
    payload: SequenceConfirmation,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    if payload.sampling_confirmed and payload.fps is None:
        raise HTTPException(400, "fps is required when sampling_confirmed is true")
    with db() as conn:
        dataset = conn.execute(
            "SELECT id FROM datasets WHERE id = ? AND project_id = ?", (dataset_id, project_id)
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        sequences = conn.execute(
            "SELECT id, requires_manifest FROM sequences WHERE dataset_id = ?", (dataset_id,)
        ).fetchall()
        if not sequences:
            raise HTTPException(409, "Dataset has no sequence candidate")
        if any(row["requires_manifest"] for row in sequences):
            raise HTTPException(409, "Sequence mapping is ambiguous; upload a sequence manifest")

        now = int(time.time())
        for sequence in sequences:
            conn.execute(
                """UPDATE sequences
                      SET confirmed_at = ?, confirmed_project_id = ?, confirmation_source = ?,
                          fps = ?, fps_source = ?, sampling_confirmed = ?
                    WHERE id = ?""",
                (
                    now, project_id, payload.confirmation_source, payload.fps,
                    "user_input" if payload.fps is not None else None,
                    int(payload.sampling_confirmed), sequence["id"],
                ),
            )
            if payload.sampling_confirmed and payload.fps:
                frames = conn.execute(
                    """SELECT id FROM images WHERE sequence_id = ?
                         ORDER BY original_frame_index, file_name""", (sequence["id"],)
                ).fetchall()
                for sampled_index, frame in enumerate(frames):
                    conn.execute(
                        """UPDATE images SET timestamp_us = ?, timestamp_source = 'fps_derived'
                             WHERE id = ? AND timestamp_us IS NULL""",
                        (round(sampled_index * 1_000_000 / payload.fps), frame["id"]),
                    )
        return _sequence_summary(conn, dataset_id)


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

    root = resolve_data_path(dataset["storage_path"])
    target = (root / image_path).resolve()

    if root not in target.parents or not target.is_file() or target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "Image not found")

    return FileResponse(target)
