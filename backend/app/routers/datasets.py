from __future__ import annotations

import sqlite3
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from app.core.config import DATA_DIR, MAX_DECODED_VIDEO_BYTES, MAX_VIDEO_FRAMES
from app.core.security import project_guard
from app.schemas.dataset import SequenceConfirmation
from app.application.datasets import archive, video, sequences, images
from app.shared.video_import import execute_video_import

router = APIRouter(prefix="/api/projects/{project_id}/datasets", tags=["datasets"])


@router.post("", status_code=201)
def upload_dataset(
    project_id: str,
    file: Annotated[UploadFile, File(description="ZIP containing image files")],
    schema_id: Annotated[str, Form()],
    annotation_file: Annotated[UploadFile | None, File(description="CVAT XML, COCO JSON, landmark-qa/v1 or cleaned JSON")] = None,
    sequence_manifest_file: Annotated[UploadFile | None, File(description="Optional sequence_manifest.json override")] = None,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return archive.upload_dataset(project_id, file, schema_id, annotation_file, sequence_manifest_file, data_dir=DATA_DIR)


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
    result = video.upload_video_dataset(project_id, file, schema_id, annotation_file, data_dir=DATA_DIR, max_frames=MAX_VIDEO_FRAMES, max_decoded_bytes=MAX_DECODED_VIDEO_BYTES)
    background.add_task(execute_video_import, result["import_job_id"])
    return result


@router.get("/{dataset_id}/import")
def get_video_import(
    project_id: str, dataset_id: str, _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return video.get_video_import(project_id, dataset_id)


@router.post("/{dataset_id}/import/retry", status_code=202)
def retry_video_import(
    project_id: str, dataset_id: str, background: BackgroundTasks,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    result = video.retry_video_import(project_id, dataset_id)
    background.add_task(execute_video_import, result["import_job_id"])
    return result


@router.get("/{dataset_id}/sequences")
def get_sequence_summary(
    project_id: str,
    dataset_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return sequences.get_sequence_summary(project_id, dataset_id)


@router.post("/{dataset_id}/sequences/confirm")
def confirm_sequences(
    project_id: str,
    dataset_id: str,
    payload: SequenceConfirmation,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return sequences.confirm_sequences(project_id, dataset_id, payload)


@router.get("/{dataset_id}/images/{image_path:path}")
def get_image(
    project_id: str,
    dataset_id: str,
    image_path: str,
    _: sqlite3.Row = Depends(project_guard),
) -> FileResponse:
    return FileResponse(images.get_image(project_id, dataset_id, image_path))
