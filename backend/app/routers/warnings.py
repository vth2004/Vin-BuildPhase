from __future__ import annotations

import sqlite3
from fastapi import APIRouter, Depends, Query, Response
from app.core.security import project_guard
from app.schemas.warning import ReviewWarning
from app.application.pose import review, export

router = APIRouter(prefix="/api/projects/{project_id}", tags=["warnings"])


@router.get("/runs/{run_id}/warnings")
def list_warnings(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> list[dict]:
    return review.list_warnings(project_id, run_id)


@router.post("/warnings/{warning_id}/review")
def review_warning(
    project_id: str,
    warning_id: str,
    payload: ReviewWarning,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return review.review_warning(project_id, warning_id, payload)


@router.get("/runs/{run_id}/export")
def export_dataset(
    project_id: str,
    run_id: str,
    response: Response,
    allow_partial: bool = Query(False),
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    headers: dict[str, str] = {}
    payload = export.export_dataset(project_id, run_id, allow_partial, headers)
    response.headers.update(headers)
    return payload


@router.get("/runs/{run_id}/warnings/{warning_id}/keypoints")
def get_warning_keypoints(
    project_id: str,
    run_id: str,
    warning_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return review.get_warning_keypoints(project_id, run_id, warning_id)


@router.get("/runs/{run_id}/images/{image_name:path}/keypoints")
def get_image_keypoints(
    project_id: str,
    run_id: str,
    image_name: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    return review.get_image_keypoints(project_id, run_id, image_name)
