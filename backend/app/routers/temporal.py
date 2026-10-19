from __future__ import annotations

import sqlite3
from typing import Any
from fastapi import APIRouter, Depends, Query, Response
from app.core.security import project_guard
from app.schemas.temporal import TemporalReviewRequest
from app.application.temporal import review, query, audit

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/temporal", tags=["temporal"])


@router.get("/reviews")
def list_temporal_reviews(project_id: str, run_id: str,
                          _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return review.list_temporal_reviews(project_id, run_id)


@router.put("/results/{result_id}/review")
def set_temporal_review(project_id: str, run_id: str, result_id: str, payload: TemporalReviewRequest,
                        _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return review.set_temporal_review(project_id, run_id, result_id, payload)


@router.delete("/results/{result_id}/review", status_code=204)
def clear_temporal_review(project_id: str, run_id: str, result_id: str,
                          _: sqlite3.Row = Depends(project_guard)) -> None:
    return review.clear_temporal_review(project_id, run_id, result_id)


@router.get("/export/audit.json")
def export_temporal_audit_json(project_id: str, run_id: str,
                               _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return audit.export_temporal_audit_json(project_id, run_id)


@router.get("/export/audit.csv")
def export_temporal_audit_csv(project_id: str, run_id: str,
                              _: sqlite3.Row = Depends(project_guard)) -> Response:
    content = audit.export_temporal_audit_csv(project_id, run_id)
    return Response(content, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="temporal-audit-{run_id[:8]}.csv"'})


@router.get("/results")
def list_temporal_results(project_id: str, run_id: str, page: int = Query(1, ge=1),
                          page_size: int = Query(50, ge=1, le=200), sequence_id: str | None = None,
                          track_id: str | None = None, frame_index: int | None = None,
                          keypoint_id: int | None = None, estimate_status: str | None = None,
                          evidence: str | None = None, warning_only: bool = False,
                          _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return query.list_temporal_results(project_id, run_id, page, page_size, sequence_id, track_id, frame_index, keypoint_id, estimate_status, evidence, warning_only)


@router.get("/results/{result_id}")
def temporal_result_detail(project_id: str, run_id: str, result_id: str,
                           _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return query.temporal_result_detail(project_id, run_id, result_id)


@router.get("/summary")
def temporal_summary(project_id: str, run_id: str,
                     _: sqlite3.Row = Depends(project_guard)) -> dict[str, Any]:
    return query.temporal_summary(project_id, run_id)
