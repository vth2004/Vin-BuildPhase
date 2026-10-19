from __future__ import annotations

import sqlite3
from typing import Any
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from app.core.security import project_guard
from .exporters.annotations import cleaned_landmark_qa, cvat_xml
from .exporters.report import generate_csv_report, generate_markdown_checklist
from .schemas.review import BatchFixRequest, PointSourceUpdate, ResultStatusUpdate
from .services import review
from .services.review import _export_results, _reports

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/face", tags=["face-vf50"])


@router.get("/results")
def list_results(
    project_id: str,
    run_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("severity", pattern="^(severity|frame_index|nme)$"),
    filter_rule: str | None = None,
    result_kind: str | None = Query("annotated_face"),
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.list_results(project_id, run_id, page, page_size, sort_by, filter_rule, result_kind)


@router.get("/results/{result_id}")
def get_result(
    project_id: str,
    run_id: str,
    result_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.get_result(project_id, run_id, result_id)


@router.put("/results/{result_id}/points/{point_id}")
def set_point_source(
    project_id: str,
    run_id: str,
    result_id: str,
    point_id: int,
    payload: PointSourceUpdate,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.set_point_source(project_id, run_id, result_id, point_id, payload)


@router.put("/results/{result_id}/status")
def set_result_status(
    project_id: str,
    run_id: str,
    result_id: str,
    payload: ResultStatusUpdate,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.set_result_status(project_id, run_id, result_id, payload)


@router.post("/batch-preview")
def preview_batch_fix(
    project_id: str,
    run_id: str,
    payload: BatchFixRequest,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.preview_batch_fix(project_id, run_id, payload)


@router.post("/batch-apply")
def apply_batch_fix(
    project_id: str,
    run_id: str,
    payload: BatchFixRequest,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return review.apply_batch_fix(project_id, run_id, payload)


@router.get("/export/cleaned-json")
def export_cleaned_json(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> JSONResponse:
    run, results = _export_results(project_id, run_id)
    return JSONResponse(
        cleaned_landmark_qa(run["dataset_name"], results),
        headers={"Content-Disposition": f'attachment; filename="vf50_{run_id}.json"'},
    )


@router.get("/export/cvat-xml")
def export_cvat_xml(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> Response:
    _, results = _export_results(project_id, run_id)
    return Response(
        cvat_xml(results), media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="vf50_{run_id}.xml"'},
    )


@router.get("/export/csv")
def export_csv(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> Response:
    _, results = _export_results(project_id, run_id)
    return Response(
        generate_csv_report(_reports(results)), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="vf50_{run_id}.csv"'},
    )


@router.get("/export/checklist")
def export_checklist(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> PlainTextResponse:
    run, results = _export_results(project_id, run_id)
    return PlainTextResponse(
        generate_markdown_checklist(run["dataset_name"], _reports(results)), media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="vf50_{run_id}_checklist.md"'},
    )
