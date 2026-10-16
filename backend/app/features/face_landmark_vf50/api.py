from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.database import db
from app.core.security import project_guard

from . import RUN_MODE
from .exporters.annotations import cleaned_landmark_qa, cvat_xml
from .exporters.report import generate_csv_report, generate_markdown_checklist
from .rules.validation import RuleViolation, check_all_rules
from .schemas.review import BatchFixRequest, PointSourceUpdate, ResultStatusUpdate
from .scoring.quality import evaluate_frame_quality

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/face", tags=["face-vf50"])


def _face_run(project_id: str, run_id: str) -> sqlite3.Row:
    with db() as conn:
        run = conn.execute(
            """SELECT r.*, d.name AS dataset_name
               FROM runs r JOIN datasets d ON d.id = r.dataset_id
               WHERE r.id = ? AND r.project_id = ? AND r.mode = ?""",
            (run_id, project_id, RUN_MODE),
        ).fetchone()
    if not run:
        raise HTTPException(404, "Face run not found")
    return run


def _json_mapping(value: str | None) -> dict[int, dict[str, Any]]:
    raw = json.loads(value or "{}")
    return {int(key): point for key, point in raw.items()}


def _point_sources(conn: sqlite3.Connection, result_id: str) -> dict[int, str]:
    return {
        int(row["point_id"]): row["source"]
        for row in conn.execute(
            "SELECT point_id, source FROM face_point_reviews WHERE result_id = ?", (result_id,)
        ).fetchall()
    }


def _reviewed_points(human: dict[int, dict[str, Any]], model: dict[int, dict[str, Any]], sources: dict[int, str]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for point_id in range(50):
        default_source = "human" if human else "model"
        source = sources.get(point_id, default_source)
        selected = model.get(point_id) if source == "model" else human.get(point_id)
        if selected is None:
            selected = human.get(point_id) or model.get(point_id) or {
                "id": point_id, "name": str(point_id), "x": None, "y": None, "state": "outside"
            }
        output[point_id] = dict(selected)
        output[point_id]["id"] = point_id
        output[point_id]["source"] = source
    return output


def _violation_dict(violation: RuleViolation) -> dict[str, Any]:
    return {
        "code": violation.code,
        "rule_name": violation.rule_name,
        "level": violation.level,
        "severity": "critical" if violation.level == "ERROR" else "major",
        "message": violation.message,
        "points": violation.points,
        "gl_section": violation.gl_section,
    }


def _recalculate_result(conn: sqlite3.Connection, result_id: str) -> None:
    row = conn.execute(
        """SELECT f.*, i.width, i.height
           FROM face_frame_results f JOIN images i ON i.id = f.image_id WHERE f.id = ?""",
        (result_id,),
    ).fetchone()
    if not row:
        return
    stored_iod = float(row["iod"])
    human = _json_mapping(row["human_keypoints_json"])
    model = _json_mapping(row["model_keypoints_json"])
    reviewed = _reviewed_points(human, model, _point_sources(conn, result_id))
    run = conn.execute("SELECT temporal_enabled FROM runs WHERE id = ?", (row["run_id"],)).fetchone()
    if run and run["temporal_enabled"] and row["annotation_id"]:
        track = conn.execute("SELECT internal_track_id, tracking_status FROM annotations WHERE id = ?",
                             (row["annotation_id"],)).fetchone()
        previous_row = conn.execute(
            """SELECT f.* FROM face_frame_results f JOIN annotations a ON a.id = f.annotation_id
               WHERE f.run_id = ? AND a.internal_track_id = ? AND a.tracking_status = 'confirmed'
                 AND f.frame_index < ? ORDER BY f.frame_index DESC LIMIT 1""",
            (row["run_id"], track["internal_track_id"], row["frame_index"]),
        ).fetchone() if track and track["internal_track_id"] and track["tracking_status"] == "confirmed" else None
    else:
        previous_row = conn.execute(
            """SELECT * FROM face_frame_results
               WHERE run_id = ? AND face_index = ? AND frame_index < ?
               ORDER BY frame_index DESC LIMIT 1""",
            (row["run_id"], row["face_index"], row["frame_index"]),
        ).fetchone()
    previous = None
    if previous_row:
        previous_human = _json_mapping(previous_row["human_keypoints_json"])
        previous_model = _json_mapping(previous_row["model_keypoints_json"])
        previous = _reviewed_points(
            previous_human, previous_model, _point_sources(conn, previous_row["id"])
        )
    violations = [
        _violation_dict(item) for item in check_all_rules(
            reviewed,
            prev_keypoints_by_id=previous,
            model_keypoints_by_id=model or None,
            image_width=row["width"] or 1280,
            image_height=row["height"] or 720,
            iod=stored_iod,
        )
    ]
    quality = evaluate_frame_quality(reviewed, model or None, iod=stored_iod)
    severity = sum(3.0 if item["level"] == "ERROR" else 2.0 for item in violations)
    error_count = sum(1 for item in violations if item["level"] == "ERROR")
    violation_count = len(violations)
    res_kind = row["result_kind"] if "result_kind" in row.keys() else "annotated_face"
    nme_val = quality["nme"] if res_kind != "unlabeled_face" else None
    if res_kind == "unlabeled_face":
        quality["nme"] = None
        quality["nme_pct"] = None
        quality["passed_nme"] = None

    conn.execute(
        """UPDATE face_frame_results SET iod = ?, nme = ?, quality_json = ?, violations_json = ?,
             severity = ?, error_count = ?, violation_count = ?, case_type = ?, ai_reliability = ?, case_label = ?,
             case_description = ?, updated_at = ? WHERE id = ?""",
        (
            stored_iod, nme_val, json.dumps(quality, ensure_ascii=False),
            json.dumps(violations, ensure_ascii=False), severity, error_count, violation_count,
            quality["case_type"], float(quality["ai_reliability"]), quality["case_label"],
            quality["case_description"], int(time.time()), result_id,
        ),
    )


def _result_detail(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    human = _json_mapping(row["human_keypoints_json"])
    model = _json_mapping(row["model_keypoints_json"])
    sources = _point_sources(conn, row["id"])
    detail = dict(row)
    detail["human_keypoints"] = human
    detail["model_keypoints"] = model
    detail["point_sources"] = sources
    detail["reviewed_keypoints"] = _reviewed_points(human, model, sources)
    detail["violations"] = json.loads(row["violations_json"] or "[]")
    detail["quality"] = json.loads(row["quality_json"] or "{}")
    for key in ("human_keypoints_json", "model_keypoints_json", "violations_json", "quality_json"):
        detail.pop(key, None)
    return detail


def _load_result(project_id: str, run_id: str, result_id: str) -> dict[str, Any]:
    _face_run(project_id, run_id)
    with db() as conn:
        row = conn.execute(
            """SELECT f.*, i.file_name, i.width, i.height
               FROM face_frame_results f JOIN images i ON i.id = f.image_id
               WHERE f.id = ? AND f.run_id = ?""",
            (result_id, run_id),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Face result not found")
        return _result_detail(conn, row)


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
    _face_run(project_id, run_id)
    order = {
        "severity": "f.severity DESC, f.error_count DESC, f.frame_index ASC",
        "frame_index": "f.frame_index ASC, f.face_index ASC",
        "nme": "CASE WHEN f.nme IS NULL THEN 1 ELSE 0 END ASC, f.nme DESC, f.frame_index ASC",
    }[sort_by]
    where = ["f.run_id = ?"]
    params: list[Any] = [run_id]
    if result_kind == "annotated_face":
        # annotation_id is authoritative. Older runs created before result_kind
        # existed can otherwise show unmatched MediaPipe detections or individual
        # skeleton fragments as fake faces.
        where.append("f.annotation_id IS NOT NULL")
        where.append("f.nme IS NOT NULL")
    elif result_kind == "unlabeled_face":
        where.append("f.annotation_id IS NULL")
    if filter_rule:
        where.append("f.violations_json LIKE ?")
        params.append(f'%"code": "{filter_rule}"%')
    predicate = " AND ".join(where)
    with db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM face_frame_results f WHERE {predicate}", params).fetchone()[0]
        unlabeled_count = conn.execute(
            "SELECT COUNT(*) FROM face_frame_results WHERE run_id = ? AND result_kind = 'unlabeled_face'",
            (run_id,),
        ).fetchone()[0]
        rows = conn.execute(
            f"""SELECT f.id, f.frame_index, f.face_index, f.iod, f.nme, f.severity,
                       f.error_count, f.violation_count, f.result_kind, f.iod_source, f.detector_confidence,
                       f.status, f.case_type, f.ai_reliability, f.case_label,
                       f.quality_json, f.violations_json, i.file_name, i.width, i.height
                FROM face_frame_results f JOIN images i ON i.id = f.image_id
                WHERE {predicate} ORDER BY {order} LIMIT ? OFFSET ?""",
            [*params, page_size, (page - 1) * page_size],
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["violations"] = json.loads(item.pop("violations_json") or "[]")
        quality_data = json.loads(item.pop("quality_json") or "{}")
        item["quality"] = quality_data
        item["e_max"] = quality_data.get("e_max")
        item["e_p90"] = quality_data.get("e_p90")
        item["exceed_count"] = quality_data.get("exceed_count", 0)
        item["has_outlier_point"] = quality_data.get("has_outlier_point", False)
        item["evaluated_points_count"] = quality_data.get("evaluated_points_count", 0)
        items.append(item)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "unlabeled_count": unlabeled_count,
        "items": items,
    }


@router.get("/results/{result_id}")
def get_result(
    project_id: str,
    run_id: str,
    result_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    return _load_result(project_id, run_id, result_id)


@router.put("/results/{result_id}/points/{point_id}")
def set_point_source(
    project_id: str,
    run_id: str,
    result_id: str,
    point_id: int,
    payload: PointSourceUpdate,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    if point_id < 0 or point_id >= 50:
        raise HTTPException(400, "point_id must be between 0 and 49")
    detail = _load_result(project_id, run_id, result_id)
    source_points = detail[f"{payload.source}_keypoints"]
    point = source_points.get(point_id)
    if not point or (payload.source == "model" and (point.get("x") is None or point.get("y") is None)):
        raise HTTPException(400, f"No usable {payload.source} coordinates for point {point_id}")
    now = int(time.time())
    with db() as conn:
        conn.execute(
            """INSERT INTO face_point_reviews (id, result_id, point_id, source, reviewed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(result_id, point_id) DO UPDATE SET source = excluded.source, reviewed_at = excluded.reviewed_at""",
            (str(uuid.uuid4()), result_id, point_id, payload.source, now),
        )
        status = "fixed" if payload.source == "model" or any(
            row[0] == "model" for row in conn.execute(
                "SELECT source FROM face_point_reviews WHERE result_id = ?", (result_id,)
            ).fetchall()
        ) else "reviewed"
        conn.execute("UPDATE face_frame_results SET status = ?, updated_at = ? WHERE id = ?", (status, now, result_id))
        _recalculate_result(conn, result_id)
    return _load_result(project_id, run_id, result_id)


@router.put("/results/{result_id}/status")
def set_result_status(
    project_id: str,
    run_id: str,
    result_id: str,
    payload: ResultStatusUpdate,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    _load_result(project_id, run_id, result_id)
    with db() as conn:
        conn.execute(
            "UPDATE face_frame_results SET status = ?, updated_at = ? WHERE id = ?",
            (payload.status, int(time.time()), result_id),
        )
    return _load_result(project_id, run_id, result_id)


def _batch_candidates(run_id: str, payload: BatchFixRequest) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with db() as conn:
        rows = conn.execute("SELECT * FROM face_frame_results WHERE run_id = ?", (run_id,)).fetchall()
        for row in rows:
            human = _json_mapping(row["human_keypoints_json"])
            model = _json_mapping(row["model_keypoints_json"])
            if not human or not model:
                continue
            violations = json.loads(row["violations_json"] or "[]")
            quality = json.loads(row["quality_json"] or "{}")
            point_ids: set[int] = set()
            if payload.only_violated:
                for violation in violations:
                    point_ids.update(int(point_id) for point_id in violation.get("points", []) if 0 <= int(point_id) < 50)
                for point_id, evaluation in quality.get("point_evaluations", {}).items():
                    if evaluation.get("exceeded"):
                        point_ids.add(int(point_id))
                if float(row["nme"]) < payload.min_nme and not violations:
                    point_ids.clear()
            else:
                point_ids.update(model)
            usable = sorted(point_id for point_id in point_ids if model.get(point_id, {}).get("x") is not None)
            if usable:
                candidates.append({"result_id": row["id"], "point_ids": usable})
    return candidates


@router.post("/batch-preview")
def preview_batch_fix(
    project_id: str,
    run_id: str,
    payload: BatchFixRequest,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    _face_run(project_id, run_id)
    candidates = _batch_candidates(run_id, payload)
    return {
        "result_count": len(candidates),
        "point_count": sum(len(item["point_ids"]) for item in candidates),
        "changes": candidates,
    }


@router.post("/batch-apply")
def apply_batch_fix(
    project_id: str,
    run_id: str,
    payload: BatchFixRequest,
    _: sqlite3.Row = Depends(project_guard),
) -> dict[str, Any]:
    _face_run(project_id, run_id)
    if not payload.confirm:
        raise HTTPException(400, "Batch fix requires confirm=true after preview")
    candidates = _batch_candidates(run_id, payload)
    now = int(time.time())
    with db() as conn:
        for candidate in candidates:
            for point_id in candidate["point_ids"]:
                conn.execute(
                    """INSERT INTO face_point_reviews (id, result_id, point_id, source, reviewed_at)
                       VALUES (?, ?, ?, 'model', ?)
                       ON CONFLICT(result_id, point_id) DO UPDATE SET source = 'model', reviewed_at = excluded.reviewed_at""",
                    (str(uuid.uuid4()), candidate["result_id"], point_id, now),
                )
            conn.execute(
                "UPDATE face_frame_results SET status = 'fixed', updated_at = ? WHERE id = ?",
                (now, candidate["result_id"]),
            )
            _recalculate_result(conn, candidate["result_id"])
    return {
        "result_count": len(candidates),
        "point_count": sum(len(item["point_ids"]) for item in candidates),
    }


def _export_results(project_id: str, run_id: str) -> tuple[sqlite3.Row, list[dict[str, Any]]]:
    run = _face_run(project_id, run_id)
    with db() as conn:
        rows = conn.execute(
            """SELECT f.*, i.file_name, i.width, i.height
               FROM face_frame_results f JOIN images i ON i.id = f.image_id
               WHERE f.run_id = ? ORDER BY f.frame_index, f.face_index""",
            (run_id,),
        ).fetchall()
        results = [_result_detail(conn, row) for row in rows]
    return run, results


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


def _reports(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "file_name": result["file_name"],
        "errors_count": sum(1 for item in result["violations"] if item.get("level") == "ERROR"),
        "warnings_count": sum(1 for item in result["violations"] if item.get("level") == "WARNING"),
        "violations": result["violations"],
        "quality": result["quality"],
    } for result in results]


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
