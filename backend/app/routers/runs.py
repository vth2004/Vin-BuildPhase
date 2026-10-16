from __future__ import annotations

import sqlite3
import time
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.database import db
from app.core.security import project_guard
from app.features.face_landmark_vf50 import RUN_MODE as FACE_RUN_MODE, SCHEMA_ID as FACE_SCHEMA_ID
from app.shared.jobs import dispatch_run
from app.shared.temporal import TEMPORAL_ALGORITHM_VERSION, validate_window_size

router = APIRouter(prefix="/api/projects/{project_id}/runs", tags=["runs"])


@router.get("")
def list_runs(
    project_id: str,
    dataset_id: str | None = None,
    schema_id: str | None = None,
    _: sqlite3.Row = Depends(project_guard),
) -> list[dict]:
    clauses = ["r.project_id = ?"]
    params: list[str] = [project_id]
    if dataset_id:
        clauses.append("r.dataset_id = ?")
        params.append(dataset_id)
    if schema_id:
        clauses.append("d.schema_id = ?")
        params.append(schema_id)
    with db() as conn:
        rows = conn.execute(
            f"""SELECT r.id, r.dataset_id, r.status, r.progress, r.mode, r.error_message,
                       r.temporal_enabled, r.temporal_window_size, r.temporal_algorithm_version,
                       r.created_at, d.name AS dataset_name, d.schema_id
                FROM runs r JOIN datasets d ON d.id = r.dataset_id
                WHERE {' AND '.join(clauses)} ORDER BY r.created_at DESC""",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


@router.post("", status_code=202)
def create_run(
    project_id: str,
    dataset_id: str,
    background: BackgroundTasks,
    temporal_enabled: bool = False,
    window_size: int = 15,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id, schema_id, status FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        if dataset["status"] != "ready":
            raise HTTPException(409, f"Dataset is not ready (status: {dataset['status']})")
        try:
            validate_window_size(window_size)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if temporal_enabled:
            sequences = conn.execute(
                "SELECT confirmed_at, requires_manifest FROM sequences WHERE dataset_id = ?", (dataset_id,)
            ).fetchall()
            if not sequences or any(not row["confirmed_at"] or row["requires_manifest"] for row in sequences):
                raise HTTPException(409, "Temporal QA requires confirmed sequence metadata")
        run_id = str(uuid.uuid4())
        if dataset["schema_id"] == "vf_humanpose17_v1":
            target_mode = "K2"
        elif dataset["schema_id"] == FACE_SCHEMA_ID:
            target_mode = FACE_RUN_MODE
        else:
            raise HTTPException(400, f"Unsupported dataset schema: {dataset['schema_id']}")
        conn.execute(
            """INSERT INTO runs (
                id, project_id, dataset_id, status, progress, mode, created_at,
                temporal_enabled, temporal_window_size, temporal_algorithm_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, project_id, dataset_id, "pending", 0, target_mode, int(time.time()),
             int(temporal_enabled), window_size, TEMPORAL_ALGORITHM_VERSION if temporal_enabled else None),
        )
    background.add_task(dispatch_run, run_id)
    return {
        "id": run_id,
        "status": "pending",
        "mode": target_mode,
        "temporal_enabled": temporal_enabled,
        "temporal_window_size": window_size,
        "temporal_algorithm_version": TEMPORAL_ALGORITHM_VERSION if temporal_enabled else None,
    }


@router.get("/{run_id}")
def get_run(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        run = conn.execute(
            """SELECT id, status, progress, dataset_id, mode, error_message,
                      temporal_enabled, temporal_window_size, temporal_algorithm_version
               FROM runs WHERE id = ? AND project_id = ?""",
            (run_id, project_id),
        ).fetchone()
    if not run:
        raise HTTPException(404, "Run not found")
    return dict(run)
