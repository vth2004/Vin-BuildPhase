from __future__ import annotations

import sqlite3
import time
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.database import db
from app.core.security import project_guard
from app.services.mock_service import execute_run

router = APIRouter(prefix="/api/projects/{project_id}/runs", tags=["runs"])


@router.post("", status_code=202)
def create_run(
    project_id: str,
    dataset_id: str,
    background: BackgroundTasks,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id, schema_id FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        run_id = str(uuid.uuid4())
        target_mode = "K2" if dataset["schema_id"] == "vf_humanpose17_v1" else "mock"
        conn.execute(
            """INSERT INTO runs (
                id, project_id, dataset_id, status, progress, mode, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, project_id, dataset_id, "pending", 0, target_mode, int(time.time())),
        )
    background.add_task(execute_run, run_id)
    return {
        "id": run_id,
        "status": "pending",
        "mode": target_mode,
    }


@router.get("/{run_id}")
def get_run(
    project_id: str,
    run_id: str,
    _: sqlite3.Row = Depends(project_guard),
) -> dict:
    with db() as conn:
        run = conn.execute(
            "SELECT id, status, progress, dataset_id, mode FROM runs WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
    if not run:
        raise HTTPException(404, "Run not found")
    return dict(run)
