from __future__ import annotations

import secrets
import sqlite3
import time
import uuid
from fastapi import APIRouter, Depends

from app.core.database import db
from app.core.security import digest, project_guard
from app.schemas.project import CreateProject

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("", status_code=201)
def create_project(payload: CreateProject) -> dict:
    project_id = str(uuid.uuid4())
    recovery_key = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?)",
            (project_id, payload.name.strip(), digest(recovery_key), int(time.time())),
        )
    return {"id": project_id, "name": payload.name.strip(), "recovery_key": recovery_key}


@router.get("/{project_id}")
def get_project(project_id: str, _: sqlite3.Row = Depends(project_guard)) -> dict:
    with db() as conn:
        project = conn.execute("SELECT id, name, created_at FROM projects WHERE id = ?", (project_id,)).fetchone()
        datasets = conn.execute(
            "SELECT id, name, status, image_count, schema_id FROM datasets WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return {"project": dict(project), "datasets": [dict(row) for row in datasets]}
