from __future__ import annotations

import json
import sqlite3
from typing import Any
from fastapi import HTTPException
from app.core.database import db


def _run(project_id: str, run_id: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    if not row["temporal_enabled"]:
        raise HTTPException(409, "Temporal QA is disabled for this run")
    return row
