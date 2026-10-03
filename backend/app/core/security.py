from __future__ import annotations

import hashlib
import secrets
import sqlite3
from typing import Annotated
from fastapi import Header, HTTPException

from app.core.database import db


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def project_guard(
    project_id: str,
    authorization: Annotated[str | None, Header()] = None,
    token: str | None = None,
) -> sqlite3.Row:
    secret = None
    if authorization and authorization.startswith("Bearer "):
        secret = authorization[7:]
    elif token:
        secret = token

    if not secret:
        raise HTTPException(401, "Missing project recovery key")
    with db() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project or not secrets.compare_digest(project["secret_hash"], digest(secret)):
        raise HTTPException(403, "Invalid project recovery key")
    return project
