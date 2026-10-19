from __future__ import annotations

import sqlite3
import shutil
import time
from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
from app.core.config import DATA_DIR, DB_PATH


def store_data_path(path: Path) -> str:
    """Persist a path relative to the configured data root."""
    return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()


def resolve_data_path(stored: str) -> Path:
    """Resolve new relative paths and still read legacy absolute paths."""
    path = Path(stored)
    if path.is_absolute():
        return path.resolve()
    root = DATA_DIR.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Stored data path escapes LANDMARK_DATA_DIR")
    return resolved


from app.core.migrations import apply_migrations
from app.core.migrations.versions import (
    FACE_MIGRATION,
    FACE_MIGRATION_V2,
    TEMPORAL_METADATA_MIGRATION,
    TEMPORAL_TRACKING_MIGRATION,
    TEMPORAL_RESULTS_MIGRATION,
    TEMPORAL_REVIEWS_MIGRATION,
    VIDEO_INGESTION_MIGRATION,
    VIDEO_IMPORT_JOBS_MIGRATION,
    AI_DRAFT_MIGRATION,
    MIGRATION_VERSIONS,
)


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _needs_migration(version: str) -> bool:
    if not DB_PATH.is_file():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            has_migrations = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
            ).fetchone()
            if not has_migrations:
                return True
            return conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
            ).fetchone() is None
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def _needs_face_migration() -> bool:
    return _needs_migration(FACE_MIGRATION)


def _backup_database_before_migration(version: str = FACE_MIGRATION) -> None:
    if not _needs_migration(version):
        return
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{DB_PATH.stem}-before-{version}-{timestamp}{DB_PATH.suffix}"
    shutil.copy2(DB_PATH, target)


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for version in MIGRATION_VERSIONS:
        if _needs_migration(version):
            _backup_database_before_migration(version)
            break
    with db() as conn:
        apply_migrations(conn)
