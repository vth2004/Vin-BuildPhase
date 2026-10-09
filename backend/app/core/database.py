from __future__ import annotations

import sqlite3
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from app.core.config import DATA_DIR, DB_PATH


FACE_MIGRATION = "001_face_landmark_vf50"


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


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _needs_face_migration() -> bool:
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
                "SELECT 1 FROM schema_migrations WHERE version = ?", (FACE_MIGRATION,)
            ).fetchone() is None
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def _backup_database_before_migration() -> None:
    if not _needs_face_migration():
        return
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{DB_PATH.stem}-before-{FACE_MIGRATION}-{timestamp}{DB_PATH.suffix}"
    shutil.copy2(DB_PATH, target)


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _backup_database_before_migration()
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY, applied_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, secret_hash TEXT NOT NULL,
          created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
          status TEXT NOT NULL, image_count INTEGER NOT NULL DEFAULT 0,
          storage_path TEXT NOT NULL, schema_id TEXT, annotation_path TEXT, created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
          id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, file_name TEXT NOT NULL,
          width INTEGER, height INTEGER, UNIQUE(dataset_id, file_name)
        );
        CREATE TABLE IF NOT EXISTS annotations (
          id TEXT PRIMARY KEY, image_id TEXT NOT NULL, label TEXT NOT NULL,
          keypoints_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
          status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
          mode TEXT DEFAULT 'K2',
          created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS warnings (
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_name TEXT NOT NULL,
          keypoint TEXT NOT NULL, warning_type TEXT NOT NULL, suspicion REAL NOT NULL,
          human_x REAL NOT NULL, human_y REAL NOT NULL, suggested_x REAL NOT NULL, suggested_y REAL NOT NULL,
          review TEXT, annotation_id TEXT,
          reliability REAL DEFAULT 1.0, delta REAL DEFAULT 0.0
        );
        """)
        # Lightweight migration for existing databases
        existing = _column_names(conn, "datasets")
        if "schema_id" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN schema_id TEXT")
        if "annotation_path" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN annotation_path TEXT")
        existing_warnings = _column_names(conn, "warnings")
        if "annotation_id" not in existing_warnings:
            conn.execute("ALTER TABLE warnings ADD COLUMN annotation_id TEXT")
        if "reliability" not in existing_warnings:
            conn.execute("ALTER TABLE warnings ADD COLUMN reliability REAL DEFAULT 1.0")
        if "delta" not in existing_warnings:
            conn.execute("ALTER TABLE warnings ADD COLUMN delta REAL DEFAULT 0.0")
        existing_runs = _column_names(conn, "runs")
        if "mode" not in existing_runs:
            conn.execute("ALTER TABLE runs ADD COLUMN mode TEXT DEFAULT 'K2'")
        if "error_message" not in existing_runs:
            conn.execute("ALTER TABLE runs ADD COLUMN error_message TEXT")

        existing_images = _column_names(conn, "images")
        if "frame_index" not in existing_images:
            conn.execute("ALTER TABLE images ADD COLUMN frame_index INTEGER")

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS face_frame_results (
          id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          image_id TEXT NOT NULL,
          annotation_id TEXT,
          face_index INTEGER NOT NULL DEFAULT 0,
          frame_index INTEGER NOT NULL DEFAULT 0,
          human_keypoints_json TEXT,
          model_keypoints_json TEXT,
          iod REAL NOT NULL DEFAULT 0.0,
          nme REAL NOT NULL DEFAULT 0.0,
          quality_json TEXT NOT NULL DEFAULT '{}',
          violations_json TEXT NOT NULL DEFAULT '[]',
          severity REAL NOT NULL DEFAULT 0.0,
          error_count INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'pending',
          case_type TEXT NOT NULL DEFAULT 'normal',
          ai_reliability REAL NOT NULL DEFAULT 1.0,
          case_label TEXT NOT NULL DEFAULT '',
          case_description TEXT NOT NULL DEFAULT '',
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(run_id, image_id, face_index),
          FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
          FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
          FOREIGN KEY(annotation_id) REFERENCES annotations(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS face_point_reviews (
          id TEXT PRIMARY KEY,
          result_id TEXT NOT NULL,
          point_id INTEGER NOT NULL CHECK(point_id >= 0 AND point_id < 50),
          source TEXT NOT NULL CHECK(source IN ('human', 'model')),
          reviewed_at INTEGER NOT NULL,
          UNIQUE(result_id, point_id),
          FOREIGN KEY(result_id) REFERENCES face_frame_results(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_face_results_run ON face_frame_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_face_results_image ON face_frame_results(image_id);
        CREATE INDEX IF NOT EXISTS idx_face_results_status ON face_frame_results(run_id, status);
        CREATE INDEX IF NOT EXISTS idx_face_results_severity ON face_frame_results(run_id, severity DESC);
        CREATE INDEX IF NOT EXISTS idx_face_reviews_result ON face_point_reviews(result_id);
        """)
        face_result_columns = _column_names(conn, "face_frame_results")
        if "quality_json" not in face_result_columns:
            conn.execute("ALTER TABLE face_frame_results ADD COLUMN quality_json TEXT NOT NULL DEFAULT '{}'")
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (FACE_MIGRATION, int(time.time())),
        )
