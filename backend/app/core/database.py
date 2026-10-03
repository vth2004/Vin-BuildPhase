from __future__ import annotations

import sqlite3
from app.core.config import DATA_DIR, DB_PATH


def db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript("""
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
          created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS warnings (
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_name TEXT NOT NULL,
          keypoint TEXT NOT NULL, warning_type TEXT NOT NULL, suspicion REAL NOT NULL,
          human_x REAL NOT NULL, human_y REAL NOT NULL, suggested_x REAL NOT NULL, suggested_y REAL NOT NULL,
          review TEXT, annotation_id TEXT
        );
        """)
        # Lightweight migration for existing databases
        existing = {row[1] for row in conn.execute("PRAGMA table_info(datasets)")}
        if "schema_id" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN schema_id TEXT")
        if "annotation_path" not in existing:
            conn.execute("ALTER TABLE datasets ADD COLUMN annotation_path TEXT")
        existing_warnings = {row[1] for row in conn.execute("PRAGMA table_info(warnings)")}
        if "annotation_id" not in existing_warnings:
            conn.execute("ALTER TABLE warnings ADD COLUMN annotation_id TEXT")
