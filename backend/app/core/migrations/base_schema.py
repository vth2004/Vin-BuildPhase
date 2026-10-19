"""Core tables and additive columns, including legacy database upgrades."""
import sqlite3
from .utils import column_names as _column_names


def apply(conn: sqlite3.Connection) -> None:
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
    if "evidence" not in existing_warnings:
        conn.execute("ALTER TABLE warnings ADD COLUMN evidence TEXT")
    existing_runs = _column_names(conn, "runs")
    if "mode" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN mode TEXT DEFAULT 'K2'")
    if "error_message" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN error_message TEXT")
    if "temporal_enabled" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN temporal_enabled INTEGER NOT NULL DEFAULT 0")
    if "temporal_window_size" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN temporal_window_size INTEGER NOT NULL DEFAULT 15")
    if "temporal_algorithm_version" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN temporal_algorithm_version TEXT")
    if "input_mode" not in existing_runs:
        conn.execute("ALTER TABLE runs ADD COLUMN input_mode TEXT NOT NULL DEFAULT 'annotated'")

    existing_images = _column_names(conn, "images")
    if "frame_index" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN frame_index INTEGER")
    if "sequence_id" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN sequence_id TEXT")
    if "original_frame_index" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN original_frame_index INTEGER")
    if "timestamp_us" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN timestamp_us INTEGER")
    if "timestamp_source" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN timestamp_source TEXT NOT NULL DEFAULT 'unavailable'")
    if "frame_index_source" not in existing_images:
        conn.execute("ALTER TABLE images ADD COLUMN frame_index_source TEXT NOT NULL DEFAULT 'import_order'")

    existing_annotations = _column_names(conn, "annotations")
    if "source_instance_id" not in existing_annotations:
        conn.execute("ALTER TABLE annotations ADD COLUMN source_instance_id TEXT")
    if "source_track_id" not in existing_annotations:
        conn.execute("ALTER TABLE annotations ADD COLUMN source_track_id TEXT")
    if "internal_track_id" not in existing_annotations:
        conn.execute("ALTER TABLE annotations ADD COLUMN internal_track_id TEXT")
    if "tracking_status" not in existing_annotations:
        conn.execute("ALTER TABLE annotations ADD COLUMN tracking_status TEXT NOT NULL DEFAULT 'untracked'")
