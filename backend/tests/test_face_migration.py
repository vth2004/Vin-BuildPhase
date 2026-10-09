from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core import database


class FaceMigrationTests(unittest.TestCase):
    def test_migration_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            db_path = data_dir / "legacy.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript("""
                    CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, secret_hash TEXT NOT NULL, created_at INTEGER NOT NULL);
                    CREATE TABLE datasets (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, image_count INTEGER NOT NULL DEFAULT 0, storage_path TEXT NOT NULL, schema_id TEXT, annotation_path TEXT, created_at INTEGER NOT NULL);
                    CREATE TABLE images (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, file_name TEXT NOT NULL, width INTEGER, height INTEGER, UNIQUE(dataset_id, file_name));
                    CREATE TABLE annotations (id TEXT PRIMARY KEY, image_id TEXT NOT NULL, label TEXT NOT NULL, keypoints_json TEXT NOT NULL);
                    CREATE TABLE runs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, dataset_id TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, mode TEXT DEFAULT 'K2', created_at INTEGER NOT NULL);
                    CREATE TABLE warnings (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_name TEXT NOT NULL, keypoint TEXT NOT NULL, warning_type TEXT NOT NULL, suspicion REAL NOT NULL, human_x REAL NOT NULL, human_y REAL NOT NULL, suggested_x REAL NOT NULL, suggested_y REAL NOT NULL, review TEXT, annotation_id TEXT, reliability REAL DEFAULT 1.0, delta REAL DEFAULT 0.0);
                    INSERT INTO projects VALUES ('p1', 'Existing', 'hash', 1);
                """)
                conn.commit()
            finally:
                conn.close()
            with patch.object(database, "DATA_DIR", data_dir), patch.object(database, "DB_PATH", db_path):
                database.init_db()
            conn = sqlite3.connect(db_path)
            try:
                self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0])
                image_columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
                self.assertIn("frame_index", image_columns)
                self.assertIsNotNone(conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='face_frame_results'"
                ).fetchone())
                self.assertIsNotNone(conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=?", (database.FACE_MIGRATION,)
                ).fetchone())
            finally:
                conn.close()
            self.assertEqual(1, len(list((data_dir / "backups").glob("*.sqlite3"))))


if __name__ == "__main__":
    unittest.main()
