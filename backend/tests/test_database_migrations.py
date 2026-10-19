from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core import database
from app.core.migrations.versions import MIGRATION_VERSIONS


class DatabaseMigrationTests(unittest.TestCase):
    def test_fresh_schema_matches_pre_refactor_schema_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(database, "DATA_DIR", root), patch.object(database, "DB_PATH", root / "db.sqlite3"):
                database.init_db()
                with database.db() as conn:
                    before = [tuple(row) for row in conn.execute("SELECT * FROM schema_migrations ORDER BY version")]
                database.init_db()
                with database.db() as conn:
                    schema = [(row["type"], row["name"], re.sub(r"\s+", " ", row["sql"].strip()))
                              for row in conn.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name")]
                    after = [tuple(row) for row in conn.execute("SELECT * FROM schema_migrations ORDER BY version")]
                    self.assertEqual(before, after)
                    self.assertEqual(list(MIGRATION_VERSIONS), [row[0] for row in after])
                # Captured from d4a142c before splitting the initializer: includes tables,
                # columns, indexes, uniqueness constraints and foreign keys.
                digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False).encode()).hexdigest()
                self.assertEqual("72ee9c533b1ec7455afe511c02a92d5b74359d8942f4d6ddcc28850c36081c98", digest)

    def test_legacy_face_metrics_upgrade_preserves_results_and_recalculates_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "legacy.sqlite3"
            with patch.object(database, "DATA_DIR", root), patch.object(database, "DB_PATH", target):
                database.init_db()
                with closing(sqlite3.connect(target)) as conn, conn:
                    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='face_frame_results'").fetchone()[0]
                    sql = sql.replace("nme REAL,", "nme REAL NOT NULL,")
                    for column in ("violation_count", "result_kind", "iod_source", "detector_confidence"):
                        sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith(column + " "))
                    conn.execute("DROP TABLE face_frame_results")
                    conn.execute(sql)
                    conn.execute("DELETE FROM schema_migrations WHERE version=?", (database.FACE_MIGRATION_V2,))
                    conn.execute("INSERT INTO projects VALUES ('p','Demo','hash',1)")
                    conn.execute("INSERT INTO datasets (id,project_id,name,status,storage_path,created_at) VALUES ('d','p','Demo','ready','images',1)")
                    conn.execute("INSERT INTO images (id,dataset_id,file_name) VALUES ('i','d','frame.jpg')")
                    conn.execute("INSERT INTO annotations (id,image_id,label,keypoints_json) VALUES ('a','i','face','[]')")
                    conn.execute("INSERT INTO runs (id,project_id,dataset_id,status,created_at) VALUES ('r','p','d','completed',1)")
                    conn.execute("""INSERT INTO face_frame_results
                        (id,run_id,image_id,annotation_id,human_keypoints_json,model_keypoints_json,nme,
                         quality_json,violations_json,error_count,created_at,updated_at)
                        VALUES ('f','r','i','a','{}','{}',0.1,?, ?,2,1,1)""",
                        (json.dumps({"evaluated_points_count": 1}), json.dumps([{"level": "ERROR"}, {"level": "WARNING"}])))
                database.init_db()
                database.init_db()
                with database.db() as conn:
                    row = conn.execute("SELECT * FROM face_frame_results WHERE id='f'").fetchone()
                    self.assertEqual("a", row["annotation_id"])
                    self.assertEqual(1, row["error_count"])
                    self.assertEqual(2, row["violation_count"])
                    self.assertEqual("annotated_face", row["result_kind"])
                    self.assertEqual(0.1, row["nme"])
                    self.assertEqual([], conn.execute("PRAGMA foreign_key_check").fetchall())
                self.assertEqual(1, len(list((root / "backups").glob("*.sqlite3"))))
