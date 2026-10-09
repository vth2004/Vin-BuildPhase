from __future__ import annotations

import json
import sqlite3
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from app.core.config import DATA_DIR, DB_PATH


FACE_MIGRATION = "001_face_landmark_vf50"
FACE_MIGRATION_V2 = "002_face_metrics_and_unlabeled"


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
    if _needs_migration(FACE_MIGRATION):
        _backup_database_before_migration(FACE_MIGRATION)
    elif _needs_migration(FACE_MIGRATION_V2):
        _backup_database_before_migration(FACE_MIGRATION_V2)
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
          nme REAL,
          quality_json TEXT NOT NULL DEFAULT '{}',
          violations_json TEXT NOT NULL DEFAULT '[]',
          severity REAL NOT NULL DEFAULT 0.0,
          error_count INTEGER NOT NULL DEFAULT 0,
          violation_count INTEGER NOT NULL DEFAULT 0,
          result_kind TEXT NOT NULL DEFAULT 'annotated_face',
          iod_source TEXT NOT NULL DEFAULT 'legacy',
          detector_confidence REAL DEFAULT NULL,
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

        applied_versions = {
            r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        if FACE_MIGRATION not in applied_versions:
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (FACE_MIGRATION, int(time.time())),
            )

        if FACE_MIGRATION_V2 not in applied_versions:
            table_info = conn.execute("PRAGMA table_info(face_frame_results)").fetchall()
            col_info = {col[1]: {"type": col[2], "notnull": col[3]} for col in table_info}
            needs_recreate = (
                "violation_count" not in col_info
                or col_info.get("nme", {}).get("notnull", 0) == 1
            )
            if needs_recreate:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("""
                CREATE TABLE face_frame_results_new (
                  id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  image_id TEXT NOT NULL,
                  annotation_id TEXT,
                  face_index INTEGER NOT NULL DEFAULT 0,
                  frame_index INTEGER NOT NULL DEFAULT 0,
                  human_keypoints_json TEXT,
                  model_keypoints_json TEXT,
                  iod REAL NOT NULL DEFAULT 0.0,
                  nme REAL,
                  quality_json TEXT NOT NULL DEFAULT '{}',
                  violations_json TEXT NOT NULL DEFAULT '[]',
                  severity REAL NOT NULL DEFAULT 0.0,
                  error_count INTEGER NOT NULL DEFAULT 0,
                  violation_count INTEGER NOT NULL DEFAULT 0,
                  result_kind TEXT NOT NULL DEFAULT 'annotated_face',
                  iod_source TEXT NOT NULL DEFAULT 'legacy',
                  detector_confidence REAL DEFAULT NULL,
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
                """)
                existing_rows = conn.execute("SELECT * FROM face_frame_results").fetchall()
                for row in existing_rows:
                    row_dict = dict(row)
                    v_raw = row_dict.get("violations_json")
                    v_list = json.loads(v_raw) if v_raw else []
                    old_err = row_dict.get("error_count", 0)
                    violation_cnt = len(v_list) if v_list else old_err
                    actual_errors = sum(
                        1 for v in v_list if isinstance(v, dict) and v.get("level") == "ERROR"
                    )
                    ann_id = row_dict.get("annotation_id")
                    res_kind = "annotated_face" if ann_id else "unlabeled_face"
                    q_raw = row_dict.get("quality_json")
                    q_data = json.loads(q_raw) if q_raw else {}
                    eval_cnt = q_data.get("evaluated_points_count", 0)
                    nme_val = row_dict.get("nme")
                    if (
                        res_kind == "unlabeled_face"
                        or eval_cnt == 0
                        or not row_dict.get("human_keypoints_json")
                        or not row_dict.get("model_keypoints_json")
                    ):
                        nme_val = None
                        if "nme" in q_data:
                            q_data["nme"] = None
                        if "nme_pct" in q_data:
                            q_data["nme_pct"] = None
                        if "passed_nme" in q_data:
                            q_data["passed_nme"] = None

                    conn.execute("""
                    INSERT INTO face_frame_results_new (
                      id, run_id, image_id, annotation_id, face_index, frame_index,
                      human_keypoints_json, model_keypoints_json, iod, nme,
                      quality_json, violations_json, severity, error_count, violation_count,
                      result_kind, iod_source, detector_confidence, status,
                      case_type, ai_reliability, case_label, case_description,
                      created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                      row_dict["id"], row_dict["run_id"], row_dict["image_id"], ann_id,
                      row_dict.get("face_index", 0), row_dict.get("frame_index", 0),
                      row_dict.get("human_keypoints_json"), row_dict.get("model_keypoints_json"),
                      row_dict.get("iod", 0.0), nme_val,
                      json.dumps(q_data, ensure_ascii=False), row_dict.get("violations_json", "[]"),
                      row_dict.get("severity", 0.0), actual_errors, violation_cnt,
                      res_kind, "legacy", None, row_dict.get("status", "pending"),
                      row_dict.get("case_type", "normal"), row_dict.get("ai_reliability", 1.0),
                      row_dict.get("case_label", ""), row_dict.get("case_description", ""),
                      row_dict["created_at"], row_dict["updated_at"],
                    ))
                conn.execute("DROP TABLE face_frame_results")
                conn.execute("ALTER TABLE face_frame_results_new RENAME TO face_frame_results")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_face_results_run ON face_frame_results(run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_face_results_image ON face_frame_results(image_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_face_results_status ON face_frame_results(run_id, status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_face_results_severity ON face_frame_results(run_id, severity DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_face_results_kind ON face_frame_results(run_id, result_kind)")
                conn.execute("PRAGMA foreign_keys = ON")

            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (FACE_MIGRATION_V2, int(time.time())),
            )
