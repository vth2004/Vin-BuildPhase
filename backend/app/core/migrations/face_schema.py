"""Initial Face tables and migration marker."""
import sqlite3
import time
from .versions import FACE_MIGRATION


def apply(conn: sqlite3.Connection, applied_versions: set[str]) -> None:
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

    if FACE_MIGRATION not in applied_versions:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (FACE_MIGRATION, int(time.time())),
        )
