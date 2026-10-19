"""Sequence, temporal, video import and AI draft schema."""
import sqlite3
from .utils import column_names as _column_names


def apply(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sequences (
      id TEXT PRIMARY KEY,
      dataset_id TEXT NOT NULL,
      name TEXT NOT NULL,
      source TEXT NOT NULL,
      fps REAL,
      fps_source TEXT,
      frame_order_source TEXT NOT NULL,
      confirmed_at INTEGER,
      confirmed_project_id TEXT,
      confirmation_source TEXT,
      mapping_snapshot_json TEXT NOT NULL DEFAULT '{}',
      sampling_confirmed INTEGER NOT NULL DEFAULT 0,
      requires_manifest INTEGER NOT NULL DEFAULT 0,
      import_status TEXT NOT NULL DEFAULT 'ready',
      media_path TEXT,
      created_at INTEGER NOT NULL,
      UNIQUE(dataset_id, name)
    );
    CREATE INDEX IF NOT EXISTS idx_sequences_dataset ON sequences(dataset_id);
    CREATE INDEX IF NOT EXISTS idx_images_sequence_frame ON images(sequence_id, original_frame_index);
    CREATE TABLE IF NOT EXISTS temporal_observations (
      id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence_id TEXT,
      segment_id INTEGER NOT NULL, internal_track_id TEXT NOT NULL,
      image_id TEXT NOT NULL, detection_index INTEGER NOT NULL,
      bbox_json TEXT, keypoints_json TEXT NOT NULL,
      raw_keypoints_json TEXT,
      observation_source TEXT NOT NULL, supporting_model_count INTEGER NOT NULL,
      supporting_models_json TEXT NOT NULL, ensemble_disagreement REAL NOT NULL DEFAULT 0.0,
      tracking_status TEXT NOT NULL DEFAULT 'confirmed',
      UNIQUE(run_id, image_id, detection_index)
    );
    CREATE INDEX IF NOT EXISTS idx_temporal_observation_track
      ON temporal_observations(run_id, sequence_id, segment_id, internal_track_id);
    CREATE INDEX IF NOT EXISTS idx_temporal_observation_image
      ON temporal_observations(run_id, image_id);
    CREATE TABLE IF NOT EXISTS temporal_results (
      id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence_id TEXT,
      segment_id INTEGER NOT NULL, internal_track_id TEXT NOT NULL,
      image_id TEXT NOT NULL, annotation_id TEXT, frame_index INTEGER NOT NULL,
      keypoint_id INTEGER NOT NULL, raw_x REAL, raw_y REAL,
      human_x REAL, human_y REAL,
      temporal_x REAL, temporal_y REAL, estimate_status TEXT NOT NULL,
      method TEXT NOT NULL, unavailable_reason TEXT, temporal_quality REAL NOT NULL,
      temporal_score REAL, support_count INTEGER NOT NULL, residual REAL,
      gap_frames INTEGER, gap_us INTEGER, normalization_source TEXT, time_source TEXT NOT NULL,
      evidence TEXT, observation_source TEXT, supporting_model_count INTEGER NOT NULL DEFAULT 0,
      trajectory_json TEXT NOT NULL DEFAULT '[]', created_at INTEGER NOT NULL,
      UNIQUE(run_id, image_id, internal_track_id, keypoint_id)
    );
    CREATE INDEX IF NOT EXISTS idx_temporal_results_filter
      ON temporal_results(run_id, sequence_id, internal_track_id, frame_index, keypoint_id);
    CREATE INDEX IF NOT EXISTS idx_temporal_results_status
      ON temporal_results(run_id, estimate_status, evidence);
    CREATE TABLE IF NOT EXISTS video_import_jobs (
      id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, attempt INTEGER NOT NULL,
      status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
      error_message TEXT, staging_path TEXT NOT NULL,
      max_frames INTEGER NOT NULL, max_decoded_bytes INTEGER NOT NULL,
      created_at INTEGER NOT NULL, started_at INTEGER, finished_at INTEGER,
      UNIQUE(dataset_id, attempt)
    );
    CREATE INDEX IF NOT EXISTS idx_video_import_jobs_dataset
      ON video_import_jobs(dataset_id, attempt DESC);
    CREATE TABLE IF NOT EXISTS ai_draft_results (
      id TEXT PRIMARY KEY, run_id TEXT NOT NULL, image_id TEXT NOT NULL,
      schema_id TEXT NOT NULL, instance_index INTEGER NOT NULL,
      internal_track_id TEXT, keypoints_json TEXT NOT NULL,
      provenance_json TEXT NOT NULL, qc_json TEXT NOT NULL DEFAULT '{}',
      visibility_confirmed INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
      UNIQUE(run_id, image_id, instance_index)
    );
    CREATE INDEX IF NOT EXISTS idx_ai_drafts_run ON ai_draft_results(run_id, image_id);
    """)
    sequence_columns = _column_names(conn, "sequences")
    for column, definition in (
        ("video_codec", "TEXT"), ("video_width", "INTEGER"), ("video_height", "INTEGER"),
        ("video_frame_count", "INTEGER"), ("rotation_degrees", "INTEGER"),
    ):
        if column not in sequence_columns:
            conn.execute(f"ALTER TABLE sequences ADD COLUMN {column} {definition}")
    if "raw_keypoints_json" not in _column_names(conn, "temporal_observations"):
        conn.execute("ALTER TABLE temporal_observations ADD COLUMN raw_keypoints_json TEXT")
    temporal_result_columns = _column_names(conn, "temporal_results")
    if "human_x" not in temporal_result_columns:
        conn.execute("ALTER TABLE temporal_results ADD COLUMN human_x REAL")
    if "human_y" not in temporal_result_columns:
        conn.execute("ALTER TABLE temporal_results ADD COLUMN human_y REAL")

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS temporal_review_decisions (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      annotation_id TEXT NOT NULL,
      keypoint_id INTEGER NOT NULL,
      temporal_result_id TEXT NOT NULL,
      source TEXT NOT NULL CHECK(source IN ('human', 'temporal')),
      selected_x REAL NOT NULL,
      selected_y REAL NOT NULL,
      temporal_x REAL,
      temporal_y REAL,
      reviewed_at INTEGER NOT NULL,
      reviewed_project_id TEXT NOT NULL,
      UNIQUE(run_id, annotation_id, keypoint_id),
      FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
      FOREIGN KEY(annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_temporal_reviews_run
      ON temporal_review_decisions(run_id, source);
    """)

    conn.execute("""
      UPDATE images
         SET original_frame_index = COALESCE(original_frame_index, frame_index),
             frame_index_source = COALESCE(NULLIF(frame_index_source, ''), 'import_order'),
             timestamp_source = COALESCE(NULLIF(timestamp_source, ''), 'unavailable')
    """)
