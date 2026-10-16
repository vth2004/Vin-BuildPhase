from __future__ import annotations

import io
import json
import sqlite3
import sys
import tempfile
import time
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.features.human_pose.services import scoring as pose_scoring
from app.main import app
from app.routers import datasets as datasets_router
from app.shared.jobs import dispatch_run
from app.shared.temporal import infer_sequences


def _coco(file_names: list[str], *, track_id: str | None = None) -> dict:
    images = [
        {"id": index + 1, "file_name": name, "width": 64, "height": 64}
        for index, name in enumerate(file_names)
    ]
    annotations = []
    for index, image in enumerate(images):
        annotation = {
            "id": 100 + index,
            "image_id": image["id"],
            "category_id": 1,
            "keypoints": [value for point in range(17) for value in (10 + point, 20 + point, 2)],
        }
        if track_id is not None:
            annotation["attributes"] = {"track_id": track_id}
        annotations.append(annotation)
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "keypoints": [str(i) for i in range(17)]}],
    }


def _archive(file_names: list[str], manifest: dict | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in file_names:
            archive.writestr(name, b"not-an-image-but-upload-metadata-is-still-testable")
        if manifest is not None:
            archive.writestr("sequence_manifest.json", json.dumps(manifest))
    return output.getvalue()


class TemporalMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.db_path = self.data_dir / "test.sqlite3"
        self.patchers = [
            patch.object(database, "DATA_DIR", self.data_dir),
            patch.object(database, "DB_PATH", self.db_path),
            patch.object(datasets_router, "DATA_DIR", self.data_dir),
        ]
        for patcher in self.patchers:
            patcher.start()
        database.init_db()
        database.init_db()  # migration must be idempotent
        self.client = TestClient(app)
        project = self.client.post("/api/projects", json={"name": "temporal metadata"}).json()
        self.project_id = project["id"]
        self.headers = {"Authorization": f"Bearer {project['recovery_key']}"}

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def _upload(self, names: list[str], manifest: dict | None = None, track_id: str | None = None):
        return self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={
                "file": ("frames.zip", _archive(names, manifest), "application/zip"),
                "annotation_file": ("annotations.json", json.dumps(_coco(names, track_id=track_id)), "application/json"),
            },
        )

    def test_migration_adds_temporal_metadata_and_backfill_defaults(self) -> None:
        with database.db() as conn:
            self.assertIsNotNone(conn.execute("SELECT 1 FROM schema_migrations WHERE version='003_temporal_metadata'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM schema_migrations WHERE version='004_temporal_tracking'").fetchone())
            image_columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
            annotation_columns = {row[1] for row in conn.execute("PRAGMA table_info(annotations)")}
            run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            observation_columns = {row[1] for row in conn.execute("PRAGMA table_info(temporal_observations)")}
        self.assertTrue({"sequence_id", "original_frame_index", "timestamp_us", "frame_index_source"} <= image_columns)
        self.assertTrue({"source_instance_id", "source_track_id", "internal_track_id", "tracking_status"} <= annotation_columns)
        self.assertTrue({"temporal_enabled", "temporal_window_size", "temporal_algorithm_version"} <= run_columns)
        self.assertTrue({"internal_track_id", "observation_source", "supporting_model_count", "tracking_status"} <= observation_columns)

    def test_filename_candidate_requires_confirmation_and_derives_sampled_time(self) -> None:
        response = self._upload(["clip_001.jpg", "clip_002.jpg", "clip_003.jpg"])
        self.assertEqual(201, response.status_code, response.text)
        summary = response.json()["sequence_summary"]
        self.assertFalse(summary["temporal_ready"])
        self.assertEqual("filename_inferred", summary["sequences"][0]["frame_order_source"])
        confirmed = self.client.post(
            f"/api/projects/{self.project_id}/datasets/{response.json()['id']}/sequences/confirm",
            headers=self.headers,
            json={"fps": 5, "sampling_confirmed": True},
        )
        self.assertEqual(200, confirmed.status_code, confirmed.text)
        self.assertTrue(confirmed.json()["temporal_ready"])
        timestamps = [frame["timestamp_us"] for frame in confirmed.json()["sequences"][0]["frames"]]
        self.assertEqual([0, 200000, 400000], timestamps)

    def test_manifest_is_authoritative_and_source_ids_are_preserved(self) -> None:
        names = ["odd-name-a.jpg", "odd-name-b.jpg", "other-sequence.jpg"]
        manifest = {"images": [
            {"file_name": names[0], "sequence": "demo", "frame_index": 10, "timestamp_us": 1000},
            {"file_name": names[1], "sequence": "demo", "frame_index": 20, "timestamp_us": 2000},
            {"file_name": names[2], "sequence": "other", "frame_index": 0, "timestamp_us": 0},
        ]}
        response = self._upload(names, manifest=manifest, track_id="source-7")
        self.assertEqual(201, response.status_code, response.text)
        self.assertTrue(response.json()["sequence_summary"]["temporal_ready"])
        self.assertEqual(2, len(response.json()["sequence_summary"]["sequences"]))
        with database.db() as conn:
            annotation = conn.execute("SELECT source_instance_id, source_track_id FROM annotations LIMIT 1").fetchone()
            image = conn.execute(
                "SELECT original_frame_index, frame_index_source, timestamp_source FROM images WHERE file_name = ?",
                (names[0],),
            ).fetchone()
        self.assertEqual("100", annotation["source_instance_id"])
        self.assertEqual("source-7", annotation["source_track_id"])
        self.assertEqual((10, "manifest", "manifest"), tuple(image))

    def test_ambiguous_filename_requires_manifest(self) -> None:
        response = self._upload(["cam1_frame_001.jpg", "cam1_frame_002.jpg"])
        self.assertEqual(201, response.status_code, response.text)
        summary = response.json()["sequence_summary"]
        self.assertTrue(summary["sequences"][0]["requires_manifest"])
        confirmed = self.client.post(
            f"/api/projects/{self.project_id}/datasets/{response.json()['id']}/sequences/confirm",
            headers=self.headers,
            json={},
        )
        self.assertEqual(409, confirmed.status_code)

    def test_coco_array_order_is_distinct_from_numeric_image_id(self) -> None:
        inference = infer_sequences([
            {"file_name": "first-frame.jpg", "coco_array_position": 0, "coco_image_id": 900},
            {"file_name": "second-frame.jpg", "coco_array_position": 1, "coco_image_id": 100},
        ])
        self.assertEqual("coco_array_order", inference.source)
        self.assertEqual([0, 1], [mapping.original_frame_index for mapping in inference.mappings])

    def test_pose_failure_is_reported_without_mock_warnings(self) -> None:
        dataset_id = self._upload(["clip_001.jpg"]).json()["id"]
        run_id = str(uuid.uuid4())
        with database.db() as conn:
            conn.execute(
                "INSERT INTO runs (id, project_id, dataset_id, status, progress, mode, created_at) VALUES (?, ?, ?, 'pending', 0, 'K2', ?)",
                (run_id, self.project_id, dataset_id, int(time.time())),
            )
        with patch.object(pose_scoring, "run_scoring", side_effect=RuntimeError("intentional scoring failure")):
            dispatch_run(run_id)
        with database.db() as conn:
            run = conn.execute("SELECT status, error_message FROM runs WHERE id = ?", (run_id,)).fetchone()
            warning_count = conn.execute("SELECT COUNT(*) FROM warnings WHERE run_id = ?", (run_id,)).fetchone()[0]
        self.assertEqual("failed", run["status"])
        self.assertIn("intentional scoring failure", run["error_message"])
        self.assertEqual(0, warning_count)


if __name__ == "__main__":
    unittest.main()
