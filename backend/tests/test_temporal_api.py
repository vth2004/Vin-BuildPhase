from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.main import app
from app.features.human_pose.services import scoring as pose_scoring
from app.routers import datasets as datasets_router
from app.shared.temporal.service import process_temporal_run


class TemporalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.patchers = [patch.object(database, "DATA_DIR", self.data_dir),
                         patch.object(database, "DB_PATH", self.data_dir / "test.sqlite3"),
                         patch.object(datasets_router, "DATA_DIR", self.data_dir)]
        for item in self.patchers:
            item.start()
        database.init_db()
        self.client = TestClient(app)
        project = self.client.post("/api/projects", json={"name": "temporal api"}).json()
        self.project_id = project["id"]
        self.headers = {"Authorization": f"Bearer {project['recovery_key']}"}
        self.dataset_id = str(uuid.uuid4())
        self.sequence_id = str(uuid.uuid4())
        now = int(time.time())
        with database.db() as conn:
            conn.execute("INSERT INTO datasets (id, project_id, name, status, storage_path, schema_id, created_at) VALUES (?, ?, 'fixture', 'ready', ?, 'vf_humanpose17_v1', ?)",
                         (self.dataset_id, self.project_id, str(self.data_dir), now))
            conn.execute("""INSERT INTO sequences (id, dataset_id, name, source, frame_order_source, confirmed_at,
                          confirmed_project_id, confirmation_source, created_at)
                          VALUES (?, ?, 'seq', 'zip', 'manifest', ?, ?, 'manifest', ?)""",
                         (self.sequence_id, self.dataset_id, now, self.project_id, now))

    def tearDown(self) -> None:
        for item in reversed(self.patchers):
            item.stop()
        self.temp.cleanup()

    def _create_run(self, enabled: bool = True, window: int = 7) -> dict:
        with patch("app.routers.runs.dispatch_run"):
            response = self.client.post(
                f"/api/projects/{self.project_id}/runs",
                headers=self.headers,
                params={"dataset_id": self.dataset_id, "temporal_enabled": str(enabled).lower(), "window_size": window},
            )
        self.assertEqual(202, response.status_code, response.text)
        return response.json()

    def _seed_track(self, run_id: str) -> None:
        track_id = "track-1"
        now = int(time.time())
        with database.db() as conn:
            for frame in range(7):
                image_id, annotation_id = str(uuid.uuid4()), str(uuid.uuid4())
                origin_x, origin_y = 100 + frame, 200
                points = {
                    1: {"id": 1, "x": origin_x + 10, "y": origin_y - 10, "conf": 0.9},
                    6: {"id": 6, "x": origin_x - 10, "y": origin_y - 50},
                    7: {"id": 7, "x": origin_x + 10, "y": origin_y - 50},
                    12: {"id": 12, "x": origin_x - 10, "y": origin_y},
                    13: {"id": 13, "x": origin_x + 10, "y": origin_y},
                }
                annotation = [dict(point, state="visible") for point in points.values()]
                if frame == 3:
                    annotation[0]["x"] += 100
                conn.execute("""INSERT INTO images (id, dataset_id, file_name, width, height, sequence_id,
                              original_frame_index, timestamp_source, frame_index_source)
                              VALUES (?, ?, ?, 640, 480, ?, ?, 'unavailable', 'manifest')""",
                             (image_id, self.dataset_id, f"{frame}.jpg", self.sequence_id, frame))
                conn.execute("""INSERT INTO annotations (id, image_id, label, keypoints_json, internal_track_id, tracking_status)
                              VALUES (?, ?, 'person', ?, ?, 'confirmed')""",
                             (annotation_id, image_id, json.dumps(annotation), track_id))
                conn.execute("""INSERT INTO temporal_observations (id, run_id, sequence_id, segment_id,
                              internal_track_id, image_id, detection_index, bbox_json, keypoints_json,
                              observation_source, supporting_model_count, supporting_models_json,
                              ensemble_disagreement, tracking_status)
                              VALUES (?, ?, ?, 0, ?, ?, 0, ?, ?, 'ensemble', 2, '["a","b"]', 0.01, 'confirmed')""",
                             (str(uuid.uuid4()), run_id, self.sequence_id, track_id, image_id,
                              json.dumps([origin_x - 30, origin_y - 70, origin_x + 30, origin_y + 20]), json.dumps(points)))

    def test_create_run_validates_and_snapshots_temporal_config(self) -> None:
        run = self._create_run()
        self.assertTrue(run["temporal_enabled"])
        self.assertEqual(7, run["temporal_window_size"])
        self.assertEqual("robust-local-fit-v2", run["temporal_algorithm_version"])
        with patch("app.routers.runs.dispatch_run"):
            invalid = self.client.post(f"/api/projects/{self.project_id}/runs", headers=self.headers,
                                       params={"dataset_id": self.dataset_id, "temporal_enabled": True, "window_size": 6})
        self.assertEqual(422, invalid.status_code)

    def test_processor_and_result_apis(self) -> None:
        run = self._create_run()
        self._seed_track(run["id"])
        process_temporal_run(run["id"], "pose", list(range(1, 18)), database.db)
        results = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/results",
                                  headers=self.headers, params={"warning_only": True})
        self.assertEqual(200, results.status_code, results.text)
        self.assertGreater(results.json()["total"], 0)
        detail_id = results.json()["items"][0]["id"]
        detail = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/results/{detail_id}", headers=self.headers)
        self.assertIn("trajectory", detail.json())
        summary = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/summary", headers=self.headers)
        self.assertEqual(119, summary.json()["total"])
        self.assertGreater(summary.json()["available"], 0)
        with database.db() as conn:
            unavailable_warnings = conn.execute("""SELECT COUNT(*) FROM warnings w JOIN temporal_results t
                ON t.run_id = w.run_id AND t.annotation_id = w.annotation_id AND t.keypoint_id = CAST(w.keypoint AS INTEGER)
                WHERE w.run_id = ? AND t.estimate_status = 'unavailable'""", (run["id"],)).fetchone()[0]
        self.assertEqual(0, unavailable_warnings)

    def test_review_decisions_are_scoped_to_annotation_and_snapshot_coordinates(self) -> None:
        run = self._create_run()
        self._seed_track(run["id"])
        process_temporal_run(run["id"], "pose", list(range(1, 18)), database.db)
        with database.db() as conn:
            first = conn.execute(
                """SELECT * FROM temporal_results WHERE run_id = ? AND estimate_status = 'available'
                   AND annotation_id IS NOT NULL ORDER BY frame_index LIMIT 1""", (run["id"],),
            ).fetchone()
        self.assertIsNotNone(first)
        endpoint = f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/results/{first['id']}/review"
        chosen = self.client.put(endpoint, headers=self.headers, json={"source": "temporal"})
        self.assertEqual(200, chosen.status_code, chosen.text)
        self.assertEqual(first["temporal_x"], chosen.json()["selected_x"])
        self.assertEqual(first["annotation_id"], chosen.json()["annotation_id"])

        with database.db() as conn:
            conn.execute("UPDATE temporal_results SET temporal_x = temporal_x + 50 WHERE id = ?", (first["id"],))
        listing = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/reviews", headers=self.headers)
        self.assertEqual(1, listing.json()["total"])
        self.assertEqual(first["temporal_x"], listing.json()["items"][0]["selected_x"])
        summary = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/summary", headers=self.headers)
        self.assertEqual(1, summary.json()["reviewed_temporal"])

        kept = self.client.put(endpoint, headers=self.headers, json={"source": "human"})
        self.assertEqual(200, kept.status_code, kept.text)
        self.assertEqual(chosen.json()["id"], kept.json()["id"])
        self.assertEqual(1, self.client.get(
            f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/summary", headers=self.headers,
        ).json()["reviewed_human"])
        self.assertEqual(204, self.client.delete(endpoint, headers=self.headers).status_code)
        self.assertEqual(0, self.client.get(
            f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/reviews", headers=self.headers,
        ).json()["total"])
        with database.db() as conn:
            second = conn.execute(
                """SELECT id, annotation_id FROM temporal_results WHERE run_id = ?
                   AND estimate_status = 'available' AND keypoint_id = ? AND annotation_id != ? LIMIT 1""",
                (run["id"], first["keypoint_id"], first["annotation_id"]),
            ).fetchone()
        self.assertIsNotNone(second)
        second_endpoint = endpoint.replace(first["id"], second["id"])
        self.assertEqual(200, self.client.put(endpoint, headers=self.headers, json={"source": "human"}).status_code)
        self.assertEqual(200, self.client.put(second_endpoint, headers=self.headers, json={"source": "human"}).status_code)
        decisions = self.client.get(f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/reviews", headers=self.headers).json()
        self.assertEqual(2, decisions["total"])
        self.assertEqual({first["annotation_id"], second["annotation_id"]}, {item["annotation_id"] for item in decisions["items"]})

    def test_review_rejects_unavailable_outside_and_cross_project(self) -> None:
        run = self._create_run()
        self._seed_track(run["id"])
        process_temporal_run(run["id"], "pose", list(range(1, 18)), database.db)
        with database.db() as conn:
            unavailable = conn.execute(
                """SELECT * FROM temporal_results WHERE run_id = ? AND estimate_status = 'available'
                   AND annotation_id IS NOT NULL LIMIT 1""", (run["id"],),
            ).fetchone()
            conn.execute("UPDATE temporal_results SET estimate_status = 'unavailable' WHERE id = ?", (unavailable["id"],))
        self.assertIsNotNone(unavailable)
        endpoint = f"/api/projects/{self.project_id}/runs/{run['id']}/temporal/results/{unavailable['id']}/review"
        self.assertEqual(409, self.client.put(endpoint, headers=self.headers, json={"source": "temporal"}).status_code)
        with database.db() as conn:
            annotation = conn.execute("SELECT keypoints_json FROM annotations WHERE id = ?", (unavailable["annotation_id"],)).fetchone()
            points = json.loads(annotation["keypoints_json"])
            next(point for point in points if point["id"] == unavailable["keypoint_id"])["state"] = "outside"
            conn.execute("UPDATE annotations SET keypoints_json = ? WHERE id = ?", (json.dumps(points), unavailable["annotation_id"]))
        self.assertEqual(409, self.client.put(endpoint, headers=self.headers, json={"source": "human"}).status_code)
        other = self.client.post("/api/projects", json={"name": "other"}).json()
        forbidden = endpoint.replace(self.project_id, other["id"])
        self.assertEqual(404, self.client.put(forbidden, headers={"Authorization": f"Bearer {other['recovery_key']}"}, json={"source": "human"}).status_code)

    def test_review_migration_is_idempotent(self) -> None:
        database.init_db()
        database.init_db()
        with database.db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = ?", (database.TEMPORAL_REVIEWS_MIGRATION,)).fetchone()[0]
        self.assertEqual(1, count)

    def test_disabled_run_has_no_temporal_access_and_cross_project_is_hidden(self) -> None:
        disabled = self._create_run(enabled=False)
        response = self.client.get(f"/api/projects/{self.project_id}/runs/{disabled['id']}/temporal/summary", headers=self.headers)
        self.assertEqual(409, response.status_code)
        other = self.client.post("/api/projects", json={"name": "other"}).json()
        response = self.client.get(f"/api/projects/{other['id']}/runs/{disabled['id']}/temporal/summary",
                                   headers={"Authorization": f"Bearer {other['recovery_key']}"})
        self.assertEqual(404, response.status_code)

    def test_temporal_off_uses_legacy_pose_path(self) -> None:
        from PIL import Image

        run = self._create_run(enabled=False)
        image_id, annotation_id = str(uuid.uuid4()), str(uuid.uuid4())
        Image.new("RGB", (64, 64), "black").save(self.data_dir / "legacy.jpg")
        points = [{"id": point_id, "x": 10 + point_id, "y": 20 + point_id, "state": "visible"}
                  for point_id in range(1, 18)]
        with database.db() as conn:
            conn.execute("INSERT INTO images (id, dataset_id, file_name, width, height) VALUES (?, ?, 'legacy.jpg', 64, 64)",
                         (image_id, self.dataset_id))
            conn.execute("INSERT INTO annotations (id, image_id, label, keypoints_json) VALUES (?, ?, 'person', ?)",
                         (annotation_id, image_id, json.dumps(points)))

        class Predictor:
            def predict(self, _path):
                return [{"bbox": (5, 5, 55, 60), "keypoints": {
                    point_id: {"id": point_id, "x": 10 + point_id, "y": 20 + point_id, "conf": 0.9}
                    for point_id in range(1, 18)}}]

        schema = {"keypoints": [{"id": point_id, "name": str(point_id)} for point_id in range(1, 18)]}
        with patch.object(pose_scoring, "setup_predictors", return_value=({"yolo26s-pose": Predictor()}, "K1")), \
             patch.object(pose_scoring, "build_pose_observations", side_effect=AssertionError("temporal path used")):
            pose_scoring.run_scoring(run["id"], database.db, lambda _schema_id: schema, mode="K1")
        with database.db() as conn:
            stored = conn.execute("SELECT status FROM runs WHERE id = ?", (run["id"],)).fetchone()
            observations = conn.execute("SELECT COUNT(*) FROM temporal_observations WHERE run_id = ?", (run["id"],)).fetchone()[0]
        self.assertEqual("completed", stored["status"])
        self.assertEqual(0, observations)

    def test_pose_run_integrates_tracking_engine_and_results(self) -> None:
        from PIL import Image

        run = self._create_run(enabled=True)
        for frame in range(7):
            image_id, annotation_id = str(uuid.uuid4()), str(uuid.uuid4())
            Image.new("RGB", (64, 64), (frame, frame, frame)).save(self.data_dir / f"frame-{frame}.jpg")
            points = [{"id": point_id, "x": 20 + point_id, "y": 30 + point_id, "state": "visible"}
                      for point_id in range(1, 18)]
            with database.db() as conn:
                conn.execute("""INSERT INTO images (id, dataset_id, file_name, width, height, sequence_id,
                              original_frame_index, timestamp_source, frame_index_source)
                              VALUES (?, ?, ?, 64, 64, ?, ?, 'unavailable', 'manifest')""",
                             (image_id, self.dataset_id, f"frame-{frame}.jpg", self.sequence_id, frame))
                conn.execute("INSERT INTO annotations (id, image_id, label, keypoints_json) VALUES (?, ?, 'person', ?)",
                             (annotation_id, image_id, json.dumps(points)))

        class Predictor:
            def predict(self, path):
                frame = int(Path(path).stem.split("-")[-1])
                keypoints = {point_id: {"id": point_id, "x": 20 + point_id + frame,
                                        "y": 30 + point_id, "conf": 0.9}
                             for point_id in range(1, 18)}
                return [{"bbox": (5 + frame, 5, 60 + frame, 63), "score": 0.9, "keypoints": keypoints}]

        schema = {"keypoints": [{"id": point_id, "name": str(point_id)} for point_id in range(1, 18)]}
        with patch.object(pose_scoring, "setup_predictors", return_value=({"yolo26s-pose": Predictor()}, "K1")):
            pose_scoring.run_scoring(run["id"], database.db, lambda _schema_id: schema, mode="K1")
        with database.db() as conn:
            observation_count = conn.execute("SELECT COUNT(*) FROM temporal_observations WHERE run_id = ?", (run["id"],)).fetchone()[0]
            result_count = conn.execute("SELECT COUNT(*) FROM temporal_results WHERE run_id = ?", (run["id"],)).fetchone()[0]
        self.assertEqual(7, observation_count)
        self.assertEqual(119, result_count)


if __name__ == "__main__":
    unittest.main()
