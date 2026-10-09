from __future__ import annotations

import io
import json
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
from app.main import app
from app.features.face_landmark_vf50.services.ingestion import parse_face_annotations
from app.routers import datasets as datasets_router


def annotation_payload() -> dict:
    points = [
        {"id": point_id, "name": str(point_id), "x": 100.0 + point_id, "y": 200.0, "state": "visible"}
        for point_id in range(50)
    ]
    return {
        "format": "landmark-qa/v1",
        "schema_id": "vf_face_landmark50_v1",
        "images": [{
            "file_name": "frame.jpg",
            "width": 1280,
            "height": 720,
            "annotations": [{"label": "face_landmarks", "keypoints": points}],
        }],
    }


class FaceApiTests(unittest.TestCase):
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
        self.client = TestClient(app)
        created = self.client.post("/api/projects", json={"name": "Face API test"}).json()
        self.project_id = created["id"]
        self.headers = {"Authorization": f"Bearer {created['recovery_key']}"}

    def tearDown(self) -> None:
        self.client.close()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def _upload(self) -> str:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("frame.jpg", b"not-needed-for-upload-validation")
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={
                "file": ("faces.zip", archive_bytes.getvalue(), "application/zip"),
                "annotation_file": ("annotations.json", json.dumps(annotation_payload()), "application/json"),
            },
        )
        self.assertEqual(201, response.status_code, response.text)
        return response.json()["id"]

    def test_upload_requires_recovery_key(self) -> None:
        response = self.client.post(f"/api/projects/{self.project_id}/datasets")
        self.assertIn(response.status_code, (401, 422))

    def test_face_failure_is_reported_without_mock_results(self) -> None:
        dataset_id = self._upload()
        created = self.client.post(
            f"/api/projects/{self.project_id}/runs?dataset_id={dataset_id}", headers=self.headers
        )
        self.assertEqual(202, created.status_code, created.text)
        run_id = created.json()["id"]
        state = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}", headers=self.headers
        )
        self.assertEqual("failed", state.json()["status"])
        self.assertTrue(state.json()["error_message"])
        with database.db() as conn:
            self.assertEqual(0, conn.execute(
                "SELECT COUNT(*) FROM face_frame_results WHERE run_id = ?", (run_id,)
            ).fetchone()[0])
            self.assertEqual(0, conn.execute(
                "SELECT COUNT(*) FROM warnings WHERE run_id = ?", (run_id,)
            ).fetchone()[0])

    def test_review_changes_export_but_not_source_annotation(self) -> None:
        dataset_id = self._upload()
        run_id = str(uuid.uuid4())
        result_id = str(uuid.uuid4())
        human = {str(point["id"]): point for point in annotation_payload()["images"][0]["annotations"][0]["keypoints"]}
        model = {key: {**point, "x": float(point["x"]) + 5.0} for key, point in human.items()}
        with database.db() as conn:
            image = conn.execute("SELECT id FROM images WHERE dataset_id = ?", (dataset_id,)).fetchone()
            annotation = conn.execute("SELECT id, keypoints_json FROM annotations WHERE image_id = ?", (image["id"],)).fetchone()
            source_before = annotation["keypoints_json"]
            conn.execute(
                "INSERT INTO runs (id, project_id, dataset_id, status, progress, mode, created_at) VALUES (?, ?, ?, 'completed', 100, 'face_mediapipe_vf50', ?)",
                (run_id, self.project_id, dataset_id, int(time.time())),
            )
            conn.execute(
                """INSERT INTO face_frame_results (
                     id, run_id, image_id, annotation_id, face_index, frame_index,
                     human_keypoints_json, model_keypoints_json, iod, nme, quality_json,
                     violations_json, severity, error_count, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 0, 0, ?, ?, 96, 0.04, '{}', '[]', 0, 0, 'pending', ?, ?)""",
                (result_id, run_id, image["id"], annotation["id"], json.dumps(human), json.dumps(model), int(time.time()), int(time.time())),
            )

        reviewed = self.client.put(
            f"/api/projects/{self.project_id}/runs/{run_id}/face/results/{result_id}/points/0",
            headers={**self.headers, "Content-Type": "application/json"},
            json={"source": "model"},
        )
        self.assertEqual(200, reviewed.status_code, reviewed.text)
        self.assertEqual("model", reviewed.json()["point_sources"]["0"])

        exported = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/face/export/cleaned-json",
            headers=self.headers,
        )
        self.assertEqual(200, exported.status_code, exported.text)
        exported_point = exported.json()["images"][0]["annotations"][0]["keypoints"][0]
        self.assertEqual(model["0"]["x"], exported_point["x"])
        round_trip = parse_face_annotations(exported.content, {"frame.jpg"})
        self.assertEqual(model["0"]["x"], round_trip[0]["annotations"][0]["keypoints"][0]["x"])
        self.assertEqual(1280, round_trip[0]["width"])
        with database.db() as conn:
            source_after = conn.execute("SELECT keypoints_json FROM annotations WHERE id = ?", (annotation["id"],)).fetchone()[0]
        self.assertEqual(source_before, source_after)


if __name__ == "__main__":
    unittest.main()
