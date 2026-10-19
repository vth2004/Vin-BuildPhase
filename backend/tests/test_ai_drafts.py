from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.main import app
from app.routers import datasets as datasets_router
from app.features.human_pose.services import scoring as pose_scoring
from app.features.face_landmark_vf50.adapters import mediapipe_face


def image_zip(name: str = "clip_000.jpg") -> bytes:
    image = io.BytesIO()
    Image.new("RGB", (96, 72), "white").save(image, format="JPEG")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, image.getvalue())
    return output.getvalue()


class AiDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patchers = [
            patch.object(database, "DATA_DIR", self.root),
            patch.object(database, "DB_PATH", self.root / "test.sqlite3"),
            patch.object(datasets_router, "DATA_DIR", self.root),
        ]
        for patcher in self.patchers:
            patcher.start()
        database.init_db()
        self.client = TestClient(app)
        project = self.client.post("/api/projects", json={"name": "ai draft"}).json()
        self.project_id = project["id"]
        self.headers = {"Authorization": f"Bearer {project['recovery_key']}"}

    def tearDown(self) -> None:
        self.client.close()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def _upload(self, schema_id: str) -> str:
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets", headers=self.headers,
            data={"schema_id": schema_id}, files={"file": ("frames.zip", image_zip(), "application/zip")},
        )
        self.assertEqual(201, response.status_code, response.text)
        return response.json()["id"]

    def test_pose_unlabeled_run_has_no_annotation_metrics_and_exports_ai_draft(self) -> None:
        dataset_id = self._upload("vf_humanpose17_v1")

        class Predictor:
            def predict(self, _path):
                return [{"bbox": (10, 5, 80, 70), "score": 0.9, "keypoints": {
                    point_id: {"id": point_id, "x": 20 + point_id, "y": 25 + point_id, "conf": 0.8}
                    for point_id in range(1, 18)
                }}]

        with patch.object(pose_scoring, "setup_predictors", return_value=({"yolo26s-pose": Predictor()}, "K1")):
            run = self.client.post(
                f"/api/projects/{self.project_id}/runs?dataset_id={dataset_id}", headers=self.headers,
            )
        self.assertEqual(202, run.status_code, run.text)
        self.assertEqual("ai_qc", run.json()["input_mode"])
        run_id = run.json()["id"]
        drafts = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts", headers=self.headers,
        )
        self.assertEqual(200, drafts.status_code, drafts.text)
        self.assertEqual(1, drafts.json()["count"])
        self.assertFalse(drafts.json()["items"][0]["visibility_confirmed"])
        warnings = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/warnings", headers=self.headers,
        )
        self.assertEqual([], warnings.json())
        exported = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export", headers=self.headers,
        ).json()
        self.assertEqual("landmark-qa/v1", exported["format"])
        self.assertEqual("AI draft", exported["label"])
        self.assertFalse(exported["visibility_confirmed"])
        self.assertNotIn("nme", exported["images"][0]["annotations"][0]["qc"])
        coco = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export/coco", headers=self.headers,
        )
        self.assertEqual(200, coco.status_code, coco.text)
        self.assertIn("ai_draft_cvat_", coco.headers["content-disposition"])
        self.assertEqual(17, len(coco.json()["categories"][0]["keypoints"]))
        self.assertEqual(51, len(coco.json()["annotations"][0]["keypoints"]))
        cvat = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export/cvat", headers=self.headers,
        )
        self.assertEqual(200, cvat.status_code, cvat.text)
        self.assertIn("ai_draft_cvat_", cvat.headers["content-disposition"])
        self.assertEqual(17, len(ET.fromstring(cvat.content).findall(".//skeleton/points")))

    def test_face_unlabeled_run_exports_unconfirmed_ai_draft(self) -> None:
        dataset_id = self._upload("vf_face_landmark50_v1")

        class Detector:
            def detect_all_faces(self, _pixels):
                coords = np.array([[15.0 + index, 20.0 + index] for index in range(50)])
                return [(coords, np.empty((0, 2)), (5.0, 5.0, 80.0, 60.0), 4800.0)]

        with patch.object(mediapipe_face, "get_detector", return_value=Detector()):
            run = self.client.post(
                f"/api/projects/{self.project_id}/runs?dataset_id={dataset_id}", headers=self.headers,
            )
        self.assertEqual(202, run.status_code, run.text)
        run_id = run.json()["id"]
        exported = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export", headers=self.headers,
        )
        self.assertEqual(200, exported.status_code, exported.text)
        annotation = exported.json()["images"][0]["annotations"][0]
        self.assertEqual("AI draft", annotation["label"])
        self.assertFalse(annotation["visibility_confirmed"])
        self.assertEqual("mediapipe", annotation["provenance"]["model"])
        coco = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export/coco", headers=self.headers,
        ).json()
        self.assertEqual(50, len(coco["categories"][0]["keypoints"]))
        self.assertEqual(150, len(coco["annotations"][0]["keypoints"]))
        self.assertEqual(50, coco["annotations"][0]["num_keypoints"])
        cvat = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/ai-drafts/export/cvat", headers=self.headers,
        )
        root = ET.fromstring(cvat.content)
        self.assertEqual(7, len(root.findall("track")))
        self.assertEqual(50, len(root.findall(".//skeleton/points")))


if __name__ == "__main__":
    unittest.main()
