from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.main import app
from app.routers import datasets as datasets_router


def create_zip(entries: list[tuple[str, bytes]]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries:
            z.writestr(name, data)
    buf.seek(0)
    return buf


class DatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        self.db_path = self.data_dir / "test.sqlite3"
        self.patchers = [
            patch.object(database, "DATA_DIR", self.data_dir),
            patch.object(database, "DB_PATH", self.db_path),
            patch.object(datasets_router, "DATA_DIR", self.data_dir),
        ]
        for p in self.patchers:
            p.start()
        database.init_db()
        self.client = TestClient(app)
        created = self.client.post("/api/projects", json={"name": "Validation test"}).json()
        self.project_id = created["id"]
        self.headers = {"Authorization": f"Bearer {created['recovery_key']}"}

    def tearDown(self) -> None:
        for p in self.patchers:
            p.stop()
        self.temp.cleanup()

    def test_reject_face_zip_in_human_pose_upload(self) -> None:
        # Giả lập file zip Face VF50 chứa annotations/person_keypoints_default.json với categories ngũ quan
        face_coco = {
            "images": [{"id": 1, "file_name": "face1.jpg", "width": 640, "height": 480}],
            "categories": [
                {"id": 1, "name": "longmaytrai", "keypoints": ["p1", "p2", "p3", "p4", "p5"]},
                {"id": 2, "name": "mattrai", "keypoints": ["p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13"]},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "keypoints": [10, 10, 2] * 5},
            ],
        }
        zip_buf = create_zip([
            ("images/face1.jpg", b"fake_image"),
            ("annotations/person_keypoints_default.json", json.dumps(face_coco).encode("utf-8")),
        ])

        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={"file": ("face_data.zip", zip_buf, "application/zip")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Face Landmark", response.json()["detail"])

    def test_reject_face_xml_in_human_pose_upload(self) -> None:
        zip_buf = create_zip([
            ("images/face1.jpg", b"fake_image"),
            ("annotations.xml", b"<annotations><version>1.1</version></annotations>"),
        ])

        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={"file": ("face_xml.zip", zip_buf, "application/zip")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Face Landmark", response.json()["detail"])

    def test_reject_human_pose_zip_in_face_landmark_upload(self) -> None:
        pose_coco = {
            "images": [{"id": 1, "file_name": "pose1.jpg", "width": 640, "height": 480}],
            "categories": [
                {"id": 1, "name": "person", "keypoints": ["k" + str(i) for i in range(17)]},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "keypoints": [10, 10, 2] * 17},
            ],
        }
        zip_buf = create_zip([
            ("images/pose1.jpg", b"fake_image"),
            ("annotations/person_keypoints_default.json", json.dumps(pose_coco).encode("utf-8")),
        ])

        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("pose_data.zip", zip_buf, "application/zip")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Human Pose", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
