from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageOps

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.main import app
from app.routers import datasets as datasets_router
from app.features.human_pose.adapters.yolo_pose import COCO_TO_VF17
from scripts.normalize_exif import normalize


def image_bytes(orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (40, 20), "navy")
    data = io.BytesIO()
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(data, format="JPEG", exif=exif)
    else:
        image.save(data, format="JPEG")
    return data.getvalue()


def archive(image: bytes) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as output:
        output.writestr("images/frame.jpg", image)
    return data.getvalue()


def coco_payload() -> dict:
    def keypoints(offset: int) -> list[float]:
        values: list[float] = []
        for index in range(17):
            visibility = 0 if index == 0 else (1 if index == 1 else 2)
            values.extend([float(offset + index), float(offset + 100 + index), visibility])
        return values

    return {
        "images": [{"id": 7, "file_name": "images/frame.jpg", "width": 40, "height": 20}],
        "categories": [{"id": 1, "name": "person", "keypoints": [name for _, name in COCO_TO_VF17.values()]}],
        "annotations": [
            {"id": 101, "image_id": 7, "category_id": 1, "keypoints": keypoints(10), "num_keypoints": 16, "extra": "keep-first"},
            {"id": 102, "image_id": 7, "category_id": 1, "keypoints": keypoints(200), "num_keypoints": 16, "extra": "keep-second"},
        ],
    }


def pose_lqa(width: int = 40, height: int = 20) -> dict:
    return {
        "format": "landmark-qa/v1",
        "schema_id": "vf_humanpose17_v1",
        "images": [{"file_name": "images/frame.jpg", "width": width, "height": height, "annotations": [{
            "label": "person",
            "keypoints": [{"id": vf_id, "name": name, "x": float(vf_id), "y": float(vf_id + 1), "state": "visible"} for vf_id, name in COCO_TO_VF17.values()],
        }]}],
    }


class PoseExportAndExifTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name) / "data"
        self.db_path = self.data_dir / "landmark_qa.sqlite3"
        self.patchers = [
            patch.object(database, "DATA_DIR", self.data_dir),
            patch.object(database, "DB_PATH", self.db_path),
            patch.object(datasets_router, "DATA_DIR", self.data_dir),
        ]
        for patcher in self.patchers:
            patcher.start()
        database.init_db()
        self.client = TestClient(app)
        created = self.client.post("/api/projects", json={"name": "pose-export"}).json()
        self.project_id = created["id"]
        self.headers = {"Authorization": f"Bearer {created['recovery_key']}"}

    def tearDown(self) -> None:
        self.client.close()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def upload(self, schema_id: str, annotation: dict, image: bytes | None = None) -> str:
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets",
            headers=self.headers,
            data={"schema_id": schema_id},
            files={
                "file": ("images.zip", archive(image or image_bytes()), "application/zip"),
                "annotation_file": ("annotation.json", json.dumps(annotation), "application/json"),
            },
        )
        self.assertEqual(201, response.status_code, response.text)
        return response.json()["id"]

    def add_run_warning(self, dataset_id: str, annotation_number: int, vf_id: int, *, annotation_id: str | None = "stored") -> tuple[str, str]:
        vf_name = dict(COCO_TO_VF17.values())[vf_id]
        with database.db() as conn:
            annotation_rows = conn.execute(
                "SELECT a.id FROM annotations a JOIN images i ON i.id=a.image_id WHERE i.dataset_id=? ORDER BY a.rowid",
                (dataset_id,),
            ).fetchall()
            run_id, warning_id = str(uuid.uuid4()), str(uuid.uuid4())
            conn.execute("INSERT INTO runs (id,project_id,dataset_id,status,progress,mode,created_at) VALUES (?,?,?,'completed',100,'K1',?)", (run_id, self.project_id, dataset_id, int(time.time())))
            conn.execute(
                """INSERT INTO warnings (id,run_id,image_name,keypoint,warning_type,suspicion,human_x,human_y,suggested_x,suggested_y,review,annotation_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?, 'use_suggestion', ?)""",
                (warning_id, run_id, "images/frame.jpg", vf_name, "test", 1, 0, 0, 9000.0 + vf_id, 8000.0 + vf_id,
                 annotation_rows[annotation_number]["id"] if annotation_id == "stored" else None),
            )
        return run_id, warning_id

    def export(self, run_id: str, suffix: str = ""):
        return self.client.get(f"/api/projects/{self.project_id}/runs/{run_id}/export{suffix}", headers=self.headers)

    def add_temporal_review(self, dataset_id: str, annotation_number: int, vf_id: int,
                            source: str = "temporal", selected: tuple[float, float] = (7001.0, 7002.0)) -> tuple[str, str]:
        now = int(time.time())
        with database.db() as conn:
            annotation = conn.execute(
                """SELECT a.id AS annotation_id, a.image_id, a.keypoints_json FROM annotations a
                   JOIN images i ON i.id = a.image_id WHERE i.dataset_id = ? ORDER BY a.rowid LIMIT 1 OFFSET ?""",
                (dataset_id, annotation_number),
            ).fetchone()
            run_id, result_id, review_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
            point = next(point for point in json.loads(annotation["keypoints_json"]) if point["id"] == vf_id)
            conn.execute(
                """INSERT INTO runs (id, project_id, dataset_id, status, progress, mode, created_at,
                   temporal_enabled, temporal_window_size, temporal_algorithm_version)
                   VALUES (?, ?, ?, 'completed', 100, 'K1', ?, 1, 15, 'robust-local-fit-v1')""",
                (run_id, self.project_id, dataset_id, now),
            )
            conn.execute(
                """INSERT INTO temporal_results (
                   id, run_id, segment_id, internal_track_id, image_id, annotation_id, frame_index, keypoint_id,
                   human_x, human_y, temporal_x, temporal_y, estimate_status, method, temporal_quality,
                   support_count, time_source, supporting_model_count, trajectory_json, created_at)
                   VALUES (?, ?, 0, 'track-1', ?, ?, 0, ?, ?, ?, 7001, 7002, 'available', 'interpolation',
                           0.9, 6, 'frame_index', 2, '[]', ?)""",
                (result_id, run_id, annotation["image_id"], annotation["annotation_id"], vf_id,
                 point.get("x"), point.get("y"), now),
            )
            conn.execute(
                """INSERT INTO temporal_review_decisions (
                   id, run_id, annotation_id, keypoint_id, temporal_result_id, source, selected_x, selected_y,
                   temporal_x, temporal_y, reviewed_at, reviewed_project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 7001, 7002, ?, ?)""",
                (review_id, run_id, annotation["annotation_id"], vf_id, result_id, source,
                 selected[0], selected[1], now, self.project_id),
            )
        return run_id, result_id

    def test_coco_export_updates_each_vf_point_on_the_correct_person_and_preserves_visibility(self) -> None:
        source = coco_payload()
        dataset_id = self.upload("vf_humanpose17_v1", source)
        # Baseline uses the parsed source without reformatting it.
        run_id, _ = self.add_run_warning(dataset_id, 1, 1)
        with database.db() as conn:
            conn.execute("UPDATE warnings SET review=NULL WHERE run_id=?", (run_id,))
        baseline = self.export(run_id)
        self.assertEqual(200, baseline.status_code)
        self.assertEqual(source, baseline.json())

        for coco_id, (vf_id, _) in COCO_TO_VF17.items():
            run_id, _ = self.add_run_warning(dataset_id, 1, vf_id)
            exported = self.export(run_id)
            self.assertEqual(200, exported.status_code, exported.text)
            self.assertEqual("1", exported.headers["X-Export-Requested"])
            self.assertEqual("1", exported.headers["X-Export-Applied"])
            payload = exported.json()
            first, second = payload["annotations"]
            self.assertEqual(source["annotations"][0], first, f"wrong person changed for VF {vf_id}")
            for point_index in range(17):
                original = source["annotations"][1]["keypoints"][point_index * 3: point_index * 3 + 3]
                actual = second["keypoints"][point_index * 3: point_index * 3 + 3]
                if point_index == coco_id:
                    self.assertEqual([9000.0 + vf_id, 8000.0 + vf_id, original[2]], actual)
                else:
                    self.assertEqual(original, actual)
            # v=0 and v=1 remain unchanged even when their x/y are reviewed.
            self.assertEqual(0, second["keypoints"][2])
            self.assertEqual(1, second["keypoints"][5])

    def test_coco_legacy_multi_person_warning_requires_allow_partial(self) -> None:
        dataset_id = self.upload("vf_humanpose17_v1", coco_payload())
        run_id, _ = self.add_run_warning(dataset_id, 0, 1, annotation_id=None)
        with database.db() as conn:
            valid_annotation_id = conn.execute(
                "SELECT a.id FROM annotations a JOIN images i ON i.id=a.image_id WHERE i.dataset_id=? ORDER BY a.rowid LIMIT 1",
                (dataset_id,),
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO warnings (id,run_id,image_name,keypoint,warning_type,suspicion,human_x,human_y,suggested_x,suggested_y,review,annotation_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?, 'use_suggestion', ?)""",
                (str(uuid.uuid4()), run_id, "images/frame.jpg", "r_eye", "test", 1, 0, 0, 777, 666, valid_annotation_id),
            )
        rejected = self.export(run_id)
        self.assertEqual(409, rejected.status_code)
        detail = rejected.json()["detail"]
        self.assertEqual(2, detail["requested"])
        self.assertEqual(1, detail["applied"])
        self.assertIn("no annotation_id", detail["skipped"][0]["reason"])
        partial = self.export(run_id, "?allow_partial=true")
        self.assertEqual(200, partial.status_code)
        self.assertEqual("1", partial.headers["X-Export-Applied"])
        self.assertEqual("1", partial.headers["X-Export-Skipped"])
        # r_eye is VF id 2, hence COCO index 2 (not index 1, left eye).
        self.assertEqual([777, 666, 2], partial.json()["annotations"][0]["keypoints"][6:9])

    def test_landmark_qa_export_behavior_is_unchanged(self) -> None:
        source = pose_lqa()
        dataset_id = self.upload("vf_humanpose17_v1", source)
        run_id, _ = self.add_run_warning(dataset_id, 0, 1)
        exported = self.export(run_id)
        self.assertEqual(200, exported.status_code)
        point = exported.json()["images"][0]["annotations"][0]["keypoints"][0]
        self.assertEqual([9001.0, 8001.0], [point["x"], point["y"]])

    def test_temporal_coco_export_updates_exact_instance_and_preserves_visibility(self) -> None:
        source = coco_payload()
        dataset_id = self.upload("vf_humanpose17_v1", source)
        run_id, result_id = self.add_temporal_review(dataset_id, 1, 3)
        with database.db() as conn:
            conn.execute("DELETE FROM temporal_review_decisions WHERE run_id = ?", (run_id,))
        reviewed = self.client.put(
            f"/api/projects/{self.project_id}/runs/{run_id}/temporal/results/{result_id}/review",
            headers={**self.headers, "Content-Type": "application/json"}, json={"source": "temporal"},
        )
        self.assertEqual(200, reviewed.status_code, reviewed.text)
        reloaded = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/temporal/reviews", headers=self.headers,
        )
        self.assertEqual("temporal", reloaded.json()["items"][0]["source"])
        exported = self.export(run_id)
        self.assertEqual(200, exported.status_code, exported.text)
        first, second = exported.json()["annotations"]
        self.assertEqual(source["annotations"][0], first)
        self.assertEqual([7001.0, 7002.0, 1], second["keypoints"][3:6])
        self.assertEqual("1", exported.headers["X-Export-Applied"])

        audit_json = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/temporal/export/audit.json", headers=self.headers,
        )
        self.assertEqual(200, audit_json.status_code, audit_json.text)
        self.assertEqual("landmark-qa/temporal-review-v1", audit_json.json()["format"])
        self.assertEqual(1, len(audit_json.json()["decisions"]))
        self.assertEqual("occluded", audit_json.json()["decisions"][0]["source_state"])
        audit_csv = self.client.get(
            f"/api/projects/{self.project_id}/runs/{run_id}/temporal/export/audit.csv", headers=self.headers,
        )
        self.assertEqual(200, audit_csv.status_code)
        self.assertIn("annotation_id", audit_csv.text.splitlines()[0])
        self.assertIn("temporal-audit-", audit_csv.headers["content-disposition"])

    def test_temporal_landmark_export_preserves_state_and_rejects_outside(self) -> None:
        source = pose_lqa()
        dataset_id = self.upload("vf_humanpose17_v1", source)
        run_id, _ = self.add_temporal_review(dataset_id, 0, 1)
        exported = self.export(run_id)
        self.assertEqual(200, exported.status_code, exported.text)
        point = exported.json()["images"][0]["annotations"][0]["keypoints"][0]
        self.assertEqual([7001.0, 7002.0, "visible"], [point["x"], point["y"], point["state"]])

        outside_source = pose_lqa()
        outside_source["images"][0]["annotations"][0]["keypoints"][0]["state"] = "outside"
        outside_source["images"][0]["annotations"][0]["keypoints"][0]["x"] = None
        outside_source["images"][0]["annotations"][0]["keypoints"][0]["y"] = None
        outside_dataset = self.upload("vf_humanpose17_v1", outside_source)
        outside_run, _ = self.add_temporal_review(outside_dataset, 0, 1)
        rejected = self.export(outside_run)
        self.assertEqual(409, rejected.status_code)
        self.assertEqual("outside keypoints cannot be replaced", rejected.json()["detail"]["skipped"][0]["reason"])

    def test_exif_orientation_is_rejected_for_pose_and_face_unless_override_enabled(self) -> None:
        rotated = image_bytes(orientation=6)
        for schema_id, annotation in (("vf_humanpose17_v1", pose_lqa()), ("vf_face_landmark50_v1", {
            "format": "landmark-qa/v1", "schema_id": "vf_face_landmark50_v1",
            "images": [{"file_name": "images/frame.jpg", "width": 40, "height": 20, "annotations": [{"label": "face_landmarks", "keypoints": [{"id": i, "x": 5, "y": 4, "state": "visible"} for i in range(50)]}]}],
        })):
            response = self.client.post(
                f"/api/projects/{self.project_id}/datasets", headers=self.headers, data={"schema_id": schema_id},
                files={"file": ("rotated.zip", archive(rotated), "application/zip"), "annotation_file": ("annotations.json", json.dumps(annotation), "application/json")},
            )
            self.assertEqual(400, response.status_code)
            self.assertIn("Orientation=6", response.json()["detail"])
        with patch.dict(os.environ, {"ALLOW_EXIF_ROTATED": "true"}):
            dataset_id = self.upload("vf_humanpose17_v1", pose_lqa(), rotated)
        self.assertTrue(dataset_id)

    def test_normalize_exif_writes_new_pixels_without_orientation_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "normalized"
            source.mkdir()
            (source / "rotated.jpg").write_bytes(image_bytes(orientation=6))
            written = normalize(source, destination)
            self.assertEqual([destination / "rotated.jpg"], written)
            with Image.open(source / "rotated.jpg") as original:
                self.assertEqual((40, 20), original.size)
                self.assertEqual(6, original.getexif().get(274))
            with Image.open(destination / "rotated.jpg") as normalized:
                self.assertEqual((20, 40), normalized.size)
                self.assertNotIn(274, normalized.getexif())


if __name__ == "__main__":
    unittest.main()
