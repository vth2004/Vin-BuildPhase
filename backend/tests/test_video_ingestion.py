from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.core import database
from app.main import app
from app.routers import datasets as datasets_router
from app.shared.video import VideoDecodeError, decode_video
from app.shared.video_import import recover_interrupted_video_imports


def small_avi(directory: Path, frame_count: int = 4) -> bytes:
    target = directory / "fixture.avi"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (48, 32))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV build cannot create MJPG AVI fixture")
    for index in range(frame_count):
        writer.write(np.full((32, 48, 3), index * 30, dtype=np.uint8))
    writer.release()
    return target.read_bytes()


class VideoIngestionApiTests(unittest.TestCase):
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
        database.init_db()
        self.client = TestClient(app)
        project = self.client.post("/api/projects", json={"name": "video ingestion"}).json()
        self.project_id = project["id"]
        self.headers = {"Authorization": f"Bearer {project['recovery_key']}"}

    def tearDown(self) -> None:
        self.client.close()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()

    def test_video_upload_decodes_confirmed_sequence_and_metadata(self) -> None:
        with database.db() as conn:
            self.assertIsNotNone(conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = ?", (database.VIDEO_INGESTION_MIGRATION,),
            ).fetchone())
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={"file": ("walk.avi", small_avi(self.root), "video/x-msvideo")},
        )
        self.assertEqual(202, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("processing", payload["status"])
        job = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{payload['id']}/import", headers=self.headers,
        ).json()
        self.assertEqual("ready", job["job"]["status"])
        self.assertEqual(100, job["job"]["progress"])
        summary = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{payload['id']}/sequences", headers=self.headers,
        ).json()
        self.assertTrue(summary["temporal_ready"])
        sequence = summary["sequences"][0]
        self.assertEqual("video", sequence["source"])
        self.assertEqual("video_decode", sequence["frame_order_source"])
        self.assertEqual([1, 1, 1], sequence["frame_deltas"])
        self.assertEqual([48, 32, 4], [sequence["video_width"], sequence["video_height"], sequence["video_frame_count"]])
        self.assertTrue(sequence["video_codec"])
        with database.db() as conn:
            rows = conn.execute(
                """SELECT file_name, original_frame_index, timestamp_us, timestamp_source
                   FROM images WHERE dataset_id = ? ORDER BY original_frame_index""", (payload["id"],),
            ).fetchall()
            media_path = conn.execute("SELECT media_path FROM sequences WHERE dataset_id = ?", (payload["id"],)).fetchone()[0]
        self.assertEqual(list(range(4)), [row["original_frame_index"] for row in rows])
        self.assertTrue(all(row["timestamp_source"] in {"video_timestamp", "fps_derived"} for row in rows))
        self.assertFalse(Path(media_path).is_absolute())
        self.assertTrue(database.resolve_data_path(media_path).is_file())
        image_response = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{payload['id']}/images/{rows[0]['file_name']}", headers=self.headers,
        )
        self.assertEqual(200, image_response.status_code)
        with Image.open(io.BytesIO(image_response.content)) as image:
            self.assertEqual((48, 32), image.size)

    def test_video_data_paths_survive_moving_the_data_directory(self) -> None:
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={"file": ("walk.avi", small_avi(self.root), "video/x-msvideo")},
        )
        self.assertEqual(202, response.status_code, response.text)
        dataset_id = response.json()["id"]
        with database.db() as conn:
            dataset = conn.execute("SELECT storage_path FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
            sequence = conn.execute("SELECT media_path FROM sequences WHERE dataset_id = ?", (dataset_id,)).fetchone()
            image = conn.execute("SELECT file_name FROM images WHERE dataset_id = ? LIMIT 1", (dataset_id,)).fetchone()
        self.assertFalse(Path(dataset["storage_path"]).is_absolute())
        self.assertFalse(Path(sequence["media_path"]).is_absolute())
        with tempfile.TemporaryDirectory() as temporary:
            relocated = Path(temporary) / "landmark-data"
            shutil.copytree(self.root, relocated)
            with patch.object(database, "DATA_DIR", relocated), patch.object(database, "DB_PATH", relocated / "test.sqlite3"):
                image_response = self.client.get(
                    f"/api/projects/{self.project_id}/datasets/{dataset_id}/images/{image['file_name']}",
                    headers=self.headers,
                )
                self.assertEqual(200, image_response.status_code, image_response.text)
                self.assertTrue(database.resolve_data_path(sequence["media_path"]).is_file())

    def test_corrupt_and_unsupported_video_do_not_create_dataset(self) -> None:
        corrupt = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("broken.mp4", b"not a video", "video/mp4")},
        )
        self.assertEqual(202, corrupt.status_code)
        corrupt_job = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{corrupt.json()['id']}/import", headers=self.headers,
        ).json()
        self.assertEqual("failed", corrupt_job["job"]["status"])
        unsupported = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("clip.mkv", b"data", "video/x-matroska")},
        )
        self.assertEqual(400, unsupported.status_code)
        with database.db() as conn:
            self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM datasets WHERE project_id = ?", (self.project_id,)).fetchone()[0])

    def test_pose_annotation_maps_by_explicit_original_frame_index(self) -> None:
        points = [
            {"id": point_id, "name": str(point_id), "x": float(point_id), "y": float(point_id + 1), "state": "visible"}
            for point_id in range(1, 18)
        ]
        annotation = {
            "format": "landmark-qa/v1", "schema_id": "vf_humanpose17_v1",
            "images": [{
                "file_name": "must-not-be-used.jpg", "frame_index": 2,
                "annotations": [{"id": "person-7", "track_id": "track-3", "keypoints": points}],
            }],
        }
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_humanpose17_v1"},
            files={
                "file": ("walk.avi", small_avi(self.root), "video/x-msvideo"),
                "annotation_file": ("pose.json", json.dumps(annotation).encode(), "application/json"),
            },
        )
        self.assertEqual(202, response.status_code, response.text)
        payload = response.json()
        with database.db() as conn:
            row = conn.execute(
                """SELECT i.original_frame_index, a.source_instance_id, a.source_track_id
                     FROM annotations a JOIN images i ON i.id = a.image_id
                    WHERE i.dataset_id = ?""", (payload["id"],),
            ).fetchone()
            annotation_path = conn.execute(
                "SELECT annotation_path FROM datasets WHERE id = ?", (payload["id"],),
            ).fetchone()[0]
        self.assertEqual([2, "person-7", "track-3"], list(row))
        self.assertEqual("must-not-be-used.jpg", json.loads(database.resolve_data_path(annotation_path).read_text())["images"][0]["file_name"])

    def test_face_annotation_maps_by_explicit_original_frame_index(self) -> None:
        points = [
            {"id": point_id, "x": float(point_id), "y": float(point_id + 1), "state": "visible"}
            for point_id in range(50)
        ]
        annotation = {
            "format": "landmark-qa/v1", "schema_id": "vf_face_landmark50_v1",
            "images": [{
                "file_name": "unrelated.png", "original_frame_index": 1,
                "annotations": [{"id": "face-1", "track_id": "face-track", "keypoints": points}],
            }],
        }
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={
                "file": ("face.avi", small_avi(self.root), "video/x-msvideo"),
                "annotation_file": ("face.json", json.dumps(annotation).encode(), "application/json"),
            },
        )
        self.assertEqual(202, response.status_code, response.text)
        with database.db() as conn:
            frame = conn.execute(
                """SELECT i.original_frame_index FROM annotations a JOIN images i ON i.id = a.image_id
                    WHERE i.dataset_id = ?""", (response.json()["id"],),
            ).fetchone()[0]
        self.assertEqual(1, frame)

    def test_video_annotation_rejects_missing_or_out_of_range_frame_index(self) -> None:
        for image in (
            {"file_name": "frame_000001.jpg", "annotations": []},
            {"frame_index": 99, "annotations": []},
        ):
            annotation = {
                "format": "landmark-qa/v1", "schema_id": "vf_face_landmark50_v1", "images": [image],
            }
            response = self.client.post(
                f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
                data={"schema_id": "vf_face_landmark50_v1"},
                files={
                    "file": ("face.avi", small_avi(self.root), "video/x-msvideo"),
                    "annotation_file": ("face.json", json.dumps(annotation).encode(), "application/json"),
                },
            )
            self.assertEqual(202, response.status_code, response.text)
            job = self.client.get(
                f"/api/projects/{self.project_id}/datasets/{response.json()['id']}/import", headers=self.headers,
            ).json()
            self.assertEqual("failed", job["job"]["status"])
        with database.db() as conn:
            self.assertEqual(2, conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE project_id = ?", (self.project_id,),
            ).fetchone()[0])

    def test_decoded_frame_quota_fails_and_cleans_staging(self) -> None:
        with patch.object(datasets_router, "MAX_VIDEO_FRAMES", 2):
            response = self.client.post(
                f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
                data={"schema_id": "vf_humanpose17_v1"},
                files={"file": ("walk.avi", small_avi(self.root), "video/x-msvideo")},
            )
        self.assertEqual(202, response.status_code, response.text)
        payload = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{response.json()['id']}/import", headers=self.headers,
        ).json()
        self.assertEqual("failed", payload["job"]["status"])
        self.assertIn("frame quota", payload["job"]["error_message"])
        with database.db() as conn:
            storage = database.resolve_data_path(conn.execute(
                "SELECT storage_path FROM datasets WHERE id = ?", (response.json()["id"],),
            ).fetchone()[0])
            staging = database.resolve_data_path(conn.execute(
                "SELECT staging_path FROM video_import_jobs WHERE dataset_id = ?", (response.json()["id"],),
            ).fetchone()[0])
        self.assertFalse(storage.exists())
        self.assertFalse(staging.exists())

    def test_failed_import_can_retry_as_new_attempt(self) -> None:
        response = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("broken.avi", b"broken", "video/x-msvideo")},
        )
        dataset_id = response.json()["id"]
        with database.db() as conn:
            storage = database.resolve_data_path(conn.execute(
                "SELECT storage_path FROM datasets WHERE id = ?", (dataset_id,),
            ).fetchone()[0])
        (storage.parent / "source_video.avi").write_bytes(small_avi(self.root))
        retry = self.client.post(
            f"/api/projects/{self.project_id}/datasets/{dataset_id}/import/retry", headers=self.headers,
        )
        self.assertEqual(202, retry.status_code, retry.text)
        self.assertEqual(2, retry.json()["attempt"])
        status = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{dataset_id}/import", headers=self.headers,
        ).json()
        self.assertEqual("ready", status["job"]["status"])
        with database.db() as conn:
            attempts = conn.execute(
                "SELECT attempt, status FROM video_import_jobs WHERE dataset_id = ? ORDER BY attempt", (dataset_id,),
            ).fetchall()
        self.assertEqual([(1, "failed"), (2, "ready")], [tuple(row) for row in attempts])

    def test_startup_recovery_marks_pending_and_processing_jobs_interrupted(self) -> None:
        for attempt, status in enumerate(("pending", "processing"), start=1):
            dataset_id = str(attempt)
            staging = self.root / f"staging-{attempt}"
            staging.mkdir()
            with database.db() as conn:
                conn.execute(
                    """INSERT INTO datasets
                         (id, project_id, name, status, image_count, storage_path, schema_id, created_at)
                       VALUES (?, ?, ?, 'processing', 0, ?, 'vf_humanpose17_v1', 1)""",
                    (dataset_id, self.project_id, f"video-{attempt}", str(self.root / f"images-{attempt}")),
                )
                conn.execute(
                    """INSERT INTO video_import_jobs
                         (id, dataset_id, attempt, status, progress, staging_path, max_frames,
                          max_decoded_bytes, created_at)
                       VALUES (?, ?, 1, ?, 10, ?, 10, 1000, 1)""",
                    (f"job-{attempt}", dataset_id, status, str(staging)),
                )
        self.assertEqual(2, recover_interrupted_video_imports())
        with database.db() as conn:
            jobs = conn.execute("SELECT status FROM video_import_jobs ORDER BY id").fetchall()
            datasets = conn.execute(
                "SELECT status FROM datasets WHERE id IN ('1', '2') ORDER BY id"
            ).fetchall()
        self.assertEqual(["interrupted", "interrupted"], [row[0] for row in jobs])
        self.assertEqual(["interrupted", "interrupted"], [row[0] for row in datasets])
        self.assertFalse((self.root / "staging-1").exists())
        self.assertFalse((self.root / "staging-2").exists())


class VideoDecoderTests(unittest.TestCase):
    class Capture:
        def __init__(self, frames: list[np.ndarray], fps: float = 5.0):
            self.frames, self.fps, self.position, self.released = frames, fps, 0, False

        def isOpened(self):
            return True

        def read(self):
            if self.position >= len(self.frames):
                return False, None
            frame = self.frames[self.position]
            self.position += 1
            return True, frame

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return self.fps
            if prop == cv2.CAP_PROP_FOURCC:
                return cv2.VideoWriter_fourcc(*"TEST")
            if prop == getattr(cv2, "CAP_PROP_ORIENTATION_META", -999):
                return 90
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 0
            return 0

        def release(self):
            self.released = True

    def test_invalid_video_timestamps_fall_back_to_fps_and_rotation_is_recorded(self) -> None:
        capture = self.Capture([np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory, patch("app.shared.video.cv2.VideoCapture", return_value=capture):
            result = decode_video(Path(directory) / "input.mp4", Path(directory) / "frames")
        self.assertEqual([0, 200000, 400000], [frame.timestamp_us for frame in result.frames])
        self.assertTrue(all(frame.timestamp_source == "fps_derived" for frame in result.frames))
        self.assertEqual(90, result.rotation_degrees)
        self.assertEqual("TEST", result.codec)
        self.assertTrue(capture.released)

    def test_frame_size_change_is_rejected(self) -> None:
        capture = self.Capture([
            np.zeros((20, 30, 3), dtype=np.uint8), np.zeros((21, 30, 3), dtype=np.uint8),
        ])
        with tempfile.TemporaryDirectory() as directory, patch("app.shared.video.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(VideoDecodeError, "size changed"):
                decode_video(Path(directory) / "input.mov", Path(directory) / "frames")
        self.assertTrue(capture.released)

    def test_decoded_pixel_byte_quota_is_enforced(self) -> None:
        capture = self.Capture([np.zeros((20, 30, 3), dtype=np.uint8)])
        with tempfile.TemporaryDirectory() as directory, patch("app.shared.video.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(VideoDecodeError, "storage quota"):
                decode_video(
                    Path(directory) / "input.mov", Path(directory) / "frames",
                    max_decoded_bytes=20 * 30 * 3 - 1,
                )
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
