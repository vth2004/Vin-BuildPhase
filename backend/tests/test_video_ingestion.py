from __future__ import annotations

import io
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
        self.assertEqual(201, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("ready", payload["status"])
        self.assertEqual(4, payload["image_count"])
        self.assertTrue(payload["sequence_summary"]["temporal_ready"])
        sequence = payload["sequence_summary"]["sequences"][0]
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
        self.assertTrue(Path(media_path).is_file())
        image_response = self.client.get(
            f"/api/projects/{self.project_id}/datasets/{payload['id']}/images/{rows[0]['file_name']}", headers=self.headers,
        )
        self.assertEqual(200, image_response.status_code)
        with Image.open(io.BytesIO(image_response.content)) as image:
            self.assertEqual((48, 32), image.size)

    def test_corrupt_and_unsupported_video_do_not_create_dataset(self) -> None:
        corrupt = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("broken.mp4", b"not a video", "video/mp4")},
        )
        self.assertEqual(400, corrupt.status_code)
        unsupported = self.client.post(
            f"/api/projects/{self.project_id}/datasets/video", headers=self.headers,
            data={"schema_id": "vf_face_landmark50_v1"},
            files={"file": ("clip.mkv", b"data", "video/x-matroska")},
        )
        self.assertEqual(400, unsupported.status_code)
        with database.db() as conn:
            self.assertEqual(0, conn.execute("SELECT COUNT(*) FROM datasets WHERE project_id = ?", (self.project_id,)).fetchone()[0])


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


if __name__ == "__main__":
    unittest.main()
