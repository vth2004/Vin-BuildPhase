from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import HTTPException, UploadFile

from app.shared.datasets.archive import safe_extract_images


def upload_zip(entries: list[tuple[str, bytes]]) -> UploadFile:
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for name, value in entries:
            archive.writestr(name, value)
    content.seek(0)
    return UploadFile(file=content, filename="dataset.zip")


class SafeZipTests(unittest.TestCase):
    def test_preserves_relative_paths_and_finds_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "images"
            images, annotation = safe_extract_images(upload_zip([
                ("nested/frame.jpg", b"image"),
                ("annotations.xml", b"<annotations />"),
            ]), destination)
            self.assertEqual(["nested/frame.jpg"], images)
            self.assertEqual(b"<annotations />", annotation)
            self.assertTrue((destination / "nested" / "frame.jpg").is_file())

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(HTTPException):
                safe_extract_images(upload_zip([("../escape.jpg", b"image")]), Path(directory) / "images")

    def test_rejects_windows_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(HTTPException):
                safe_extract_images(upload_zip([("C:/escape.jpg", b"image")]), Path(directory) / "images")

    def test_rejects_case_insensitive_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(HTTPException):
                safe_extract_images(upload_zip([
                    ("frames/A.jpg", b"one"),
                    ("frames/a.jpg", b"two"),
                ]), Path(directory) / "images")


if __name__ == "__main__":
    unittest.main()
