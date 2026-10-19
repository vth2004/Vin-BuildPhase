from __future__ import annotations

import sys
import unittest
from pathlib import Path
from fastapi import HTTPException

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.application.datasets.formats import parse_annotations
from app.shared.video import VideoAnnotationError


class DatasetFormatTests(unittest.TestCase):
    def test_unlabeled_archives_keep_image_order_for_both_features(self) -> None:
        for schema in ("vf_face_landmark50_v1", "vf_humanpose17_v1"):
            with self.subTest(schema=schema):
                records = parse_annotations(schema, None, ["z.jpg", "a.jpg"], source="archive")
                self.assertEqual(["z.jpg", "a.jpg"], [row["file_name"] for row in records])
                self.assertTrue(all(row["annotations"] == [] for row in records))

    def test_unknown_schema_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as error:
            parse_annotations("missing", None, [], source="archive")
        self.assertEqual(400, error.exception.status_code)

    def test_pose_xml_uses_source_specific_validation(self) -> None:
        with self.assertRaises(HTTPException) as error:
            parse_annotations("vf_humanpose17_v1", b"<annotations/>", [], source="archive")
        self.assertEqual(400, error.exception.status_code)
        with self.assertRaises(VideoAnnotationError):
            parse_annotations("vf_humanpose17_v1", b"<annotations/>", [], source="video")

    def test_invalid_archive_json_keeps_client_error(self) -> None:
        with self.assertRaises(HTTPException) as error:
            parse_annotations("vf_humanpose17_v1", b"{broken", [], source="archive")
        self.assertEqual("Annotation file is not valid JSON", error.exception.detail)
