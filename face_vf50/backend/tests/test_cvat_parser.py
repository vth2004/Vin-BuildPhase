import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from face_vf50.backend.app.io.cvat_parser import parse_cvat_xml, parse_annotation_payload, resolve_point_id
except ModuleNotFoundError:
    from app.io.cvat_parser import parse_cvat_xml, parse_annotation_payload, resolve_point_id


class TestCVATParser(unittest.TestCase):
    def test_sublabel_id_mapping(self):
        # 117 should map to 0 (longmaytrai)
        self.assertEqual(resolve_point_id("longmaytrai", raw_label="0", label_id="117"), 0)
        # 123 should map to 5 (longmayphai point 0)
        self.assertEqual(resolve_point_id("longmayphai", raw_label="0", label_id="123"), 5)
        # 134 should map to 14 (mattrai point 0)
        self.assertEqual(resolve_point_id("mattrai", raw_label="0", label_id="134"), 14)
        # 143 should map to 22 (matphai point 0)
        self.assertEqual(resolve_point_id("matphai", raw_label="0", label_id="143"), 22)
        # 152 should map to 30 (moingoai point 0)
        self.assertEqual(resolve_point_id("moingoai", raw_label="0", label_id="152"), 30)
        # 165 should map to 42 (moitrong point 0)
        self.assertEqual(resolve_point_id("moitrong", raw_label="0", label_id="165"), 42)
        # 172 should map to 49 (moitrong point 7)
        self.assertEqual(resolve_point_id("moitrong", raw_label="7", label_id="172"), 49)

    def test_skeleton_sublabel_names_no_overwrite(self):
        # When CVAT exports sublabels named '0'..'4' inside skeletons
        # longmaytrai local 0 -> 0
        self.assertEqual(resolve_point_id("longmaytrai", raw_label="0"), 0)
        # longmayphai local 0 -> 5 (MUST NOT overwrite 0!)
        self.assertEqual(resolve_point_id("longmayphai", raw_label="0"), 5)
        # longmayphai local 4 -> 9
        self.assertEqual(resolve_point_id("longmayphai", raw_label="4"), 9)
        # songmui local 0 -> 10
        self.assertEqual(resolve_point_id("songmui", raw_label="0"), 10)
        # moingoai local 11 -> 41
        self.assertEqual(resolve_point_id("moingoai", raw_label="11"), 41)

    def test_global_id_preservation(self):
        # If exported with global IDs (e.g. 5 for longmayphai)
        self.assertEqual(resolve_point_id("longmayphai", raw_label="5"), 5)
        self.assertEqual(resolve_point_id("longmayphai", raw_label="9"), 9)
        self.assertEqual(resolve_point_id("moitrong", raw_label="49"), 49)

    def test_parse_cvat_xml_full(self):
        sample_xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="media_1791431084108.png" width="1280" height="720">
    <skeleton label="longmaytrai">
      <points label="0" label_id="117" points="480.0,260.0" outside="0" occluded="0" />
      <points label="1" label_id="118" points="495.0,260.0" outside="0" occluded="0" />
      <points label="2" label_id="119" points="510.0,260.0" outside="0" occluded="0" />
      <points label="3" label_id="120" points="525.0,260.0" outside="0" occluded="0" />
      <points label="4" label_id="121" points="540.0,260.0" outside="0" occluded="0" />
    </skeleton>
    <skeleton label="longmayphai">
      <points label="0" label_id="123" points="560.0,260.0" outside="0" occluded="0" />
      <points label="1" label_id="124" points="575.0,260.0" outside="0" occluded="0" />
      <points label="2" label_id="125" points="590.0,260.0" outside="0" occluded="0" />
      <points label="3" label_id="126" points="605.0,260.0" outside="0" occluded="0" />
      <points label="4" label_id="127" points="620.0,260.0" outside="0" occluded="0" />
    </skeleton>
  </image>
</annotations>
"""
        parsed = parse_cvat_xml(sample_xml)
        self.assertEqual(len(parsed), 1)
        kpts = parsed[0]["keypoints"]
        # Must contain points 0..9 without collision
        self.assertIn(0, kpts)
        self.assertIn(4, kpts)
        self.assertIn(5, kpts)
        self.assertIn(9, kpts)
        self.assertEqual(kpts[0]["x"], 480.0)
        self.assertEqual(kpts[5]["x"], 560.0)
        self.assertEqual(kpts[9]["x"], 620.0)

    def test_parse_cvat_video_tracks(self):
        sample_video_xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="longmaytrai">
    <points frame="0" points="480.0,260.0" outside="0" occluded="0" label="0" />
    <points frame="1" points="481.0,261.0" outside="0" occluded="0" label="0" />
  </track>
  <track id="1" label="longmayphai">
    <points frame="0" points="560.0,260.0" outside="0" occluded="0" label="0" />
  </track>
</annotations>
"""
        parsed = parse_cvat_xml(sample_video_xml)
        self.assertEqual(len(parsed), 2)
        # Frame 0 should have point 0 and point 5
        f0_kpts = parsed[0]["keypoints"]
        self.assertIn(0, f0_kpts)
        self.assertIn(5, f0_kpts)
        self.assertEqual(f0_kpts[5]["x"], 560.0)

    def test_parse_coco_keypoints_7_categories(self):
        sample_coco = {
            "images": [{"id": 1, "file_name": "test_img.png", "width": 1280, "height": 720}],
            "categories": [
                {"id": 1, "name": "longmaytrai", "keypoints": ["0", "1", "2", "3", "4"]},
                {"id": 7, "name": "longmayphai", "keypoints": ["5", "6", "7", "8", "9"]},
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [10.0, 20.0, 2, 15.0, 25.0, 2, 20.0, 30.0, 2, 25.0, 35.0, 2, 30.0, 40.0, 2],
                },
                {
                    "image_id": "1",  # String ID test
                    "category_id": "7",  # String category test
                    "keypoints": [50.0, 60.0, 2, 55.0, 65.0, 2, 60.0, 70.0, 2, 65.0, 75.0, 2, 70.0, 80.0, 2],
                },
            ],
        }
        import json
        payload = json.dumps(sample_coco).encode("utf-8")
        parsed = parse_annotation_payload(payload)
        self.assertEqual(len(parsed), 1)
        kpts = parsed[0]["keypoints"]
        # Must have longmaytrai (0..4) and longmayphai (5..9)
        self.assertEqual(len(kpts), 10)
        self.assertIn(0, kpts)
        self.assertIn(5, kpts)
        self.assertIn(9, kpts)
        self.assertEqual(kpts[5]["x"], 50.0)

    def test_parse_coco_keypoints_single_category(self):
        # 50 points flat under 1 single category 'face'
        flat_kpts = []
        for i in range(50):
            flat_kpts.extend([float(i * 10), float(i * 10 + 5), 2])
        sample_coco = {
            "images": [{"id": "img_01", "file_name": "face.jpg", "width": 640, "height": 480}],
            "categories": [{"id": 100, "name": "face", "keypoints": [f"point_{i}" for i in range(50)]}],
            "annotations": [{"image_id": "img_01", "category_id": 100, "keypoints": flat_kpts}],
        }
        import json
        payload = json.dumps(sample_coco).encode("utf-8")
        parsed = parse_annotation_payload(payload)
        self.assertEqual(len(parsed), 1)
        kpts = parsed[0]["keypoints"]
        self.assertEqual(len(kpts), 50)
        for i in range(50):
            self.assertIn(i, kpts)
            self.assertEqual(kpts[i]["x"], float(i * 10))


if __name__ == "__main__":
    unittest.main()
