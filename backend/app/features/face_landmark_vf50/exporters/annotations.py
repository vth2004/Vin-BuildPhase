from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

SKELETONS = [
    ("longmaytrai", range(0, 5)),
    ("longmayphai", range(5, 10)),
    ("songmui", range(10, 14)),
    ("mattrai", range(14, 22)),
    ("matphai", range(22, 30)),
    ("moingoai", range(30, 42)),
    ("moitrong", range(42, 50)),
]


def cleaned_landmark_qa(dataset_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    images: dict[str, dict[str, Any]] = {}
    for result in results:
        image = images.setdefault(result["file_name"], {
            "file_name": result["file_name"],
            "width": result.get("width"),
            "height": result.get("height"),
            "frame_index": result.get("frame_index"),
            "annotations": [],
        })
        image["annotations"].append({
            "label": "face_landmarks",
            "face_index": result.get("face_index", 0),
            "keypoints": [result["reviewed_keypoints"][point_id] for point_id in range(50)],
        })
    return {
        "format": "landmark-qa/v1",
        "schema_id": "vf_face_landmark50_v1",
        "dataset_name": dataset_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "images": sorted(images.values(), key=lambda item: (item.get("frame_index") or 0, item["file_name"])),
    }


def cvat_xml(results: list[dict[str, Any]]) -> str:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["file_name"]].append(result)
    for image_id, (file_name, faces) in enumerate(sorted(grouped.items())):
        first = faces[0]
        image = ET.SubElement(root, "image", {
            "id": str(image_id),
            "name": file_name,
            "width": str(first.get("width") or 0),
            "height": str(first.get("height") or 0),
        })
        for face in faces:
            points = face["reviewed_keypoints"]
            group_id = str(face.get("face_index", 0))
            for label, point_ids in SKELETONS:
                skeleton = ET.SubElement(image, "skeleton", {"label": label, "group_id": group_id})
                for point_id in point_ids:
                    point = points[point_id]
                    outside = point.get("state") == "outside" or point.get("x") is None or point.get("y") is None
                    value = "" if outside else f"{float(point['x']):.2f},{float(point['y']):.2f}"
                    ET.SubElement(skeleton, "points", {
                        "label": str(point_id),
                        "points": value,
                        "occluded": "1" if point.get("state") == "occluded" else "0",
                        "outside": "1" if outside else "0",
                    })
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def json_text(dataset_name: str, results: list[dict[str, Any]]) -> str:
    return json.dumps(cleaned_landmark_qa(dataset_name, results), ensure_ascii=False, indent=2)
