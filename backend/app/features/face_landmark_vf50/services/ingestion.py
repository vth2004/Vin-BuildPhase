from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..parsers.cvat import parse_annotation_payload


def _match_image(raw_name: str, image_names: set[str]) -> str:
    normalized = raw_name.replace("\\", "/").lstrip("./")
    if normalized in image_names:
        return normalized
    basename = Path(normalized).name.casefold()
    matches = [name for name in image_names if Path(name).name.casefold() == basename]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(400, f"Ambiguous annotation image name: {raw_name}")
    raise HTTPException(400, f"Annotation refers to image missing from ZIP: {raw_name}")


def _points_from_mapping(mapping: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for point_id in range(50):
        raw = mapping.get(point_id) or {}
        state = raw.get("state", "outside")
        x, y = raw.get("x"), raw.get("y")
        if state not in {"visible", "occluded", "outside"}:
            raise HTTPException(400, f"Point {point_id} has invalid state: {state}")
        if state == "outside" or x is None or y is None:
            state, x, y = "outside", None, None
        elif not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise HTTPException(400, f"Point {point_id} requires numeric coordinates")
        points.append({"id": point_id, "name": str(point_id), "x": x, "y": y, "state": state})
    return points


def _parse_single_payload(content: bytes) -> list[dict[str, Any]]:
    parsed = parse_annotation_payload(content)
    return [
        {
            "file_name": item.get("file_name", ""),
            "width": item.get("width"),
            "height": item.get("height"),
            "annotations": [{"label": "face_landmarks", "keypoints": _points_from_mapping(item.get("keypoints", {}))}],
        }
        for item in parsed
        if item.get("file_name")
    ]


def parse_face_annotations(content: bytes, image_names: set[str]) -> list[dict[str, Any]]:
    """Normalize supported VF50 formats to the core dataset representation."""
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    # Preserve multiple COCO face annotations by parsing each annotation independently.
    if isinstance(payload, dict) and isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
        output_by_name: dict[str, dict[str, Any]] = {}
        for image in payload["images"]:
            image_id = image.get("id")
            matching = [annotation for annotation in payload["annotations"] if annotation.get("image_id") == image_id]
            for annotation in matching:
                partial = dict(payload)
                partial["images"] = [image]
                partial["annotations"] = [annotation]
                for record in _parse_single_payload(json.dumps(partial).encode("utf-8")):
                    name = _match_image(record["file_name"], image_names)
                    target = output_by_name.setdefault(name, {
                        "file_name": name,
                        "width": record.get("width"),
                        "height": record.get("height"),
                        "annotations": [],
                    })
                    target["annotations"].extend(record["annotations"])
        output = list(output_by_name.values())
    elif isinstance(payload, dict) and payload.get("format") == "landmark-qa/v1":
        output = []
        for image in payload.get("images", []):
            name = _match_image(str(image.get("file_name", "")), image_names)
            annotations = []
            for annotation in image.get("annotations", []):
                mapping = {int(point["id"]): point for point in annotation.get("keypoints", []) if "id" in point}
                annotations.append({"label": annotation.get("label", "face_landmarks"), "keypoints": _points_from_mapping(mapping)})
            if annotations:
                output.append({
                    "file_name": name,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "annotations": annotations,
                })
    else:
        output = _parse_single_payload(content)
        for record in output:
            record["file_name"] = _match_image(record["file_name"], image_names)

    output = [record for record in output if record.get("annotations")]
    if not output:
        raise HTTPException(400, "No usable VF50 annotations match the uploaded images")
    return output
