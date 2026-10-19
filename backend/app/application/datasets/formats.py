"""Compose feature parsers here so shared infrastructure has no feature imports."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal
from fastapi import HTTPException
from app.features.face_landmark_vf50.services.ingestion import parse_face_annotations
from app.features.human_pose.services.dataset import validate_annotations
from app.shared.datasets.schemas import load_schema
from app.shared.video import VideoAnnotationError

Source = Literal["archive", "video"]
Parser = Callable[[bytes | None, dict, list[str], Source], list[dict]]


def _empty_images(images: list[str]) -> list[dict]:
    return [{"file_name": name, "width": None, "height": None, "annotations": []} for name in images]


def _face(raw: bytes | None, schema: dict, images: list[str], source: Source) -> list[dict]:
    records = parse_face_annotations(raw, set(images)) if raw else []
    if source == "video":
        return records
    by_name = {record["file_name"]: record for record in records}
    return [by_name.get(record["file_name"], record) for record in _empty_images(images)]


def _pose(raw: bytes | None, schema: dict, images: list[str], source: Source) -> list[dict]:
    if not raw:
        return _empty_images(images)
    if raw.lstrip().startswith(b"<"):
        if source == "video":
            raise VideoAnnotationError("Human Pose video annotations must use a supported JSON format")
        raise HTTPException(
            400,
            "File annotation XML này thuộc định dạng Face Landmark (CVAT XML). "
            "Phân hệ Human Pose chỉ hỗ trợ file JSON COCO 17 điểm. Vui lòng chuyển sang phân hệ Face Landmark.",
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        if source == "video":
            raise
        raise HTTPException(400, "Annotation file is not valid JSON") from error
    return validate_annotations(payload, schema, set(images))


PARSERS: dict[str, Parser] = {
    "vf_face_landmark50_v1": _face,
    "vf_humanpose17_v1": _pose,
}


def parse_annotations(schema_id: str, raw: bytes | None, images: list[str], *, source: Source) -> list[dict]:
    schema = load_schema(schema_id)
    parser = PARSERS.get(schema_id)
    if parser is None:
        raise HTTPException(400, f"Unsupported schema_id: {schema_id}")
    return parser(raw, schema, images, source)
