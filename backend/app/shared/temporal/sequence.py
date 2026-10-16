from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from fastapi import HTTPException


@dataclass(frozen=True)
class FrameMapping:
    file_name: str
    sequence_name: str
    original_frame_index: int
    timestamp_us: int | None


@dataclass(frozen=True)
class SequenceInference:
    source: str
    mappings: tuple[FrameMapping, ...]
    requires_manifest: bool = False
    reason: str | None = None


def _normalized_name(value: object) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def parse_sequence_manifest(raw: bytes, image_names: set[str]) -> SequenceInference:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(400, "sequence_manifest.json is not valid UTF-8 JSON") from error

    records: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        records = payload["images"]
    elif isinstance(payload, dict):
        for file_name, value in payload.items():
            if not isinstance(value, dict):
                raise HTTPException(400, "manifest mapping values must be objects")
            records.append({"file_name": file_name, **value})
    else:
        raise HTTPException(400, "manifest must be an object or contain an images array")

    mappings: list[FrameMapping] = []
    seen_files: set[str] = set()
    seen_frames: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise HTTPException(400, "manifest image entries must be objects")
        file_name = _normalized_name(record.get("file_name"))
        sequence = str(record.get("sequence") or record.get("sequence_name") or "").strip()
        frame = record.get("frame_index", record.get("original_frame_index"))
        if file_name not in image_names:
            raise HTTPException(400, f"manifest image is missing from ZIP: {file_name}")
        if not sequence or not isinstance(frame, int) or frame < 0:
            raise HTTPException(400, f"invalid sequence/frame for manifest image: {file_name}")
        if file_name in seen_files or (sequence, frame) in seen_frames:
            raise HTTPException(400, f"duplicate manifest file or sequence frame: {file_name}")
        timestamp_us = record.get("timestamp_us")
        if timestamp_us is None and isinstance(record.get("timestamp_ms"), (int, float)):
            timestamp_us = round(float(record["timestamp_ms"]) * 1000)
        if timestamp_us is not None and (not isinstance(timestamp_us, int) or timestamp_us < 0):
            raise HTTPException(400, f"invalid timestamp for manifest image: {file_name}")
        seen_files.add(file_name)
        seen_frames.add((sequence, frame))
        mappings.append(FrameMapping(file_name, sequence, frame, timestamp_us))

    missing = image_names - seen_files
    if missing:
        raise HTTPException(400, f"manifest does not cover all ZIP images; first missing: {sorted(missing)[0]}")
    return SequenceInference("manifest", tuple(mappings))


def infer_sequences(records: list[dict[str, Any]]) -> SequenceInference:
    """Infer a candidate mapping without claiming that file names prove chronology."""
    explicit: list[FrameMapping] = []
    if records and all(isinstance(item.get("original_frame_index"), int) for item in records):
        for item in records:
            explicit.append(FrameMapping(
                _normalized_name(item["file_name"]),
                str(item.get("sequence_name") or "default"),
                int(item["original_frame_index"]),
                item.get("timestamp_us"),
            ))
        if len({(m.sequence_name, m.original_frame_index) for m in explicit}) == len(explicit):
            return SequenceInference("annotation", tuple(explicit))

    inferred: list[FrameMapping] = []
    missing_numeric_token = False
    for item in records:
        file_name = _normalized_name(item["file_name"])
        path = PurePosixPath(file_name)
        tokens = list(re.finditer(r"\d+", path.stem))
        if len(tokens) > 1:
            return SequenceInference(
                "import_order", (), True,
                f"ambiguous filename sequence inference: {file_name}",
            )
        if not tokens:
            missing_numeric_token = True
            continue
        token = tokens[0]
        prefix = (path.stem[:token.start()] + path.stem[token.end():]).strip(" _-.") or "default"
        sequence = str(path.parent / prefix) if str(path.parent) != "." else prefix
        inferred.append(FrameMapping(file_name, sequence, int(token.group()), None))

    if inferred and missing_numeric_token:
        return SequenceInference(
            "import_order", (), True,
            "only some filenames contain an unambiguous frame number",
        )

    if inferred and len({(m.sequence_name, m.original_frame_index) for m in inferred}) == len(inferred):
        return SequenceInference("filename_inferred", tuple(inferred))

    # COCO order is only a candidate. It remains unconfirmed until the API call.
    if records and all(item.get("coco_array_position") is not None for item in records):
        mappings = tuple(
            FrameMapping(_normalized_name(item["file_name"]), "default", int(item["coco_array_position"]), None)
            for item in records
        )
        return SequenceInference("coco_array_order", mappings)

    return SequenceInference("import_order", (), True, "sequence order could not be inferred")
