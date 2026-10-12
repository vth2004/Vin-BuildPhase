"""Image metadata validation shared by dataset upload flows."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger(__name__)

_ORIENTATION_TAG = 274
_ROTATED_ORIENTATIONS = {5, 6, 7, 8}


def validate_exif_orientation(
    image_root: Path,
    records: list[dict[str, Any]],
) -> None:
    """Inspect raw image dimensions and reject EXIF-rotated images by default.

    Coordinates in annotations are intentionally compared only; this helper never
    rotates pixels or transforms coordinates.  Callers have already normalized
    records to ``file_name``, ``width`` and ``height`` by this point.
    """
    records_by_name = {str(record.get("file_name", "")).replace("\\", "/"): record for record in records}
    rotated_files: list[str] = []
    allow_rotated = os.getenv("ALLOW_EXIF_ROTATED", "").strip().lower() == "true"

    for file_name, record in records_by_name.items():
        image_path = image_root / Path(*file_name.split("/"))
        try:
            with Image.open(image_path) as image:
                raw_width, raw_height = image.size
                orientation = int(image.getexif().get(_ORIENTATION_TAG, 1) or 1)
        except Exception as error:
            logger.warning("Could not read image metadata for %s: %s", file_name, error)
            continue

        rotated_size = (
            (raw_height, raw_width)
            if orientation in _ROTATED_ORIENTATIONS
            else (raw_width, raw_height)
        )
        annotation_size = (record.get("width"), record.get("height"))
        match_kind = (
            "raw" if annotation_size == (raw_width, raw_height)
            else "rotated" if annotation_size == rotated_size
            else "neither"
        )
        logger.info(
            "Image metadata %s: raw=%sx%s orientation=%s rotated=%sx%s annotation=%s match=%s",
            file_name, raw_width, raw_height, orientation,
            rotated_size[0], rotated_size[1], annotation_size, match_kind,
        )
        if orientation != 1:
            rotated_files.append(
                f"{file_name} (EXIF Orientation={orientation}; annotation matches {match_kind})"
            )

    if rotated_files and not allow_rotated:
        raise HTTPException(
            400,
            "EXIF Orientation khác 1 không được hỗ trợ: "
            + "; ".join(rotated_files)
            + ". Hãy chuẩn hóa ảnh (xoay pixel và bỏ EXIF Orientation) trước khi đưa vào CVAT và web. "
              "Có thể tạm cho phép bằng ALLOW_EXIF_ROTATED=true.",
        )
    if rotated_files:
        logger.warning(
            "ALLOW_EXIF_ROTATED=true: accepting EXIF-rotated images without coordinate transformation: %s",
            "; ".join(rotated_files),
        )
