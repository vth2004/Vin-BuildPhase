from __future__ import annotations

import math
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


@dataclass(frozen=True)
class DecodedFrame:
    file_name: str
    frame_index: int
    timestamp_us: int | None
    timestamp_source: str
    width: int
    height: int


@dataclass(frozen=True)
class VideoDecodeResult:
    frames: tuple[DecodedFrame, ...]
    fps: float | None
    codec: str
    width: int
    height: int
    rotation_degrees: int


class VideoDecodeError(ValueError):
    pass


class VideoAnnotationError(ValueError):
    pass


def prepare_video_annotation(content: bytes, frame_count: int) -> bytes:
    """Map supported annotations to decoded names using explicit source frame indices.

    Video imports deliberately never infer a frame from an annotation filename. JSON
    records must carry ``frame_index``, ``original_frame_index`` or ``frame``. CVAT
    XML uses the video ``track/*@frame`` value or the image element ``@id``.
    """
    stripped = content.lstrip()
    if stripped.startswith(b"<"):
        try:
            root = ET.fromstring(content)
        except ET.ParseError as error:
            raise VideoAnnotationError("Annotation XML is invalid") from error
        found = False
        for image in root.findall(".//image"):
            raw_index = image.get("id")
            index = _video_frame_index(raw_index, frame_count)
            image.set("name", f"frame_{index:06d}.jpg")
            found = True
        for element in root.findall(".//*[@frame]"):
            _video_frame_index(element.get("frame"), frame_count)
            found = True
        if not found:
            raise VideoAnnotationError("Video annotation XML has no explicit frame indices")
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VideoAnnotationError("Annotation file is not valid JSON or XML") from error
    if not isinstance(payload, dict):
        raise VideoAnnotationError("Annotation JSON must be an object")

    records = payload.get("images")
    if not isinstance(records, list):
        records = payload.get("frames")
    if not isinstance(records, list) or not records:
        raise VideoAnnotationError("Video annotation JSON requires a non-empty images or frames array")
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise VideoAnnotationError("Each video annotation frame must be an object")
        raw_index = next((record.get(key) for key in ("frame_index", "original_frame_index", "frame")
                          if record.get(key) is not None), None)
        index = _video_frame_index(raw_index, frame_count)
        if index in seen:
            raise VideoAnnotationError(f"Duplicate video annotation frame index: {index}")
        seen.add(index)
        record["file_name"] = f"frame_{index:06d}.jpg"
        if "image_filename" in record:
            record["image_filename"] = record["file_name"]
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _video_frame_index(raw: object, frame_count: int) -> int:
    if isinstance(raw, bool):
        raise VideoAnnotationError("Video annotation frame index must be an integer")
    try:
        index = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise VideoAnnotationError("Video annotation requires an explicit integer frame index") from error
    if str(raw).strip() != str(index):
        raise VideoAnnotationError("Video annotation frame index must be an integer")
    if index < 0 or index >= frame_count:
        raise VideoAnnotationError(
            f"Video annotation frame index {index} is outside decoded range 0..{frame_count - 1}"
        )
    return index


def _finite_positive(value: float, maximum: float = 1000.0) -> bool:
    return math.isfinite(value) and 0 < value <= maximum


def _codec_name(fourcc: float) -> str:
    try:
        value = int(fourcc)
        text = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
        return text.strip("\x00 ") or "unknown"
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def decode_video(source: Path, output_dir: Path) -> VideoDecodeResult:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError("Video container or codec cannot be opened by OpenCV")

    fps_value = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps_value if _finite_positive(fps_value) else None
    codec = _codec_name(capture.get(cv2.CAP_PROP_FOURCC))
    orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    rotation_raw = float(capture.get(orientation_property)) if orientation_property is not None else 0.0
    rotation = int(round(rotation_raw)) if math.isfinite(rotation_raw) else 0
    if rotation not in {0, 90, 180, 270}:
        rotation = 0

    output_dir.mkdir(parents=True, exist_ok=False)
    decoded: list[tuple[str, int, float, int, int]] = []
    expected_size: tuple[int, int] | None = None
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
                raise VideoDecodeError(f"Decoded frame {frame_index} is invalid")
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if width <= 0 or height <= 0:
                raise VideoDecodeError(f"Decoded frame {frame_index} has invalid dimensions")
            if expected_size is None:
                expected_size = (width, height)
            elif expected_size != (width, height):
                raise VideoDecodeError(
                    f"Video frame size changed at frame {frame_index}: "
                    f"expected {expected_size[0]}x{expected_size[1]}, got {width}x{height}"
                )
            file_name = f"frame_{frame_index:06d}.jpg"
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise VideoDecodeError(f"Could not encode decoded frame {frame_index}")
            encoded.tofile(str(output_dir / file_name))
            decoded.append((file_name, frame_index, float(capture.get(cv2.CAP_PROP_POS_MSEC)), width, height))
            frame_index += 1
    finally:
        capture.release()

    if not decoded:
        raise VideoDecodeError("Video contains no decodable frames")

    timestamps_ms = [item[2] for item in decoded]
    timestamps_valid = all(math.isfinite(value) and value >= 0 for value in timestamps_ms)
    if len(timestamps_ms) > 1:
        timestamps_valid = timestamps_valid and all(
            current > previous for previous, current in zip(timestamps_ms, timestamps_ms[1:])
        )
    frames = []
    for file_name, frame_index, timestamp_ms, width, height in decoded:
        if timestamps_valid:
            timestamp_us, timestamp_source = round(timestamp_ms * 1000), "video_timestamp"
        elif fps:
            timestamp_us, timestamp_source = round(frame_index * 1_000_000 / fps), "fps_derived"
        else:
            timestamp_us, timestamp_source = None, "unavailable"
        frames.append(DecodedFrame(file_name, frame_index, timestamp_us, timestamp_source, width, height))

    width, height = expected_size or (0, 0)
    return VideoDecodeResult(tuple(frames), fps, codec, width, height, rotation)
