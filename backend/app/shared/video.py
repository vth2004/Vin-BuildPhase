from __future__ import annotations

import math
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
