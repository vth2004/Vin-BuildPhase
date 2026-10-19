from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

TEMPORAL_ALGORITHM_VERSION = "robust-local-fit-v2"


@dataclass(frozen=True)
class TemporalEngineConfig:
    version: str = TEMPORAL_ALGORITHM_VERSION
    huber_k: float = 1.345
    min_huber_delta: float = 0.01
    time_decay: float = 1.0
    max_normalized_radius: float = 5.0
    quality_residual_decay: float = 3.0
    max_gap_frames_with_time: int = 5
    max_gap_us: int = 200_000
    max_gap_frames_without_time: int = 3
    max_face_support_observations: int = 11
    min_face_estimate_quality: float = 0.35


DEFAULT_TEMPORAL_CONFIG = TemporalEngineConfig()


@dataclass(frozen=True)
class ObservationFrame:
    frame_index: int
    keypoints: dict[int, dict[str, Any]]
    bbox: tuple[float, float, float, float] | list[float] | None = None
    timestamp_us: int | None = None
    timestamp_source: str = "unavailable"
    image_width: int | None = None
    image_height: int | None = None
    quality: float = 1.0
    tracking_status: str = "confirmed"


@dataclass(frozen=True)
class Transform:
    origin_x: float
    origin_y: float
    scale: float
    source: str


@dataclass(frozen=True)
class TemporalEstimate:
    coordinate: tuple[float, float] | None
    estimate_status: Literal["available", "unavailable"]
    method: Literal["interpolation", "extrapolation", "unavailable"]
    unavailable_reason: str | None
    temporal_quality: float
    support_count: int
    residual: float | None
    gap_frames: int | None
    gap_us: int | None
    normalization_source: str | None
    time_source: str
    polynomial_degree: int | None
    algorithm_version: str = TEMPORAL_ALGORITHM_VERSION


def validate_window_size(window_size: int, max_window_size: int = 29) -> None:
    if (isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 5
            or window_size > max_window_size or window_size % 2 == 0):
        raise ValueError(f"window_size must be an odd integer from 5 through {max_window_size}")


def _point(frame: ObservationFrame, point_id: int) -> tuple[float, float] | None:
    point = frame.keypoints.get(point_id)
    if not point or point.get("x") is None or point.get("y") is None:
        return None
    return float(point["x"]), float(point["y"])


def _mean_points(frame: ObservationFrame, ids: range | tuple[int, ...]) -> tuple[float, float] | None:
    values = [_point(frame, point_id) for point_id in ids]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(x for x, _ in valid) / len(valid), sum(y for _, y in valid) / len(valid)


def _bbox_transform(frame: ObservationFrame) -> Transform | None:
    if not frame.bbox or len(frame.bbox) < 4:
        return None
    x1, y1, x2, y2 = map(float, frame.bbox[:4])
    scale = math.sqrt(max(0.0, (x2 - x1) * (y2 - y1)))
    if scale <= 1e-6:
        return None
    return Transform((x1 + x2) / 2, (y1 + y2) / 2, scale, "bbox")


def normalization_transform(frame: ObservationFrame, feature: Literal["pose", "face"]) -> Transform | None:
    if feature == "pose":
        right_hip, left_hip = _point(frame, 12), _point(frame, 13)
        right_shoulder, left_shoulder = _point(frame, 6), _point(frame, 7)
        if all((right_hip, left_hip, right_shoulder, left_shoulder)):
            hip = ((right_hip[0] + left_hip[0]) / 2, (right_hip[1] + left_hip[1]) / 2)
            shoulder = ((right_shoulder[0] + left_shoulder[0]) / 2, (right_shoulder[1] + left_shoulder[1]) / 2)
            scale = math.dist(hip, shoulder)
            if scale > 1e-6:
                return Transform(hip[0], hip[1], scale, "torso")
        return _bbox_transform(frame)

    left_eye, right_eye = _mean_points(frame, range(14, 22)), _mean_points(frame, range(22, 30))
    bbox = _bbox_transform(frame)
    if left_eye and right_eye and math.dist(left_eye, right_eye) > 1e-6:
        center = bbox and (bbox.origin_x, bbox.origin_y)
        if center is None:
            center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        return Transform(center[0], center[1], math.dist(left_eye, right_eye), "iod")
    if bbox:
        return bbox
    if frame.image_width and frame.image_height and frame.image_width > 0 and frame.image_height > 0:
        return Transform(0.0, 0.0, float(max(frame.image_width, frame.image_height)), "image")
    return None


def _time_axis(frames: list[ObservationFrame], target: ObservationFrame) -> tuple[list[float], float, str, bool]:
    timestamps = [frame.timestamp_us for frame in frames]
    trusted_sources = {"video", "manifest", "fps_derived"}
    valid = target.timestamp_source in trusted_sources and all(frame.timestamp_source == target.timestamp_source for frame in frames)
    valid = valid and target.timestamp_us is not None and all(value is not None and value >= 0 for value in timestamps)
    ordered = sorted([*frames, target], key=lambda frame: frame.frame_index)
    ordered_timestamps = [frame.timestamp_us for frame in ordered]
    valid = valid and all(value is not None and value >= 0 for value in ordered_timestamps)
    valid = valid and len(set(ordered_timestamps)) == len(ordered_timestamps) and ordered_timestamps == sorted(ordered_timestamps)
    if valid:
        return [float(value) / 1_000_000 for value in timestamps], float(target.timestamp_us) / 1_000_000, target.timestamp_source, True
    return [float(frame.frame_index) for frame in frames], float(target.frame_index), "frame_index", False


def _robust_fit(times: np.ndarray, values: np.ndarray, base_weights: np.ndarray, degree: int,
                target_time: float, config: TemporalEngineConfig) -> tuple[np.ndarray, float]:
    relative = times - target_time
    span = max(float(np.max(np.abs(relative))), 1e-9)
    design = np.vander(relative / span, degree + 1, increasing=True)
    weights = np.clip(base_weights, 1e-4, None)
    coefficients = np.zeros((degree + 1, 2), dtype=np.float64)
    for _ in range(12):
        root = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(design * root, values * root, rcond=None)[0]
        residual_vectors = values - design @ coefficients
        residuals = np.linalg.norm(residual_vectors, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        delta = max(config.min_huber_delta, config.huber_k * 1.4826 * mad)
        huber = np.ones_like(residuals)
        mask = residuals > delta
        huber[mask] = delta / np.maximum(residuals[mask], 1e-9)
        new_weights = np.clip(base_weights * huber, 1e-4, None)
        if np.max(np.abs(new_weights - weights)) < 1e-6:
            weights = new_weights
            break
        weights = new_weights
    residual = float(np.sqrt(np.average(np.sum((values - design @ coefficients) ** 2, axis=1), weights=weights)))
    return coefficients[0], residual


def _unavailable(reason: str, *, support: int = 0, gap_frames: int | None = None,
                 gap_us: int | None = None, normalization: str | None = None,
                 time_source: str = "frame_index") -> TemporalEstimate:
    return TemporalEstimate(None, "unavailable", "unavailable", reason, 0.0, support, None,
                            gap_frames, gap_us, normalization, time_source, None)


def estimate_keypoint(frames: list[ObservationFrame], target_position: int, point_id: int, *,
                      feature: Literal["pose", "face"], window_size: int = 15,
                      config: TemporalEngineConfig = DEFAULT_TEMPORAL_CONFIG) -> TemporalEstimate:
    validate_window_size(window_size, 69 if feature == "face" else 29)
    if target_position < 0 or target_position >= len(frames):
        raise IndexError("target_position is outside frames")
    target = frames[target_position]
    if target.tracking_status != "confirmed":
        return _unavailable("tracking_uncertain")
    target_transform = normalization_transform(target, feature)
    if target_transform is None:
        return _unavailable("normalization_unavailable")

    radius = window_size // 2
    start, stop = max(0, target_position - radius), min(len(frames), target_position + radius + 1)
    candidates = frames[start:stop]
    transforms = [normalization_transform(frame, feature) for frame in candidates]
    observations: list[tuple[ObservationFrame, Transform, tuple[float, float]]] = []
    for absolute_position, (frame, transform) in enumerate(zip(candidates, transforms), start=start):
        point = _point(frame, point_id)
        if absolute_position == target_position or frame.tracking_status != "confirmed" or transform is None or point is None:
            continue
        if transform.source != target_transform.source:
            continue
        observations.append((frame, transform, point))
    if len(observations) < 2:
        return _unavailable("low_support", support=len(observations), normalization=target_transform.source)

    if feature == "face" and len(observations) > config.max_face_support_observations:
        observations = sorted(
            observations,
            key=lambda item: abs(item[0].frame_index - target.frame_index),
        )[:config.max_face_support_observations]
        observations.sort(key=lambda item: item[0].frame_index)

    left = [item for item in observations if item[0].frame_index < target.frame_index]
    right = [item for item in observations if item[0].frame_index > target.frame_index]
    method: Literal["interpolation", "extrapolation"] = "interpolation" if left and right else "extrapolation"
    if method == "extrapolation" and (window_size < 7 or max(len(left), len(right)) < 3):
        reason = "window_too_small" if window_size < 7 else "low_support"
        return _unavailable(reason, support=len(observations), normalization=target_transform.source)

    nearest = []
    if left:
        nearest.append(max(left, key=lambda item: item[0].frame_index)[0])
    if right:
        nearest.append(min(right, key=lambda item: item[0].frame_index)[0])
    if method == "interpolation":
        nearest_left = max(left, key=lambda item: item[0].frame_index)[0]
        nearest_right = min(right, key=lambda item: item[0].frame_index)[0]
        gap_frames = max(0, nearest_right.frame_index - nearest_left.frame_index - 1)
    else:
        gap_frames = max(abs(frame.frame_index - target.frame_index) for frame in nearest)
    time_frames = [item[0] for item in observations]
    times, target_time, time_source, uses_time = _time_axis(time_frames, target)
    gap_us = None
    if uses_time:
        if method == "interpolation":
            gap_us = nearest_right.timestamp_us - nearest_left.timestamp_us
        else:
            gap_us = max(abs(frame.timestamp_us - target.timestamp_us) for frame in nearest)
        if gap_frames > config.max_gap_frames_with_time or gap_us > config.max_gap_us:
            return _unavailable("gap_too_long", support=len(observations), gap_frames=gap_frames,
                                gap_us=gap_us, normalization=target_transform.source, time_source=time_source)
    elif gap_frames > config.max_gap_frames_without_time:
        return _unavailable("gap_too_long", support=len(observations), gap_frames=gap_frames,
                            normalization=target_transform.source, time_source=time_source)

    normalized = np.asarray([((point[0] - transform.origin_x) / transform.scale,
                              (point[1] - transform.origin_y) / transform.scale)
                             for _, transform, point in observations], dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    distances = np.abs(time_values - target_time)
    time_scale = max(float(np.max(distances)), 1e-9)
    qualities = np.asarray([min(1.0, max(0.0, frame.quality)) for frame, _, _ in observations])
    base_weights = np.exp(-config.time_decay * distances / time_scale) * np.maximum(qualities, 0.05)
    degree = 2 if len(observations) >= 5 else 1
    predicted, residual = _robust_fit(time_values, normalized, base_weights, degree, target_time, config)
    if not np.all(np.isfinite(predicted)) or np.linalg.norm(predicted) > config.max_normalized_radius:
        return _unavailable("geometry_implausible", support=len(observations), gap_frames=gap_frames,
                            gap_us=gap_us, normalization=target_transform.source, time_source=time_source)
    coordinate = (target_transform.origin_x + float(predicted[0]) * target_transform.scale,
                  target_transform.origin_y + float(predicted[1]) * target_transform.scale)
    support_factor = min(1.0, len(observations) / 5.0)
    quality = support_factor * float(np.mean(qualities)) * math.exp(-min(residual, 5.0) * config.quality_residual_decay)
    if feature == "face":
        side_balance = min(len(left), len(right)) / max(len(left), len(right)) if method == "interpolation" else 0.6
        gap_penalty = math.exp(-0.15 * max(gap_frames or 0, 0))
        quality *= side_balance * gap_penalty * math.exp(-5.0 * min(residual, 1.0))
        if quality < config.min_face_estimate_quality:
            return _unavailable("low_quality", support=len(observations), gap_frames=gap_frames,
                                gap_us=gap_us, normalization=target_transform.source, time_source=time_source)
    return TemporalEstimate(coordinate, "available", method, None, min(1.0, max(0.0, quality)),
                            len(observations), residual, gap_frames, gap_us,
                            target_transform.source, time_source, degree)


def estimate_track(frames: list[ObservationFrame], point_ids: list[int], *, feature: Literal["pose", "face"],
                   window_size: int = 15, config: TemporalEngineConfig = DEFAULT_TEMPORAL_CONFIG) -> dict[tuple[int, int], TemporalEstimate]:
    """Return an explicit estimate or rejection reason for every requested frame/keypoint pair."""
    validate_window_size(window_size, 69 if feature == "face" else 29)
    return {
        (position, point_id): estimate_keypoint(frames, position, point_id, feature=feature,
                                                window_size=window_size, config=config)
        for position in range(len(frames))
        for point_id in point_ids
    }
