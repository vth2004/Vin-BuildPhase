"""Shared temporal metadata, observation matching and tracking."""

from .sequence import SequenceInference, infer_sequences, parse_sequence_manifest
from .tracking import Assignment, DetectionTracker, Match, assign_one_to_one, continuity_cost
from .segmentation import is_scene_cut, luminance_histogram
from .engine import (
    TEMPORAL_ALGORITHM_VERSION, DEFAULT_TEMPORAL_CONFIG, ObservationFrame, TemporalEngineConfig, TemporalEstimate,
    estimate_keypoint, estimate_track, normalization_transform, validate_window_size,
)

__all__ = [
    "SequenceInference", "infer_sequences", "parse_sequence_manifest",
    "Assignment", "DetectionTracker", "Match", "assign_one_to_one", "continuity_cost",
    "is_scene_cut", "luminance_histogram",
    "TEMPORAL_ALGORITHM_VERSION", "DEFAULT_TEMPORAL_CONFIG", "ObservationFrame", "TemporalEngineConfig", "TemporalEstimate",
    "estimate_keypoint", "estimate_track", "normalization_transform", "validate_window_size",
]
