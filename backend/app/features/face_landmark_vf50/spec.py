"""Config loader and single source of truth for VinFast Face Landmark VF-50.

Loads threshold definitions from configs/face_landmark_vf50/vf50_spec.yaml
with caching for high performance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

SPEC_FILE = Path(__file__).resolve().parents[3] / "configs" / "face_landmark_vf50" / "vf50_spec.yaml"


@lru_cache(maxsize=1)
def load_vf50_spec() -> dict[str, Any]:
    """Nạp cấu hình spec và ngưỡng kiểm định VF-50 từ YAML (có cache)."""
    if not SPEC_FILE.is_file():
        raise FileNotFoundError(f"Cannot find VF50 specification file at: {SPEC_FILE}")
    with open(SPEC_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_anchor_points() -> set[int]:
    spec = load_vf50_spec()
    return set(spec.get("anchor_points", [0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36]))


def get_scoring_tolerances() -> dict[str, float]:
    spec = load_vf50_spec()
    return spec.get("scoring_tolerances", {
        "anchor_point_iod_ratio": 0.03,
        "contour_point_iod_ratio": 0.05,
        "nme_max_threshold": 0.035,
        "severity_critical_multiplier": 1.8,
        "suspicion_normalizer_multiplier": 2.5,
        "occluded_suspicion_discount": 0.6,
        "min_iod_for_norm": 10.0,
    })


def get_case_classification_spec() -> dict[str, Any]:
    spec = load_vf50_spec()
    return spec.get("case_classification", {
        "scale_close_min_iod": 80.0,
        "scale_normal_min_iod": 38.0,
        "sunglasses_min_eye_occluded": 4,
        "sunglasses_ai_reliability": 0.45,
        "occluded_min_total_occluded": 6,
        "occluded_ai_reliability": 0.58,
        "far_max_iod": 35.0,
        "far_ai_reliability": 0.65,
        "normal_ai_reliability": 0.95,
    })


def get_rules_spec() -> dict[str, Any]:
    spec = load_vf50_spec()
    return spec.get("rules", {
        "scale_factor": {"base_iod": 96.0, "min_scale": 0.25, "max_scale": 2.5},
        "r04_eyebrow_monotonic": {"min_points_evaluated": 8, "min_tolerance_px": 1.5, "scale_factor_multiplier": 5.0},
        "r06_eyelid_tilt": {"min_tolerance_px": 0.8, "scale_factor_multiplier": 2.0},
        "r08_nose_bridge": {"max_ratio": 1.55, "min_segment_length_px": 1.0},
        "r09_inner_lip": {
            "corner_x_min_tolerance_px": 0.8,
            "corner_x_scale_factor_multiplier": 2.0,
            "contour_tolerance_multiplier": 2.0,
        },
        "r10_skeleton_visibility": {
            "mattrai": 4, "matphai": 4, "longmaytrai": 3, "longmayphai": 3,
            "moingoai": 6, "moitrong": 4, "songmui": 1,
        },
        "r12_coincident_points": {"min_threshold_px": 0.4, "iod_ratio_multiplier": 0.012},
        "r13_temporal_jump": {"min_jump_px": 5.0, "scale_factor_multiplier": 15.0},
        "r14_occluded_drift": {"max_drift_iod_ratio": 0.15},
    })


def get_geometry_spec() -> dict[str, Any]:
    spec = load_vf50_spec()
    return spec.get("geometry", {
        "iod": {
            "min_eye_points": 4,
            "min_eye_center_dist": 10.0,
            "outer_corners_min_dist": 5.0,
            "outer_corners_ratio": 0.65,
            "face_width_min_points": 8,
            "face_width_min_px": 10.0,
            "face_width_min_result_px": 12.0,
            "face_width_ratio": 0.42,
            "default_iod_px": 96.0,
        }
    })


def get_matching_spec() -> dict[str, Any]:
    spec = load_vf50_spec()
    return spec.get("matching", {
        "cutoff_iod_multiplier": 2.0,
    })


def get_hungarian_matching_spec() -> dict[str, Any]:
    spec = load_vf50_spec()
    return spec.get("hungarian_matching", {
        "centroid_distance_cutoff_iod_multiplier": 2.0,
        "max_distance_px_fallback": 150.0,
    })
