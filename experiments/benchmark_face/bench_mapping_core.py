"""Config-driven public-dataset landmark mapping into VF50.

This benchmark module intentionally reuses the production arc-length primitive
without changing production code. It does not import either Pose feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from backend.app.features.face_landmark_vf50.adapters.mapper import resample_polyline


VF50_ANCHORS = {0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36}
VF50_GROUPS = {
    "longmaytrai": tuple(range(0, 5)),
    "longmayphai": tuple(range(5, 10)),
    "songmui": tuple(range(10, 14)),
    "mattrai": tuple(range(14, 22)),
    "matphai": tuple(range(22, 30)),
    "moingoai": tuple(range(30, 42)),
    "moitrong": tuple(range(42, 50)),
}


def load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_mapping(config)
    return config


def validate_mapping(config: dict[str, Any]) -> None:
    count = int(config["source_point_count"])
    assigned: dict[int, str] = {}
    for group in config["groups"]:
        vf_ids = [int(value) for value in group["vf_ids"]]
        source_ids = [int(value) for value in group["source_ids"]]
        if group["method"] == "direct" and len(vf_ids) != len(source_ids):
            raise ValueError(f"Direct group {group['name']} has unequal lengths")
        if group["method"] == "resample" and len(vf_ids) != len(group["fractions"]):
            raise ValueError(f"Resample group {group['name']} has unequal lengths")
        if min(source_ids) < 0 or max(source_ids) >= count:
            raise ValueError(f"Source ID outside 0..{count - 1}: {source_ids}")
        for vf_id in vf_ids:
            if vf_id in assigned:
                raise ValueError(f"VF ID {vf_id} assigned twice")
            assigned[vf_id] = group["name"]
    if set(assigned) != set(range(50)):
        raise ValueError(f"Mapping must assign VF IDs 0..49; got {sorted(assigned)}")
    anchors = {int(key): int(value) for key, value in config["anchors"].items()}
    if set(anchors) != VF50_ANCHORS:
        raise ValueError("Mapping must declare all 12 VF50 anchors")


def map_points(source_points: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    points = np.asarray(source_points, dtype=np.float64)
    expected = int(config["source_point_count"])
    if points.ndim != 2 or points.shape[0] != expected or points.shape[1] < 2:
        raise ValueError(f"Expected ({expected}, >=2), got {points.shape}")
    result = np.full((50, 2), np.nan, dtype=np.float64)
    for group in config["groups"]:
        vf_ids = np.asarray(group["vf_ids"], dtype=int)
        source_ids = np.asarray(group["source_ids"], dtype=int)
        polyline = points[source_ids, :2]
        if group["method"] == "direct":
            mapped = polyline
        elif group["method"] == "resample":
            mapped = resample_polyline(polyline, group["fractions"])[:, :2]
        else:
            raise ValueError(f"Unknown method: {group['method']}")
        result[vf_ids] = mapped
    return result


def mapping_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in config["groups"]:
        for offset, vf_id in enumerate(group["vf_ids"]):
            source = group["source_ids"][offset] if group["method"] == "direct" else group["source_ids"]
            rows.append({
                "vf_id": int(vf_id),
                "skeleton": group["name"],
                "method": group["method"],
                "source": source,
                "fraction": None if group["method"] == "direct" else float(group["fractions"][offset]),
                "anchor": int(vf_id) in VF50_ANCHORS,
            })
    return sorted(rows, key=lambda row: row["vf_id"])


def noise_floor_by_skeleton(
    source_points: np.ndarray,
    config: dict[str, Any],
    iod: float,
    fraction_delta: float = 0.02,
) -> dict[str, list[float]]:
    """Measure mapping sensitivity by perturbing only non-anchor fractions.

    Direct mappings are definitionally zero. For resampled contour points, two
    symmetric fraction perturbations approximate a small segmentation/start
    convention change while anchor endpoints remain fixed.
    """
    baseline = map_points(source_points, config)
    variants: list[np.ndarray] = []
    for sign in (-1.0, 1.0):
        variant = yaml.safe_load(yaml.safe_dump(config))
        for group in variant["groups"]:
            if group["method"] != "resample":
                continue
            anchor_ids = set(group.get("anchors", []))
            group["fractions"] = [
                value if vf_id in anchor_ids else min(1.0, max(0.0, value + sign * fraction_delta))
                for vf_id, value in zip(group["vf_ids"], group["fractions"])
            ]
        variants.append(map_points(source_points, variant))
    denominator = max(float(iod), 10.0)
    output: dict[str, list[float]] = {}
    for name, ids in VF50_GROUPS.items():
        values: list[float] = []
        for variant in variants:
            values.extend((np.linalg.norm(variant[list(ids)] - baseline[list(ids)], axis=1) / denominator).tolist())
        output[name] = values
    return output
