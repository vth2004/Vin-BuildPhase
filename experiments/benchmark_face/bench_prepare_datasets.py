"""Prepare public WFLW and 300W annotations as landmark-qa/v1 copies."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .bench_mapping_core import load_mapping, map_points


HERE = Path(__file__).resolve().parent


def _keypoints(mapped: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"id": idx, "name": str(idx), "x": float(x), "y": float(y), "state": "visible"}
        for idx, (x, y) in enumerate(mapped)
    ]


def _split_by_image(paths: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(paths))
    rng = random.Random(seed)
    rng.shuffle(unique)
    dev_count = len(unique) // 2
    return {path: ("dev" if index < dev_count else "test") for index, path in enumerate(unique)}


def prepare_wflw(root: Path, seed: int) -> dict[str, Any]:
    config = load_mapping(HERE / "configs" / "wflw98_to_vf50.yaml")
    annotation_root = root / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
    image_root = root / "WFLW_images"
    rows: list[tuple[str, dict[str, Any]]] = []
    attribute_names = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
    for official_split in ("train", "test"):
        path = annotation_root / f"list_98pt_rect_attr_{official_split}.txt"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.strip().split()
            if len(fields) != 207:
                raise ValueError(f"{path}:{line_number}: expected 207 fields, got {len(fields)}")
            source = np.asarray([float(value) for value in fields[:196]], dtype=np.float64).reshape(98, 2)
            bbox = [float(value) for value in fields[196:200]]
            attrs = {name: bool(int(value)) for name, value in zip(attribute_names, fields[200:206])}
            relative = fields[206].replace("\\", "/")
            rows.append((relative, {
                "label": "face_landmarks",
                "source_dataset": "WFLW",
                "source_official_split": official_split,
                "source_line": line_number,
                "source_bbox": bbox,
                "attributes": attrs,
                "keypoints": _keypoints(map_points(source, config)),
                "source_points": source.tolist(),
            }))
    split_map = _split_by_image([row[0] for row in rows], seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relative, annotation in rows:
        grouped[relative].append(annotation)
    images = []
    for relative in sorted(grouped):
        image_path = image_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({
            "file_name": relative,
            "source_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "split": split_map[relative],
            "annotations": grouped[relative],
        })
    return {
        "format": "landmark-qa/v1",
        "schema_id": "vf_face_landmark50_v1",
        "dataset": "WFLW",
        "seed": seed,
        "split_unit": "image",
        "images": images,
    }


def _read_pts(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    start = lines.index("{") + 1
    end = lines.index("}")
    values = [[float(value) for value in line.split()] for line in lines[start:end] if line.strip()]
    points = np.asarray(values, dtype=np.float64)
    if points.shape != (68, 2):
        raise ValueError(f"{path}: expected 68x2, got {points.shape}")
    return points - 1.0  # Official 300W annotations use MATLAB pixel origin (1, 1).


def prepare_300w(root: Path, seed: int) -> dict[str, Any]:
    config = load_mapping(HERE / "configs" / "ibug68_to_vf50.yaml")
    rows = []
    for pts_path in sorted(root.rglob("*.pts")):
        image_path = next((candidate for candidate in (pts_path.with_suffix(".png"), pts_path.with_suffix(".jpg")) if candidate.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"No image for {pts_path}")
        relative = image_path.relative_to(root).as_posix()
        source = _read_pts(pts_path)
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append({
            "file_name": relative,
            "source_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "annotations": [{
                "label": "face_landmarks",
                "source_dataset": "300W",
                "source_annotation": pts_path.relative_to(root).as_posix(),
                "attributes": {"indoor": "01_Indoor" in relative, "outdoor": "02_Outdoor" in relative},
                "keypoints": _keypoints(map_points(source, config)),
                "source_points": source.tolist(),
            }],
        })
    split_map = _split_by_image([row["file_name"] for row in rows], seed)
    for row in rows:
        row["split"] = split_map[row["file_name"]]
    return {
        "format": "landmark-qa/v1",
        "schema_id": "vf_face_landmark50_v1",
        "dataset": "300W-private",
        "seed": seed,
        "split_unit": "image",
        "coordinate_conversion": "source MATLAB (1,1) to image (0,0)",
        "images": rows,
    }


def _write_dataset(dataset: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for split in ("dev", "test"):
        paths = [image["file_name"] for image in dataset["images"] if image["split"] == split]
        (output.parent / f"{dataset['dataset'].lower()}_{split}_images.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wflw-root", type=Path, required=True)
    parser.add_argument("--w300-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _write_dataset(prepare_wflw(args.wflw_root, args.seed), args.output / "wflw_landmark_qa_v1.json")
    _write_dataset(prepare_300w(args.w300_root, args.seed), args.output / "300w_landmark_qa_v1.json")


if __name__ == "__main__":
    main()
