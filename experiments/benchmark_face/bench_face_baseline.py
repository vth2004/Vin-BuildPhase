"""Run mapping-noise and MediaPipe-vs-GT baseline on prepared public data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.app.features.face_landmark_vf50.adapters.mediapipe_face import MediaPipeFaceDetector
from backend.app.features.face_landmark_vf50.rules.geometry import compute_iod, compute_iod_with_source
from backend.app.features.face_landmark_vf50.scoring.quality import evaluate_frame_quality

from .bench_mapping_core import VF50_GROUPS, load_mapping, noise_floor_by_skeleton


HERE = Path(__file__).resolve().parent


def _mapping(points: list[dict[str, Any]] | np.ndarray) -> dict[int, dict[str, Any]]:
    if isinstance(points, np.ndarray):
        return {idx: {"id": idx, "name": str(idx), "x": float(x), "y": float(y), "state": "visible"} for idx, (x, y) in enumerate(points)}
    return {int(point["id"]): point for point in points}


def _centroid(mapping: dict[int, dict[str, Any]]) -> tuple[float, float]:
    values = [(float(point["x"]), float(point["y"])) for point in mapping.values() if point.get("x") is not None and point.get("y") is not None]
    return statistics.fmean(value[0] for value in values), statistics.fmean(value[1] for value in values)


def _match(gt: list[dict[int, dict[str, Any]]], detected: list[np.ndarray]) -> dict[int, int]:
    if not gt or not detected:
        return {}
    cost = np.asarray([[math.dist(_centroid(g), _centroid(_mapping(d))) for d in detected] for g in gt])
    rows, cols = linear_sum_assignment(cost)
    matches = {}
    for row, col in zip(rows, cols):
        gt_iod = compute_iod(gt[row])
        cutoff = 2.0 * gt_iod if gt_iod > 10.0 else 150.0
        if cost[row, col] <= cutoff:
            matches[int(row)] = int(col)
    return matches


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95))}


def mapping_noise(dataset: dict[str, Any]) -> dict[str, Any]:
    config_name = "wflw98_to_vf50.yaml" if dataset["dataset"] == "WFLW" else "ibug68_to_vf50.yaml"
    config = load_mapping(HERE / "configs" / config_name)
    grouped: dict[str, list[float]] = defaultdict(list)
    for image in dataset["images"]:
        for annotation in image["annotations"]:
            source = np.asarray(annotation["source_points"], dtype=np.float64)
            human = _mapping(annotation["keypoints"])
            iod = compute_iod(human)
            for skeleton, values in noise_floor_by_skeleton(source, config, iod).items():
                grouped[skeleton].extend(values)
    return {name: {**_percentiles(values), "max": max(values) if values else None, "count": len(values)} for name, values in grouped.items()}


def run_dataset(dataset_path: Path, detector: MediaPipeFaceDetector, rows_writer: csv.DictWriter) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    totals = defaultdict(int)
    nmes: dict[str, list[float]] = defaultdict(list)
    latencies = []
    for image_index, image in enumerate(dataset["images"], start=1):
        frame = cv2.imread(image["source_path"])
        if frame is None:
            totals["decode_fail_images"] += 1
            continue
        start = time.perf_counter()
        detections_full = detector.detect_all_faces(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)
        detections = [item[0] for item in detections_full]
        gt = [_mapping(annotation["keypoints"]) for annotation in image["annotations"]]
        matches = _match(gt, detections)
        totals["images"] += 1
        totals["gt_faces"] += len(gt)
        totals["detected_faces"] += len(detections)
        totals["matched_faces"] += len(matches)
        if detections:
            totals["images_with_detection"] += 1
        for gt_index, human in enumerate(gt):
            annotation = image["annotations"][gt_index]
            matched = gt_index in matches
            row = {
                "dataset": dataset["dataset"], "split": image["split"], "file_name": image["file_name"],
                "face_index": gt_index, "matched": int(matched), "nme_runtime": "", "nme_gt_iod": "",
                "iod_runtime": "", "iod_source": "", "case_type": "miss", "attributes": json.dumps(annotation.get("attributes", {}), sort_keys=True),
            }
            if matched:
                model = _mapping(detections[matches[gt_index]])
                iod_runtime, iod_source = compute_iod_with_source(model, human)
                quality = evaluate_frame_quality(human, model, iod=iod_runtime)
                gt_quality = evaluate_frame_quality(human, model, iod=compute_iod(human))
                row.update(nme_runtime=quality["nme"], nme_gt_iod=gt_quality["nme"], iod_runtime=iod_runtime, iod_source=iod_source, case_type=quality["case_type"])
                slices = ["all", image["split"], quality["case_type"]]
                slices.extend(name for name, enabled in annotation.get("attributes", {}).items() if enabled)
                for name in slices:
                    nmes[name].append(float(quality["nme"]))
            rows_writer.writerow(row)
        if image_index % 250 == 0:
            print(f"{dataset['dataset']}: {image_index}/{len(dataset['images'])}", flush=True)
    return {
        "dataset": dataset["dataset"],
        "counts": dict(totals),
        "image_detection_coverage": totals["images_with_detection"] / max(1, totals["images"]),
        "face_detection_coverage": totals["matched_faces"] / max(1, totals["gt_faces"]),
        "face_detection_miss_rate": 1.0 - totals["matched_faces"] / max(1, totals["gt_faces"]),
        "nme_by_slice": {name: {**_percentiles(values), "count": len(values)} for name, values in sorted(nmes.items())},
        "latency_ms": _percentiles(latencies),
        "mapping_noise_floor_by_skeleton": mapping_noise(dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detector = MediaPipeFaceDetector()
    csv_path = args.output / "face_baseline_raw.csv"
    fields = ["dataset", "split", "file_name", "face_index", "matched", "nme_runtime", "nme_gt_iod", "iod_runtime", "iod_source", "case_type", "attributes"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        summaries = [run_dataset(path, detector, writer) for path in sorted(args.prepared.glob("*_landmark_qa_v1.json"))]
    (args.output / "face_baseline_summary.json").write_text(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
