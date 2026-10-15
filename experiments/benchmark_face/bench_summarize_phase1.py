"""Create Phase-1 audit tables from raw per-face MediaPipe results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[tuple[str, str], dict[str, object]] = defaultdict(lambda: {"total": 0, "matched": 0, "nme": []})
    with args.raw.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            attributes = json.loads(row["attributes"])
            slices = {"all", row["split"]}
            slices.update(name for name, enabled in attributes.items() if enabled)
            matched = row["matched"] == "1"
            for slice_name in slices:
                group = groups[(row["dataset"], slice_name)]
                group["total"] = int(group["total"]) + 1
                if matched:
                    group["matched"] = int(group["matched"]) + 1
                    cast_values = group["nme"]
                    assert isinstance(cast_values, list)
                    cast_values.append(float(row["nme_runtime"]))
    slices = []
    for (dataset, slice_name), group in sorted(groups.items()):
        total, matched = int(group["total"]), int(group["matched"])
        values = group["nme"]
        assert isinstance(values, list)
        slices.append({
            "dataset": dataset, "slice": slice_name, "total_faces": total, "matched_faces": matched,
            "detection_recall": matched / total if total else None, "nme_runtime": stats(values),
        })
    original = json.loads(args.summary.read_text(encoding="utf-8"))
    output = {"raw_csv": str(args.raw), "baseline_summary": str(args.summary), "slices": slices, **original}
    args.json_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Face Benchmark — Phase 1 Summary", "",
        "> NME chỉ được tính trên face mà MediaPipe phát hiện và ghép thành công. Detection recall dùng toàn bộ GT face, vì vậy phải đọc hai chỉ số cùng nhau.", "",
        f"Raw per-face: `{args.raw.as_posix()}`  ", f"Raw aggregate: `{args.summary.as_posix()}`", "",
        "## Detection coverage tổng thể", "",
        "| Dataset | Images | GT faces | Images có detection | Image coverage | Matched faces | Face coverage | Miss rate | Latency median | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in original["summaries"]:
        counts = summary["counts"]
        lines.append(
            f"| {summary['dataset']} | {counts['images']} | {counts['gt_faces']} | {counts['images_with_detection']} | "
            f"{100*summary['image_detection_coverage']:.2f}% | {counts['matched_faces']} | "
            f"{100*summary['face_detection_coverage']:.2f}% | {100*summary['face_detection_miss_rate']:.2f}% | "
            f"{summary['latency_ms']['median']:.2f} ms | {summary['latency_ms']['p95']:.2f} ms |"
        )
    lines.extend(["",
        "## MediaPipe coverage và NME theo lát", "",
        "| Dataset | Slice | GT face | Matched | Detection recall | NME mean | NME median | NME P90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in slices:
        nme = item["nme_runtime"]
        fmt = lambda value: "chưa đo" if value is None else f"{100 * float(value):.2f}%"
        lines.append(
            f"| {item['dataset']} | {item['slice']} | {item['total_faces']} | {item['matched_faces']} | "
            f"{100 * float(item['detection_recall']):.2f}% | {fmt(nme['mean'])} | {fmt(nme['median'])} | {fmt(nme['p90'])} |"
        )
    lines.extend(["", "## Mapping noise floor", "", "Đơn vị dưới đây là phần trăm IOD; direct mapping có algorithmic floor bằng 0.", "",
                  "| Dataset | Skeleton | Median | P95 | Max |", "|---|---|---:|---:|---:|"])
    for summary in original["summaries"]:
        for skeleton, values in summary["mapping_noise_floor_by_skeleton"].items():
            lines.append(f"| {summary['dataset']} | {skeleton} | {100*values['median']:.3f}% | {100*values['p95']:.3f}% | {100*values['max']:.3f}% |")
    lines.extend(["", "## Giới hạn", "", "- WFLW attributes là nhãn cấp face, không phải visibility từng điểm.",
                  "- NME có selection bias do chỉ tính được khi detector tìm và ghép đúng face.",
                  "- WFLW direct 98→50 không nội suy; noise floor 0 chỉ nói về thuật toán mapping, không nói annotation protocol giữa WFLW và VF50 hoàn toàn tương đương.",
                  "- 300W-68 cần nội suy mí mắt 6→8; đây là nguồn mapping noise đo được."])
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
