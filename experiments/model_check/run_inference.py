import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import time
import psutil
from pathlib import Path
from typing import Any
import numpy as np

SANDBOX_DIR = Path(__file__).resolve().parent
DATA_DIR = SANDBOX_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CACHE_DIR = SANDBOX_DIR / "cache"
RESULTS_DIR = SANDBOX_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from models_manager import setup_all_models, BasePoseModel


def run_model_inference_on_split(
    model_name: str,
    model: BasePoseModel,
    split_items: list[dict[str, Any]],
    split_name: str = "dev",
) -> dict[str, Any]:
    """
    Chạy inference có cache theo (model_name, image_id).
    Ghi nhận latency, memory và kết quả dự đoán.
    """
    model_cache_dir = CACHE_DIR / model_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    cached_hits = 0
    new_inferences = 0

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 * 1024)

    total_images = len(split_items)
    print(f"\n[*] Đang chạy Model [{model_name}] trên tập [{split_name}] ({total_images} ảnh)...")

    for idx, item in enumerate(split_items):
        img_id = item["image_id"]
        file_name = item["file_name"]
        cache_file = model_cache_dir / f"{img_id}.json"

        if cache_file.is_file():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                latencies.append(cached_data.get("latency_ms", 0.0))
                cached_hits += 1
                continue
            except Exception:
                pass

        # Chưa có cache -> chạy inference thực tế
        img_path = IMAGES_DIR / file_name
        if not img_path.is_file():
            continue

        persons, latency_ms = model.predict(img_path)
        latencies.append(latency_ms)
        new_inferences += 1

        # Lưu cache
        cache_data = {
            "image_id": img_id,
            "file_name": file_name,
            "model_name": model_name,
            "latency_ms": round(latency_ms, 2),
            "persons": persons,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        if (idx + 1) % 50 == 0 or idx == total_images - 1:
            print(f"    Tiến độ: {idx + 1}/{total_images} | Latency trung bình gần nhất: {np.mean(latencies[-50:]):.1f} ms")

    mem_after = proc.memory_info().rss / (1024 * 1024)

    stats = {
        "model_name": model_name,
        "split_name": split_name,
        "total_images": total_images,
        "new_inferences": new_inferences,
        "cached_hits": cached_hits,
        "model_size_mb": model.get_model_size_mb(),
        "latency_mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "latency_median_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "latency_min_ms": round(float(np.min(latencies)), 2) if latencies else 0.0,
        "latency_max_ms": round(float(np.max(latencies)), 2) if latencies else 0.0,
        "memory_delta_mb": round(mem_after - mem_before, 2),
    }

    return stats


def main():
    manifest_file = DATA_DIR / "split_manifest.json"
    if not manifest_file.is_file():
        print(f"[LỖI] Chưa có {manifest_file}. Hãy chạy prepare_dataset.py trước!")
        sys.exit(1)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models = setup_all_models()
    if not models:
        print("[LỖI] Không có model nào được tải thành công!")
        sys.exit(1)

    speed_reports: dict[str, Any] = {"dev": {}, "test": {}}

    dev_items = manifest["dev"]
    test_items = manifest["test"]

    for split_name, items in [("dev", dev_items), ("test", test_items)]:
        print(f"\n==========================================")
        print(f"=== CHẠY INFERENCE TRÊN TẬP: {split_name.upper()} ({len(items)} ảnh) ===")
        print(f"==========================================")
        for m_name, m_inst in models.items():
            stats = run_model_inference_on_split(m_name, m_inst, items, split_name=split_name)
            speed_reports[split_name][m_name] = stats
            print(f"[{split_name.upper()} - {m_name}] Median={stats['latency_median_ms']} ms, P95={stats['latency_p95_ms']} ms, Cached={stats['cached_hits']}/{stats['total_images']}")

    out_file = RESULTS_DIR / "speed_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(speed_reports, f, indent=2)
    print(f"\n[OK] Đã ghi nhận benchmark tốc độ vào: {out_file}")


if __name__ == "__main__":
    main()
