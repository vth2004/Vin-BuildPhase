from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import time
import uuid
import sqlite3
from app.core.database import db, init_db
from app.services.dataset_service import load_schema
from app.services.scoring import run_scoring
from app.services.ensemble import clear_prediction_cache


def run_benchmark_on_humanpose1():
    init_db()

    with db() as conn:
        datasets = conn.execute(
            "SELECT * FROM datasets WHERE name LIKE '%humanpose%' ORDER BY created_at DESC"
        ).fetchall()

        dataset = None
        for d in datasets:
            if Path(d["storage_path"]).exists():
                dataset = d
                break

        if not dataset:
            print("[LỖI] Không tìm thấy dataset 'humanpose1' hợp lệ có file trên đĩa!")
            return

        project_id = dataset["project_id"]
        dataset_id = dataset["id"]
        image_count = dataset["image_count"]

    print(f"[*] Đã tìm thấy dataset: {dataset['name']} (ID: {dataset_id})")
    print(f"    - Thuộc project: {project_id}")
    print(f"    - Số lượng ảnh: {image_count}")

    # =========================================================================
    # 1. CHẠY THỬ CHẾ ĐỘ K=1 (CHẾ ĐỘ NHANH)
    # =========================================================================
    run_id_k1 = str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            """INSERT INTO runs (id, project_id, dataset_id, status, progress, mode, created_at)
               VALUES (?, ?, ?, 'pending', 0, 'K1', ?)""",
            (run_id_k1, project_id, dataset_id, int(time.time())),
        )

    print("\n-------------------------------------------------------------")
    print("[*] Đang chạy kiểm thử CHẾ ĐỘ K=1 (YOLO26s đơn lẻ)...")
    clear_prediction_cache()
    t0_k1 = time.perf_counter()
    run_scoring(run_id_k1, db, load_schema, mode="K1")
    t1_k1 = time.perf_counter()
    total_time_k1 = t1_k1 - t0_k1
    avg_latency_k1 = (total_time_k1 / max(image_count, 1)) * 1000.0

    with db() as conn:
        warnings_k1 = conn.execute(
            """SELECT image_name, keypoint, warning_type, suspicion, human_x, human_y,
                      suggested_x, suggested_y, reliability, delta
               FROM warnings WHERE run_id = ? ORDER BY suspicion DESC LIMIT 50""",
            (run_id_k1,),
        ).fetchall()
        total_warn_k1 = conn.execute(
            "SELECT COUNT(*) FROM warnings WHERE run_id = ?", (run_id_k1,)
        ).fetchone()[0]

    print(f"[OK] Hoàn tất K=1:")
    print(f"     - Tổng thời gian: {total_time_k1:.2f}s")
    print(f"     - Thời gian trung bình/ảnh: {avg_latency_k1:.1f} ms/ảnh")
    print(f"     - Tổng số cảnh báo phát hiện (≥20%): {total_warn_k1}")

    # =========================================================================
    # 2. CHẠY THỬ CHẾ ĐỘ K=2 (ENSEMBLE MẶC ĐỊNH: YOLO26s + RTMPose-m)
    # =========================================================================
    run_id_k2 = str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            """INSERT INTO runs (id, project_id, dataset_id, status, progress, mode, created_at)
               VALUES (?, ?, ?, 'pending', 0, 'K2', ?)""",
            (run_id_k2, project_id, dataset_id, int(time.time())),
        )

    print("\n-------------------------------------------------------------")
    print("[*] Đang chạy kiểm thử CHẾ ĐỘ K=2 (Ensemble YOLO26s + RTMPose-m)...")
    clear_prediction_cache()
    t0_k2 = time.perf_counter()
    run_scoring(run_id_k2, db, load_schema, mode="K2")
    t1_k2 = time.perf_counter()
    total_time_k2 = t1_k2 - t0_k2
    avg_latency_k2 = (total_time_k2 / max(image_count, 1)) * 1000.0

    with db() as conn:
        warnings_k2 = conn.execute(
            """SELECT image_name, keypoint, warning_type, suspicion, human_x, human_y,
                      suggested_x, suggested_y, reliability, delta
               FROM warnings WHERE run_id = ? ORDER BY suspicion DESC LIMIT 50""",
            (run_id_k2,),
        ).fetchall()
        total_warn_k2 = conn.execute(
            "SELECT COUNT(*) FROM warnings WHERE run_id = ?", (run_id_k2,)
        ).fetchone()[0]

    print(f"[OK] Hoàn tất K=2:")
    print(f"     - Tổng thời gian: {total_time_k2:.2f}s")
    print(f"     - Thời gian trung bình/ảnh: {avg_latency_k2:.1f} ms/ảnh")
    print(f"     - Tổng số cảnh báo phát hiện (≥20%): {total_warn_k2}")

    # =========================================================================
    # 3. XUẤT BẢNG TOP CẢNH BÁO
    # =========================================================================
    print("\n" + "=" * 90)
    print("BẢNG SO SÁNH HIỆU NĂNG TỔNG QUAN TRÊN HUMANPOSE1 (20 ẢNH)")
    print("=" * 90)
    print(f"{'Chế độ':<20} | {'Thời gian/ảnh':<18} | {'Tổng cảnh báo':<16} | {'Ghi chú'}")
    print("-" * 90)
    print(f"{'K=1 (Nhanh)':<20} | {avg_latency_k1:6.1f} ms/ảnh{'':<6} | {total_warn_k1:<16} | YOLO26s + OKS đơn lẻ")
    print(f"{'K=2 (Ensemble)':<20} | {avg_latency_k2:6.1f} ms/ảnh{'':<6} | {total_warn_k2:<16} | YOLO26s + RTMPose-m (e * R^0.5)")
    print("=" * 90)

    # In top 20 của K=1 và K=2 ra console
    print("\n=== TOP CẢNH BÁO CỦA K=1 (Đơn model): ===")
    print(f"{'STT':<4} | {'Ảnh':<20} | {'Khớp':<12} | {'Điểm nghi ngờ':<14} | {'Loại lỗi':<14} | {'Tọa độ gợi ý'}")
    print("-" * 85)
    for i, w in enumerate(warnings_k1[:20], 1):
        print(f"{i:<4} | {w['image_name']:<20} | {w['keypoint']:<12} | {w['suspicion']*100:5.1f}%{'':<8} | {w['warning_type']:<14} | ({w['suggested_x']}, {w['suggested_y']})")

    print("\n=== TOP CẢNH BÁO CỦA K=2 (Ensemble K=2): ===")
    print(f"{'STT':<4} | {'Ảnh':<20} | {'Khớp':<12} | {'Điểm':<8} | {'Tin cậy R':<10} | {'Bất đồng δ':<12} | {'Loại lỗi':<14} | {'Tham chiếu p*'}")
    print("-" * 95)
    for i, w in enumerate(warnings_k2[:20], 1):
        rel = f"{w['reliability']*100:4.1f}%" if w['reliability'] is not None else "N/A"
        delta_str = f"{w['delta']:.4f}" if w['delta'] is not None else "0.0"
        print(f"{i:<4} | {w['image_name']:<20} | {w['keypoint']:<12} | {w['suspicion']*100:5.1f}% | {rel:<10} | {delta_str:<12} | {w['warning_type']:<14} | ({w['suggested_x']}, {w['suggested_y']})")

    # Lưu kết quả đầy đủ ra file Markdown để người dùng xem bằng mắt
    out_md = backend_dir / "scripts" / "TOP_50_WARNINGS_HUMANPOSE1.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# KẾT QUẢ ĐỐI CHIẾU TOP CẢNH BÁO TRÊN TẬP HUMANPOSE1 (K=1 vs K=2)\n\n")
        f.write(f"- **Tập dữ liệu:** `humanpose1` ({image_count} ảnh VinFast)\n")
        f.write(f"- **Thời gian K=1:** {avg_latency_k1:.1f} ms/ảnh (Tổng cảnh báo: {total_warn_k1})\n")
        f.write(f"- **Thời gian K=2:** {avg_latency_k2:.1f} ms/ảnh (Tổng cảnh báo: {total_warn_k2})\n\n")

        f.write("## 1. Top Cảnh báo Chế độ K=2 (Ensemble YOLO26s + RTMPose-m)\n\n")
        f.write("| STT | Tên ảnh | Khớp | Điểm nghi ngờ | Độ tin cậy ($R$) | Bất đồng ($\\delta$) | Loại cảnh báo | Người gán | Tham chiếu $p^*$ |\n")
        f.write("|:---:|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|\n")
        for i, w in enumerate(warnings_k2, 1):
            rel = f"{w['reliability']*100:4.1f}%" if w['reliability'] is not None else "N/A"
            delta_str = f"{w['delta']:.4f}" if w['delta'] is not None else "0.0"
            f.write(f"| {i} | `{w['image_name']}` | `{w['keypoint']}` | **{w['suspicion']*100:.1f}%** | {rel} | {delta_str} | `{w['warning_type']}` | ({w['human_x']}, {w['human_y']}) | ({w['suggested_x']}, {w['suggested_y']}) |\n")

        f.write("\n## 2. Top Cảnh báo Chế độ K=1 (Đơn model YOLO26s + OKS)\n\n")
        f.write("| STT | Tên ảnh | Khớp | Điểm nghi ngờ | Loại cảnh báo | Người gán | Model gợi ý |\n")
        f.write("|:---:|:---|:---|:---:|:---|:---:|:---:|\n")
        for i, w in enumerate(warnings_k1, 1):
            f.write(f"| {i} | `{w['image_name']}` | `{w['keypoint']}` | **{w['suspicion']*100:.1f}%** | `{w['warning_type']}` | ({w['human_x']}, {w['human_y']}) | ({w['suggested_x']}, {w['suggested_y']}) |\n")

    print(f"\n[OK] Đã xuất toàn bộ bảng so sánh Top 50 ra file: {out_md}")


if __name__ == "__main__":
    run_benchmark_on_humanpose1()
