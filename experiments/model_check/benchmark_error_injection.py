from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import math
import random
from pathlib import Path
from typing import Any
import numpy as np

SANDBOX_DIR = Path(__file__).resolve().parent
DATA_DIR = SANDBOX_DIR / "data"
CACHE_DIR = SANDBOX_DIR / "cache"
RESULTS_DIR = SANDBOX_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COCO_TO_VF17 = {
    0: 1, 1: 3, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 9, 8: 8,
    9: 11, 10: 10, 11: 13, 12: 12, 13: 15, 14: 14, 15: 17, 16: 16
}

# COCO OKS sigma_i cho từng khớp
COCO_SIGMAS = {
    1: 0.026, 2: 0.025, 3: 0.025, 4: 0.035, 5: 0.035,
    6: 0.079, 7: 0.079, 8: 0.072, 9: 0.072, 10: 0.062, 11: 0.062,
    12: 0.107, 13: 0.107, 14: 0.087, 15: 0.087, 16: 0.089, 17: 0.089
}

# Cặp đối xứng Trái - Phải: (Right_ID, Left_ID)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
PAIR_LOOKUP = {}
for r_id, l_id in SYMMETRIC_PAIRS:
    PAIR_LOOKUP[r_id] = l_id
    PAIR_LOOKUP[l_id] = r_id


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Fast Mann-Whitney U AUROC
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = np.sum(ranks[:n_pos])
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    total_pos = int(np.sum(y_true == 1))
    if total_pos == 0:
        return (0.0, 0.0)
    top_indices = np.argsort(-y_score)[:k]
    hits = int(np.sum(y_true[top_indices] == 1))
    prec = hits / max(k, 1)
    rec = hits / total_pos
    return (round(prec, 4), round(rec, 4))


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    # Scale score về [0, 1] nếu cần
    scores = np.clip(y_score, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (scores >= low) & (scores < high if i < n_bins - 1 else scores <= high)
        bin_size = np.sum(mask)
        if bin_size > 0:
            avg_conf = np.mean(scores[mask])
            avg_acc = np.mean(y_true[mask])
            ece += (bin_size / n) * abs(avg_acc - avg_conf)
    return round(float(ece), 4)


def inject_errors_into_instances(instances: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    injected: list[dict[str, Any]] = []

    # Gom theo (image_id, ann_idx) để có thể xử lý swap error giữa 2 khớp trong cùng 1 người
    person_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for inst in instances:
        key = (inst["image_id"], inst["ann_idx"])
        person_groups.setdefault(key, []).append(inst)

    for (img_id, ann_idx), group in person_groups.items():
        scale = group[0]["scale"]
        gt_by_id = {item["vf_id"]: item["gt"] for item in group}

        # Quyết định khớp nào bị lỗi: tỷ lệ 25% lỗi
        for item in group:
            vf_id = item["vf_id"]
            gx, gy = item["gt"]
            is_error = rng.random() < 0.25

            if not is_error:
                # Negative: Jitter nhỏ bình thường (độ lệch giải phẫu / rung tay nhẹ)
                jitter_r = rng.gauss(0, 0.012 * scale)
                jitter_ang = rng.uniform(0, 2 * math.pi)
                hx = gx + jitter_r * math.cos(jitter_ang)
                hy = gy + jitter_r * math.sin(jitter_ang)
                label = 0
                err_type = "clean"
            else:
                label = 1
                # Positive: 70% dịch chuyển lớn (0.1 - 0.3 * s), 30% swap trái phải
                if rng.random() < 0.30 and vf_id in PAIR_LOOKUP and PAIR_LOOKUP[vf_id] in gt_by_id:
                    # Swap error
                    other_id = PAIR_LOOKUP[vf_id]
                    ox, oy = gt_by_id[other_id]
                    # Hoán đổi sang tọa độ khớp đối diện + jitter nhỏ
                    hx = ox + rng.gauss(0, 0.01 * scale)
                    hy = oy + rng.gauss(0, 0.01 * scale)
                    err_type = "swap"
                else:
                    # Spatial shift: 0.10 * s -> 0.30 * s
                    shift_dist = rng.uniform(0.10, 0.30) * scale
                    ang = rng.uniform(0, 2 * math.pi)
                    hx = gx + shift_dist * math.cos(ang)
                    hy = gy + shift_dist * math.sin(ang)
                    err_type = "shift"

            new_item = dict(item)
            new_item["human_pt"] = (hx, hy)
            new_item["label"] = label
            new_item["error_type"] = err_type
            injected.append(new_item)

    return injected


def evaluate_scorer(
    instances: list[dict[str, Any]],
    scorer_func,
) -> tuple[float, dict[int, tuple[float, float]], float]:
    y_true = np.array([inst["label"] for inst in instances])
    y_scores = np.array([scorer_func(inst) for inst in instances])

    auroc = compute_auroc(y_true, y_scores)
    pr_k = {k: compute_precision_recall_at_k(y_true, y_scores, k) for k in [20, 50, 100]}
    ece = compute_ece(y_true, y_scores)
    return round(auroc, 4), pr_k, ece


def run_error_injection_benchmark():
    from evaluate_benchmark import extract_keypoint_instances, load_cached_predictions

    manifest_file = DATA_DIR / "split_manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Đọc tốc độ đo thực tế từ speed_benchmark.json
    speed_file = RESULTS_DIR / "speed_benchmark.json"
    speed_data = json.load(open(speed_file, "r", encoding="utf-8")) if speed_file.exists() else {}

    available_models = ["yolo26s-pose", "yolov8n-pose", "yolov8m-pose", "rtmpose-m"]

    # 1. Trích xuất instances từ Dev và Test
    dev_preds = load_cached_predictions(available_models, [it["image_id"] for it in manifest["dev"]])
    dev_instances = extract_keypoint_instances(manifest["dev"], dev_preds, available_models)

    test_preds = load_cached_predictions(available_models, [it["image_id"] for it in manifest["test"]])
    test_instances = extract_keypoint_instances(manifest["test"], test_preds, available_models)

    # Tiêm lỗi
    dev_injected = inject_errors_into_instances(dev_instances, seed=42)
    test_injected = inject_errors_into_instances(test_instances, seed=123)

    print(f"[*] Số lượng keypoint Dev: {len(dev_injected)} (Lỗi: {sum(x['label'] for x in dev_injected)})")
    print(f"[*] Số lượng keypoint Test: {len(test_injected)} (Lỗi: {sum(x['label'] for x in test_injected)})")

    # Đọc trọng số w_k fit trên dev từ rigorous_benchmark_test.json
    rig_file = RESULTS_DIR / "rigorous_benchmark_test.json"
    if rig_file.exists():
        rig_data = json.load(open(rig_file, "r", encoding="utf-8"))
        dev_weights = rig_data["dev_weights"]["raw_w_k"]
    else:
        dev_weights = {"yolo26s-pose": 2028.9, "yolov8n-pose": 1082.0, "yolov8m-pose": 1826.8, "rtmpose-m": 2860.0}

    # 2. TUNE HYPERPARAMETERS TRÊN TẬP DEV (Grid search: lambda, tau, alpha)
    print("\n[*] Đang tuning siêu tham số (lambda, tau, alpha) trên DEV set...")
    lambda_cands = [0.5, 1.0, 2.0]
    tau_cands = [0.03, 0.05, 0.08]
    alpha_cands = [0.5, 1.0, 2.0]

    best_score_dev = -1.0
    best_params = {"lambda": 1.0, "tau": 0.05, "alpha": 1.0}

    # Helper scoring cho K=3 ensemble
    def make_k3_scorer(lam, tau, alpha, use_r):
        models = ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"]
        def scorer(inst):
            active_m = [m for m in models if m in inst["models"]]
            if not active_m:
                return 0.0
            hx, hy = inst["human_pt"]
            scale = inst["scale"]
            vf_id = inst["vf_id"]
            sigma_i = COCO_SIGMAS.get(vf_id, 0.070)
            k_i = 2.0 * sigma_i

            # Consensus p*
            sum_w = sum(dev_weights[m] * inst["models"][m]["conf"] for m in active_m)
            if sum_w <= 0:
                sum_w = 1.0
            px = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["x"] for m in active_m) / sum_w
            py = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["y"] for m in active_m) / sum_w

            dist_norm = math.hypot(hx - px, hy - py) / scale
            disp = sum(
                dev_weights[m] * inst["models"][m]["conf"] * ((inst["models"][m]["x"] - px)**2 + (inst["models"][m]["y"] - py)**2)
                for m in active_m
            ) / sum_w
            delta = math.sqrt(max(0.0, disp)) / scale

            # OKS adaptive
            e = 1.0 - math.exp(- (dist_norm ** 2) / (2.0 * (k_i**2 + lam * (delta**2))))
            if not use_r:
                return e

            # R_i
            c_bar = sum(dev_weights[m] * inst["models"][m]["conf"] for m in active_m) / sum(dev_weights[m] for m in active_m)
            r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
            return e * (r ** alpha)

        return scorer

    for l_val in lambda_cands:
        for t_val in tau_cands:
            for a_val in alpha_cands:
                sc_func = make_k3_scorer(l_val, t_val, a_val, use_r=True)
                auroc_val, _, _ = evaluate_scorer(dev_injected, sc_func)
                if auroc_val > best_score_dev:
                    best_score_dev = auroc_val
                    best_params = {"lambda": l_val, "tau": t_val, "alpha": a_val}

    print(f"[OK] Bộ tham số tối ưu tìm được trên DEV: {best_params} (AUROC Dev = {best_score_dev})")

    # 3. ĐÁNH GIÁ CÁC CẤU HÌNH TRÊN TẬP TEST
    print("\n=== [BƯỚC 3: ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST] ===")

    # Scorer 1: M0 với x4.5 heuristic
    def scorer_m0_heur(inst):
        if "yolo26s-pose" not in inst["models"]:
            return 0.0
        m = inst["models"]["yolo26s-pose"]
        hx, hy = inst["human_pt"]
        dist = math.hypot(hx - m["x"], hy - m["y"]) / inst["scale"]
        return min(1.0, 4.5 * dist) * m["conf"]

    # Scorer 2: M0 + OKS (sigma_i)
    def scorer_m0_oks(inst):
        if "yolo26s-pose" not in inst["models"]:
            return 0.0
        m = inst["models"]["yolo26s-pose"]
        hx, hy = inst["human_pt"]
        dist = math.hypot(hx - m["x"], hy - m["y"]) / inst["scale"]
        k_i = 2.0 * COCO_SIGMAS.get(inst["vf_id"], 0.070)
        e = 1.0 - math.exp(- (dist**2) / (2.0 * (k_i**2)))
        return e * m["conf"]

    # Scorer 3: M3 đơn + OKS
    def scorer_m3_oks(inst):
        if "rtmpose-m" not in inst["models"]:
            return 0.0
        m = inst["models"]["rtmpose-m"]
        hx, hy = inst["human_pt"]
        dist = math.hypot(hx - m["x"], hy - m["y"]) / inst["scale"]
        k_i = 2.0 * COCO_SIGMAS.get(inst["vf_id"], 0.070)
        e = 1.0 - math.exp(- (dist**2) / (2.0 * (k_i**2)))
        return e * m["conf"]

    # Scorer 4a: M0 + M3 (chỉ dùng e)
    def scorer_m0_m3_pure_e(inst):
        models = ["yolo26s-pose", "rtmpose-m"]
        active = [m for m in models if m in inst["models"]]
        if not active:
            return 0.0
        hx, hy = inst["human_pt"]
        scale = inst["scale"]
        k_i = 2.0 * COCO_SIGMAS.get(inst["vf_id"], 0.070)
        sum_w = sum(dev_weights[m] * inst["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["x"] for m in active) / sum_w
        py = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["y"] for m in active) / sum_w
        dist = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * inst["models"][m]["conf"] * ((inst["models"][m]["x"] - px)**2 + (inst["models"][m]["y"] - py)**2) for m in active) / sum_w
        delta = math.sqrt(max(0.0, disp)) / scale
        return 1.0 - math.exp(- (dist**2) / (2.0 * (k_i**2 + best_params["lambda"] * delta**2)))

    # Scorer 4b: M0 + M3 (dùng e * R^alpha)
    def scorer_m0_m3_with_r(inst):
        models = ["yolo26s-pose", "rtmpose-m"]
        active = [m for m in models if m in inst["models"]]
        if not active:
            return 0.0
        hx, hy = inst["human_pt"]
        scale = inst["scale"]
        k_i = 2.0 * COCO_SIGMAS.get(inst["vf_id"], 0.070)
        sum_w = sum(dev_weights[m] * inst["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["x"] for m in active) / sum_w
        py = sum(dev_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["y"] for m in active) / sum_w
        dist = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * inst["models"][m]["conf"] * ((inst["models"][m]["x"] - px)**2 + (inst["models"][m]["y"] - py)**2) for m in active) / sum_w
        delta = math.sqrt(max(0.0, disp)) / scale
        e = 1.0 - math.exp(- (dist**2) / (2.0 * (k_i**2 + best_params["lambda"] * delta**2)))
        c_bar = sum(dev_weights[m] * inst["models"][m]["conf"] for m in active) / sum(dev_weights[m] for m in active)
        r = c_bar * math.exp(- (delta**2) / (2.0 * (best_params["tau"]**2)))
        return e * (r ** best_params["alpha"])

    # Scorer 5a: M0 + M2 + M3 (chỉ dùng e)
    scorer_k3_pure_e = make_k3_scorer(best_params["lambda"], best_params["tau"], best_params["alpha"], use_r=False)
    # Scorer 5b: M0 + M2 + M3 (dùng e * R^alpha)
    scorer_k3_with_r = make_k3_scorer(best_params["lambda"], best_params["tau"], best_params["alpha"], use_r=True)

    configs_to_test = [
        ("M0 với x4.5 (Heuristic cũ)", scorer_m0_heur, 129.2),
        ("M0 + OKS (Chuẩn COCO)", scorer_m0_oks, 129.2),
        ("M3 đơn + OKS (RTMPose)", scorer_m3_oks, 370.7),
        ("M0 + M3 [chỉ dùng e]", scorer_m0_m3_pure_e, 129.2 + 370.7),
        ("M0 + M3 [e * R^alpha]", scorer_m0_m3_with_r, 129.2 + 370.7),
        ("M0 + M2 + M3 (K=3) [chỉ dùng e]", scorer_k3_pure_e, 129.2 + 253.9 + 370.7),
        ("M0 + M2 + M3 (K=3) [e * R^alpha]", scorer_k3_with_r, 129.2 + 253.9 + 370.7),
    ]

    benchmark_qa_results: list[dict[str, Any]] = []

    for name, func, lat in configs_to_test:
        auroc, pr_k, ece = evaluate_scorer(test_injected, func)
        res = {
            "config_name": name,
            "AUROC": auroc,
            "ECE": ece,
            "Latency_ms_per_img": round(lat, 1),
            "P@20": pr_k[20][0], "R@20": pr_k[20][1],
            "P@50": pr_k[50][0], "R@50": pr_k[50][1],
            "P@100": pr_k[100][0], "R@100": pr_k[100][1],
        }
        benchmark_qa_results.append(res)
        print(f"[{name}] AUROC: {auroc} | ECE: {ece} | P@20: {pr_k[20][0]} | P@50: {pr_k[50][0]} | Latency: {lat:.1f} ms")

    out_file = RESULTS_DIR / "error_injection_qa_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "tuned_hyperparameters_dev": best_params,
            "benchmark_qa_results": benchmark_qa_results
        }, f, indent=2)

    print(f"\n[OK] Đã hoàn thành benchmark phát hiện lỗi (Error Injection QA). File xuất: {out_file}")


if __name__ == "__main__":
    run_error_injection_benchmark()
