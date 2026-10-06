from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from scipy import stats

SANDBOX_DIR = Path(__file__).resolve().parent
DATA_DIR = SANDBOX_DIR / "data"
CACHE_DIR = SANDBOX_DIR / "cache"
RESULTS_DIR = SANDBOX_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# COCO 0..16 sang VinFast VF17 (1..17)
COCO_TO_VF17 = {
    0: 1, 1: 3, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 9, 8: 8,
    9: 11, 10: 10, 11: 13, 12: 12, 13: 15, 14: 14, 15: 17, 16: 16
}

VF17_NAMES = {
    1: "nose", 2: "r_eye", 3: "l_eye", 4: "r_ear", 5: "l_ear",
    6: "r_shoulder", 7: "l_shoulder", 8: "r_elbow", 9: "l_elbow",
    10: "r_wrist", 11: "l_wrist", 12: "r_hip", 13: "l_hip",
    14: "r_knee", 15: "l_knee", 16: "r_ankle", 17: "l_ankle"
}


def load_cached_predictions(model_names: list[str], image_ids: list[int]) -> dict[str, dict[int, Any]]:
    all_preds: dict[str, dict[int, Any]] = {m: {} for m in model_names}
    for m in model_names:
        m_dir = CACHE_DIR / m
        if not m_dir.exists():
            continue
        for img_id in image_ids:
            c_file = m_dir / f"{img_id}.json"
            if c_file.is_file():
                try:
                    with open(c_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_preds[m][img_id] = data.get("persons", [])
                except Exception:
                    pass
    return all_preds


def match_gt_to_predicted_person(
    gt_kpts_vf: dict[int, tuple[float, float]],
    predicted_persons: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not predicted_persons:
        return None
    if len(predicted_persons) == 1:
        return predicted_persons[0]

    gt_points = list(gt_kpts_vf.values())
    if not gt_points:
        return predicted_persons[0]

    gt_cx = sum(p[0] for p in gt_points) / len(gt_points)
    gt_cy = sum(p[1] for p in gt_points) / len(gt_points)

    best_match = None
    min_dist = float("inf")

    for p in predicted_persons:
        if p.get("bbox"):
            x1, y1, x2, y2 = p["bbox"]
            pcx = (x1 + x2) / 2
            pcy = (y1 + y2) / 2
        else:
            m_pts = [(kp["x"], kp["y"]) for kp in p.get("keypoints", {}).values() if kp.get("x") is not None]
            if not m_pts:
                continue
            pcx = sum(pt[0] for pt in m_pts) / len(m_pts)
            pcy = sum(pt[1] for pt in m_pts) / len(m_pts)

        dist = math.hypot(gt_cx - pcx, gt_cy - pcy)
        if dist < min_dist:
            min_dist = dist
            best_match = p

    return best_match


def extract_keypoint_instances(split_items: list[dict[str, Any]], all_preds: dict[str, dict[int, Any]], available_models: list[str]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for item in split_items:
        img_id = item["image_id"]
        annotations = item.get("annotations", [])

        for ann_idx, ann in enumerate(annotations):
            bbox = ann["bbox"]
            w, h = max(10.0, bbox[2]), max(10.0, bbox[3])
            scale = math.sqrt(w * h)

            raw_kpts = ann["keypoints"]
            gt_vf_kpts: dict[int, tuple[float, float]] = {}
            for coco_idx in range(17):
                x = raw_kpts[coco_idx * 3]
                y = raw_kpts[coco_idx * 3 + 1]
                v = raw_kpts[coco_idx * 3 + 2]
                if v == 2:  # Chỉ lấy keypoint có v=2
                    vf_id = COCO_TO_VF17[coco_idx]
                    gt_vf_kpts[vf_id] = (float(x), float(y))

            if not gt_vf_kpts:
                continue

            model_matched: dict[str, dict[str, Any] | None] = {}
            for m in available_models:
                preds_for_img = all_preds[m].get(img_id, [])
                match = match_gt_to_predicted_person(gt_vf_kpts, preds_for_img)
                model_matched[m] = match

            for vf_id, (gx, gy) in gt_vf_kpts.items():
                inst = {
                    "image_id": img_id,
                    "ann_idx": ann_idx,
                    "vf_id": vf_id,
                    "vf_name": VF17_NAMES[vf_id],
                    "scale": scale,
                    "gt": (gx, gy),
                    "models": {},
                }
                for m in available_models:
                    matched_person = model_matched[m]
                    if matched_person:
                        m_kpts = matched_person.get("keypoints", {})
                        pt = m_kpts.get(vf_id) or m_kpts.get(str(vf_id))
                        if pt and pt.get("x") is not None and pt.get("y") is not None:
                            mx, my = float(pt["x"]), float(pt["y"])
                            conf = float(pt.get("conf", 1.0))
                            err = math.hypot(mx - gx, my - gy) / scale
                            inst["models"][m] = {
                                "x": mx,
                                "y": my,
                                "conf": conf,
                                "err": err,
                            }
                instances.append(inst)
    return instances


def compute_mpe_for_instances(inst_list: list[dict[str, Any]], model_name: str) -> float:
    errs = [inst["models"][model_name]["err"] for inst in inst_list if model_name in inst["models"]]
    return float(np.median(errs)) if errs else float("nan")


def compute_combo_error_for_inst(inst: dict[str, Any], combo: list[str], weights: dict[str, float]) -> float:
    active_m = [m for m in combo if m in inst["models"]]
    if not active_m:
        return float("nan")
    gx, gy = inst["gt"]
    scale = inst["scale"]
    sum_w = sum(weights[m] * inst["models"][m]["conf"] for m in active_m)
    if sum_w <= 0:
        sum_w = 1.0
    px = sum(weights[m] * inst["models"][m]["conf"] * inst["models"][m]["x"] for m in active_m) / sum_w
    py = sum(weights[m] * inst["models"][m]["conf"] * inst["models"][m]["y"] for m in active_m) / sum_w
    return math.hypot(px - gx, py - gy) / scale


def run_full_benchmark():
    manifest_file = DATA_DIR / "split_manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_items = manifest["dev"]
    test_items = manifest["test"]
    available_models = ["yolo26s-pose", "yolov8n-pose", "yolov8m-pose", "rtmpose-m"]

    # 1. Đọc dữ liệu Dev set & Fit w_k = 1 / MPE_k^2
    all_dev_ids = [it["image_id"] for it in dev_items]
    dev_preds = load_cached_predictions(available_models, all_dev_ids)
    dev_instances = extract_keypoint_instances(dev_items, dev_preds, available_models)

    dev_mpe: dict[str, float] = {}
    mpe_weights: dict[str, float] = {}
    print("=== [BƯỚC 1: FIT TRỌNG SỐ W_K TRÊN TẬP DEV] ===")
    for m in available_models:
        mpe_val = compute_mpe_for_instances(dev_instances, m)
        dev_mpe[m] = round(mpe_val, 4)
        mpe_weights[m] = 1.0 / (mpe_val ** 2) if mpe_val > 0 else 1.0
        print(f"    - Dev MPE[{m}]: {dev_mpe[m]} -> Trọng số w_k (1/MPE^2): {mpe_weights[m]:.1f}")

    # Chuẩn hóa trọng số tương đối (cho w_M0 = 1.0)
    w_base = mpe_weights["yolo26s-pose"]
    norm_weights = {m: round(mpe_weights[m] / w_base, 3) for m in available_models}
    print(f"    - Normalized Weights (relative to M0): {norm_weights}")

    # 2. Đọc dữ liệu Test set (MỌI BÁO CÁO CHÍNH THỨC SẼ NẰM TRÊN TẬP NÀY)
    all_test_ids = [it["image_id"] for it in test_items]
    test_preds = load_cached_predictions(available_models, all_test_ids)
    test_instances = extract_keypoint_instances(test_items, test_preds, available_models)
    print(f"\n=== [BƯỚC 2: ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ({len(test_items)} ảnh)] ===")
    print(f"    - Tổng số keypoints hợp lệ (v=2) trên Test: {len(test_instances)}")

    # Gom instances theo image_id phục vụ Cluster-Bootstrap
    instances_by_img: dict[int, list[dict[str, Any]]] = {}
    for inst in test_instances:
        instances_by_img.setdefault(inst["image_id"], []).append(inst)

    unique_test_img_ids = list(instances_by_img.keys())
    n_images = len(unique_test_img_ids)

    # 3. Hiệu năng đơn lẻ trên TEST kèm 95% Cluster-Bootstrap CI
    test_single_models: dict[str, Any] = {}
    for m in available_models:
        all_errs = [inst["models"][m]["err"] for inst in test_instances if m in inst["models"]]
        mpe_val = float(np.median(all_errs))
        
        # Per-joint MPE trên test
        per_joint = {}
        for vf_id in range(1, 18):
            j_errs = [inst["models"][m]["err"] for inst in test_instances if m in inst["models"] and inst["vf_id"] == vf_id]
            per_joint[VF17_NAMES[vf_id]] = round(float(np.median(j_errs)), 4) if j_errs else 0.0

        test_single_models[m] = {
            "MPE_test": round(mpe_val, 4),
            "Mean_test": round(float(np.mean(all_errs)), 4),
            "Per_Joint_MPE": per_joint,
        }

    # 4. Cluster-Bootstrap theo Ảnh (1000 resamples) cho Hiệu số cặp (Paired Differences)
    print("\n[*] Đang chạy Cluster-Bootstrap theo Image ID (1000 resamples) cho các cặp đối đầu...")
    rng = np.random.default_rng(42)
    B = 1000

    boot_single_mpe: dict[str, list[float]] = {m: [] for m in available_models}
    boot_combo_mpe: dict[str, list[float]] = {
        "M0+M1_w": [],
        "M0+M3_w": [],
        "M0+M2+M3_w": [],
    }

    # Lưu paired differences
    paired_diffs = {
        "Delta_M0_minus_M2": [],
        "Delta_M0_minus_M3": [],
        "Delta_M3_minus_M0M3": [],
        "Delta_M0M3_minus_M0M2M3": [],
        "Delta_M0_minus_M0M1w": [],
    }

    combos_def = {
        "M0+M1_w": ["yolo26s-pose", "yolov8n-pose"],
        "M0+M3_w": ["yolo26s-pose", "rtmpose-m"],
        "M0+M2+M3_w": ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"],
    }

    for _ in range(B):
        # Sample cluster = image_id with replacement
        sampled_img_ids = rng.choice(unique_test_img_ids, size=n_images, replace=True)
        sample_insts = []
        for i_id in sampled_img_ids:
            sample_insts.extend(instances_by_img[i_id])

        # Tính MPE từng model đơn lẻ trên mẫu
        cur_mpe: dict[str, float] = {}
        for m in available_models:
            errs = [inst["models"][m]["err"] for inst in sample_insts if m in inst["models"]]
            m_val = float(np.median(errs)) if errs else float("nan")
            cur_mpe[m] = m_val
            boot_single_mpe[m].append(m_val)

        # Tính MPE các combo
        cur_combo: dict[str, float] = {}
        for c_key, c_models in combos_def.items():
            c_errs = [compute_combo_error_for_inst(inst, c_models, mpe_weights) for inst in sample_insts]
            c_errs = [e for e in c_errs if not math.isnan(e)]
            c_val = float(np.median(c_errs)) if c_errs else float("nan")
            cur_combo[c_key] = c_val
            boot_combo_mpe[c_key].append(c_val)

        # Paired differences trên cùng mẫu bootstrap:
        paired_diffs["Delta_M0_minus_M2"].append(cur_mpe["yolo26s-pose"] - cur_mpe["yolov8m-pose"])
        paired_diffs["Delta_M0_minus_M3"].append(cur_mpe["yolo26s-pose"] - cur_mpe["rtmpose-m"])
        paired_diffs["Delta_M3_minus_M0M3"].append(cur_mpe["rtmpose-m"] - cur_combo["M0+M3_w"])
        paired_diffs["Delta_M0M3_minus_M0M2M3"].append(cur_combo["M0+M3_w"] - cur_combo["M0+M2+M3_w"])
        paired_diffs["Delta_M0_minus_M0M1w"].append(cur_mpe["yolo26s-pose"] - cur_combo["M0+M1_w"])

    # Tính 95% Cluster CI cho từng model và combo
    for m in available_models:
        low = float(np.percentile(boot_single_mpe[m], 2.5))
        high = float(np.percentile(boot_single_mpe[m], 97.5))
        test_single_models[m]["Cluster_95_CI"] = [round(low, 4), round(high, 4)]

    # Tổng kết Paired Differences
    paired_summary: dict[str, Any] = {}
    for pair_key, diff_vals in paired_diffs.items():
        low = float(np.percentile(diff_vals, 2.5))
        high = float(np.percentile(diff_vals, 97.5))
        med = float(np.median(diff_vals))
        p_val = 2.0 * min(np.mean(np.array(diff_vals) > 0), np.mean(np.array(diff_vals) < 0))
        paired_summary[pair_key] = {
            "median_diff": round(med, 5),
            "ci_95": [round(low, 5), round(high, 5)],
            "significant": not (low <= 0.0 <= high),
            "p_value_approx": round(p_val, 4),
        }

    # 5. Chi tiết hóa ma trận tương quan sai số J_ab trên TEST
    print("[*] Đang phân tích ma trận tương quan sai số J_ab (Counts, Marginals, Cond Prob, Cluster CI)...")
    thresholds = [0.05, 0.10, 0.15, 0.20]
    j_analysis: dict[str, Any] = {}

    for t in thresholds:
        j_analysis[str(t)] = []
        model_pairs = [
            ("yolo26s-pose", "yolov8m-pose"),
            ("yolo26s-pose", "rtmpose-m"),
            ("yolo26s-pose", "yolov8n-pose"),
            ("yolov8m-pose", "rtmpose-m"),
        ]

        for m_a, m_b in model_pairs:
            common = [inst for inst in test_instances if m_a in inst["models"] and m_b in inst["models"]]
            n_tot = len(common)
            e_a = np.array([inst["models"][m_a]["err"] > t for inst in common])
            e_b = np.array([inst["models"][m_b]["err"] > t for inst in common])

            n_a = int(np.sum(e_a))
            n_b = int(np.sum(e_b))
            n_ab = int(np.sum(e_a & e_b))

            p_a = n_a / n_tot if n_tot > 0 else 0.0
            p_b = n_b / n_tot if n_tot > 0 else 0.0
            p_ab = n_ab / n_tot if n_tot > 0 else 0.0

            p_b_given_a = n_ab / n_a if n_a > 0 else 0.0
            j_val = (p_ab / (p_a * p_b)) if (p_a * p_b) > 1e-6 else 1.0

            # Cluster bootstrap CI cho J_ab
            boot_j = []
            for _ in range(500):
                sampled_i_ids = rng.choice(unique_test_img_ids, size=n_images, replace=True)
                s_insts = []
                for i_id in sampled_i_ids:
                    s_insts.extend(instances_by_img[i_id])
                s_common = [inst for inst in s_insts if m_a in inst["models"] and m_b in inst["models"]]
                if not s_common:
                    continue
                s_ea = np.array([inst["models"][m_a]["err"] > t for inst in s_common])
                s_eb = np.array([inst["models"][m_b]["err"] > t for inst in s_common])
                spa = np.mean(s_ea)
                spb = np.mean(s_eb)
                spab = np.mean(s_ea & s_eb)
                if spa * spb > 1e-6:
                    boot_j.append(spab / (spa * spb))

            j_low = float(np.percentile(boot_j, 2.5)) if boot_j else j_val
            j_high = float(np.percentile(boot_j, 97.5)) if boot_j else j_val

            j_analysis[str(t)].append({
                "pair": f"{m_a} & {m_b}",
                "N_total": n_tot,
                "N_a": n_a,
                "P_a": round(p_a, 4),
                "N_b": n_b,
                "P_b": round(p_b, 4),
                "N_ab": n_ab,
                "P_ab": round(p_ab, 4),
                "P_b_given_a": round(p_b_given_a, 4),
                "J_ab": round(j_val, 3),
                "J_ab_95_CI": [round(j_low, 3), round(j_high, 3)],
            })

    # 6. Đánh giá Ensemble Combinations trên TEST
    combo_results: list[dict[str, Any]] = []
    combo_test_list = [
        ("M0 đơn", ["yolo26s-pose"]),
        ("M0 + M1 (Equal)", ["yolo26s-pose", "yolov8n-pose"]),
        ("M0 + M1 (Inverse w)", ["yolo26s-pose", "yolov8n-pose"]),
        ("M0 + M2 (Inverse w)", ["yolo26s-pose", "yolov8m-pose"]),
        ("M0 + M3 (Inverse w)", ["yolo26s-pose", "rtmpose-m"]),
        ("M0 + M2 + M3 (Inverse w)", ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"]),
    ]

    for c_label, c_models in combo_test_list:
        use_eq = "Equal" in c_label
        c_weights = {m: 1.0 for m in c_models} if use_eq else mpe_weights

        errs = []
        disags = []
        true_errs = []
        for inst in test_instances:
            active_m = [m for m in c_models if m in inst["models"]]
            if not active_m:
                continue
            err_val = compute_combo_error_for_inst(inst, c_models, c_weights)
            errs.append(err_val)

            if len(active_m) >= 2:
                # Tính disagreement delta
                gx, gy = inst["gt"]
                scale = inst["scale"]
                sum_w = sum(c_weights[m] * inst["models"][m]["conf"] for m in active_m)
                px = sum(c_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["x"] for m in active_m) / sum_w
                py = sum(c_weights[m] * inst["models"][m]["conf"] * inst["models"][m]["y"] for m in active_m) / sum_w
                disp = sum(
                    c_weights[m] * inst["models"][m]["conf"] * ((inst["models"][m]["x"] - px)**2 + (inst["models"][m]["y"] - py)**2)
                    for m in active_m
                ) / sum_w
                delta_i = math.sqrt(max(0.0, disp)) / scale
                disags.append(delta_i)
                true_errs.append(err_val)

        mpe_c = float(np.median(errs))
        sp_corr = 0.0
        p_val = 1.0
        if len(disags) > 10:
            sp = stats.spearmanr(disags, true_errs)
            sp_corr = float(sp.statistic)
            p_val = float(sp.pvalue)

        combo_results.append({
            "name": c_label,
            "models": c_models,
            "K": len(c_models),
            "weights_type": "equal" if use_eq else "inverse_mpe2",
            "MPE_test": round(mpe_c, 4),
            "Spearman_corr": round(sp_corr, 4),
            "Spearman_p": round(p_val, 6),
        })

    # Lưu kết quả đầy đủ
    report_data = {
        "dataset": "COCO val2017 (500 filtered images, 250 dev / 250 test, seed 42)",
        "dev_weights": {
            "dev_MPE": dev_mpe,
            "raw_w_k": {m: round(mpe_weights[m], 1) for m in available_models},
            "norm_w_k": norm_weights,
        },
        "test_single_models": test_single_models,
        "paired_differences": paired_summary,
        "J_analysis": j_analysis,
        "combo_results": combo_results,
    }

    out_file = RESULTS_DIR / "rigorous_benchmark_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Đã hoàn thành đo đạc chuẩn mực trên TEST set. File xuất: {out_file}")
    return report_data


if __name__ == "__main__":
    run_full_benchmark()
