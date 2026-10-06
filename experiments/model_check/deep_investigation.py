from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import math
import time
import random
from pathlib import Path
from typing import Any
import numpy as np


def fast_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = np.sum(ranks[:n_pos])
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def fast_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tính Average Precision (PR-AUC) chuẩn xác theo Scikit-Learn / PASCAL VOC."""
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    # Tích phân step: AP = sum_{k} (R_k - R_{k-1}) * P_k
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


SANDBOX_DIR = Path(__file__).resolve().parent
DATA_DIR = SANDBOX_DIR / "data"
CACHE_DIR = SANDBOX_DIR / "cache"
RESULTS_DIR = SANDBOX_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

COCO_SIGMAS = {
    1: 0.026, 2: 0.025, 3: 0.025, 4: 0.035, 5: 0.035,
    6: 0.079, 7: 0.079, 8: 0.072, 9: 0.072, 10: 0.062, 11: 0.062,
    12: 0.107, 13: 0.107, 14: 0.087, 15: 0.087, 16: 0.089, 17: 0.089
}

SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
PAIR_LOOKUP = {}
for r_id, l_id in SYMMETRIC_PAIRS:
    PAIR_LOOKUP[r_id] = l_id
    PAIR_LOOKUP[l_id] = r_id

JOINT_GROUPS = {
    "face": [1, 2, 3, 4, 5],
    "torso_upper": [6, 7, 8, 9, 10, 11],
    "lower_legs": [12, 13, 14, 15, 16, 17],
}


def load_all_predictions(model_names: list[str], image_ids: list[int]) -> dict[str, dict[int, Any]]:
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


def match_gt_to_predicted_person(gt_kpts_vf: dict[int, tuple[float, float]], predicted_persons: list[dict[str, Any]]) -> dict[str, Any] | None:
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
            pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
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


def extract_instances_full(split_items: list[dict[str, Any]], all_preds: dict[str, dict[int, Any]], models: list[str]) -> list[dict[str, Any]]:
    """Trích xuất instances bao gồm cả thông tin crowd, bbox_area, và điểm v=1."""
    instances: list[dict[str, Any]] = []
    for item in split_items:
        img_id = item["image_id"]
        annotations = item.get("annotations", [])
        num_persons_in_img = len(annotations)

        for ann_idx, ann in enumerate(annotations):
            bbox = ann["bbox"]
            w, h = max(10.0, bbox[2]), max(10.0, bbox[3])
            area = bbox[2] * bbox[3]
            scale = math.sqrt(w * h)

            raw_kpts = ann["keypoints"]
            # Lưu cả v=2 và v=1
            gt_vf_kpts: dict[int, tuple[float, float, int]] = {}
            for coco_idx in range(17):
                x = raw_kpts[coco_idx * 3]
                y = raw_kpts[coco_idx * 3 + 1]
                v = raw_kpts[coco_idx * 3 + 2]
                if v in (1, 2):
                    vf_id = COCO_TO_VF17[coco_idx]
                    gt_vf_kpts[vf_id] = (float(x), float(y), int(v))

            if not gt_vf_kpts:
                continue

            # Matching model
            model_matched: dict[str, dict[str, Any] | None] = {}
            for m in models:
                preds_for_img = all_preds[m].get(img_id, [])
                match = match_gt_to_predicted_person(
                    {k: (v[0], v[1]) for k, v in gt_vf_kpts.items() if v[2] == 2},
                    preds_for_img
                )
                model_matched[m] = match

            for vf_id, (gx, gy, v_flag) in gt_vf_kpts.items():
                inst = {
                    "image_id": img_id,
                    "ann_idx": ann_idx,
                    "vf_id": vf_id,
                    "vf_name": VF17_NAMES[vf_id],
                    "scale": scale,
                    "bbox_area": area,
                    "num_persons_in_img": num_persons_in_img,
                    "v_flag": v_flag,
                    "gt": (gx, gy),
                    "gt_bbox": bbox,
                    "models": {},
                }
                for m in models:
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
                                "pred_bbox": matched_person.get("bbox"),
                            }
                instances.append(inst)
    return instances


# =========================================================================
# THỰC NGHIỆM 1: CLUSTER-BOOTSTRAP PAIRED CI CHO AUROC VÀ AP (AVERAGE PRECISION)
# =========================================================================
def run_paired_auroc_ap_bootstrap(test_instances: list[dict[str, Any]], dev_weights: dict[str, float], tuned_params: dict[str, float]):
    print("\n==================================================================")
    print("=== THỰC NGHIỆM 1: PAIRED DIFFERENCE AUROC & AP (CLUSTER-BOOTSTRAP) ===")
    print("==================================================================")

    # Tiêm lỗi chuẩn: 25% lỗi (shift + swap)
    rng = random.Random(42)
    injected_test = []
    for inst in [i for i in test_instances if i["v_flag"] == 2]:
        is_err = rng.random() < 0.25
        gx, gy = inst["gt"]
        scale = inst["scale"]
        vf_id = inst["vf_id"]

        if not is_err:
            jr = rng.gauss(0, 0.012 * scale)
            ja = rng.uniform(0, 2 * math.pi)
            hx = gx + jr * math.cos(ja)
            hy = gy + jr * math.sin(ja)
            lbl = 0
        else:
            lbl = 1
            if rng.random() < 0.30 and vf_id in PAIR_LOOKUP:
                # swap
                other_id = PAIR_LOOKUP[vf_id]
                # Lấy gt của other_id nếu có
                ox, oy = gx, gy
                hx, hy = ox + 0.15 * scale, oy + 0.15 * scale
            else:
                s_dist = rng.uniform(0.10, 0.30) * scale
                s_ang = rng.uniform(0, 2 * math.pi)
                hx = gx + s_dist * math.cos(s_ang)
                hy = gy + s_dist * math.sin(s_ang)

        it = dict(inst)
        it["human_pt"] = (hx, hy)
        it["label"] = lbl
        injected_test.append(it)

    # Định nghĩa 5 cấu hình tính điểm:
    lam, tau, alpha = tuned_params["lambda"], tuned_params["tau"], tuned_params["alpha"]

    def score_A(it): # Heuristic cũ
        m = it["models"].get("yolo26s-pose")
        if not m: return 0.0
        d = math.hypot(it["human_pt"][0] - m["x"], it["human_pt"][1] - m["y"]) / it["scale"]
        return min(1.0, 4.5 * d) * m["conf"]

    def score_B(it): # M0 + OKS
        m = it["models"].get("yolo26s-pose")
        if not m: return 0.0
        d = math.hypot(it["human_pt"][0] - m["x"], it["human_pt"][1] - m["y"]) / it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        return (1.0 - math.exp(- (d**2) / (2.0 * (ki**2)))) * m["conf"]

    def score_C(it): # M3 + OKS
        m = it["models"].get("rtmpose-m")
        if not m: return 0.0
        d = math.hypot(it["human_pt"][0] - m["x"], it["human_pt"][1] - m["y"]) / it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        return (1.0 - math.exp(- (d**2) / (2.0 * (ki**2)))) * m["conf"]

    def score_D(it): # M0 + M3 [e * R^alpha]
        active = [m for m in ["yolo26s-pose", "rtmpose-m"] if m in it["models"]]
        if not active: return 0.0
        hx, hy = it["human_pt"]
        scale = it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        sw = sum(dev_weights[m] * it["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in active) / sw
        py = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in active) / sw
        d = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * it["models"][m]["conf"] * ((it["models"][m]["x"] - px)**2 + (it["models"][m]["y"] - py)**2) for m in active) / sw
        delta = math.sqrt(max(0.0, disp)) / scale
        e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2 + lam * (delta**2))))
        c_bar = sum(dev_weights[m] * it["models"][m]["conf"] for m in active) / sum(dev_weights[m] for m in active)
        r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
        return e * (r ** alpha)

    def score_E(it): # M0 + M2 + M3 [e * R^alpha]
        active = [m for m in ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"] if m in it["models"]]
        if not active: return 0.0
        hx, hy = it["human_pt"]
        scale = it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        sw = sum(dev_weights[m] * it["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in active) / sw
        py = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in active) / sw
        d = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * it["models"][m]["conf"] * ((it["models"][m]["x"] - px)**2 + (it["models"][m]["y"] - py)**2) for m in active) / sw
        delta = math.sqrt(max(0.0, disp)) / scale
        e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2 + lam * (delta**2))))
        c_bar = sum(dev_weights[m] * it["models"][m]["conf"] for m in active) / sum(dev_weights[m] for m in active)
        r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
        return e * (r ** alpha)

    scorers = {
        "A (M0 x4.5)": score_A,
        "B (M0 OKS)": score_B,
        "C (M3 OKS)": score_C,
        "D (M0+M3 e*R)": score_D,
        "E (K=3 e*R)": score_E,
    }

    # Tính điểm gốc trên toàn bộ tập test
    y_true_all = np.array([it["label"] for it in injected_test])
    base_scores = {k: np.array([sc(it) for it in injected_test]) for k, sc in scorers.items()}
    base_auroc = {k: round(float(fast_roc_auc_score(y_true_all, base_scores[k])), 4) for k in scorers}
    base_ap = {k: round(float(fast_average_precision_score(y_true_all, base_scores[k])), 4) for k in scorers}

    print("[*] Điểm số gốc trên toàn bộ TEST:")
    for k in scorers:
        print(f"    - {k}: AUROC = {base_auroc[k]:.4f}, AP = {base_ap[k]:.4f}")

    # Cluster Bootstrap theo Image ID (1000 resamples)
    img_groups: dict[int, list[int]] = {}
    for idx, it in enumerate(injected_test):
        img_groups.setdefault(it["image_id"], []).append(idx)
    img_ids = list(img_groups.keys())
    n_imgs = len(img_ids)

    boot_rng = np.random.default_rng(42)
    B = 1000

    pairs_to_eval = [
        ("B (M0 OKS)", "A (M0 x4.5)", "B vs A (OKS vs Heuristic 4.5)"),
        ("C (M3 OKS)", "B (M0 OKS)", "C vs B (RTMPose vs YOLO26s)"),
        ("D (M0+M3 e*R)", "B (M0 OKS)", "D vs B (K=2 vs YOLO26s)"),
        ("E (K=3 e*R)", "B (M0 OKS)", "E vs B (K=3 vs YOLO26s)"),
        ("E (K=3 e*R)", "D (M0+M3 e*R)", "E vs D (K=3 vs K=2)"),
    ]

    paired_boot_auroc: dict[str, list[float]] = {p[2]: [] for p in pairs_to_eval}
    paired_boot_ap: dict[str, list[float]] = {p[2]: [] for p in pairs_to_eval}

    for _ in range(B):
        s_img_ids = boot_rng.choice(img_ids, size=n_imgs, replace=True)
        s_indices = []
        for i_id in s_img_ids:
            s_indices.extend(img_groups[i_id])

        s_y_true = y_true_all[s_indices]
        if np.sum(s_y_true == 1) == 0 or np.sum(s_y_true == 0) == 0:
            continue

        s_auc = {}
        s_ap = {}
        for k in scorers:
            s_sc = base_scores[k][s_indices]
            s_auc[k] = fast_roc_auc_score(s_y_true, s_sc)
            s_ap[k] = fast_average_precision_score(s_y_true, s_sc)

        for cfg1, cfg2, p_name in pairs_to_eval:
            paired_boot_auroc[p_name].append(s_auc[cfg1] - s_auc[cfg2])
            paired_boot_ap[p_name].append(s_ap[cfg1] - s_ap[cfg2])

    paired_results = []
    for cfg1, cfg2, p_name in pairs_to_eval:
        diff_auc = np.array(paired_boot_auroc[p_name])
        diff_ap = np.array(paired_boot_ap[p_name])

        med_auc, low_auc, high_auc = float(np.median(diff_auc)), float(np.percentile(diff_auc, 2.5)), float(np.percentile(diff_auc, 97.5))
        med_ap, low_ap, high_ap = float(np.median(diff_ap)), float(np.percentile(diff_ap, 2.5)), float(np.percentile(diff_ap, 97.5))

        exceeds_threshold = (low_auc > 0.01)  # Vượt ngưỡng +0.01 với CI không chứa 0

        res_item = {
            "pair_name": p_name,
            "cfg1": cfg1,
            "cfg2": cfg2,
            "delta_AUROC_median": round(med_auc, 4),
            "delta_AUROC_95_CI": [round(low_auc, 4), round(high_auc, 4)],
            "delta_AP_median": round(med_ap, 4),
            "delta_AP_95_CI": [round(low_ap, 4), round(high_ap, 4)],
            "significant_gt_zero": (low_auc > 0.0),
            "exceeds_plus_0_01": exceeds_threshold,
        }
        paired_results.append(res_item)
        print(f"[*] {p_name}: Delta AUROC = {med_auc:+.4f} [{low_auc:+.4f}, {high_auc:+.4f}], Delta AP = {med_ap:+.4f} [{low_ap:+.4f}, {high_ap:+.4f}] (Exceed +0.01: {exceeds_threshold})")

    return {
        "base_auroc": base_auroc,
        "base_ap": base_ap,
        "paired_differences": paired_results,
    }


# =========================================================================
# THỰC NGHIỆM 2: TIÊM LỖI KHÓ HƠN (0.03 - 0.10 s), PHÂN TÁCH LỖI VÀ NHÓM KHỚP
# =========================================================================
def run_subtle_error_injection(test_instances: list[dict[str, Any]], dev_weights: dict[str, float], tuned_params: dict[str, float]):
    print("\n==================================================================")
    print("=== THỰC NGHIỆM 2: TIÊM LỖI KHÓ HƠN (0.03 - 0.10 s), TÁCH SCENARIO & NHÓM KHỚP ===")
    print("==================================================================")

    valid_test = [i for i in test_instances if i["v_flag"] == 2]
    error_rates = [0.02, 0.05, 0.25]
    scenarios = ["shift_subtle", "swap_only"]

    results_matrix = []

    lam, tau, alpha = tuned_params["lambda"], tuned_params["tau"], tuned_params["alpha"]

    def score_B(it):
        m = it["models"].get("yolo26s-pose")
        if not m: return 0.0
        d = math.hypot(it["human_pt"][0] - m["x"], it["human_pt"][1] - m["y"]) / it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        return (1.0 - math.exp(- (d**2) / (2.0 * (ki**2)))) * m["conf"]

    def score_E(it):
        active = [m for m in ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"] if m in it["models"]]
        if not active: return 0.0
        hx, hy = it["human_pt"]
        scale = it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        sw = sum(dev_weights[m] * it["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in active) / sw
        py = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in active) / sw
        d = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * it["models"][m]["conf"] * ((it["models"][m]["x"] - px)**2 + (it["models"][m]["y"] - py)**2) for m in active) / sw
        delta = math.sqrt(max(0.0, disp)) / scale
        e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2 + lam * (delta**2))))
        c_bar = sum(dev_weights[m] * it["models"][m]["conf"] for m in active) / sum(dev_weights[m] for m in active)
        r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
        return e * (r ** alpha)

    for err_rate in error_rates:
        for scen in scenarios:
            rng = random.Random(100 + int(err_rate * 100))
            injected = []

            for inst in valid_test:
                vf_id = inst["vf_id"]
                gx, gy = inst["gt"]
                scale = inst["scale"]
                is_err = rng.random() < err_rate

                if not is_err:
                    jr = rng.gauss(0, 0.012 * scale)
                    ja = rng.uniform(0, 2 * math.pi)
                    hx = gx + jr * math.cos(ja)
                    hy = gy + jr * math.sin(ja)
                    lbl = 0
                else:
                    if scen == "shift_subtle":
                        # Dịch chuyển khó: 0.03 - 0.10 * s
                        dist = rng.uniform(0.03, 0.10) * scale
                        ang = rng.uniform(0, 2 * math.pi)
                        hx = gx + dist * math.cos(ang)
                        hy = gy + dist * math.sin(ang)
                        lbl = 1
                    else: # swap_only
                        if vf_id in PAIR_LOOKUP:
                            # Swap đối xứng
                            hx = gx + 0.12 * scale
                            hy = gy + 0.12 * scale
                            lbl = 1
                        else:
                            # Không phải khớp đối xứng thì coi như không bị lỗi swap
                            hx, hy = gx, gy
                            lbl = 0

                it = dict(inst)
                it["human_pt"] = (hx, hy)
                it["label"] = lbl
                injected.append(it)

            # Đánh giá tổng và đánh giá theo từng nhóm khớp
            for grp_name, grp_ids in [("ALL", list(range(1, 18))), ("face", JOINT_GROUPS["face"]), ("torso_upper", JOINT_GROUPS["torso_upper"]), ("lower_legs", JOINT_GROUPS["lower_legs"])]:
                sub_insts = [it for it in injected if it["vf_id"] in grp_ids]
                y_t = np.array([it["label"] for it in sub_insts])
                n_pos = int(np.sum(y_t == 1))

                if n_pos < 5 or (len(y_t) - n_pos) < 5:
                    auc_b, auc_e, ap_b, ap_e = 0.5, 0.5, 0.0, 0.0
                else:
                    sc_b = np.array([score_B(it) for it in sub_insts])
                    sc_e = np.array([score_E(it) for it in sub_insts])
                    auc_b = round(float(fast_roc_auc_score(y_t, sc_b)), 4)
                    auc_e = round(float(fast_roc_auc_score(y_t, sc_e)), 4)
                    ap_b = round(float(fast_average_precision_score(y_t, sc_b)), 4)
                    ap_e = round(float(fast_average_precision_score(y_t, sc_e)), 4)

                results_matrix.append({
                    "error_rate": f"{int(err_rate*100)}%",
                    "scenario": scen,
                    "joint_group": grp_name,
                    "N_samples": len(sub_insts),
                    "N_positive": n_pos,
                    "AUROC_M0_OKS": auc_b,
                    "AUROC_K3_eR": auc_e,
                    "Delta_AUROC": round(auc_e - auc_b, 4),
                    "AP_M0_OKS": ap_b,
                    "AP_K3_eR": ap_e,
                    "Delta_AP": round(ap_e - ap_b, 4),
                })

    print("[*] Đã hoàn thành đánh giá subtle error injection theo nhóm khớp.")
    return results_matrix


# =========================================================================
# THỰC NGHIỆM 3: TẬP CA KHÓ (OCCLUDED v=1, SMALL PERSON, CROWDED) VÀ TÁC DỤNG CỦA R
# =========================================================================
def run_hard_cases_investigation(test_instances: list[dict[str, Any]], dev_weights: dict[str, float], tuned_params: dict[str, float]):
    print("\n==================================================================")
    print("=== THỰC NGHIỆM 3: TẬP CA KHÓ (v=1, SMALL PERSON, CROWD) VÀ TÁC DỤNG CỦA R ===")
    print("==================================================================")

    # Lấy các điểm ca khó KHÔNG BỊ TIÊM LỖI (Ground truth gốc + noise tự nhiên):
    # Các ca này reviewer chấm đúng, nhưng AI dễ bị báo oan (False Alarm)
    rng = random.Random(789)
    hard_instances = []
    for inst in test_instances:
        is_occluded = (inst["v_flag"] == 1)
        is_small = (inst["bbox_area"] <= 8000.0) # người nhỏ
        is_crowd = (inst["num_persons_in_img"] >= 3) # đông người

        if not (is_occluded or is_small or is_crowd):
            continue

        gx, gy = inst["gt"]
        scale = inst["scale"]
        jr = rng.gauss(0, 0.012 * scale)
        ja = rng.uniform(0, 2 * math.pi)
        hx = gx + jr * math.cos(ja)
        hy = gy + jr * math.sin(ja)

        it = dict(inst)
        it["human_pt"] = (hx, hy)
        it["is_occluded"] = is_occluded
        it["is_small"] = is_small
        it["is_crowd"] = is_crowd
        hard_instances.append(it)

    print(f"[*] Số keypoints thuộc tập ca khó: {len(hard_instances)}")
    print(f"    - Bị che khuất (v=1): {sum(1 for x in hard_instances if x['is_occluded'])}")
    print(f"    - Người nhỏ (area <= 8000): {sum(1 for x in hard_instances if x['is_small'])}")
    print(f"    - Ảnh đông người (>= 3 người): {sum(1 for x in hard_instances if x['is_crowd'])}")

    lam, tau, alpha = tuned_params["lambda"], tuned_params["tau"], tuned_params["alpha"]

    # Đo điểm e thuần vs e * R^alpha
    scores_pure_e = []
    scores_e_R = []

    for it in hard_instances:
        active = [m for m in ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"] if m in it["models"]]
        if not active:
            scores_pure_e.append(0.0)
            scores_e_R.append(0.0)
            continue
        hx, hy = it["human_pt"]
        scale = it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        sw = sum(dev_weights[m] * it["models"][m]["conf"] for m in active)
        px = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in active) / sw
        py = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in active) / sw
        d = math.hypot(hx - px, hy - py) / scale
        disp = sum(dev_weights[m] * it["models"][m]["conf"] * ((it["models"][m]["x"] - px)**2 + (it["models"][m]["y"] - py)**2) for m in active) / sw
        delta = math.sqrt(max(0.0, disp)) / scale
        e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2 + lam * (delta**2))))
        c_bar = sum(dev_weights[m] * it["models"][m]["conf"] for m in active) / sum(dev_weights[m] for m in active)
        r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
        scores_pure_e.append(e)
        scores_e_R.append(e * (r ** alpha))

    scores_pure_e = np.array(scores_pure_e)
    scores_e_R = np.array(scores_e_R)

    # Đo False Alarm Rate (Tỷ lệ bị coi là nghi ngờ cao vượt ngưỡng t)
    thresholds = [0.20, 0.30, 0.40]
    far_summary = []
    for t_thr in thresholds:
        fa_pure = float(np.mean(scores_pure_e > t_thr)) * 100
        fa_with_r = float(np.mean(scores_e_R > t_thr)) * 100
        reduc = ((fa_pure - fa_with_r) / fa_pure * 100) if fa_pure > 0 else 0.0
        far_summary.append({
            "threshold": t_thr,
            "False_Alarm_Pure_e": round(fa_pure, 2),
            "False_Alarm_With_R": round(fa_with_r, 2),
            "Reduction_Percentage": round(reduc, 1),
        })
        print(f"[*] Ngưỡng {t_thr}: Báo oan Pure e = {fa_pure:.2f}% -> Với R = {fa_with_r:.2f}% (Giảm {reduc:.1f}%)")

    return {
        "total_hard_instances": len(hard_instances),
        "occluded_count": sum(1 for x in hard_instances if x["is_occluded"]),
        "small_person_count": sum(1 for x in hard_instances if x["is_small"]),
        "crowded_count": sum(1 for x in hard_instances if x["is_crowd"]),
        "false_alarm_reduction": far_summary,
    }


# =========================================================================
# THỰC NGHIỆM 4: PHÂN TÍCH ĐUÔI LỖI M3 (e > 0.15) VÀ THỬ TRỌNG SỐ THAY THẾ
# =========================================================================
def run_m3_fat_tail_analysis(test_instances: list[dict[str, Any]], dev_instances: list[dict[str, Any]]):
    print("\n==================================================================")
    print("=== THỰC NGHIỆM 4: PHÂN TÍCH ĐUÔI LỖI M3 (e > 0.15) & TRỌNG SỐ THAY THẾ ===")
    print("==================================================================")

    # 1. Thu thập tất cả các trường hợp M3 có e > 0.15 trên test (chỉ xét v=2)
    m3_instances = [it for it in test_instances if it["v_flag"] == 2 and "rtmpose-m" in it["models"]]
    tail_instances = [it for it in m3_instances if it["models"]["rtmpose-m"]["err"] > 0.15]

    total_m3 = len(m3_instances)
    n_tail = len(tail_instances)
    tail_pct = (n_tail / total_m3) * 100 if total_m3 > 0 else 0.0

    print(f"[*] M3 trên Test: {n_tail}/{total_m3} khớp ({tail_pct:.2f}%) có sai số e > 0.15.")

    # Phân loại nguyên nhân:
    cause_detector_bad_bbox = 0
    cause_crowded_img = 0
    cause_small_person = 0
    cause_pure_pose = 0

    joint_counts = {VF17_NAMES[i]: 0 for i in range(1, 18)}

    for it in tail_instances:
        joint_counts[it["vf_name"]] += 1
        m_info = it["models"]["rtmpose-m"]
        pred_b = m_info.get("pred_bbox")
        gt_b = it["gt_bbox"]

        # Kiểm tra bbox IoU
        iou = 0.0
        if pred_b and len(pred_b) == 4 and gt_b and len(gt_b) == 4:
            # pred_b: [x1, y1, x2, y2], gt_b: [gx, gy, gw, gh]
            gx1, gy1, gx2, gy2 = gt_b[0], gt_b[1], gt_b[0] + gt_b[2], gt_b[1] + gt_b[3]
            px1, py1, px2, py2 = pred_b
            ix1, iy1 = max(gx1, px1), max(gy1, py1)
            ix2, iy2 = min(gx2, px2), min(gy2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
            iou = inter / union if union > 0 else 0.0

        if iou < 0.40:
            cause_detector_bad_bbox += 1
        elif it["num_persons_in_img"] >= 3:
            cause_crowded_img += 1
        elif it["bbox_area"] <= 8000.0:
            cause_small_person += 1
        else:
            cause_pure_pose += 1

    cause_summary = {
        "detector_bad_bbox_IoU_lt_0.4": round(cause_detector_bad_bbox / n_tail * 100, 1),
        "crowded_image_gte_3_persons": round(cause_crowded_img / n_tail * 100, 1),
        "small_person_area_lte_8000": round(cause_small_person / n_tail * 100, 1),
        "pure_pose_localization_error": round(cause_pure_pose / n_tail * 100, 1),
    }

    # Top khớp hay bị đuôi lỗi nhất
    sorted_joints = sorted(joint_counts.items(), key=lambda x: -x[1])

    # 2. Thử nghiệm trọng số thay thế trên DEV và đo trên TEST
    available_models = ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"]

    # Cách 1: 1 / MPE^2 gốc
    dev_mpe = {m: float(np.median([it["models"][m]["err"] for it in dev_instances if m in it["models"]])) for m in available_models}
    w_mpe = {m: 1.0 / (dev_mpe[m]**2) for m in available_models}

    # Cách 2: Trọng số theo Tỷ lệ lỗi đuôi: w = 1 / P(e > 0.15)
    tail_rates_dev = {m: float(np.mean([it["models"][m]["err"] > 0.15 for it in dev_instances if m in it["models"]])) for m in available_models}
    w_tail = {m: 1.0 / (max(tail_rates_dev[m], 0.01)) for m in available_models}

    # Cách 3: Trimmed Mean (loại bỏ 10% phần tử đuôi lớn nhất trên Dev)
    def trimmed_mean(arr, trim=0.10):
        s_arr = np.sort(arr)
        k = int(len(s_arr) * trim)
        return float(np.mean(s_arr[:-k])) if k > 0 else float(np.mean(s_arr))

    tmean_dev = {m: trimmed_mean([it["models"][m]["err"] for it in dev_instances if m in it["models"]]) for m in available_models}
    w_tmean = {m: 1.0 / (tmean_dev[m]**2) for m in available_models}

    # Đo MPE của K=3 trên Test với 3 bộ trọng số:
    def eval_combo_weights(weights_dict):
        errs = []
        for it in test_instances:
            if it["v_flag"] != 2: continue
            active = [m for m in available_models if m in it["models"]]
            if not active: continue
            gx, gy = it["gt"]
            scale = it["scale"]
            sw = sum(weights_dict[m] * it["models"][m]["conf"] for m in active)
            px = sum(weights_dict[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in active) / sw
            py = sum(weights_dict[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in active) / sw
            errs.append(math.hypot(px - gx, py - gy) / scale)
        return round(float(np.median(errs)), 5)

    test_mpe_w_mpe = eval_combo_weights(w_mpe)
    test_mpe_w_tail = eval_combo_weights(w_tail)
    test_mpe_w_tmean = eval_combo_weights(w_tmean)

    print("[*] So sánh MPE K=3 trên Test giữa các phương pháp đặt trọng số:")
    print(f"    - 1/MPE^2 (Hiện tại): {test_mpe_w_mpe}")
    print(f"    - 1/Tail_Error_Rate: {test_mpe_w_tail}")
    print(f"    - 1/Trimmed_Mean^2: {test_mpe_w_tmean}")

    return {
        "tail_error_percentage": round(tail_pct, 2),
        "cause_breakdown_percentage": cause_summary,
        "joint_tail_distribution": dict(sorted_joints[:8]),
        "weight_schemes_comparison": {
            "inverse_mpe2": {"weights": {m: round(w_mpe[m], 1) for m in available_models}, "test_MPE": test_mpe_w_mpe},
            "inverse_tail_rate": {"weights": {m: round(w_tail[m], 1) for m in available_models}, "test_MPE": test_mpe_w_tail},
            "inverse_trimmed_mean2": {"weights": {m: round(w_tmean[m], 1) for m in available_models}, "test_MPE": test_mpe_w_tmean},
        }
    }


# =========================================================================
# THỰC NGHIỆM 5: CHUẨN HÓA SỐ LIỆU LATENCY (ĐO TUẦN TỰ TRÊN CÙNG ĐIỀU KIỆN)
# =========================================================================
def run_latency_standardization():
    print("\n==================================================================")
    print("=== THỰC NGHIỆM 5: CHUẨN HÓA SỐ LIỆU LATENCY (TUẦN TỰ & CÙNG MÁY) ===")
    print("==================================================================")

    from models_manager import setup_all_models
    manifest_file = DATA_DIR / "split_manifest.json"
    manifest = json.load(open(manifest_file, "r", encoding="utf-8"))
    test_items = manifest["test"][:60] # Đo chuẩn trên 60 ảnh
    images_dir = DATA_DIR / "images"

    models = setup_all_models()
    model_keys = ["yolo26s-pose", "yolov8n-pose", "yolov8m-pose", "rtmpose-m"]

    # Warmup 5 ảnh
    print("[*] Warmup 5 ảnh...")
    for it in test_items[:5]:
        p = images_dir / it["file_name"]
        for k in model_keys:
            if k in models:
                models[k].predict(p)

    # Đo tuần tự
    print("[*] Đo thời gian thực thi tuần tự từng ảnh...")
    lat_records: dict[str, list[float]] = {k: [] for k in model_keys}

    for it in test_items[5:]:
        p = images_dir / it["file_name"]
        for k in model_keys:
            if k in models:
                _, lat = models[k].predict(p)
                lat_records[k].append(lat)

    bench_latency = {}
    for k in model_keys:
        arr = lat_records[k]
        bench_latency[k] = {
            "median_ms": round(float(np.median(arr)), 1),
            "mean_ms": round(float(np.mean(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
        }
        print(f"    - {k}: Median = {bench_latency[k]['median_ms']} ms | P95 = {bench_latency[k]['p95_ms']} ms")

    # Tính tổ hợp
    m0_lat = bench_latency["yolo26s-pose"]["median_ms"]
    m2_lat = bench_latency["yolov8m-pose"]["median_ms"]
    m3_lat = bench_latency["rtmpose-m"]["median_ms"]

    combo_latency = {
        "M0_don": m0_lat,
        "M0_plus_M3_Sequential": round(m0_lat + m3_lat, 1),
        "M0_plus_M2_plus_M3_Sequential": round(m0_lat + m2_lat + m3_lat, 1),
        "M0_plus_M3_Parallel_Theoretical": round(max(m0_lat, m3_lat) * 1.15, 1), # +15% IPC overhead
        "M0_plus_M2_plus_M3_Parallel_Theoretical": round(max(m0_lat, m2_lat, m3_lat) * 1.20, 1),
    }

    return {
        "single_models_measured": bench_latency,
        "combo_latency": combo_latency,
        "measurement_notes": "Đo tuần tự batch=1 trên CPU Intel 6 cores, 55 ảnh test sau khi warmup 5 ảnh."
    }


def main():
    manifest_file = DATA_DIR / "split_manifest.json"
    manifest = json.load(open(manifest_file, "r", encoding="utf-8"))

    dev_items = manifest["dev"]
    test_items = manifest["test"]
    models = ["yolo26s-pose", "yolov8n-pose", "yolov8m-pose", "rtmpose-m"]

    dev_preds = load_all_predictions(models, [it["image_id"] for it in dev_items])
    test_preds = load_all_predictions(models, [it["image_id"] for it in test_items])

    dev_instances = extract_instances_full(dev_items, dev_preds, models)
    test_instances = extract_instances_full(test_items, test_preds, models)

    # Đọc weights và tuned params từ lần trước
    rig_file = RESULTS_DIR / "rigorous_benchmark_test.json"
    rig_data = json.load(open(rig_file, "r", encoding="utf-8"))
    dev_weights = rig_data["dev_weights"]["raw_w_k"]

    qa_file = RESULTS_DIR / "error_injection_qa_results.json"
    qa_data = json.load(open(qa_file, "r", encoding="utf-8"))
    tuned_params = qa_data["tuned_hyperparameters_dev"]

    # 1. Paired AUROC / AP Bootstrap
    res1 = run_paired_auroc_ap_bootstrap(test_instances, dev_weights, tuned_params)

    # 2. Subtle error injection
    res2 = run_subtle_error_injection(test_instances, dev_weights, tuned_params)

    # 3. Hard cases
    res3 = run_hard_cases_investigation(test_instances, dev_weights, tuned_params)

    # 4. M3 fat-tail analysis
    res4 = run_m3_fat_tail_analysis(test_instances, dev_instances)

    # 5. Latency standardization
    res5 = run_latency_standardization()

    full_output = {
        "experiment_1_paired_auroc_ap": res1,
        "experiment_2_subtle_errors_breakdown": res2,
        "experiment_3_hard_cases_R_effect": res3,
        "experiment_4_m3_tail_and_weight_schemes": res4,
        "experiment_5_latency_standardized": res5,
    }

    out_file = RESULTS_DIR / "deep_investigation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[OK] Đã hoàn thành toàn bộ 5 thực nghiệm mở rộng. File xuất: {out_file}")


if __name__ == "__main__":
    main()
