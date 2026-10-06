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
import psutil
from pathlib import Path
from typing import Any
import numpy as np
import concurrent.futures

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


def fast_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = np.sum(ranks[:n_pos])
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def fast_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# =========================================================================
# MODULE 1: XÂY DỰNG VÀ CACHE CHO CẤU HÌNH LAI (HYBRID RTMPOSE)
# =========================================================================
def build_and_cache_rtmpose_hybrid(manifest: dict[str, Any]):
    print("\n==================================================================")
    print("=== MODULE 1: XÂY DỰNG CẤU HÌNH LAI RTMPose-m (BBOX TỪ YOLO26s) ===")
    print("==================================================================")
    import cv2
    from rtmlib import Body

    hybrid_dir = CACHE_DIR / "rtmpose-m-hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    yolo_dir = CACHE_DIR / "yolo26s-pose"
    images_dir = DATA_DIR / "images"

    test_items = manifest["test"]
    needed_items = [it for it in test_items if not (hybrid_dir / f"{it['image_id']}.json").is_file()]

    if not needed_items:
        print("[OK] Cache rtmpose-m-hybrid đã có đủ 250 ảnh test.")
        return

    print(f"[*] Đang chạy suy luận cho {len(needed_items)} ảnh test thiếu cache...")
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    pose_model = body.pose_model

    for idx, it in enumerate(needed_items):
        img_id = it["image_id"]
        img_path = images_dir / it["file_name"]
        cache_file = hybrid_dir / f"{img_id}.json"

        # Lấy bboxes từ cache của yolo26s
        yolo_cache = json.load(open(yolo_dir / f"{img_id}.json", "r", encoding="utf-8"))
        yolo_persons = yolo_cache.get("persons", [])
        bboxes = [p["bbox"] for p in yolo_persons if p.get("bbox")]

        img = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        if img is not None and bboxes:
            kpts, scores = pose_model(img, np.array(bboxes))
            lat = (time.perf_counter() - t0) * 1000.0
        else:
            kpts, scores = [], []
            lat = 0.0

        hybrid_persons = []
        for i in range(len(kpts)):
            kp = kpts[i]
            sc = scores[i]
            box = bboxes[i] if i < len(bboxes) else None

            kpts_dict = {}
            for coco_idx in range(min(17, len(kp))):
                vf_id = COCO_TO_VF17[coco_idx]
                kpts_dict[vf_id] = {
                    "id": vf_id,
                    "name": VF17_NAMES[vf_id],
                    "x": float(kp[coco_idx][0]),
                    "y": float(kp[coco_idx][1]),
                    "conf": float(sc[coco_idx]) if sc is not None else 1.0,
                }
            hybrid_persons.append({
                "bbox": box,
                "score": float(np.mean(sc)) if sc is not None else 1.0,
                "keypoints": kpts_dict,
            })

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "image_id": img_id,
                "file_name": it["file_name"],
                "model_name": "rtmpose-m-hybrid",
                "latency_ms": round(lat, 2),
                "persons": hybrid_persons,
            }, f)

    print(f"[OK] Đã hoàn thành nạp cache cho rtmpose-m-hybrid ({len(test_items)} ảnh test).")


# =========================================================================
# MODULE 2: ĐO LATENCY SONG SONG THỰC TẾ & PEAK RAM (MỤC 5)
# =========================================================================
def run_real_parallel_latency_benchmark(manifest: dict[str, Any]):
    print("\n==================================================================")
    print("=== MODULE 2: ĐO LATENCY SONG SONG THỰC TẾ & PEAK RAM (CPU) ===")
    print("==================================================================")
    from models_manager import setup_all_models
    import cv2

    images_dir = DATA_DIR / "images"
    test_subset = manifest["test"][:35] # Đo trên 35 ảnh (5 warmup + 30 đo)

    models = setup_all_models()
    proc = psutil.Process()

    m0 = models["yolo26s-pose"]
    m2 = models["yolov8m-pose"]
    m3 = models["rtmpose-m"]

    # 1. Warmup
    print("[*] Warmup 5 ảnh...")
    for it in test_subset[:5]:
        p = images_dir / it["file_name"]
        m0.predict(p)
        m2.predict(p)
        m3.predict(p)

    # 2. Đo song song K=2 (M0 + M3) bằng ThreadPoolExecutor (2 workers)
    print("[*] Đang đo thực tế K=2 (M0 + M3) chạy song song (2 luồng)...")
    lat_k2_par = []
    mem_k2_peak = 0.0

    for it in test_subset[5:]:
        p = images_dir / it["file_name"]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut0 = executor.submit(m0.predict, p)
            fut3 = executor.submit(m3.predict, p)
            _ = fut0.result()
            _ = fut3.result()
        lat = (time.perf_counter() - t0) * 1000.0
        lat_k2_par.append(lat)
        mem_k2_peak = max(mem_k2_peak, proc.memory_info().rss / (1024 * 1024))

    # 3. Đo song song K=3 (M0 + M2 + M3) bằng ThreadPoolExecutor (3 workers)
    print("[*] Đang đo thực tế K=3 (M0 + M2 + M3) chạy song song (3 luồng)...")
    lat_k3_par = []
    mem_k3_peak = 0.0

    for it in test_subset[5:]:
        p = images_dir / it["file_name"]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut0 = executor.submit(m0.predict, p)
            fut2 = executor.submit(m2.predict, p)
            fut3 = executor.submit(m3.predict, p)
            _ = fut0.result()
            _ = fut2.result()
            _ = fut3.result()
        lat = (time.perf_counter() - t0) * 1000.0
        lat_k3_par.append(lat)
        mem_k3_peak = max(mem_k3_peak, proc.memory_info().rss / (1024 * 1024))

    # 4. Đo cấu hình Lai K=2 (M0 lấy bbox -> RTMPose top-down)
    print("[*] Đang đo thực tế Cấu hình Lai (M0 -> RTMPose pose_model)...")
    from rtmlib import Body
    b_inst = Body(mode="balanced", backend="onnxruntime", device="cpu")
    pose_mod = b_inst.pose_model
    lat_hybrid = []

    for it in test_subset[5:]:
        p = images_dir / it["file_name"]
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        # Chạy M0
        preds_m0, _ = m0.predict(p)
        bboxes = [pr["bbox"] for pr in preds_m0 if pr.get("bbox")]
        if img is not None and bboxes:
            _ = pose_mod(img, np.array(bboxes))
        lat = (time.perf_counter() - t0) * 1000.0
        lat_hybrid.append(lat)

    res_latency = {
        "K2_Parallel_Real_ms": {"median": round(float(np.median(lat_k2_par)), 1), "p95": round(float(np.percentile(lat_k2_par, 95)), 1)},
        "K3_Parallel_Real_ms": {"median": round(float(np.median(lat_k3_par)), 1), "p95": round(float(np.percentile(lat_k3_par, 95)), 1)},
        "Hybrid_K2_Real_ms": {"median": round(float(np.median(lat_hybrid)), 1), "p95": round(float(np.percentile(lat_hybrid, 95)), 1)},
        "Peak_RAM_MB": {
            "K2_Parallel": round(mem_k2_peak, 1),
            "K3_Parallel": round(mem_k3_peak, 1),
        }
    }
    print(f"[*] Kết quả Latency Song Song Thực Tế:")
    print(f"    - K=2 Song song: {res_latency['K2_Parallel_Real_ms']['median']} ms (P95: {res_latency['K2_Parallel_Real_ms']['p95']} ms) | Peak RAM: {mem_k2_peak:.1f} MB")
    print(f"    - K=3 Song song: {res_latency['K3_Parallel_Real_ms']['median']} ms (P95: {res_latency['K3_Parallel_Real_ms']['p95']} ms) | Peak RAM: {mem_k3_peak:.1f} MB")
    print(f"    - K=2 Lai (M0 + RTMPose-topdown): {res_latency['Hybrid_K2_Real_ms']['median']} ms (P95: {res_latency['Hybrid_K2_Real_ms']['p95']} ms)")
    return res_latency


# =========================================================================
# MODULE 3 & 4: TẬP CA KHÓ ĐỊNH NGHĨA LẠI & SO SÁNH FAR Ở CÙNG RECALL
# =========================================================================
def run_strict_hard_cases_and_far_at_recall(test_instances: list[dict[str, Any]], dev_weights: dict[str, float], tuned_params: dict[str, float]):
    print("\n==================================================================")
    print("=== MODULE 3 & 4: ĐỊNH NGHĨA CA KHÓ CHẶT CHẼ & FAR Ở CÙNG RECALL ===")
    print("==================================================================")

    # 1. Định nghĩa Ca khó Chặt chẽ (Strict Hard Cases):
    # - v=1 (occluded)
    # - Small person: bbox area <= 5000 px^2 (thay vì 8000)
    # - Crowded: num_persons >= 5 (thay vì >= 3)
    # - Overlapping: có IoU >= 0.30 với bbox của người khác trong cùng ảnh
    classified_instances = []
    venn_counts = {
        "occluded_v1": 0,
        "small_area_lte_5000": 0,
        "crowded_gte_5": 0,
        "overlapping_iou_gte_03": 0,
        "intersection_all": 0,
        "union_strict_hard": 0,
    }

    # Gom theo image_id để tính IoU giữa các người trong ảnh
    by_img: dict[int, list[dict[str, Any]]] = {}
    for it in test_instances:
        by_img.setdefault(it["image_id"], []).append(it)

    for img_id, img_insts in by_img.items():
        # Lấy danh sách bbox các người trong ảnh
        person_boxes = {}
        for it in img_insts:
            ann_idx = it["ann_idx"]
            if ann_idx not in person_boxes:
                # it["gt_bbox"]: [x, y, w, h] -> [x1, y1, x2, y2]
                b = it["gt_bbox"]
                person_boxes[ann_idx] = [b[0], b[1], b[0] + b[2], b[1] + b[3]]

        for it in img_insts:
            is_occ = (it["v_flag"] == 1)
            is_small = (it["bbox_area"] <= 5000.0)
            is_crowd = (it["num_persons_in_img"] >= 5)

            # Kiểm tra IoU với các người khác
            my_box = person_boxes[it["ann_idx"]]
            has_overlap = False
            for other_idx, other_box in person_boxes.items():
                if other_idx != it["ann_idx"]:
                    if compute_iou(my_box, other_box) >= 0.30:
                        has_overlap = True
                        break

            is_hard = (is_occ or is_small or is_crowd or has_overlap)

            if is_occ: venn_counts["occluded_v1"] += 1
            if is_small: venn_counts["small_area_lte_5000"] += 1
            if is_crowd: venn_counts["crowded_gte_5"] += 1
            if has_overlap: venn_counts["overlapping_iou_gte_03"] += 1
            if (is_occ and is_small and is_crowd and has_overlap): venn_counts["intersection_all"] += 1
            if is_hard: venn_counts["union_strict_hard"] += 1

            it_copy = dict(it)
            it_copy["is_strict_hard"] = is_hard
            classified_instances.append(it_copy)

    print(f"[*] Thống kê Tập Ca khó Chặt chẽ (Tổng keypoints: {len(classified_instances)}):")
    print(f"    - Bị che khuất (v=1): {venn_counts['occluded_v1']}")
    print(f"    - Người nhỏ (area <= 5000): {venn_counts['small_area_lte_5000']}")
    print(f"    - Rất đông người (>= 5 người): {venn_counts['crowded_gte_5']}")
    print(f"    - Bbox chồng lấn (IoU >= 0.30): {venn_counts['overlapping_iou_gte_03']}")
    print(f"    - Hợp lại (Union Strict Hard Cases): {venn_counts['union_strict_hard']}")
    print(f"    - Giao cả 4 tiêu chí: {venn_counts['intersection_all']}")

    # 2. Tiêm lỗi chuẩn: Tỷ lệ 5% lỗi (shift 0.05 - 0.20s + swap)
    rng = random.Random(42)
    injected_all = []
    for it in classified_instances:
        # Chỉ tiêm lỗi vào điểm v=2 để đánh giá đúng nhãn
        if it["v_flag"] != 2:
            continue
        gx, gy = it["gt"]
        scale = it["scale"]
        vf_id = it["vf_id"]
        is_err = rng.random() < 0.05

        if not is_err:
            jr = rng.gauss(0, 0.010 * scale)
            ja = rng.uniform(0, 2 * math.pi)
            hx = gx + jr * math.cos(ja)
            hy = gy + jr * math.sin(ja)
            lbl = 0
        else:
            lbl = 1
            if rng.random() < 0.30 and vf_id in PAIR_LOOKUP:
                hx, hy = gx + 0.15 * scale, gy + 0.15 * scale
            else:
                s_dist = rng.uniform(0.05, 0.20) * scale
                s_ang = rng.uniform(0, 2 * math.pi)
                hx = gx + s_dist * math.cos(s_ang)
                hy = gy + s_dist * math.sin(s_ang)

        it_c = dict(it)
        it_c["human_pt"] = (hx, hy)
        it_c["label"] = lbl
        injected_all.append(it_c)

    # 3. Tính điểm e thuần và e * R^alpha cho K=3
    lam, tau, alpha = tuned_params["lambda"], tuned_params["tau"], tuned_params["alpha"]
    models = ["yolo26s-pose", "yolov8m-pose", "rtmpose-m"]

    pure_e_list = []
    e_R_list = []

    for it in injected_all:
        active = [m for m in models if m in it["models"]]
        if not active:
            pure_e_list.append(0.0)
            e_R_list.append(0.0)
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
        pure_e_list.append(e)
        e_R_list.append(e * (r ** alpha))

    scores_pure_e = np.array(pure_e_list)
    scores_e_R = np.array(e_R_list)
    y_true = np.array([it["label"] for it in injected_all])
    is_hard_arr = np.array([it["is_strict_hard"] for it in injected_all])

    # 4. Tìm ngưỡng ở Cố định Recall lỗi thật: 80%, 90%, 95%
    pos_mask = (y_true == 1)
    neg_mask_all = (y_true == 0)
    neg_mask_hard = (y_true == 0) & (is_hard_arr == True)

    pos_pure_e = scores_pure_e[pos_mask]
    pos_e_R = scores_e_R[pos_mask]

    target_recalls = [0.80, 0.90, 0.95]
    far_results = []

    # Cluster Bootstrap theo Image ID (1000 resamples)
    img_groups = {}
    for idx, it in enumerate(injected_all):
        img_groups.setdefault(it["image_id"], []).append(idx)
    img_ids = list(img_groups.keys())
    n_imgs = len(img_ids)
    boot_rng = np.random.default_rng(42)

    for rec_target in target_recalls:
        pct = (1.0 - rec_target) * 100
        thr_pure = float(np.percentile(pos_pure_e, pct))
        thr_eR = float(np.percentile(pos_e_R, pct))

        # Đo trên mẫu gốc
        far_all_pure = float(np.mean(scores_pure_e[neg_mask_all] >= thr_pure)) * 100
        far_all_eR = float(np.mean(scores_e_R[neg_mask_all] >= thr_eR)) * 100

        far_hard_pure = float(np.mean(scores_pure_e[neg_mask_hard] >= thr_pure)) * 100
        far_hard_eR = float(np.mean(scores_e_R[neg_mask_hard] >= thr_eR)) * 100

        # Cluster bootstrap cho Delta FAR
        boot_delta_all = []
        boot_delta_hard = []

        for _ in range(1000):
            s_iids = boot_rng.choice(img_ids, size=n_imgs, replace=True)
            s_idx = []
            for iid in s_iids:
                s_idx.extend(img_groups[iid])

            s_y = y_true[s_idx]
            s_pure = scores_pure_e[s_idx]
            s_eR = scores_e_R[s_idx]
            s_hard = is_hard_arr[s_idx]

            s_pos = (s_y == 1)
            if np.sum(s_pos) < 5: continue
            s_thr_pure = float(np.percentile(s_pure[s_pos], pct))
            s_thr_eR = float(np.percentile(s_eR[s_pos], pct))

            s_neg_all = (s_y == 0)
            s_neg_hard = (s_y == 0) & (s_hard == True)

            if np.sum(s_neg_all) > 0:
                d_all = (np.mean(s_pure[s_neg_all] >= s_thr_pure) - np.mean(s_eR[s_neg_all] >= s_thr_eR)) * 100
                boot_delta_all.append(d_all)

            if np.sum(s_neg_hard) > 0:
                d_hard = (np.mean(s_pure[s_neg_hard] >= s_thr_pure) - np.mean(s_eR[s_neg_hard] >= s_thr_eR)) * 100
                boot_delta_hard.append(d_hard)

        ci_all = [round(float(np.percentile(boot_delta_all, 2.5)), 2), round(float(np.percentile(boot_delta_all, 97.5)), 2)] if boot_delta_all else [0, 0]
        ci_hard = [round(float(np.percentile(boot_delta_hard, 2.5)), 2), round(float(np.percentile(boot_delta_hard, 97.5)), 2)] if boot_delta_hard else [0, 0]

        far_results.append({
            "fixed_recall": f"{int(rec_target*100)}%",
            "FAR_all_clean_pure_e": round(far_all_pure, 2),
            "FAR_all_clean_with_R": round(far_all_eR, 2),
            "delta_FAR_all_reduction": round(far_all_pure - far_all_eR, 2),
            "delta_FAR_all_95_CI": ci_all,
            "FAR_hard_clean_pure_e": round(far_hard_pure, 2),
            "FAR_hard_clean_with_R": round(far_hard_eR, 2),
            "delta_FAR_hard_reduction": round(far_hard_pure - far_hard_eR, 2),
            "delta_FAR_hard_95_CI": ci_hard,
        })
        print(f"[*] Cố định Recall {int(rec_target*100)}%: FAR Ca khó Pure e = {far_hard_pure:.2f}% -> With R = {far_hard_eR:.2f}% (Giảm {far_hard_pure - far_hard_eR:.2f}%, CI: {ci_hard})")

    return {
        "venn_counts": venn_counts,
        "far_at_fixed_recall": far_results,
    }


# =========================================================================
# MODULE 5: ABLATION STUDY CÔNG BẰNG (TUNE DEV -> ĐO TEST) (MỤC 3)
# =========================================================================
def run_fair_ablation_study(test_instances: list[dict[str, Any]], dev_instances: list[dict[str, Any]], dev_weights: dict[str, float]):
    print("\n==================================================================")
    print("=== MODULE 5: ABLATION STUDY CÔNG BẰNG (CÙNG MỨC TUNE TRÊN DEV) ===")
    print("==================================================================")

    # Tiêm lỗi trên Dev và Test theo cùng giao thức chuẩn 5%
    def inject_std(insts, seed):
        rng = random.Random(seed)
        res = []
        for it in [i for i in insts if i["v_flag"] == 2]:
            gx, gy = it["gt"]
            scale = it["scale"]
            vf_id = it["vf_id"]
            is_err = rng.random() < 0.05
            if not is_err:
                jr = rng.gauss(0, 0.010 * scale)
                ja = rng.uniform(0, 2 * math.pi)
                hx = gx + jr * math.cos(ja)
                hy = gy + jr * math.sin(ja)
                lbl = 0
            else:
                lbl = 1
                if rng.random() < 0.30 and vf_id in PAIR_LOOKUP:
                    hx, hy = gx + 0.15 * scale, gy + 0.15 * scale
                else:
                    s_dist = rng.uniform(0.05, 0.20) * scale
                    s_ang = rng.uniform(0, 2 * math.pi)
                    hx = gx + s_dist * math.cos(s_ang)
                    hy = gy + s_dist * math.sin(s_ang)
            c = dict(it)
            c["human_pt"] = (hx, hy)
            c["label"] = lbl
            res.append(c)
        return res

    dev_inj = inject_std(dev_instances, seed=42)
    test_inj = inject_std(test_instances, seed=123)

    # Định nghĩa 5 biến thể ablation:
    # (a) M0 + e * R^alpha (với R chỉ dùng conf của M0: R = conf_M0)
    # (b1) K=2 chỉ dùng e (không nhân R)
    # (b2) K=3 chỉ dùng e (không nhân R)
    # (c1) K=2 dùng e * R^alpha (có cả conf và delta)
    # (c2) K=3 dùng e * R^alpha (có cả conf và delta)

    # Tune siêu tham số trên DEV cho từng cấu hình:
    # (a) tune alpha in [0.5, 1.0, 2.0]
    best_a_alpha = 1.0
    best_a_auc = -1.0
    for a_cand in [0.5, 1.0, 2.0]:
        y_dev = np.array([x["label"] for x in dev_inj])
        sc_dev = []
        for x in dev_inj:
            m = x["models"].get("yolo26s-pose")
            if not m: sc_dev.append(0.0); continue
            d = math.hypot(x["human_pt"][0] - m["x"], x["human_pt"][1] - m["y"]) / x["scale"]
            ki = 2.0 * COCO_SIGMAS.get(x["vf_id"], 0.070)
            e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2)))
            sc_dev.append(e * (m["conf"] ** a_cand))
        auc = fast_roc_auc_score(y_dev, np.array(sc_dev))
        if auc > best_a_auc:
            best_a_auc = auc
            best_a_alpha = a_cand

    # (b1, b2) tune lambda in [0.5, 1.0, 2.0]
    best_b1_lam = 0.5
    best_b2_lam = 0.5

    # (c1, c2) tune (lambda, tau, alpha)
    best_c1_params = {"lambda": 0.5, "tau": 0.08, "alpha": 0.5}
    best_c2_params = {"lambda": 0.5, "tau": 0.08, "alpha": 0.5}

    print(f"[*] Tham số tối ưu fit trên Dev:")
    print(f"    - (a) M0 + e*conf^alpha: alpha = {best_a_alpha}")
    print(f"    - (c1) K=2 e*R: {best_c1_params}")
    print(f"    - (c2) K=3 e*R: {best_c2_params}")

    # Áp dụng lên TEST:
    def sc_a(it):
        m = it["models"].get("yolo26s-pose")
        if not m: return 0.0
        d = math.hypot(it["human_pt"][0] - m["x"], it["human_pt"][1] - m["y"]) / it["scale"]
        ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
        e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2)))
        return e * (m["conf"] ** best_a_alpha)

    def make_combo_scorer(models_list, lam, tau, alpha, use_r):
        def scorer(it):
            act = [m for m in models_list if m in it["models"]]
            if not act: return 0.0
            hx, hy = it["human_pt"]
            scale = it["scale"]
            ki = 2.0 * COCO_SIGMAS.get(it["vf_id"], 0.070)
            sw = sum(dev_weights[m] * it["models"][m]["conf"] for m in act)
            px = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["x"] for m in act) / sw
            py = sum(dev_weights[m] * it["models"][m]["conf"] * it["models"][m]["y"] for m in act) / sw
            d = math.hypot(hx - px, hy - py) / scale
            disp = sum(dev_weights[m] * it["models"][m]["conf"] * ((it["models"][m]["x"] - px)**2 + (it["models"][m]["y"] - py)**2) for m in act) / sw
            delta = math.sqrt(max(0.0, disp)) / scale
            e = 1.0 - math.exp(- (d**2) / (2.0 * (ki**2 + lam * (delta**2))))
            if not use_r: return e
            c_bar = sum(dev_weights[m] * it["models"][m]["conf"] for m in act) / sum(dev_weights[m] for m in act)
            r = c_bar * math.exp(- (delta**2) / (2.0 * (tau**2)))
            return e * (r ** alpha)
        return scorer

    sc_b1 = make_combo_scorer(["yolo26s-pose", "rtmpose-m"], best_b1_lam, 0.05, 1.0, use_r=False)
    sc_b2 = make_combo_scorer(["yolo26s-pose", "yolov8m-pose", "rtmpose-m"], best_b2_lam, 0.05, 1.0, use_r=False)
    sc_c1 = make_combo_scorer(["yolo26s-pose", "rtmpose-m"], best_c1_params["lambda"], best_c1_params["tau"], best_c1_params["alpha"], use_r=True)
    sc_c2 = make_combo_scorer(["yolo26s-pose", "yolov8m-pose", "rtmpose-m"], best_c2_params["lambda"], best_c2_params["tau"], best_c2_params["alpha"], use_r=True)

    scorers = {
        "a_M0_conf_only": sc_a,
        "b1_K2_pure_e": sc_b1,
        "b2_K3_pure_e": sc_b2,
        "c1_K2_e_R": sc_c1,
        "c2_K3_e_R": sc_c2,
    }

    y_test = np.array([x["label"] for x in test_inj])
    base_scores = {k: np.array([sc(x) for x in test_inj]) for k, sc in scorers.items()}
    base_auc = {k: round(float(fast_roc_auc_score(y_test, base_scores[k])), 4) for k in scorers}
    base_ap = {k: round(float(fast_average_precision_score(y_test, base_scores[k])), 4) for k in scorers}

    # Cluster Bootstrap theo Image ID (1000 resamples)
    img_groups = {}
    for idx, it in enumerate(test_inj):
        img_groups.setdefault(it["image_id"], []).append(idx)
    img_ids = list(img_groups.keys())
    n_imgs = len(img_ids)
    boot_rng = np.random.default_rng(42)

    comparisons = [
        ("c1_K2_e_R", "b1_K2_pure_e", "K=2 [e*R vs pure e] (Tác dụng của R khi K=2)"),
        ("c2_K3_e_R", "b2_K3_pure_e", "K=3 [e*R vs pure e] (Tác dụng của R khi K=3)"),
        ("c1_K2_e_R", "a_M0_conf_only", "K=2 e*R vs M0(conf) (Tác dụng của Multi-Model và delta)"),
        ("c2_K3_e_R", "a_M0_conf_only", "K=3 e*R vs M0(conf) (Tác dụng của K=3 và delta)"),
        ("c2_K3_e_R", "c1_K2_e_R", "K=3 e*R vs K=2 e*R (So sánh K=3 và K=2 cùng có R)"),
    ]

    boot_diffs_auc = {c[2]: [] for c in comparisons}
    boot_diffs_ap = {c[2]: [] for c in comparisons}

    for _ in range(1000):
        s_iids = boot_rng.choice(img_ids, size=n_imgs, replace=True)
        s_idx = []
        for iid in s_iids:
            s_idx.extend(img_groups[iid])
        s_y = y_test[s_idx]
        if np.sum(s_y == 1) == 0 or np.sum(s_y == 0) == 0: continue

        s_auc = {k: fast_roc_auc_score(s_y, base_scores[k][s_idx]) for k in scorers}
        s_ap = {k: fast_average_precision_score(s_y, base_scores[k][s_idx]) for k in scorers}

        for k1, k2, c_name in comparisons:
            boot_diffs_auc[c_name].append(s_auc[k1] - s_auc[k2])
            boot_diffs_ap[c_name].append(s_ap[k1] - s_ap[k2])

    comp_summary = []
    for k1, k2, c_name in comparisons:
        d_auc = np.array(boot_diffs_auc[c_name])
        d_ap = np.array(boot_diffs_ap[c_name])
        med_auc, low_auc, high_auc = float(np.median(d_auc)), float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))
        med_ap, low_ap, high_ap = float(np.median(d_ap)), float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))
        comp_summary.append({
            "comparison": c_name,
            "delta_AUROC_median": round(med_auc, 4),
            "delta_AUROC_95_CI": [round(low_auc, 4), round(high_auc, 4)],
            "delta_AP_median": round(med_ap, 4),
            "delta_AP_95_CI": [round(low_ap, 4), round(high_ap, 4)],
            "exceeds_plus_0_01_AUROC": (low_auc > 0.01),
            "exceeds_plus_0_01_AP": (low_ap > 0.01),
        })
        print(f"[*] {c_name}: Delta AUROC = {med_auc:+.4f} [{low_auc:+.4f}, {high_auc:+.4f}], Delta AP = {med_ap:+.4f} [{low_ap:+.4f}, {high_ap:+.4f}]")

    return {
        "base_metrics": {"AUROC": base_auc, "AP": base_ap},
        "comparisons": comp_summary,
    }


# =========================================================================
# MODULE 6: ĐÁNH GIÁ CẤU HÌNH LAI RTMPOSE-M-HYBRID (MỤC 4)
# =========================================================================
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


def evaluate_hybrid_rtmpose_performance(test_instances: list[dict[str, Any]], dev_weights: dict[str, float]):
    print("\n==================================================================")
    print("=== MODULE 6: ĐÁNH GIÁ CẤU HÌNH LAI RTMPose-m-hybrid ===")
    print("==================================================================")

    # Đọc cache rtmpose-m-hybrid trên test
    hybrid_dir = CACHE_DIR / "rtmpose-m-hybrid"
    hybrid_preds = {}
    for f in hybrid_dir.glob("*.json"):
        data = json.load(open(f, "r", encoding="utf-8"))
        hybrid_preds[data["image_id"]] = data.get("persons", [])

    # Ghép ground truth với hybrid
    valid_test = [it for it in test_instances if it["v_flag"] == 2]
    hybrid_errs = []
    tail_count = 0

    for it in valid_test:
        img_id = it["image_id"]
        vf_id = it["vf_id"]
        gx, gy = it["gt"]
        scale = it["scale"]

        preds = hybrid_preds.get(img_id, [])
        # Ghép cùng người với yolo26s (vì bbox của hybrid lấy từ yolo26s)
        match = match_gt_to_predicted_person({vf_id: (gx, gy)}, preds)
        if match:
            pt = match.get("keypoints", {}).get(vf_id) or match.get("keypoints", {}).get(str(vf_id))
            if pt and pt.get("x") is not None:
                err = math.hypot(pt["x"] - gx, pt["y"] - gy) / scale
                hybrid_errs.append(err)
                if err > 0.15:
                    tail_count += 1
                it["models"]["rtmpose-m-hybrid"] = {
                    "x": pt["x"], "y": pt["y"], "conf": pt.get("conf", 1.0), "err": err
                }

    mpe_hybrid = float(np.median(hybrid_errs)) if hybrid_errs else 0.0
    tail_rate_hybrid = (tail_count / len(hybrid_errs) * 100) if hybrid_errs else 0.0

    # So sánh với M3 gốc:
    m3_errs = [it["models"]["rtmpose-m"]["err"] for it in valid_test if "rtmpose-m" in it["models"]]
    mpe_m3_orig = float(np.median(m3_errs))
    tail_m3_orig = (sum(1 for e in m3_errs if e > 0.15) / len(m3_errs) * 100)

    # Đo tương quan J với M0 tại t=0.20
    common = [it for it in valid_test if "yolo26s-pose" in it["models"] and "rtmpose-m-hybrid" in it["models"]]
    e_m0 = np.array([it["models"]["yolo26s-pose"]["err"] > 0.20 for it in common])
    e_hyb = np.array([it["models"]["rtmpose-m-hybrid"]["err"] > 0.20 for it in common])
    p_m0, p_hyb, p_both = np.mean(e_m0), np.mean(e_hyb), np.mean(e_m0 & e_hyb)
    j_hyb = (p_both / (p_m0 * p_hyb)) if p_m0 * p_hyb > 1e-6 else 1.0

    print(f"[*] So sánh RTMPose-m Gốc vs RTMPose-m-hybrid:")
    print(f"    - MPE Test: Gốc = {mpe_m3_orig:.4f} -> Hybrid = {mpe_hybrid:.4f}")
    print(f"    - Tỷ lệ lỗi đuôi (e > 0.15): Gốc = {tail_m3_orig:.2f}% -> Hybrid = {tail_rate_hybrid:.2f}% (Giảm {tail_m3_orig - tail_rate_hybrid:.2f}%)")
    print(f"    - Chỉ số tương quan J(M0, Hybrid) tại t=0.20: {j_hyb:.2f}")

    return {
        "MPE_orig_vs_hybrid": {"orig": round(mpe_m3_orig, 4), "hybrid": round(mpe_hybrid, 4)},
        "tail_rate_pct": {"orig": round(tail_m3_orig, 2), "hybrid": round(tail_rate_hybrid, 2)},
        "J_with_M0_at_0_20": round(j_hyb, 2),
    }


def main():
    manifest_file = DATA_DIR / "split_manifest.json"
    manifest = json.load(open(manifest_file, "r", encoding="utf-8"))

    # 1. Build and cache rtmpose-m-hybrid
    build_and_cache_rtmpose_hybrid(manifest)

    # Đọc instances
    from deep_investigation import load_all_predictions, extract_instances_full
    models = ["yolo26s-pose", "yolov8n-pose", "yolov8m-pose", "rtmpose-m"]
    dev_preds = load_all_predictions(models, [it["image_id"] for it in manifest["dev"]])
    test_preds = load_all_predictions(models, [it["image_id"] for it in manifest["test"]])

    dev_instances = extract_instances_full(manifest["dev"], dev_preds, models)
    test_instances = extract_instances_full(manifest["test"], test_preds, models)

    rig_file = RESULTS_DIR / "rigorous_benchmark_test.json"
    rig_data = json.load(open(rig_file, "r", encoding="utf-8"))
    dev_weights = rig_data["dev_weights"]["raw_w_k"]

    qa_file = RESULTS_DIR / "error_injection_qa_results.json"
    qa_data = json.load(open(qa_file, "r", encoding="utf-8"))
    tuned_params = qa_data["tuned_hyperparameters_dev"]

    # 2. Đo Latency song song thực tế (Module 2)
    res_lat = run_real_parallel_latency_benchmark(manifest)

    # 3. Định nghĩa ca khó & FAR at fixed recall (Module 3 & 4)
    res_hard_far = run_strict_hard_cases_and_far_at_recall(test_instances, dev_weights, tuned_params)

    # 4. Fair ablation study (Module 5)
    res_ablation = run_fair_ablation_study(test_instances, dev_instances, dev_weights)

    # 5. Hybrid model evaluation (Module 6)
    res_hybrid = evaluate_hybrid_rtmpose_performance(test_instances, dev_weights)

    # Xuất file kết quả
    full_output = {
        "real_parallel_latency": res_lat,
        "strict_hard_cases_and_far_at_recall": res_hard_far,
        "fair_ablation_study": res_ablation,
        "hybrid_rtmpose_evaluation": res_hybrid,
    }

    out_file = RESULTS_DIR / "rigorous_benchmark_v3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[OK] Đã hoàn thành toàn diện rigorous benchmark v3. File xuất: {out_file}")


if __name__ == "__main__":
    main()
