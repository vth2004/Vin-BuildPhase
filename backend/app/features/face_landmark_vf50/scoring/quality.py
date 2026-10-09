from __future__ import annotations

import math
from typing import Any

from ..rules.geometry import compute_iod


ANCHOR_POINTS = {0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36}


def evaluate_frame_quality(
    human_kpts: dict[int, dict[str, Any]],
    model_kpts: dict[int, dict[str, Any]] | None,
    iod: float | None = None,
) -> dict[str, Any]:
    """
    Tính toán sai số chuẩn hóa theo IOD và NME cho một frame ảnh.
    Theo Mục 8 GL:
      - 12 Điểm neo: dung sai <= 3% IOD (~3 px)
      - Điểm contour còn lại: dung sai <= 5% IOD (~5 px)
      - NME toàn ảnh: <= 0.035 (~3.4 px)
    """
    if iod is None:
        iod = compute_iod(human_kpts)
    from ..spec import get_anchor_points, get_scoring_tolerances, get_case_classification_spec

    anchor_points = get_anchor_points()
    scoring_tol = get_scoring_tolerances()
    anchor_ratio = scoring_tol.get("anchor_point_iod_ratio", 0.03)
    contour_ratio = scoring_tol.get("contour_point_iod_ratio", 0.05)
    crit_multiplier = scoring_tol.get("severity_critical_multiplier", 1.8)
    susp_multiplier = scoring_tol.get("suspicion_normalizer_multiplier", 2.5)
    occ_discount = scoring_tol.get("occluded_suspicion_discount", 0.6)
    min_iod_norm = scoring_tol.get("min_iod_for_norm", 10.0)
    nme_threshold = scoring_tol.get("nme_max_threshold", 0.035)

    case_spec = get_case_classification_spec()
    close_min = case_spec.get("scale_close_min_iod", 80.0)
    normal_min = case_spec.get("scale_normal_min_iod", 38.0)
    sunglasses_min_occ = case_spec.get("sunglasses_min_eye_occluded", 4)
    sunglasses_rel = case_spec.get("sunglasses_ai_reliability", 0.45)
    occluded_min_occ = case_spec.get("occluded_min_total_occluded", 6)
    occluded_rel = case_spec.get("occluded_ai_reliability", 0.58)
    far_max_iod = case_spec.get("far_max_iod", 35.0)
    far_rel = case_spec.get("far_ai_reliability", 0.65)
    normal_rel = case_spec.get("normal_ai_reliability", 0.95)

    point_evaluations: dict[int, dict[str, Any]] = {}
    deviations: list[float] = []

    if model_kpts:
        for pt_id in range(50):
            h = human_kpts.get(pt_id)
            m = model_kpts.get(pt_id)

            if not h or h.get("state") == "outside" or h.get("x") is None or h.get("y") is None:
                continue
            if not m or m.get("x") is None or m.get("y") is None:
                continue

            hx, hy = float(h["x"]), float(h["y"])
            mx, my = float(m["x"]), float(m["y"])

            dist_px = math.hypot(hx - mx, hy - my)
            e_norm = dist_px / max(iod, min_iod_norm)
            deviations.append(e_norm)

            is_anchor = pt_id in anchor_points
            tolerance = anchor_ratio if is_anchor else contour_ratio
            exceeded = e_norm > tolerance

            # Phân loại mức độ nghiêm trọng theo tỉ lệ IOD
            if not exceeded:
                sev = "NORMAL"
            elif e_norm <= tolerance * crit_multiplier:
                sev = "WARNING"
            else:
                sev = "CRITICAL"

            # Tính độ nghi ngờ chuẩn hóa [0.0, 1.0]
            # Điểm Occluded đòi hỏi ước lượng giải phẫu, hạ trọng số phạt 40%
            raw_suspicion = min(1.0, e_norm / (tolerance * susp_multiplier))
            if h.get("state") == "occluded":
                raw_suspicion *= occ_discount

            point_evaluations[pt_id] = {
                "id": pt_id,
                "dist_px": round(dist_px, 1),
                "norm_error": round(e_norm, 4),
                "norm_error_pct": round(e_norm * 100.0, 1),
                "tolerance_pct": anchor_ratio * 100.0 if is_anchor else contour_ratio * 100.0,
                "max_px_allowed": round(tolerance * iod, 1),
                "is_anchor": is_anchor,
                "tolerance": tolerance,
                "exceeded": exceeded,
                "severity": sev,
                "suspicion": round(raw_suspicion, 3),
                "human_xy": (round(hx, 1), round(hy, 1)),
                "suggested_xy": (round(mx, 1), round(my, 1)),
            }

    if deviations:
        nme: float | None = sum(deviations) / len(deviations)
        passed_nme: bool | None = nme <= nme_threshold
        nme_pct: float | None = nme * 100.0
        e_max: float | None = float(max(deviations))
        import numpy as np
        e_p90: float | None = float(np.percentile(deviations, 90))
        exceed_count = sum(1 for p in point_evaluations.values() if p.get("exceeded"))
        has_outlier_point = any(p.get("severity") == "CRITICAL" for p in point_evaluations.values())
    else:
        nme = None
        passed_nme = None
        nme_pct = None
        e_max = None
        e_p90 = None
        exceed_count = 0
        has_outlier_point = False

    face_scale = "CLOSE" if iod >= close_min else ("NORMAL" if iod >= normal_min else "FAR")

    # Phát hiện "Ca khó / Góc khuất" (Hard Case / Occlusion Detection)
    eye_pts = set(range(14, 30))
    eye_occ_count = sum(
        1 for pid in eye_pts
        if (h := human_kpts.get(pid)) and h.get("state") in ("occluded", "outside")
    )
    total_occ_count = sum(
        1 for pid, h in human_kpts.items()
        if h and h.get("state") in ("occluded", "outside")
    )

    case_type = "normal"
    ai_reliability = normal_rel
    case_label = "Bình thường"
    case_description = "Ảnh rõ nét, đầy đủ thông tin mặt, AI tin cậy cao."

    if eye_occ_count >= sunglasses_min_occ:
        case_type = "sunglasses"
        ai_reliability = sunglasses_rel
        case_label = "🕶️ Ca khó (Kính râm / Che mắt)"
        case_description = (
            "Tài xế đeo kính râm, che khuất vùng mắt. Độ tin cậy Model AI giảm còn 45%. "
            "Ưu tiên kiểm tra trạng thái 'occluded' thay vì coi là sai lệch gán nhãn."
        )
    elif total_occ_count >= occluded_min_occ:
        case_type = "occluded"
        ai_reliability = occluded_rel
        case_label = "⚠️ Ca khó (Nhiều điểm bị che)"
        case_description = (
            f"Khuôn mặt có {total_occ_count} điểm bị che khuất (tóc/tay lái/vật cản). "
            "Dung sai giải phẫu được nới lỏng."
        )
    elif iod < far_max_iod:
        case_type = "far"
        ai_reliability = far_rel
        case_label = "🔍 Ca khó (Mặt nhỏ ở xa)"
        case_description = f"Khoảng cách chụp xa (IOD={iod:.1f}px < 35px), dung sai chuẩn hóa theo pixel rất hẹp."

    return {
        "iod": round(iod, 1),
        "face_scale": face_scale,
        "max_acceptable_px_anchor": round(0.03 * iod, 1),
        "max_acceptable_px_contour": round(0.05 * iod, 1),
        "scale_explanation": (
            f"Khuôn mặt {face_scale} (IOD={iod:.1f}px): "
            f"Dung sai điểm neo <= {0.03 * iod:.1f}px (3% IOD), "
            f"điểm viền <= {0.05 * iod:.1f}px (5% IOD)."
        ),
        "nme": round(nme, 4) if nme is not None else None,
        "nme_pct": round(nme_pct, 2) if nme_pct is not None else None,
        "passed_nme": passed_nme,
        "evaluated_points_count": len(deviations),
        "e_max": round(e_max, 4) if e_max is not None else None,
        "e_p90": round(e_p90, 4) if e_p90 is not None else None,
        "exceed_count": exceed_count,
        "has_outlier_point": has_outlier_point,
        "point_evaluations": point_evaluations,
        "case_type": case_type,
        "ai_reliability": round(ai_reliability, 2),
        "case_label": case_label,
        "case_description": case_description,
    }
