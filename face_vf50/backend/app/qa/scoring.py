from __future__ import annotations

import math
from typing import Any

try:
    from app.qa.geometry import compute_iod
except ImportError:
    from .geometry import compute_iod


ANCHOR_POINTS = {0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36}


def evaluate_frame_quality(
    human_kpts: dict[int, dict[str, Any]],
    model_kpts: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Tính toán sai số chuẩn hóa theo IOD và NME cho một frame ảnh.
    Theo Mục 8 GL:
      - 12 Điểm neo: dung sai <= 3% IOD (~3 px)
      - Điểm contour còn lại: dung sai <= 5% IOD (~5 px)
      - NME toàn ảnh: <= 0.035 (~3.4 px)
    """
    iod = compute_iod(human_kpts)
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
            e_norm = dist_px / max(iod, 10.0)
            deviations.append(e_norm)

            is_anchor = pt_id in ANCHOR_POINTS
            tolerance = 0.03 if is_anchor else 0.05
            exceeded = e_norm > tolerance

            # Phân loại mức độ nghiêm trọng theo tỉ lệ IOD
            if not exceeded:
                sev = "NORMAL"
            elif e_norm <= tolerance * 1.8:
                sev = "WARNING"
            else:
                sev = "CRITICAL"

            # Tính độ nghi ngờ chuẩn hóa [0.0, 1.0]
            # Điểm Occluded đòi hỏi ước lượng giải phẫu, hạ trọng số phạt 40%
            raw_suspicion = min(1.0, e_norm / (tolerance * 2.5))
            if h.get("state") == "occluded":
                raw_suspicion *= 0.6

            point_evaluations[pt_id] = {
                "id": pt_id,
                "dist_px": round(dist_px, 1),
                "norm_error": round(e_norm, 4),
                "norm_error_pct": round(e_norm * 100.0, 1),
                "tolerance_pct": 3.0 if is_anchor else 5.0,
                "max_px_allowed": round(tolerance * iod, 1),
                "is_anchor": is_anchor,
                "tolerance": tolerance,
                "exceeded": exceeded,
                "severity": sev,
                "suspicion": round(raw_suspicion, 3),
                "human_xy": (round(hx, 1), round(hy, 1)),
                "suggested_xy": (round(mx, 1), round(my, 1)),
            }

    nme = (sum(deviations) / len(deviations)) if deviations else 0.0
    passed_nme = nme <= 0.035 if deviations else True

    face_scale = "CLOSE" if iod >= 80.0 else ("NORMAL" if iod >= 38.0 else "FAR")

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
    ai_reliability = 0.95
    case_label = "Bình thường"
    case_description = "Ảnh rõ nét, đầy đủ thông tin mặt, AI tin cậy cao."

    if eye_occ_count >= 4:
        case_type = "sunglasses"
        ai_reliability = 0.45
        case_label = "🕶️ Ca khó (Kính râm / Che mắt)"
        case_description = (
            "Tài xế đeo kính râm, che khuất vùng mắt. Độ tin cậy Model AI giảm còn 45%. "
            "Ưu tiên kiểm tra trạng thái 'occluded' thay vì coi là sai lệch gán nhãn."
        )
    elif total_occ_count >= 6:
        case_type = "occluded"
        ai_reliability = 0.58
        case_label = "⚠️ Ca khó (Nhiều điểm bị che)"
        case_description = (
            f"Khuôn mặt có {total_occ_count} điểm bị che khuất (tóc/tay lái/vật cản). "
            "Dung sai giải phẫu được nới lỏng."
        )
    elif iod < 35.0:
        case_type = "far"
        ai_reliability = 0.65
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
        "nme": round(nme, 4),
        "nme_pct": round(nme * 100.0, 2),
        "passed_nme": passed_nme,
        "evaluated_points_count": len(deviations),
        "point_evaluations": point_evaluations,
        "case_type": case_type,
        "ai_reliability": round(ai_reliability, 2),
        "case_label": case_label,
        "case_description": case_description,
    }
