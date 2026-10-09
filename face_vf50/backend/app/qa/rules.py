from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from app.qa.geometry import (
        compute_centroid,
        compute_head_tilt_angle_rad,
        compute_iod,
        euclidean_dist,
        has_self_intersection,
        point_in_polygon,
        rotate_point,
    )
except ImportError:
    from .geometry import (
        compute_centroid,
        compute_head_tilt_angle_rad,
        compute_iod,
        euclidean_dist,
        has_self_intersection,
        point_in_polygon,
        rotate_point,
    )


@dataclass
class RuleViolation:
    code: str
    rule_name: str
    level: str  # "ERROR" hoặc "WARNING"
    message: str
    points: list[int]
    gl_section: str


def check_all_rules(
    keypoints_by_id: dict[int, dict[str, Any]],
    prev_keypoints_by_id: dict[int, dict[str, Any]] | None = None,
    model_keypoints_by_id: dict[int, dict[str, Any]] | None = None,
    image_width: int = 1280,
    image_height: int = 720,
) -> list[RuleViolation]:
    """
    Chạy toàn bộ 14 bộ luật (R01 - R14) kiểm tra nhãn Face VF-50 theo đúng Guideline BTC v1.3.
    """
    violations: list[RuleViolation] = []

    # Chuẩn hóa khoảng cách hình học theo IOD (Scale-Invariant Factor)
    # Tránh nghịch lý: vật gần lệch 10px thì chấp nhận được, vật xa lệch 10px là lỗi nặng
    iod_val = compute_iod(keypoints_by_id)
    scale_factor = max(0.25, min(2.5, iod_val / 96.0))

    # =========================================================================
    # R01: ĐẦY ĐỦ CẤU TRÚC 50 ĐIỂM (0 đến 49)
    # =========================================================================
    missing_ids = [i for i in range(50) if i not in keypoints_by_id]
    if missing_ids:
        violations.append(
            RuleViolation(
                code="R01",
                rule_name="Đầy đủ cấu trúc",
                level="ERROR",
                message=f"Thiếu {len(missing_ids)} điểm keypoint: {missing_ids[:5]}...",
                points=missing_ids,
                gl_section="Mục 2.2, 7.1",
            )
        )

    # =========================================================================
    # R02: CẢNH BÁO QUÊN SỬA TRẠNG THÁI (CẢ 50 ĐIỂM ĐỀU LÀ VISIBLE)
    # Theo Mục 1.1 (#2): Máy đặt sẵn cả 50 điểm ở trạng thái Visible cho mọi ảnh.
    # =========================================================================
    visible_count = sum(
        1 for kp in keypoints_by_id.values() if kp.get("state") == "visible" or kp.get("state") is None
    )
    if visible_count == 50 and len(keypoints_by_id) == 50:
        violations.append(
            RuleViolation(
                code="R02",
                rule_name="Quên sửa trạng thái",
                level="WARNING",
                message="Tất cả 50 điểm đều ở trạng thái Visible. Nhóm cần rà soát lại các điểm bị che (Occluded) hoặc ngoài khung (Outside).",
                points=[],
                gl_section="Mục 1.1 (#2), Mục 9",
            )
        )

    # =========================================================================
    # R03: ĐẢO TRÁI - PHẢI THEO KHUNG NHÌN ẢNH
    # Theo Mục 2.1: longmaytrai và mattrai nằm về phía trái của khuôn mặt (x nhỏ hơn).
    # =========================================================================
    l_brow_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(0, 5)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    r_brow_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(5, 10)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    l_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(14, 22)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    r_eye_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(22, 30)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]

    if l_brow_pts and r_brow_pts:
        c_lb = compute_centroid(l_brow_pts)
        c_rb = compute_centroid(r_brow_pts)
        if c_lb[0] >= c_rb[0]:
            violations.append(
                RuleViolation(
                    code="R03",
                    rule_name="Đảo Trái - Phải Lông mày",
                    level="ERROR",
                    message="longmaytrai nằm ở bên phải của longmayphai trên ảnh. Quy ước VinFast bắt buộc trái/phải theo khung nhìn ảnh.",
                    points=list(range(0, 10)),
                    gl_section="Mục 2.1, Mục 9",
                )
            )

    if l_eye_pts and r_eye_pts:
        c_le = compute_centroid(l_eye_pts)
        c_re = compute_centroid(r_eye_pts)
        if c_le[0] >= c_re[0]:
            violations.append(
                RuleViolation(
                    code="R03",
                    rule_name="Đảo Trái - Phải Mắt",
                    level="ERROR",
                    message="mattrai nằm ở bên phải của matphai trên ảnh. Quy ước VinFast bắt buộc trái/phải theo khung nhìn ảnh.",
                    points=list(range(14, 30)),
                    gl_section="Mục 2.1, Mục 9",
                )
            )

    # =========================================================================
    # R04: THỨ TỰ TỌA ĐỘ X LÔNG MÀY TĂNG DẦN (MẶT GẦN CHÍNH DIỆN)
    # Theo Mục 3.1: Toạ độ x của 10 điểm lông mày 0 đến 9 tăng dần liên tục.
    # =========================================================================
    brow_xs = [
        (i, float(keypoints_by_id[i]["x"]))
        for i in range(10)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    if len(brow_xs) >= 8:
        # Kiểm tra xem có cặp nào x phía sau nhỏ hơn x phía trước quá dung sai không (thích ứng theo IOD)
        tol_brow_x = max(1.5, 5.0 * scale_factor)
        inversions = []
        for idx in range(len(brow_xs) - 1):
            cur_id, cur_x = brow_xs[idx]
            next_id, next_x = brow_xs[idx + 1]
            if next_x < cur_x - tol_brow_x:
                inversions.append((cur_id, next_id))
        if inversions:
            bad_pts = list({p for pair in inversions for p in pair})
            violations.append(
                RuleViolation(
                    code="R04",
                    rule_name="Thứ tự x Lông mày",
                    level="ERROR",
                    message=f"Tọa độ x của lông mày không tăng dần từ 0 đến 9 (nghi nhầm thứ tự điểm hoặc nhầm nhóm): cặp {inversions}.",
                    points=bad_pts,
                    gl_section="Mục 3.1, 5.2",
                )
            )

    # =========================================================================
    # R05: KHÓE MẮT CỰC TRỊ TRÊN TRỤC X
    # Theo Mục 3.3: mattrai điểm 14 có x nhỏ nhất, 18 có x lớn nhất.
    #               matphai điểm 22 có x nhỏ nhất, 26 có x lớn nhất.
    # =========================================================================
    for eye_name, start_id, end_id, min_id, max_id in [
        ("mattrai", 14, 21, 14, 18),
        ("matphai", 22, 29, 22, 26),
    ]:
        eye_pts_dict = {
            i: float(keypoints_by_id[i]["x"])
            for i in range(start_id, end_id + 1)
            if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
        }
        if len(eye_pts_dict) == 8:
            actual_min_id = min(eye_pts_dict, key=eye_pts_dict.get)
            actual_max_id = max(eye_pts_dict, key=eye_pts_dict.get)
            if actual_min_id != min_id:
                violations.append(
                    RuleViolation(
                        code="R05",
                        rule_name=f"Khóe mắt cực trị {eye_name}",
                        level="ERROR",
                        message=f"Điểm {min_id} của {eye_name} phải có tọa độ x nhỏ nhất (khóe trái nhất), nhưng điểm {actual_min_id} lại nhỏ hơn.",
                        points=[min_id, actual_min_id],
                        gl_section="Mục 3.3",
                    )
                )
            if actual_max_id != max_id:
                violations.append(
                    RuleViolation(
                        code="R05",
                        rule_name=f"Khóe mắt cực trị {eye_name}",
                        level="ERROR",
                        message=f"Điểm {max_id} của {eye_name} phải có tọa độ x lớn nhất (khóe phải nhất), nhưng điểm {actual_max_id} lại lớn hơn.",
                        points=[max_id, actual_max_id],
                        gl_section="Mục 3.3",
                    )
                )

    # =========================================================================
    # R06: MÍ TRÊN PHẢI CAO HƠN HOẶC BẰNG MÍ DƯỚI (ĐÃ BÙ GÓC NGHIÊNG ĐẦU)
    # Theo Mục 3.3: Các cặp mí trên/dưới đối diện (15-21, 16-20, 17-19) và (23-29, 24-28, 25-27).
    # Trong tọa độ pixel ảnh, "cao hơn" nghĩa là tọa độ y nhỏ hơn.
    # =========================================================================
    head_angle = compute_head_tilt_angle_rad(keypoints_by_id)
    face_center = compute_centroid(l_eye_pts + r_eye_pts) if (l_eye_pts and r_eye_pts) else (640.0, 360.0)

    eyelid_pairs = [(15, 21), (16, 20), (17, 19), (23, 29), (24, 28), (25, 27)]
    for top_id, btm_id in eyelid_pairs:
        if (
            top_id in keypoints_by_id
            and btm_id in keypoints_by_id
            and keypoints_by_id[top_id].get("x") is not None
            and keypoints_by_id[btm_id].get("x") is not None
            and keypoints_by_id[top_id].get("state") != "outside"
            and keypoints_by_id[btm_id].get("state") != "outside"
        ):
            pt_top = (float(keypoints_by_id[top_id]["x"]), float(keypoints_by_id[top_id]["y"]))
            pt_btm = (float(keypoints_by_id[btm_id]["x"]), float(keypoints_by_id[btm_id]["y"]))

            # Xoay về hệ tọa độ thẳng đứng của mặt
            rot_top = rotate_point(pt_top, face_center, head_angle)
            rot_btm = rotate_point(pt_btm, face_center, head_angle)

            # Mí trên phải cao hơn mí dưới (rot_top[1] <= rot_btm[1] + dung sai theo IOD)
            tol_eyelid = max(0.8, 2.0 * scale_factor)
            if rot_top[1] > rot_btm[1] + tol_eyelid:
                violations.append(
                    RuleViolation(
                        code="R06",
                        rule_name="Mí trên thấp hơn mí dưới",
                        level="ERROR",
                        message=f"Mí trên (điểm {top_id}) bị lộn xuống dưới mí dưới (điểm {btm_id}) sau khi bù góc nghiêng đầu.",
                        points=[top_id, btm_id],
                        gl_section="Mục 3.3, 7.1",
                    )
                )

    # =========================================================================
    # R07: CONTOUR MẮT KHÔNG TỰ CẮT (HÌNH CHỮ X)
    # Theo Mục 3.3 & 9: Hai mắt tạo thành vòng kín, không có đường bắt chéo.
    # =========================================================================
    for eye_name, start_id, end_id in [("mattrai", 14, 21), ("matphai", 22, 29)]:
        poly = [
            (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
            for i in range(start_id, end_id + 1)
            if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
        ]
        if len(poly) == 8 and has_self_intersection(poly):
            violations.append(
                RuleViolation(
                    code="R07",
                    rule_name=f"Mắt bị bắt chéo (chữ X) {eye_name}",
                    level="ERROR",
                    message=f"Đa giác {eye_name} bị tự cắt tạo thành hình chữ X. Kiểm tra lại thứ tự nối điểm quanh mắt.",
                    points=list(range(start_id, end_id + 1)),
                    gl_section="Mục 3.3, 5.2, Mục 9",
                )
            )

    # =========================================================================
    # R08: TỈ LỆ 3 ĐOẠN SỐNG MŨI
    # Theo Mục 3.2: 4 điểm chia đều dọc sống mũi. Ba đoạn 10-11, 11-12, 12-13 gần bằng nhau.
    # Điểm 13 là chân sống mũi, không phải chóp mũi.
    # =========================================================================
    nose_pts = [
        keypoints_by_id.get(i)
        for i in range(10, 14)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    if len(nose_pts) == 4:
        p10 = (float(keypoints_by_id[10]["x"]), float(keypoints_by_id[10]["y"]))
        p11 = (float(keypoints_by_id[11]["x"]), float(keypoints_by_id[11]["y"]))
        p12 = (float(keypoints_by_id[12]["x"]), float(keypoints_by_id[12]["y"]))
        p13 = (float(keypoints_by_id[13]["x"]), float(keypoints_by_id[13]["y"]))

        seg1 = euclidean_dist(p10, p11)
        seg2 = euclidean_dist(p11, p12)
        seg3 = euclidean_dist(p12, p13)

        segs = [seg1, seg2, seg3]
        if min(segs) > 1.0:
            ratio = max(segs) / min(segs)
            if ratio > 1.55:
                violations.append(
                    RuleViolation(
                        code="R08",
                        rule_name="Sống mũi chia không đều",
                        level="WARNING",
                        message=f"3 đoạn sống mũi chênh lệch lớn (tỉ lệ max/min = {ratio:.2f} > 1.55). Lưu ý điểm 13 là chân sống mũi, không kéo xuống chóp mũi.",
                        points=[10, 11, 12, 13],
                        gl_section="Mục 3.2, 7.1",
                    )
                )

    # =========================================================================
    # R09: MÔI TRONG NẰM HOÀN TOÀN TRONG MÔI NGOÀI
    # Theo Mục 3.4: moitrong luôn nằm hoàn toàn bên trong moingoai; x42 > x30 và x46 < x36.
    # =========================================================================
    p30 = keypoints_by_id.get(30)
    p36 = keypoints_by_id.get(36)
    p42 = keypoints_by_id.get(42)
    p46 = keypoints_by_id.get(46)

    if (
        p30 and p36 and p42 and p46
        and p30.get("x") is not None and p36.get("x") is not None
        and p42.get("x") is not None and p46.get("x") is not None
        and p30.get("state") != "outside" and p36.get("state") != "outside"
        and p42.get("state") != "outside" and p46.get("state") != "outside"
    ):
        tol_lip = max(0.8, 2.0 * scale_factor)
        if float(p42["x"]) < float(p30["x"]) - tol_lip:
            violations.append(
                RuleViolation(
                    code="R09",
                    rule_name="Khóe môi trong lệch ngoài",
                    level="ERROR",
                    message="Khoé trong trái (điểm 42) lại nằm ngoài khoé miệng ngoài trái (điểm 30) trên trục x.",
                    points=[30, 42],
                    gl_section="Mục 3.4",
                )
            )
        if float(p46["x"]) > float(p36["x"]) + tol_lip:
            violations.append(
                RuleViolation(
                    code="R09",
                    rule_name="Khóe môi trong lệch ngoài",
                    level="ERROR",
                    message="Khoé trong phải (điểm 46) lại nằm ngoài khoé miệng ngoài phải (điểm 36) trên trục x.",
                    points=[36, 46],
                    gl_section="Mục 3.4",
                )
            )

    # Kiểm tra trọng tâm môi trong có nằm trong đa giác môi ngoài không
    outer_lip_poly = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(30, 42)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    inner_lip_pts = [
        (float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(42, 50)
        if i in keypoints_by_id and keypoints_by_id[i].get("x") is not None and keypoints_by_id[i].get("state") != "outside"
    ]
    if len(outer_lip_poly) >= 8 and len(inner_lip_pts) >= 4:
        c_inner = compute_centroid(inner_lip_pts)
        if not point_in_polygon(c_inner, outer_lip_poly):
            violations.append(
                RuleViolation(
                    code="R09",
                    rule_name="Môi trong rơi ra ngoài môi ngoài",
                    level="ERROR",
                    message="Tâm cụm điểm moitrong rơi ra ngoài đa giác viền moingoai.",
                    points=list(range(30, 50)),
                    gl_section="Mục 3.4, 7.1",
                )
            )

    # =========================================================================
    # R10: NGƯỠNG QUYẾT ĐỊNH CHO CẢ SKELETON (MỤC 4.2)
    # Nếu thấy ít hơn 4/8 điểm mắt -> phải Outside cả mắt.
    # Nếu thấy ít hơn 3/5 điểm lông mày -> phải Outside cả lông mày.
    # =========================================================================
    for skel_name, start_id, end_id, threshold, total_pts in [
        ("mattrai", 14, 21, 4, 8),
        ("matphai", 22, 29, 4, 8),
        ("longmaytrai", 0, 4, 3, 5),
        ("longmayphai", 5, 9, 3, 5),
    ]:
        vis_pts = [
            i for i in range(start_id, end_id + 1)
            if i in keypoints_by_id and keypoints_by_id[i].get("state") == "visible"
        ]
        out_pts = [
            i for i in range(start_id, end_id + 1)
            if i in keypoints_by_id and keypoints_by_id[i].get("state") == "outside"
        ]
        # Nếu số điểm nhìn thấy < threshold nhưng không đánh Outside cả bộ
        if 0 < len(vis_pts) < threshold and len(out_pts) < total_pts:
            violations.append(
                RuleViolation(
                    code="R10",
                    rule_name=f"Ngưỡng ẩn cả bộ phận {skel_name}",
                    level="WARNING",
                    message=f"{skel_name} chỉ còn nhìn thấy {len(vis_pts)}/{total_pts} điểm (dưới ngưỡng {threshold}). Theo Mục 4.2 GL: phải Outside toàn bộ skeleton này.",
                    points=list(range(start_id, end_id + 1)),
                    gl_section="Mục 4.2",
                )
            )

    # =========================================================================
    # R11: ĐIỂM RƠI RA NGOÀI BIÊN ẢNH MÀ KHÔNG PHẢI OUTSIDE
    # Theo Mục 4.1 & 6.10: Điểm rơi ra ngoài biên ảnh bắt buộc đánh Outside.
    # =========================================================================
    oob_pts = []
    for pt_id, kp in keypoints_by_id.items():
        if kp.get("state") != "outside":
            x, y = kp.get("x"), kp.get("y")
            if x is not None and y is not None:
                if x < 0 or x > image_width or y < 0 or y > image_height:
                    oob_pts.append(pt_id)
    if oob_pts:
        violations.append(
            RuleViolation(
                code="R11",
                rule_name="Điểm ngoài biên ảnh",
                level="ERROR",
                message=f"Các điểm {oob_pts} rơi ra ngoài khung hình {image_width}x{image_height} nhưng không đặt trạng thái Outside.",
                points=oob_pts,
                gl_section="Mục 4.1, 6.10",
            )
        )

    # =========================================================================
    # R12: HAI ĐIỂM KHÁC NHAU TRÙNG TỌA ĐỘ KHI CÙNG VISIBLE
    # Theo Mục 6.2: Hai khớp khác nhau không thể trùng toạ độ khi cả hai đều Visible.
    # (Ngoại trừ khi nhắm mắt thì mí trên mí dưới được trùng).
    # =========================================================================
    allowed_coincident_pairs = {
        (15, 21), (16, 20), (17, 19),
        (23, 29), (24, 28), (25, 27),
        (43, 49), (44, 48), (45, 47),
    }
    visible_pts = [
        (i, float(keypoints_by_id[i]["x"]), float(keypoints_by_id[i]["y"]))
        for i in range(50)
        if i in keypoints_by_id
        and keypoints_by_id[i].get("x") is not None
        and (keypoints_by_id[i].get("state") == "visible" or keypoints_by_id[i].get("state") is None)
    ]
    iod_for_coincident = compute_iod(keypoints_by_id)
    coincident_threshold = max(0.4, 0.012 * iod_for_coincident)
    coincident_found = []
    for i in range(len(visible_pts)):
        for j in range(i + 1, len(visible_pts)):
            id1, x1, y1 = visible_pts[i]
            id2, x2, y2 = visible_pts[j]
            pair_key = (min(id1, id2), max(id1, id2))
            if pair_key not in allowed_coincident_pairs:
                if math.hypot(x1 - x2, y1 - y2) < coincident_threshold:
                    coincident_found.append(pair_key)
    if coincident_found:
        bad_pts = list({p for pair in coincident_found for p in pair})
        violations.append(
            RuleViolation(
                code="R12",
                rule_name="Trùng tọa độ bất thường",
                level="WARNING",
                message=f"Các cặp điểm {coincident_found} trùng khít tọa độ nhau khi cùng Visible.",
                points=bad_pts,
                gl_section="Mục 6.2, 7.1",
            )
        )

    # =========================================================================
    # R13: NHÃN NHẢY BẤT THƯỜNG (>15px) GIỮA 2 FRAME LIÊN TIẾP
    # Theo Mục 6.9: Độ dịch chuyển trung bình là 2.7px, phân vị 90 là 7.7px.
    # Nếu nhãn nhảy > 15px giữa hai frame liền kề -> Cảnh báo.
    # =========================================================================
    if prev_keypoints_by_id:
        jumping_pts = []
        for i in range(50):
            cur_p = keypoints_by_id.get(i)
            prev_p = prev_keypoints_by_id.get(i)
            if (
                cur_p and prev_p
                and cur_p.get("x") is not None and prev_p.get("x") is not None
                and cur_p.get("state") != "outside" and prev_p.get("state") != "outside"
            ):
                dist_jump = math.hypot(
                    float(cur_p["x"]) - float(prev_p["x"]),
                    float(cur_p["y"]) - float(prev_p["y"]),
                )
                tol_jump = max(5.0, 15.0 * scale_factor)
                if dist_jump > tol_jump:
                    jumping_pts.append((i, round(dist_jump, 1)))
        if jumping_pts:
            violations.append(
                RuleViolation(
                    code="R13",
                    rule_name="Trôi điểm frame liền kề",
                    level="WARNING",
                    message=f"Các điểm {jumping_pts} nhảy đột ngột > {tol_jump:.1f}px (ngưỡng 15% IOD) so với frame trước đó.",
                    points=[p[0] for p in jumping_pts],
                    gl_section="Mục 6.9, 7.2",
                )
            )

    # =========================================================================
    # R14: ĐIỂM OCCLUDED GIỮ NGUYÊN TỌA ĐỘ SAI CỦA PRE-LABEL
    # Theo Mục 6.1 & 7.1: Điểm Occluded đòi hỏi toạ độ ước lượng hợp lý,
    # không được giữ nguyên vị trí sai của pre-label rồi đánh Occluded.
    # =========================================================================
    if model_keypoints_by_id:
        iod_val = compute_iod(keypoints_by_id)
        bad_occluded = []
        for i in range(50):
            kp = keypoints_by_id.get(i)
            mkp = model_keypoints_by_id.get(i)
            if (
                kp and mkp
                and kp.get("state") == "occluded"
                and kp.get("x") is not None and mkp.get("x") is not None
            ):
                dist_to_m = math.hypot(
                    float(kp["x"]) - float(mkp["x"]),
                    float(kp["y"]) - float(mkp["y"]),
                )
                # Nếu lệch hơn 15% IOD (~15px) so với vị trí model gợi ý
                if dist_to_m > 0.15 * iod_val:
                    bad_occluded.append(i)
        if bad_occluded:
            violations.append(
                RuleViolation(
                    code="R14",
                    rule_name="Occluded giữ tọa độ sai",
                    level="WARNING",
                    message=f"Các điểm Occluded {bad_occluded} lệch rất xa vị trí giải phẫu dự đoán (>15% IOD). Cần ước lượng lại vị trí hợp lý.",
                    points=bad_occluded,
                    gl_section="Mục 4.1, 6.1, 7.1",
                )
            )

    return violations
