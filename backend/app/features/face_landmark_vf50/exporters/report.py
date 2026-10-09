from __future__ import annotations

import csv
import io
from typing import Any


def generate_markdown_checklist(
    dataset_name: str,
    frames_report: list[dict[str, Any]],
) -> str:
    """
    Tạo báo cáo Checklist dạng Markdown chi tiết cho học viên mở ra đối chiếu sửa tay trên CVAT.
    """
    md = []
    md.append(f"# BẢNG CHECKLIST SỬA TAY TRÊN CVAT — TASK {dataset_name}\n")
    md.append("> **Căn cứ:** Guideline Face Landmark VF-50 v1.3 (Ban Tổ Chức VinFast).\n")
    md.append("> **Nguyên tắc:** Mở từng frame trên CVAT, kiểm tra và sửa tọa độ/trạng thái theo các mục liệt kê bên dưới.\n\n")

    total_frames = len(frames_report)
    error_frames = [f for f in frames_report if f.get("errors_count", 0) > 0]
    warn_frames = [f for f in frames_report if f.get("errors_count", 0) == 0 and f.get("warnings_count", 0) > 0]

    md.append("## 1. TỔNG QUAN TÌNH TRẠNG TASK\n")
    md.append(f"- **Tổng số frame đã quét:** {total_frames}\n")
    md.append(f"- **Số frame có LỖI NGHIÊM TRỌNG (Cần sửa ngay):** {len(error_frames)}\n")
    md.append(f"- **Số frame có CẢNH BÁO (Cần rà soát):** {len(warn_frames)}\n")
    md.append(f"- **Số frame ĐẠT CHUẨN:** {total_frames - len(error_frames) - len(warn_frames)}\n\n")

    md.append("## 2. DANH SÁCH CHI TIẾT CÁC FRAME CẦN SỬA (ƯU TIÊN THEO ĐỘ LỆCH)\n\n")

    for idx, f in enumerate(frames_report, start=1):
        fn = f["file_name"]
        violations = f.get("violations", [])
        quality = f.get("quality", {})
        nme = quality.get("nme", 0.0)
        passed_nme = quality.get("passed_nme", True)
        nme_badge = "ĐẠT (<= 0.035)" if passed_nme else "**VƯỢT NGƯỠNG (> 0.035)**"

        if not violations and passed_nme:
            continue

        md.append(f"### #{idx}. Frame: `{fn}`\n")
        md.append(f"- **Chỉ số NME toàn ảnh:** {nme} ({nme_badge}) | **IOD:** {quality.get('iod', 96)} px\n")

        if violations:
            md.append("- **Các vi phạm luật Guideline:**\n")
            for v in violations:
                level_icon = "🔴 [LỖI]" if v["level"] == "ERROR" else "🟡 [CẢNH BÁO]"
                pts_str = f"(Điểm {v['points']})" if v.get("points") else ""
                md.append(f"  - {level_icon} **{v['code']} - {v['rule_name']}**: {v['message']} {pts_str} — *{v['gl_section']}*\n")

        # Các điểm lệch vượt ngưỡng so với model
        exceeded_pts = [
            p for p in quality.get("point_evaluations", {}).values() if p.get("exceeded")
        ]
        if exceeded_pts:
            md.append("- **Gợi ý điểm lệch tọa độ so với giải phẫu (> 3% hoặc > 5% IOD):**\n")
            for p in exceeded_pts[:8]:
                anchor_tag = "[Điểm neo]" if p.get("is_anchor") else "[Contour]"
                md.append(
                    f"  - Điểm `{p['id']}` {anchor_tag}: Lệch {p['dist_px']} px ({p['norm_error']*100:.1f}% IOD). "
                    f"Người gán: {p['human_xy']} -> Gợi ý tham chiếu: {p['suggested_xy']}\n"
                )

        md.append("\n---\n\n")

    return "".join(md)


def generate_csv_report(frames_report: list[dict[str, Any]]) -> str:
    """Tạo bảng CSV tóm tắt để import vào Google Sheets chia việc cho nhóm."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Tên Frame", "Số Lỗi", "Số Cảnh báo", "NME", "Đạt chuẩn NME",
        "Danh sách mã vi phạm", "Điểm lệch nặng nhất", "Trạng thái Review",
    ])

    for f in frames_report:
        fn = f["file_name"]
        violations = f.get("violations", [])
        err_c = sum(1 for v in violations if v["level"] == "ERROR")
        warn_c = sum(1 for v in violations if v["level"] == "WARNING")
        q = f.get("quality", {})
        nme = q.get("nme", 0.0)
        passed = "ĐẠT" if q.get("passed_nme", True) else "VƯỢT NGƯỠNG"
        v_codes = "; ".join(v["code"] for v in violations)

        exceeded_pts = [
            str(p["id"]) for p in q.get("point_evaluations", {}).values() if p.get("exceeded")
        ]
        exceeded_str = ", ".join(exceeded_pts[:5])

        writer.writerow([
            fn, err_c, warn_c, nme, passed, v_codes, exceeded_str, "Chưa sửa",
        ])

    return output.getvalue()
