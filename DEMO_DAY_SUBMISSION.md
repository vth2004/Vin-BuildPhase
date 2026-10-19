# Nội dung nộp Demo Day — Đội 005 · WAR

Dùng file này để copy từng trường vào form. Các mục đánh dấu `[CẦN ĐIỀN]` là phần chỉ đội mới có (video/slide/ảnh).

---

## 1. Tên sản phẩm

```
Landmark QA/QC
```

## 2. Tóm tắt sản phẩm (1–2 câu, hiển thị trên thẻ Demo Day)

```
Landmark QA/QC là nền tảng web tự động kiểm định chất lượng nhãn keypoint, đối chiếu nhãn do người gán với dự đoán AI đa mô hình và các luật hình học để khoanh vùng điểm nghi lỗi cho reviewer. Hệ thống hỗ trợ Human Pose 17 điểm (COCO-17) và Face Landmark VF50 (50 điểm khuôn mặt), có review theo từng điểm, QA/QC theo chuỗi frame/video (Temporal) và xuất dữ liệu đã làm sạch.
```

---

## 3. Mô tả dự án (Markdown)

```markdown
## Bài toán

Dữ liệu huấn luyện model thị giác (pose toàn thân, landmark khuôn mặt) chất lượng cao phụ thuộc trực tiếp vào khâu gán nhãn thủ công. Quy trình hiện tại gặp các lỗi điển hình: tọa độ lệch, **hoán đổi trái/phải**, sai cấu trúc giải phẫu (mí mắt, môi, sống mũi), điểm vượt biên ảnh, và **trôi/giật nhãn giữa các frame liên tiếp**. Kiểm tra 100% bằng người vừa tốn thời gian vừa thiếu nhất quán, và càng khó khi dữ liệu là chuỗi video. Vấn đề không phải là kiểm tra mọi điểm, mà là **ưu tiên đúng những điểm đáng nghi** để reviewer ra quyết định nhanh và có căn cứ.

**Mục tiêu sản phẩm:** tự động phát hiện và xếp hạng ca nghi lỗi (không thay thế người kiểm định), giữ nguyên annotation gốc bất biến, hỗ trợ review từng điểm và xuất dữ liệu đã làm sạch cùng báo cáo kiểm toán.

## Giải pháp kỹ thuật

Kiến trúc: **FastAPI + SQLite** ở backend, **React/TypeScript (Vite)** ở frontend, chạy **CPU-only, offline** (không gửi ảnh lên cloud). Hai feature nghiệp vụ tách độc lập, dùng chung lõi (project/dataset/run, upload ZIP an toàn, hàng đợi job nền).

**1) Human Pose 17 điểm — ensemble đa mô hình.** Thay vì tin một model, hệ thống hợp nhất YOLO26s-pose và RTMPose-m (chế độ K2, tùy chọn K3 thêm YOLOv8m) thành vị trí đồng thuận `p*` theo trọng số nghịch bình phương sai số. Từ đó tính:
- độ bất đồng giữa các model `δ`,
- điểm nghi ngờ chuẩn COCO OKS `e`,
- độ tin cậy `R` (phạt khi model bỏ sót người),
- điểm xếp hạng `Score = e·R^α`, kèm **phát hiện swap trái/phải**.

Chỉ các điểm vượt ngưỡng mới vào hàng chờ review, giúp giảm mạnh báo động oan.

**2) Face Landmark VF50 — ánh xạ model + luật hình học.** MediaPipe Face Landmarker trả 478 điểm, được **resample theo chiều dài cung** về đúng 50 điểm VF50 (7 nhóm giải phẫu). Hệ thống ghép nhiều khuôn mặt bằng thuật toán **Hungarian** với ngưỡng cắt theo IOD, tính **NME chuẩn hóa**, các chỉ số cấp khuôn mặt (`e_max`, `e_p90`, `exceed_count`, `has_outlier_point` để tránh "bình quân che lấp lỗi nặng"), và **14 luật kiểm định R01–R14** (cấu trúc, đảo trái/phải, thứ tự contour, sống mũi, môi trong/ngoài, trùng tọa độ, nhảy frame...). IOD được tính một lần và giữ bất biến trong suốt vòng đời review để thang đo nhất quán.

**3) Temporal QA/QC (điểm khác biệt).** Ngoài kiểm tra từng ảnh, hệ thống phân tích **chuyển động qua nhiều frame**: tự suy sequence/thứ tự frame từ ZIP (hoặc nhận `sequence_manifest.json` khi tên file mơ hồ), nối detection giữa các frame thành track, rồi **hồi quy quỹ đạo cục bộ robust (Huber/IRLS, bậc 2 fallback bậc 1, leave-one-out)** để gợi ý vị trí keypoint. Từ đây phát hiện nhãn bị giật bất thường, và reviewer có thể **chấp nhận tọa độ Temporal theo từng điểm**; export chỉ áp dụng đúng quyết định đã review, giữ nguyên trạng thái `visible/occluded/outside`.

**4) Luồng AI draft cho dữ liệu chưa có nhãn.** Với ZIP/video không annotation, hệ thống chạy QC trên chính dự đoán AI, cảnh báo dự đoán bất ổn định và xuất `landmark-qa/v1` kèm sequence/track, provenance model và cờ `visibility_confirmed=false` (reviewer phải xác nhận trước khi dùng như ground truth).

**5) An toàn & bảo mật.** Mọi API thuộc project yêu cầu **recovery key**; giải nén ZIP có kiểm tra path traversal, symlink, đường dẫn trùng, số file và tổng dung lượng. **Annotation nguồn không bao giờ bị sửa tại chỗ** — chỉ thêm quyết định review và dựng file export từ nhãn gốc.

### Bằng chứng định lượng

Benchmark trên dữ liệu công khai (COCO val2017, CrowdPose, WFLW, 300W, 300VW), chia DEV/TEST theo ảnh, chọn ngưỡng trên DEV và khóa trước khi đánh giá TEST:

- **Pose (COCO TEST):** chế độ K2 so với K1 tăng chất lượng phát hiện lỗi rõ rệt — AUPRC 0.1895 → **0.2831**, precision 0.2436 → **0.3893**, FPR 0.0576 → **0.0272**; đổi lại độ trễ tăng (median ~86 ms → ~505 ms/ảnh trên CPU). Xu hướng K2 > K1 giữ nguyên trên CrowdPose (ngoài miền COCO).
- **Face (TEST, 1.300 face, 39.000 kịch bản lỗi):** các luật then chốt đạt F1 rất cao — R01, R10, R14 = 1.000; R13 = 0.993; R03 = 0.912; R11 = 0.936. **R13 trên video thật (300VW)** cho thấy FPR còn cao (10.57%), đã được ghi nhận là hạn chế thay vì tô hồng.

## Tính khả thi

- **Chi phí vận hành thấp:** inference **CPU-only**, không cần GPU, không cần cloud — phù hợp chạy nội bộ và bảo mật dữ liệu. Backend dùng SQLite (một file, dễ sao lưu).
- **Đã kiểm thử:** bộ test backend (unittest) + frontend build chạy trước mỗi lần bàn giao; migration schema có **backup timestamped** và chạy lại idempotent, không mất dữ liệu cũ.
- **Có cơ chế an toàn:** chế độ fallback khi model phụ lỗi (`K1_fallback`), chặn tạo run khi dataset chưa `ready`, import video chạy nền có **staging + quota** (mặc định 10.000 frame / 5 GB), lỗi thì cleanup và không publish dataset nửa vời.
- **Hạn chế được ghi nhận trung thực:** ngưỡng Temporal/scene-cut/gap chưa hiệu chỉnh; Face còn R02/R13 yếu; chưa có benchmark temporal có ground truth; job nền chưa bền qua restart (chỉ phát hiện+retry). Đây là các mục đã nằm trong roadmap, không phải phát hiện muộn.

## Hướng phát triển

1. **Đo trên lỗi thật:** thu thập tập lỗi gán nhãn thực tế được adjudicate độc lập, khóa làm TEST thực địa (thay vì chỉ lỗi tiêm nhân tạo).
2. **Benchmark Temporal có ground truth:** đo precision/recall/FPR của khôi phục điểm bị che, coverage, ID switch, thời gian/RAM — chỉ công bố hiệu quả khi có bằng chứng.
3. **Pilot downstream:** chứng minh dữ liệu sau làm sạch giúp cải thiện chất lượng model huấn luyện (AP/NME) so với nhãn thô.
4. **Mở rộng độ phủ:** bổ sung dataset occlusion/visibility thật (COFW-68, MERL-RAV); secondary-model bridge qua đoạn mất detection; re-identification; parser CVAT XML cho Pose.
5. **Hiệu chỉnh ngưỡng theo từng miền dữ liệu:** thay vì một ngưỡng chung, tinh chỉnh theo ánh sáng, góc máy, mật độ người và đặc thù từng bộ dữ liệu.
```

---

## 4. Video giới thiệu sản phẩm

```
[CẦN ĐIỀN] — dán link YouTube (unlisted) hoặc Google Drive
```

Gợi ý kịch bản ~3 phút:
1. **(0:00–0:20)** Vấn đề: nhãn keypoint thủ công có lỗi lệch tọa độ, swap trái/phải, trôi nhãn.
2. **(0:20–0:50)** Mở project → upload ZIP Pose + annotation → chạy QA → overlay Human/Model và hàng chờ cảnh báo.
3. **(0:50–1:20)** Review một ca swap: chọn "Dùng điểm Model", xuất annotation sạch.
4. **(1:20–2:00)** Face VF50: 14 luật, NME, chọn nguồn từng điểm, batch fix có preview.
5. **(2:00–2:40)** Temporal: upload ZIP/video chuỗi frame → xác nhận sequence → bật Temporal QA → quỹ đạo + gợi ý điểm → chấp nhận điểm Temporal → export + audit JSON.
6. **(2:40–3:00)** Kết quả benchmark (K2 cải thiện precision/FPR) và hướng phát triển.

## 5. Slide thuyết trình

```
[CẦN ĐIỀN] — dán link Google Slides (chia sẻ công khai) hoặc file PDF
```

Bản 5 slide đã dựng sẵn ở `reports/assets/demo_slides/` (kèm `LandmarkQA_QC_slides.pdf` + PNG từng slide 1920×1080): Bìa → Bài toán → Giải pháp → Kết quả → Đã dùng được & hướng phát triển.

## 6. Ảnh đại diện (thumbnail Demo Day)

```
[CẦN ĐIỀN] — link ảnh hoặc Google Drive
```

Bản đã tạo sẵn: `reports/assets/demo_day_thumbnail.png` (1920×1080), kèm bản 1280×720 và 1200×630. Chỉ cần upload lên Drive rồi dán link.
