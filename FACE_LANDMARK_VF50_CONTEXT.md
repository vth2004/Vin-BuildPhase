# FACE LANDMARK VF-50 CONTEXT & ARCHITECTURE SPECIFICATION

> **Feature:** VinFast Driver Monitoring System — Face Landmark 50 Keypoints QA (`vf_face_landmark50_v1`)  
> **Căn cứ nghiệp vụ:** Guideline Kiểm định Dữ liệu Face Landmark VF-50 v1.3 (Ban Tổ Chức VinFast).

---

## 1. Tổng quan Feature Face Landmark VF-50
Hệ thống **Face Landmark VF-50 QA** tự động thẩm định chất lượng gán nhãn 50 điểm mốc khuôn mặt của tài xế trên ảnh cabin xe, phát hiện các lỗi sai lệch tọa độ, đảo ngược đối xứng, sai cấu trúc giải phẫu và trôi nhãn video.

- **Tiêu chuẩn nhãn:** 50 keypoints (ID 0–49) chia thành 7 nhóm giải phẫu:
  - Lông mày trái (0–4), Lông mày phải (5–9)
  - Sống mũi (10–13)
  - Mắt trái (14–21), Mắt phải (22–29)
  - Môi ngoài (30–41), Môi trong (42–49)
- **Engine suy luận:** Google MediaPipe Face Landmarker (CPU Delegate, ~15–20ms/frame, 478 landmarks 3D dense map về 50 điểm VinFast qua spline curve resampling).
- **Run Mode Dispatch:** Mode `face_mediapipe_vf50` được đăng ký trong registry `app/shared/jobs/runner.py` và dispatch trực tiếp tới `app/features/face_landmark_vf50/services/run.py:execute_face_run`.

---

## 2. Nguồn Sự Thật Duy Nhất Cho Ngưỡng (Single Source of Truth)
Toàn bộ ngưỡng kiểm định và chấm điểm được tập trung trong file cấu hình:
- **Tệp cấu hình:** `backend/configs/face_landmark_vf50/vf50_spec.yaml`
- **Module nạp (có cache):** `backend/app/features/face_landmark_vf50/spec.py`
  - `load_vf50_spec()`: nạp YAML có `@lru_cache(maxsize=1)`
  - `get_anchor_points()`: 12 điểm neo quan trọng (0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36)
  - `get_scoring_tolerances()`: dung sai IOD (neo 3%, viền 5%, NME tối đa 3.5%, critical multiplier 1.8)
  - `get_case_classification_spec()`: phân loại ca khó (sunglasses, occluded, far, close)
  - `get_rules_spec()`: toàn bộ tham số của 14 luật R01–R14 (R08 = 1.55; R09 per-point tol; R13 jump)
  - `get_hungarian_matching_spec()`: ngưỡng cắt ghép cặp centroid (2.0 * IOD, fallback 150px)

---

## 3. Nguồn Tính IOD & Chuỗi Dự Phòng (IOD Source Priority)
Khoảng cách giữa hai đồng tử (Inter-Ocular Distance — IOD) là thước đo chuẩn hóa kích thước khuôn mặt:
- **Nguyên tắc:** Tính **MỘT LẦN** khi chạy run từ Model (AI detection).
- **Chuỗi ưu tiên (Fallback chain):**
  1. `model`: Tâm mắt trái Model tới tâm mắt phải Model.
  2. `human`: Nếu không có Model, tính từ mắt Human.
  3. `outer_corners`: Khoảng cách khóe mắt ngoài chia 0.65.
  4. `bbox`: Bounding box khuôn mặt nhân 0.42.
  5. `default`: 96.0 px.
- **Lưu trữ:** Lưu vào `face_frame_results.iod` cùng cột nguồn gốc `iod_source` (`'model' | 'human' | 'bbox' | 'default' | 'legacy'`).
- **Tính bất biến khi Review:** Khi reviewer chuyển đổi nguồn điểm giữa Human và Model trên giao diện, hệ thống tái sử dụng IOD đã lưu trong database, không tính lại từ reviewed keypoints để đảm bảo tính nhất quán của thang đo.

---

## 4. Hệ Thống Scoring & Chỉ Số Cấp Khuôn Mặt
Hệ thống tính toán chất lượng gán nhãn kết hợp cả mức trung bình và các chỉ số cực trị:

### 4.1. Chuẩn Hóa NME (Normalized Mean Error)
- Khi có điểm so sánh: $NME = \frac{1}{N} \sum_{i=1}^N \frac{\|h_i - m_i\|}{\max(IOD, 10)}$.
- **Khi không có điểm so sánh (hoặc mặt chưa nhãn):**
  - Trả về `nme = None`, `passed_nme = None`, `nme_pct = None`.
  - Trên API, UI, CSV, Checklist hiển thị rõ `N/A`.
  - Bộ lọc SQL: `ORDER BY CASE WHEN f.nme IS NULL THEN 1 ELSE 0 END ASC, f.nme DESC/ASC` để các kết quả `N/A` luôn nằm ở cuối danh sách.
  - Bộ lọc pass/fail không tính kết quả `N/A` là pass.

### 4.2. Bộ Chỉ Số Cấp Mặt Mới (Face-level Metrics)
Để tránh hiện tượng "bình quân che lấp lỗi nặng" (ví dụ: 49 điểm cực tốt $e=0.02$ + 1 điểm lệch rất xa $e=0.5$ thì NME vẫn $\approx 0.0296 \le 0.035$ qua mặt NME):
- **`e_max`:** Sai số chuẩn hóa lớn nhất trong các điểm đánh giá.
- **`e_p90`:** Phân vị 90 của sai số chuẩn hóa.
- **`exceed_count`:** Số điểm có $e > \text{tolerance}$ (3% IOD cho điểm neo, 5% IOD cho contour).
- **`has_outlier_point`:** Cờ dị biệt, bật `True` khi có **ít nhất 1 điểm đạt mức CRITICAL** ($e > 1.8 \times \text{tolerance}$). Giúp người kiểm định bắt ngay các trường hợp điểm dị biệt nghiêm trọng.

---

## 5. Ghép Cặp Tối Ưu Bipartite (Hungarian Matching)
Thay thế phép ghép tham lam (greedy) cũ bằng thuật toán Hungarian toàn cục:
- **Thư viện:** `scipy.optimize.linear_sum_assignment`.
- **Ma trận chi phí:** Khoảng cách Euclidean giữa centroid của từng Annotation và Detection.
- **Ngưỡng cắt (Cutoff Threshold):** $2.0 \times \text{IOD}$ (hoặc 150 px). Nếu khoảng cách tối ưu vượt ngưỡng cắt, cặp mặt không bị ép ghép mà tách thành 1 Annotation độc lập và 1 Unlabeled Detection.
- **Đường dẫn nhanh (Fast-path):** Khi ảnh chỉ có 1 Annotation và 1 Detection, so khớp trực tiếp theo ngưỡng cắt.

---

## 6. Phân Loại Mặt Chưa Gán Nhãn (`unlabeled_face`)
Khi MediaPipe phát hiện khuôn mặt trên ảnh nhưng không có nhãn annotation tương ứng (ví dụ: hành khách ngồi cạnh tài xế hoặc người ngoài xe):
- **Cột phân loại:** `result_kind = 'unlabeled_face'` (so với `'annotated_face'`).
- **NME & Đánh giá:** `nme = None`, `passed_nme = None`, không tính vào thống kê NME của dataset.
- **Giao diện & API:** Có bộ lọc riêng (`result_kind=annotated_face | unlabeled_face | all`), hiển thị số đếm `unlabeled_count`, gắn nhãn riêng trên danh sách kết quả.

---

## 7. Đổi Tên & Phân Tách `error_count` / `violation_count`
- **`violation_count`:** Tổng số vi phạm luật (bao gồm cả WARNING và ERROR).
- **`error_count`:** Chỉ tính các vi phạm có mức độ nghiêm trọng `ERROR` (vi phạm cấu trúc bắt buộc theo guideline).
- Toàn bộ API, giao diện bảng danh sách, chi tiết và xuất báo cáo CSV/Checklist hiển thị đồng bộ `error_count` lỗi / `violation_count` vi phạm.

---

## 8. Cải Tiến Các Luật Kiểm Định Guideline
- **R08 (Sống mũi):** Đồng bộ ngưỡng tỉ lệ đoạn sống mũi `max_ratio = 1.55` trong cả code và `vf50_spec.yaml`.
- **R09 (Môi trong vs Môi ngoài):** Thay kiểm tra trọng tâm centroid bằng kiểm tra **từng điểm môi trong (42–49)**:
  - Nếu điểm nằm ngoài đa giác viền môi ngoài (30–41), tính khoảng cách tối thiểu tới các cạnh đa giác (`min_distance_to_polygon`).
  - Vi phạm nếu khoảng cách vượt dung sai $tol = \text{tolerance\_px} \times \text{scale\_factor}$ (mặc định tạm thời 2.0 px * scale_factor).
  - Báo cáo rõ danh sách các ID điểm vi phạm cụ thể.
- **R13 (Nhảy điểm giữa frame):**
  - Chỉ kích hoạt khi dataset có `frame_index` thực tế và 2 frame xử lý liên tiếp có $\Delta \text{frame\_index} = 1$.
  - Ghép khuôn mặt giữa frame $k$ và $k-1$ theo khoảng cách centroid gần nhất ($\le 1.5 \times \text{IOD}$).
  - Nếu không có frame trước liền kề hoặc mặt trôi quá xa, bỏ qua rule (không báo pass/fail giả).

---

## 9. Cơ Sở Dữ Liệu & Migration Schema V2
- **Tên migration:** `002_face_metrics_and_unlabeled`
- **Cơ chế:** Tạo bản sao lưu timestamped trong `data/backups/`, thực hiện tái cấu trúc bảng `face_frame_results` bằng giao dịch SQLite atomicity an toàn:
  - `nme REAL` (nullable).
  - Cột mới: `violation_count INTEGER`, `result_kind TEXT`, `iod_source TEXT`, `detector_confidence REAL`.
  - Chỉ mục mới: `idx_face_results_kind ON face_frame_results(run_id, result_kind)`.
- **Dữ liệu cũ:** Di chuyển toàn vẹn: `violation_count = error_count` cũ, tính lại `error_count = COUNT(level == 'ERROR')`, gán `result_kind` tương ứng và đặt `nme = NULL` cho các bản ghi không có điểm đánh giá.
