# Tài liệu ngữ cảnh AI: Face Landmark VF50

Tài liệu này tóm tắt riêng feature Face Landmark VF50 đang có trong dự án Landmark QA. Mục tiêu là cung cấp đủ ngữ cảnh nghiệp vụ và kỹ thuật cho một AI hỗ trợ đọc, phân tích hoặc sửa feature mà không cần gửi toàn bộ repository. Nội dung mô tả code hiện tại; khi có khác biệt, hành vi trong source code là chuẩn.

## 1. Mục tiêu và phạm vi

Feature nhận ZIP ảnh và annotation khuôn mặt (nếu có), chạy MediaPipe Face Landmarker để dự đoán landmark, chuyển kết quả thành chuẩn VinFast VF50, kiểm tra guideline và so sánh nhãn người với gợi ý model. Người review có thể chọn nguồn tọa độ theo từng điểm, rà soát ca khó, rồi xuất annotation đã review.

- Schema: `vf_face_landmark50_v1`.
- Run mode: `face_mediapipe_vf50`.
- UI: route `/face` trên frontend hiện tại.
- Backend: FastAPI, SQLite, MediaPipe Tasks Vision.
- Face có feature folder riêng tại `backend/app/features/face_landmark_vf50/`. Một số hạ tầng như DB, project/recovery-key, dataset và job dispatch được dùng chung; logic nghiệp vụ của Face nằm trong feature folder này.
- Không cần hiểu hay gửi model weights để hiểu thuật toán; model `.task` là artifact tải riêng, không nằm trong Git.

## 2. Cấu trúc code Face

```text
backend/app/features/face_landmark_vf50/
├── adapters/       # MediaPipe, ánh xạ 478 landmark -> VF50
├── parsers/        # CVAT/COCO/Landmark QA và chuẩn hóa input
├── rules/          # Hình học và 14 luật kiểm tra
├── scoring/        # IOD, NME, sai số điểm và phân loại ca khó
├── services/       # Nạp annotation và thực thi Face run
├── exporters/      # Cleaned JSON, CVAT XML, CSV, Markdown checklist
├── schemas/        # Request review/batch
└── api.py          # API kết quả, review, batch và export
```

Các điểm composition quan trọng:

- `backend/app/main.py`: đăng ký handler cho mode `face_mediapipe_vf50` và mount Face API.
- `backend/app/routers/runs.py`: tạo run theo schema dataset; Face schema được gán Face run mode.
- `backend/app/shared/jobs/`: dispatch run tới handler đã đăng ký.
- `backend/app/core/database.py`: migration bảng Face, backup DB trước migration.
- `frontend/src/features/face-landmark-vf50/`: trang Face, types, cấu hình hiển thị điểm và style.

## 3. Chuẩn annotation VF50

Tọa độ dùng pixel `(x, y)` theo kích thước ảnh gốc; ID điểm chạy từ `0` đến `49`. Trạng thái điểm hợp lệ là `visible`, `occluded`, `outside`. Với `outside`, tọa độ được chuẩn hóa thành `null` khi ingestion. Có thể có nhiều face annotation trong một ảnh.

| ID | Skeleton | Ý nghĩa / thứ tự |
|---|---|---|
| 0–4 | `longmaytrai` | Bờ trên lông mày trái theo ảnh, đuôi ngoài → đầu trong |
| 5–9 | `longmayphai` | Bờ trên lông mày phải theo ảnh, đầu trong → đuôi ngoài |
| 10–13 | `songmui` | Đỉnh sống mũi → thân → chân sống mũi; điểm 13 không phải chóp mũi |
| 14–21 | `mattrai` | Khóe ngoài, mí trên, khóe trong, mí dưới |
| 22–29 | `matphai` | Khóe trong, mí trên, khóe ngoài, mí dưới |
| 30–41 | `moingoai` | Khóe trái → viền môi trên → khóe phải → viền môi dưới |
| 42–49 | `moitrong` | Khóe trong trái → viền trong trên → khóe trong phải → viền trong dưới |

12 điểm neo dùng tolerance chặt hơn: `{0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36}`. Các điểm còn lại là contour. Skeleton mắt/lông mày/môi được biểu diễn thành đường nối; mắt, môi ngoài và môi trong là contour đóng theo schema.

## 4. Input và ingestion

ZIP ảnh được kiểm tra an toàn bởi shared ZIP extractor trước khi giải nén. Face annotation có thể nằm trong một trong các định dạng:

- CVAT XML.
- COCO JSON; parser giữ nhiều face annotation cùng ảnh.
- `landmark-qa/v1`.
- Cleaned JSON tương thích parser hiện tại.
- ZIP chỉ có ảnh: vẫn tạo dataset và chạy inference, nhưng không có đối chiếu nhãn Human.

Tên ảnh trong annotation được ghép với file trong ZIP theo đường dẫn tương đối; nếu chỉ khớp basename thì basename phải duy nhất. Thiếu file hoặc basename nhập nhằng bị từ chối. Mỗi face annotation được chuẩn hóa thành đủ 50 ID; ID thiếu được coi là `outside`. State không hợp lệ, tọa độ không phải số hoặc annotation không khớp ảnh đều gây lỗi ingestion.

Các bảng lõi `projects`, `datasets`, `images`, `annotations`, `runs` tiếp tục lưu project/dataset/ảnh/annotation/run. Annotation gốc không bị sửa khi review.

## 5. Inference và ánh xạ model

Adapter `mediapipe_face.py` dùng Google MediaPipe Tasks Vision Face Landmarker, mode `IMAGE`, giới hạn 5 mặt/ảnh, confidence detection/presence/tracking đều đặt `0.3`. Ảnh OpenCV BGR được đổi RGB rồi đưa vào detector. Mỗi face trả landmark dày MediaPipe (478 điểm); tọa độ normalized được nhân với chiều rộng/cao ảnh để ra pixel.

`mapper.py` chuyển contour thành 50 điểm bằng lấy mẫu theo độ dài cung polyline:

1. Tính độ dài từng đoạn 2D, cộng dồn thành arc length.
2. Với mỗi fraction `f`, lấy vị trí tại `f * tổng_độ_dài`.
3. Nội suy tuyến tính trên đoạn chứa vị trí đó.

Các contour nguồn của MediaPipe và tỷ lệ lấy mẫu VF50 được ghi trong `backend/app/features/face_landmark_vf50/adapters/mapper.py` và spec `backend/configs/face_landmark_vf50/vf50_spec.yaml`. Tỷ lệ chính: lông mày trái `[0, .15, .41, .73, 1]`; lông mày phải `[0, .26, .58, .85, 1]`; sống mũi `[0, .34, .67, 1]`; mỗi mắt lấy 5 điểm mí trên và 3 điểm mí dưới; môi ngoài 7 điểm trên + 5 dưới; môi trong 5 trên + 3 dưới. Điểm đầu/cuối được dùng chung ở hai cung nên tổng vẫn là 50.

Trong run hiện tại, tất cả mặt phát hiện được trả về. Annotation được ghép lần lượt với detection còn lại gần centroid nhất; centroid là trung bình các tọa độ hợp lệ (không `outside`). Annotation không có điểm dùng detection đầu tiên còn lại. Detection dư vẫn được lưu thành result không có annotation. Khi không có annotation, rule vẫn có thể chạy trên điểm model; NME không có cặp Human/Model để so sánh nên quality trả về NME `0` và `passed_nme=true` theo nhánh không có điểm đánh giá. Vì vậy `passed_nme` của run ảnh-only không có nghĩa là đã đo được độ chính xác nhãn.

Thứ tự ảnh dùng `frame_index` nếu có, sau đó tên file; nếu thiếu thì run điền frame index theo thứ tự ảnh. Kết quả frame liền kề được dùng làm ngữ cảnh cho R13 theo `face_index`.

## 6. Hình học và công thức chất lượng

### IOD

Interocular Distance là khoảng cách giữa tâm hai mắt. Tâm mỗi mắt lấy trung bình các điểm có tọa độ trong nhóm trái `14–21`, phải `22–29`. Nếu mỗi mắt có ít nhất 4 điểm và khoảng cách tâm lớn hơn 10 px, dùng:

`IOD = sqrt((cx_left - cx_right)^2 + (cy_left - cy_right)^2)`

Fallback theo thứ tự:

1. Nếu có điểm 14 và 26, khoảng cách hai khóe ngoài lớn hơn 5 px: `IOD = 0.65 * distance(14, 26)`.
2. Nếu có ít nhất 8 điểm hợp lệ và bề rộng bbox mặt lớn hơn 10 px: `IOD = max(12, 0.42 * face_width)`.
3. Nếu không có dữ liệu đủ: `IOD = 96 px`.

Các điểm `outside` vẫn có thể tham gia tính IOD nếu còn tọa độ do hàm IOD chỉ kiểm tra tọa độ; fallback bbox thì loại điểm `outside`.

### Sai số từng điểm và NME

Chỉ tính trên điểm Human không `outside`, có tọa độ, và Model có tọa độ:

```text
d_px = sqrt((x_human - x_model)^2 + (y_human - y_model)^2)
e_norm = d_px / max(IOD, 10)
tolerance = 0.03 nếu là anchor, nếu không thì 0.05
```

Điểm vượt ngưỡng khi `e_norm > tolerance`. Mức severity:

- `NORMAL`: không vượt ngưỡng.
- `WARNING`: vượt tolerance nhưng `e_norm <= 1.8 * tolerance`.
- `CRITICAL`: `e_norm > 1.8 * tolerance`.

Suspicion của điểm: `min(1, e_norm / (2.5 * tolerance))`; nếu Human state là `occluded`, nhân thêm `0.6` để giảm phạt 40%.

`NME = mean(e_norm)` của các điểm được đánh giá. NME đạt khi `NME <= 0.035`. Nếu không có điểm so sánh, NME là 0 và pass là true (không được hiểu là bằng chứng model chính xác).

### Scale và hard case

- Face scale: `CLOSE` nếu IOD ≥ 80 px; `NORMAL` nếu IOD ≥ 38 px; còn lại `FAR`.
- Hard case được xét theo thứ tự ưu tiên:
  - `sunglasses`: ít nhất 4/16 điểm mắt (`14–29`) có state `occluded` hoặc `outside`; AI reliability `0.45`.
  - `occluded`: ít nhất 6 điểm toàn mặt có state `occluded` hoặc `outside`; reliability `0.58`.
  - `far`: IOD < 35 px; reliability `0.65`.
  - bình thường: reliability `0.95`.

Reliability là metadata phân loại ca, không được nhân trực tiếp vào NME hay point suspicion.

### Severity tổng result

Mỗi violation đóng góp `3` nếu mức `ERROR`, `2` nếu mức `WARNING`; tổng lưu vào `face_frame_results.severity`. Đây là điểm ưu tiên sắp xếp review, không phải xác suất lỗi. `error_count` hiện lưu tổng số violation, bao gồm cả WARNING; muốn đếm riêng ERROR cần lọc `level`.

## 7. 14 luật guideline đang chạy

Rule code và ngưỡng chi tiết nằm trong `backend/app/features/face_landmark_vf50/rules/validation.py`. Rule trả về `ERROR` hoặc `WARNING`, danh sách điểm liên quan và mục guideline.

| Mã | Kiểm tra | Ý chính / ngưỡng |
|---|---|---|
| R01 | Đầy đủ cấu trúc | Cần đủ ID 0–49; thiếu điểm là lỗi cấu trúc. |
| R02 | Quên sửa trạng thái | Cảnh báo trường hợp cả 50 điểm đều `visible`, thường là quên gán trạng thái che khuất/ngoài khung. |
| R03 | Đảo trái-phải | Kiểm tra trật tự trái/phải theo hướng nhìn trên ảnh cho chân mày và mắt. |
| R04 | Thứ tự x lông mày | Kiểm tra thứ tự ngang chân mày trong giả định mặt gần chính diện; tolerance scale theo IOD. |
| R05 | Khóe mắt cực trị | Điểm khóe ngoài/trong phải là điểm cực trị theo trục X của contour mắt tương ứng. |
| R06 | Mí trên/mí dưới | Mí trên không được thấp hơn mí dưới sau bù góc nghiêng đầu theo đường nối tâm mắt; tolerance `max(0.8, 2 * scale_factor)` px. |
| R07 | Mắt tự cắt | Kiểm tra contour mắt tự giao nhau/hình chữ X. |
| R08 | Tỉ lệ 3 đoạn sống mũi | Kiểm tra các đoạn liên tiếp 10–11, 11–12, 12–13 không mất cân đối; ngưỡng `1.55` đã được đồng bộ trong YAML spec và validation code. |
| R09 | Môi trong so với môi ngoài | Khóe môi trong (42, 46) phải nằm trong khóe môi ngoài (30, 36) theo x. Đồng thời kiểm tra từng điểm môi trong (42–49): nếu nằm ngoài đa giác môi ngoài (30–41) quá dung sai (2.0 px * scale_factor) thì báo ERROR và liệt kê chi tiết ID điểm vi phạm. |
| R10 | Ngưỡng ẩn cả skeleton | Mắt: dưới 4/8 điểm thấy thì nên Outside toàn bộ; chân mày: dưới 3/5. Nếu còn gán một phần thấy và chưa Outside cả nhóm thì cảnh báo. |
| R11 | Tọa độ ngoài biên ảnh | Điểm không `outside` nằm ngoài `[0,width] × [0,height]` là ERROR. |
| R12 | Trùng điểm bất thường | Hai điểm visible trùng gần nhau dưới `max(0.4, 0.012 * IOD)` px bị cảnh báo, trừ các cặp mí trên/dưới cho phép trùng khi nhắm mắt. |
| R13 | Nhảy điểm giữa frame | Chỉ kích hoạt khi dataset có frame_index thực tế và 2 frame xử lý liền kề nhau (chênh frame_index = 1). Ghép cặp khuôn mặt giữa 2 frame theo khoảng cách centroid gần nhất (ngưỡng 1.5 * IOD). Khoảng nhảy điểm phải ≤ `max(5, 15 * scale_factor)` px. |
| R14 | Occluded giữ tọa độ pre-label sai | Nếu điểm Human `occluded` lệch model hơn `0.15 * IOD`, cảnh báo cần ước lượng lại tọa độ. |

Trong code, `scale_factor = clamp(IOD / 96, 0.25, 2.5)`. Một số rule có điều kiện dữ liệu đủ (ví dụ kiểm tra polygon/đường mắt); thiếu điểm có thể khiến rule không đánh giá thay vì mặc định pass/fail.

## 8. Review và ý nghĩa dữ liệu

`face_frame_results` lưu kết quả cho `(run_id, image_id, face_index)`: Human keypoints, Model keypoints, IOD (tính 1 lần từ Model, có chuỗi fallback), `iod_source` ('model'|'human'|'bbox'|'default'|'legacy'), NME (nullable, None khi không có điểm đánh giá hoặc mặt chưa nhãn), quality JSON (chứa `e_max`, `e_p90`, `exceed_count`, `has_outlier_point`), violations JSON, severity, `error_count` (chỉ cấp ERROR), `violation_count` (tổng vi phạm), `result_kind` ('annotated_face'|'unlabeled_face'), `detector_confidence`, status và metadata hard case. Reviewer đổi nguồn điểm không làm đổi IOD đã lưu.

`face_point_reviews` lưu quyết định cho từng `(result_id, point_id)`: source là `human` hoặc `model`. Source mặc định là Human nếu có annotation; nếu không có Human thì mặc định Model. Chọn Model chỉ hợp lệ khi điểm model có tọa độ. Mỗi thay đổi source sẽ tính lại rule và quality trên tập điểm đã chọn; Human annotation JSON gốc vẫn giữ nguyên.

Status result: `pending`, `reviewed`, `fixed`. Thao tác chọn Model thường chuyển status sang `fixed`; chọn Human không có lựa chọn Model nào giữ result thì chuyển `reviewed`. Status có endpoint riêng để reviewer chỉnh trực tiếp.

Batch fix có hai bước: `batch-preview`, sau đó `batch-apply` với `confirm=true`. Chế độ chỉ điểm vi phạm gom điểm từ danh sách violation và điểm vượt tolerance trong quality. Điểm không có tọa độ model bị loại. Batch apply ghi source Model và tính lại result. Đây là thao tác review dữ liệu, không sửa annotation nguồn.

## 9. API Face

Base URL path:

```text
/api/projects/{project_id}/runs/{run_id}/face
```

Mọi API project yêu cầu `Authorization: Bearer <recovery-key>` thông qua project guard.

| Method | Path | Mục đích |
|---|---|---|
| GET | `/results` | Danh sách phân trang; sort theo severity, frame_index hoặc NME; lọc `filter_rule`. |
| GET | `/results/{result_id}` | Chi tiết Human/Model/reviewed keypoints, violations, quality. |
| PUT | `/results/{result_id}/points/{point_id}` | Chọn source Human hoặc Model cho điểm 0–49. |
| PUT | `/results/{result_id}/status` | Cập nhật status pending/reviewed/fixed. |
| POST | `/batch-preview` | Xem số result/điểm sẽ áp dụng batch. |
| POST | `/batch-apply` | Áp dụng batch; bắt buộc `confirm=true`. |
| GET | `/export/cleaned-json` | Xuất `landmark-qa/v1` từ reviewed keypoints. |
| GET | `/export/cvat-xml` | Xuất CVAT XML skeleton với dimensions và group_id. |
| GET | `/export/csv` | Báo cáo tóm tắt CSV. |
| GET | `/export/checklist` | Checklist Markdown để sửa thủ công. |

Run collection dùng chung tại `/api/projects/{project_id}/runs`; có thể lọc `schema_id=vf_face_landmark50_v1`. Tạo run qua endpoint collection dựa vào schema dataset, không phải endpoint riêng trong Face API.

## 10. Frontend

Trang chính Face nằm ở `frontend/src/features/face-landmark-vf50/FaceLandmarkPage.tsx`. UI chia sẻ project/recovery session với ứng dụng, nhưng Face thao tác tại `/face`. Các khả năng hiện tại: upload dataset, tạo run và xem tiến độ, lịch sử/reopen run, danh sách result sort/filter, overlay ảnh Human/Model, xem violation/quality/hard case, chọn Human/Model theo điểm, preview/xác nhận batch và tải các định dạng export.

Model path cấu hình bằng `FACE_LANDMARKER_MODEL_PATH`; mặc định là `backend/models/face_landmarker.task`. Tải model bằng `backend/scripts/download_face_landmarker.py`; artifact không commit.

## 11. Chạy Face local

Yêu cầu Python 3.10+, Node.js/npm. Trong PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_face_landmarker.py
uvicorn app.main:app --reload --port 8000
```

Cửa sổ khác:

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

Mở `http://localhost:5173/face`; API docs ở `http://localhost:8000/docs`. Database SQLite mặc định là `backend/data/landmark_qa.sqlite3`; lần migration Face tạo backup có timestamp trước khi thay đổi schema.

## 12. Hướng dẫn cho AI đọc tài liệu này

- Giữ schema ID, run mode, 50 ID điểm, thứ tự skeleton và state hiện có khi đề xuất thay đổi.
- Tách rõ Human annotation gốc, Model suggestion và reviewed output; review chỉ lưu lựa chọn nguồn, không ghi đè annotation gốc.
- Khi phân tích quality, phân biệt violation guideline với sai số Human–Model. Một result có thể có một loại mà không có loại kia.
- NME không có giá trị đánh giá khi thiếu cặp Human/Model; NME 0 trong trường hợp đó là sentinel hiện tại, không phải độ chính xác tuyệt đối.
- `ai_reliability` là nhãn metadata ca khó, không phải confidence được dùng để hiệu chỉnh NME.
- Khi sửa rule hoặc công thức, cập nhật đồng bộ tests, UI diễn giải, tài liệu guideline và export nếu các trường/ý nghĩa thay đổi.
- Đọc source được nêu dưới đây trước khi sửa; tài liệu này là bản đồ ngữ cảnh chứ không thay thế kiểm tra implementation.

## 13. Source of truth

- Schema điểm và skeleton: `backend/configs/schemas/vf_face_landmark50_v1.json`.
- Spec tỷ lệ/anchor/tolerance: `backend/configs/face_landmark_vf50/vf50_spec.yaml`.
- MediaPipe: `backend/app/features/face_landmark_vf50/adapters/mediapipe_face.py`.
- Map 478 → 50: `backend/app/features/face_landmark_vf50/adapters/mapper.py`.
- Geometry/IOD: `backend/app/features/face_landmark_vf50/rules/geometry.py`.
- 14 rules: `backend/app/features/face_landmark_vf50/rules/validation.py`.
- NME/quality/hard case: `backend/app/features/face_landmark_vf50/scoring/quality.py`.
- Ingestion: `backend/app/features/face_landmark_vf50/services/ingestion.py` và `parsers/`.
- Run orchestration: `backend/app/features/face_landmark_vf50/services/run.py`.
- Review/API: `backend/app/features/face_landmark_vf50/api.py` và `schemas/review.py`.
- DB migration: `backend/app/core/database.py`.
- Export: `backend/app/features/face_landmark_vf50/exporters/`.
- UI: `frontend/src/features/face-landmark-vf50/`.

## 14. Công thức hình học chi tiết

Phần này bổ sung diễn giải toán học từ bộ tài liệu Face VF50 trên nhánh `dev/dung`. Công thức được ghi theo ký hiệu tổng quát; ghi chú “code hiện tại” nêu khác biệt cần nhớ khi AI phân tích hoặc thay đổi implementation.

### 14.1. Tâm mắt, IOD và scale factor

Với tập điểm mắt trái `L={14,...,21}` và mắt phải `R={22,...,29}`, tâm mắt là trung bình các điểm hợp lệ:

```text
C_L = (mean(x_i), mean(y_i)), i ∈ L hợp lệ
C_R = (mean(x_i), mean(y_i)), i ∈ R hợp lệ
IOD = ||C_L - C_R||₂
    = sqrt((C_Rx-C_Lx)² + (C_Ry-C_Ly)²)
```

“Hợp lệ” ở implementation hiện tại nghĩa là có cả x và y; hàm không loại state `outside` nếu điểm vẫn mang tọa độ. Dùng tâm hai mắt khi mỗi bên có ít nhất 4 điểm và khoảng cách lớn hơn 10 px. Fallback tiếp theo:

```text
IOD = 0.65 * ||P14 - P26||₂                         nếu khoảng cách > 5 px
IOD = max(12, 0.42 * face_width)                   nếu ≥8 điểm không-outside, face_width >10 px
IOD = 96 px                                        nếu không đủ dữ liệu
```

Scale factor cho ngưỡng pixel trong rule:

```text
s = clamp(IOD / 96, 0.25, 2.5)
clamp(z,a,b) = min(max(z,a),b)
```

IOD trung vị khoảng 96 px trên ảnh 1280×720 là mốc quy đổi, không phải hằng số đo cho mọi ảnh. Ví dụ tại IOD=96 px: 3% IOD = 2.88 px; 5% IOD = 4.80 px; NME 0.035 tương đương trung bình khoảng 3.36 px trên các điểm được tính.

### 14.2. Sai số chuẩn hóa, tolerance và NME

Với điểm i có tọa độ Human `H_i=(x_hi,y_hi)` và Model `M_i=(x_mi,y_mi)`:

```text
d_i = ||H_i-M_i||₂
e_i = d_i / max(IOD,10)
τ_i = 0.03 nếu i là anchor; 0.05 nếu i là contour
```

Đánh giá điểm: NORMAL nếu `e_i ≤ τ_i`; WARNING nếu `τ_i < e_i ≤ 1.8τ_i`; CRITICAL nếu `e_i > 1.8τ_i`. Suspicion chuẩn hóa:

```text
q_i = min(1, e_i / (2.5τ_i))
q_i = 0.6q_i nếu Human state = occluded
```

NME trên tập điểm đánh giá `V`:

```text
NME = (1 / |V|) * Σ(i∈V) e_i
V = các ID 0..49 mà Human không outside, có x/y và Model có x/y
PASS nếu NME ≤ 0.035
```

Đây là phép đo sai khác giữa nhãn người và dự đoán MediaPipe. Model chỉ là tham chiếu tự động, không phải ground truth đã được xác nhận; do đó NME thấp không tự chứng minh annotation đúng tuyệt đối. Implementation trả NME=0 và pass=true khi `V` rỗng, nên caller phải xem `evaluated_points_count` trước khi diễn giải PASS.

### 14.3. Xử lý nghiêng đầu cho kiểm tra mí

Góc nghiêng được ước lượng từ vector nối tâm mắt:

```text
θ = atan2(C_Ry - C_Ly, C_Rx - C_Lx)
```

Để đưa điểm P về hệ trục đã khử góc nghiêng, xoay quanh tâm C=(x_c,y_c) một góc `-θ`. Hàm `rotate_point(P,C,angle)` tự tính `cos(-angle)`/`sin(-angle)`; validation truyền `angle=θ`:

```text
[x']   [ cos(-θ)  -sin(-θ) ] [x-x_c]   [x_c]
[y'] = [ sin(-θ)   cos(-θ) ] [y-y_c] + [y_c]
```

Trong hệ ảnh, y tăng từ trên xuống. Với từng cặp mí trên/mí dưới `(15,21)`, `(16,20)`, `(17,19)`, `(23,29)`, `(24,28)`, `(25,27)`, vi phạm khi:

```text
y'_upper > y'_lower + ε
ε = max(0.8, 2s) px
```

Code dùng tâm mắt từ các điểm mắt hợp lệ làm tâm xoay; góc bằng 0 nếu không đủ ít nhất 4 điểm ở mỗi mắt. Bộ tài liệu cũ còn nói có một luật riêng cảnh báo góc mắt >45°; code hiện tại không có luật ngưỡng 45° độc lập.

### 14.4. Nội suy theo chiều dài cung (Arc-length resampling)

Cho polyline `P_0,...,P_m` lấy từ contour MediaPipe. Mỗi độ dài đoạn chỉ dùng x,y (nếu input có z thì z được nội suy nhưng không tham gia tính độ dài):

```text
Δs_k = sqrt((x_(k+1)-x_k)² + (y_(k+1)-y_k)²)
S_0 = 0
S_k = Σ(j=0..k-1) Δs_j
L = S_m
d = clamp(f,0,1) * L
```

Tìm k sao cho `S_k ≤ d ≤ S_(k+1)`, rồi nội suy tuyến tính:

```text
t = (d-S_k) / Δs_k
P*(f) = P_k + t(P_(k+1)-P_k)
```

Nếu chỉ có ≤1 điểm hoặc tổng chiều dài gần 0 (`L < 1e-6`), implementation lặp lại điểm đầu cho mọi fraction. `searchsorted` tìm đoạn trong danh sách chiều dài tích lũy. Fractions từng skeleton nằm trong `vf50_spec.yaml` và mapper; đây là quy tắc phân bố vị trí trên đường cong, không đảm bảo landmark nội suy luôn trùng chính xác một mốc giải phẫu nếu contour MediaPipe ban đầu lệch.

### 14.5. Giao đoạn thẳng và đa giác tự cắt

Với ba điểm A,B,C, phép thử hướng CCW trong code tương đương so dấu hai tích chéo:

```text
ccw(A,B,C) = (C_y-A_y)(B_x-A_x) > (B_y-A_y)(C_x-A_x)
```

Hai đoạn AB và CD được xem là giao nhau nếu:

```text
ccw(A,C,D) ≠ ccw(B,C,D)
AND
ccw(A,B,C) ≠ ccw(A,B,D)
```

Các cạnh kề nhau và cạnh chung đỉnh được bỏ qua. Với polygon n đỉnh, code duyệt các cặp cạnh không kề nhau (chi phí O(n²)). Lưu ý đây là phép thử giao cắt dạng orientation cơ bản, không xử lý đầy đủ mọi trường hợp suy biến/đồng tuyến như một thư viện robust geometry. Validation hiện áp dụng tự cắt cho contour mắt; không nên suy ra rằng mọi skeleton đều được kiểm tra self-intersection.

### 14.6. Kiểm tra điểm trong polygon bằng ray casting

Từ điểm cần kiểm tra `P=(x,y)`, xét tia ngang về phía +x. Với từng cạnh polygon `(x1,y1)-(x2,y2)`, cạnh được xét khi y nằm trong khoảng nửa mở `[min(y1,y2), max(y1,y2))`; tọa độ giao với tia:

```text
x_intersect = x1 + (y-y1) * (x2-x1) / (y2-y1)
```

Nếu `x ≤ x_intersect`, đảo trạng thái inside/outside. Sau khi xét các cạnh: số giao cắt lẻ là trong polygon, chẵn là ngoài. Code R09 không kiểm tra riêng từng điểm môi trong: nó tính centroid của các điểm môi trong có tọa độ rồi kiểm tra centroid có nằm trong polygon môi ngoài hay không (với ít nhất 8 điểm outer và 4 điểm inner). Đây là khác biệt so với mô tả guideline cũ “tất cả 8 điểm nằm bên trong”.

## 15. Đối chiếu tài liệu nhánh `dev/dung` với implementation hiện tại

Đã đọc 5 file Markdown đang có trong `face_vf50/docs/` trên nhánh đó: tổng quan/quy trình, 14 luật, giải phẫu 50 điểm, công thức/scoring, kiến trúc/API. Các tài liệu này hữu ích để hiểu căn cứ nghiệp vụ và giải thích công thức, nhưng là tài liệu của bản standalone trước khi tích hợp. Mã hiện tại đã đổi kiến trúc, cổng, API, và một số rule. Không dùng số hiệu rule trong tài liệu cũ như số hiệu rule hiện hành.

| Nội dung trong tài liệu cũ | Gần với implementation hiện tại | Khác biệt quan trọng |
|---|---|---|
| R01 thiếu điểm | R01 | Tương tự. |
| R02 ngoài khung | R11 | Tương tự; code dùng kích thước ảnh thực tế, không cố định 1280×720. |
| R03 lật mí | R06 | Code khử nghiêng và áp dụng tolerance theo scale. |
| R04 contour tự cắt | R07 | Code hiện kiểm contour mắt; tài liệu cũ mô tả rộng hơn. |
| R05 tỉ lệ sống mũi | R08 | Spec YAML ghi 1.45; validation code hiện dùng 1.55 (cảnh báo khi `max/min > 1.55`). |
| R06 trái-phải | R03 | Thứ tự rule đã đổi. |
| R07 môi trong | R09 | Code xét centroid môi trong, không test từng điểm. |
| R08 trạng thái che khuất | R02/R10/R14 | Tách thành heuristic cả 50 visible, ngưỡng ẩn skeleton và kiểm tra tọa độ occluded so với model. |
| R09 temporal jitter | R13 | Code dùng `max(5,15*s)` px thay vì ngưỡng tuyệt đối 15 px. |
| R10 mắt nghiêng >45° | Không có rule tương ứng riêng | Không được khẳng định hệ thống hiện reject mọi góc mắt >45°. |
| R11/R12 sai số anchor/contour | Scoring `point_evaluations` | Đây là đánh giá quality so với model, không phải rule violation có mã R11/R12 hiện tại. |
| R13 NME | Scoring `passed_nme` | NME không sinh violation tự động; result có quality field riêng. |
| R14 mắt/môi nhắm vẫn đủ điểm | R01 và các rule khác một phần | Không có rule riêng chỉ để kiểm tra mắt/môi nhắm. |
| Tên/ngưỡng severity CRITICAL/MAJOR trong tài liệu cũ | `ERROR`/`WARNING` ở rule hiện tại | Mapping hiện tại trong API: ERROR → `critical`, WARNING → `major`; code không dùng nguyên severity table cũ. |

Các điểm cần đặc biệt tránh nhầm:

- Tài liệu cũ mô tả IOD từ đủ 8 điểm mỗi mắt; code hiện dùng centroid nếu mỗi mắt có ít nhất 4 điểm.
- Spec YAML hiện đặt bridge ratio `1.45`, nhưng `validation.py` đang dùng `1.55`; không nên coi hai giá trị đã đồng nhất.
- Tài liệu cũ mô tả NME trên 50 điểm như KPI annotation; code chỉ tính trung bình trên các cặp Human/Model thực có, chuẩn hóa bằng `max(IOD,10)`.
- Tài liệu cũ có lúc ghi NME như `Σ d_i/(50*IOD)`; phép triển khai hiện tại tương đương trung bình `d_i/max(IOD,10)` trên tập điểm hợp lệ, không chia cố định cho 50.
- Tài liệu cũ ghi một số nhãn lỗi gắn với rule code không khớp cả tài liệu rule cũ lẫn implementation mới. Luôn tra `validation.py` và `quality.py` để biết hành vi thật.
- Tài liệu cũ cấm upload annotation lên CVAT vì quy trình thi cụ thể; app hiện xuất file để người dùng tải thủ công. Không có chức năng app tự đăng nhập hoặc upload/ghi đè CVAT.
- Tài liệu/API trong nhánh cũ nói port 8001/5174 và endpoint standalone (`/api/upload`, `/api/sessions/...`). Bản tích hợp hiện chạy backend 8000, frontend 5173, dùng endpoint project/run của app hiện tại.

## 16. Tài liệu nguồn đã đối chiếu

Nguồn tham khảo là Markdown trên `dev/dung/face_vf50/docs/` tại thời điểm soạn phần bổ sung:

- `01_TONG_QUAN_VA_QUY_TRINH.md`
- `02_QUY_TAC_14_LUAT_KIEM_DINH.md`
- `03_GIAI_PHAU_50_DIEM_VA_7_SKELETONS.md`
- `04_CONG_THUC_TOAN_HOC_VA_SCORING.md`
- `05_KIEN_TRUC_HE_THONG_VA_API.md`

Đường dẫn GitHub: `https://github.com/vth2004/Vin-BuildPhase/tree/dev/dung/face_vf50/docs`. Bản nguồn có một số liên kết `file:///` trỏ máy tác giả và có tài liệu mô tả kiến trúc standalone; các liên kết đó không dùng làm đường dẫn trong repo tích hợp.
