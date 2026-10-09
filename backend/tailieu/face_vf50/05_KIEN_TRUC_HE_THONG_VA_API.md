# Kiến trúc tích hợp và API Face Landmark VF50

## Kiến trúc runtime

Face Landmark là một feature của Landmark QA, không phải ứng dụng độc lập.

- Frontend: cùng SPA cổng `5173`, route `/face`.
- Backend: cùng FastAPI cổng `8000`.
- Xác thực: cùng Project và Recovery key.
- Database: cùng `landmark_qa.sqlite3`; bảng Face tách riêng khỏi bảng cảnh báo HumanPose.
- Run mode: `face_mediapipe_vf50`.

Code nghiệp vụ nằm trong `backend/app/features/face_landmark_vf50/`. Feature này không import code HumanPose; hai bên chỉ dùng hạ tầng trong `app/shared` và `app/core`.

## Luồng dữ liệu

1. Upload ZIP ảnh, tùy chọn kèm CVAT XML/JSON.
2. Parser chuẩn hóa nhãn về VF50 nhưng giữ annotation nguồn bất biến.
3. Tạo run qua API chung.
4. MediaPipe dự đoán tối đa 5 khuôn mặt và ánh xạ 478 → 50 điểm.
5. Ghép từng annotation với face gần nhất, chạy 14 luật và tính IOD/NME.
6. Review lưu lựa chọn Human/Model trong `face_point_reviews`.
7. Export dựng nhãn cleaned tại thời điểm tải, không ghi đè annotation gốc.

## Xác thực

Tất cả endpoint dưới đây yêu cầu:

```http
Authorization: Bearer <project-recovery-key>
```

## Dataset và run

### Upload Face dataset

`POST /api/projects/{project_id}/datasets` dạng multipart:

- `file`: ZIP ảnh.
- `schema_id=vf_face_landmark50_v1`.
- `annotation_file`: tùy chọn; CVAT XML, COCO JSON, `landmark-qa/v1` hoặc cleaned JSON.

### Tạo run

`POST /api/projects/{project_id}/runs?dataset_id={dataset_id}`

Dataset Face tự chọn mode `face_mediapipe_vf50`.

### Theo dõi run

`GET /api/projects/{project_id}/runs/{run_id}`

Response có `status`, `progress`, `mode` và `error_message`. Thiếu model hoặc inference lỗi làm run chuyển sang `failed`; hệ thống không tạo kết quả mock.

### Mở lại run đã có

`GET /api/projects/{project_id}/runs?schema_id=vf_face_landmark50_v1`

Trả về lịch sử Face run kèm dataset, trạng thái và lỗi gần nhất để UI `/face` có thể mở lại kết quả sau khi tải lại trang.

## Review Face

Base path:

```text
/api/projects/{project_id}/runs/{run_id}/face
```

- `GET /results`: phân trang, lọc `filter_rule`, sắp xếp theo `severity|frame_index|nme`.
- `GET /results/{result_id}`: Human, Model, reviewed keypoints, violations và quality metrics.
- `PUT /results/{result_id}/points/{point_id}`: body `{"source":"human"}` hoặc `{"source":"model"}`.
- `PUT /results/{result_id}/status`: body `{"status":"pending|reviewed|fixed"}`.
- `POST /batch-preview`: xem số result/point sẽ đổi.
- `POST /batch-apply`: cùng điều kiện preview và bắt buộc `confirm=true`.

Batch request:

```json
{
  "confirm": false,
  "only_violated": true,
  "min_nme": 0.035
}
```

## Export

- `GET /export/cleaned-json`
- `GET /export/cvat-xml`
- `GET /export/csv`
- `GET /export/checklist`

JSON sạch dùng lại định dạng `landmark-qa/v1` để có thể import round-trip. CVAT XML dùng width/height thật và `group_id` để phân biệt nhiều face trên cùng ảnh.

## Model

Chạy `python backend/scripts/download_face_landmarker.py` hoặc cấu hình `FACE_LANDMARKER_MODEL_PATH`. Model float16/v1 được kiểm tra checksum trước khi đưa vào sử dụng và không nằm trong Git.
