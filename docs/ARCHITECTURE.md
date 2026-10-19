# Kiến trúc sau refactor

## Frontend

Các page tập trung vào bố cục và kết nối sự kiện với workspace hook.

| Phần | Vị trí | Trách nhiệm |
|---|---|---|
| API JSON | `frontend/src/shared/api.ts` | Base URL, credentials, merge headers, lỗi JSON và phản hồi 204 |
| Session | `frontend/src/shared/session.ts` | Đọc/ghi/xóa khóa `landmark-session`; đọc JSON hỏng trả `null` |
| Types | `frontend/src/shared/types.ts` | Project credentials, session và dataset |
| Polling | `shared/poll.ts`, `shared/useRunMonitor.ts` | Request tuần tự; dừng khi completed/failed, đổi run hoặc rời trang |
| Human Pose | `features/human-pose/` | `useHumanPoseWorkspace` điều phối dataset/run/review; `PoseReviewCanvas` giữ state overlay riêng; badge/constants/types riêng |
| Face | `features/face-landmark-vf50/` | `useFaceWorkspace` điều phối upload/run/review/export; `points.ts` xử lý điểm; page dựng giao diện |
| Temporal | `features/temporal/` | `useTemporalWorkspace` điều phối dataset/run; các panel giữ logic review chuyên biệt |
| Styles | `src/styles.css`, `src/styles/` | Entry import theo thứ tự base → home → workspace → controls → review → canvas → responsive |

Các endpoint tải file vẫn dùng `Response.blob()` để giữ tên file và header export. Khi thêm lời gọi JSON, dùng `requestJson`; không đặt Content-Type cho FormData vì trình duyệt tự thêm multipart boundary.

## Backend

Luồng phụ thuộc chính:

```text
main / routers → application → features
                       ↓          ↓
                      core / shared
```

- `main.py` đăng ký run handlers và HTTP routers, khởi tạo DB và phục hồi video jobs khi startup.
- Router giữ route, input validation, `project_guard`, response headers và lập lịch BackgroundTasks.
- `application/datasets/` chứa upload ZIP/video, xác nhận sequence, tra ảnh và thực thi/phục hồi video import.
- `application/datasets/formats.py` là điểm chọn parser theo schema. Face và Pose không import lẫn nhau; tầng application phối hợp hai feature.
- `shared/datasets/schemas.py` đọc metadata schema; shared không import application, features hay routers.
- `application/pose/` chứa review/keypoint queries và export nhãn nguồn kèm quyết định review.
- `application/temporal/` tách kiểm tra run, query, review và audit export.
- `features/face_landmark_vf50/services/review.py` chứa review từng điểm, batch fix, tính lại chất lượng và dựng dữ liệu export của Face.
- Các service vẫn giữ `HTTPException` hiện hành để bảo toàn status code và nội dung lỗi. Chưa có tầng domain-error độc lập.

Các file shim `app/services/*.py`, `app/adapters/*.py` và bản ZIP extractor trùng trong Pose đã bị xóa. Scripts/tests hiện import trực tiếp namespace đang sở hữu logic. Video import chuyển từ `shared/video_import.py` sang `application/datasets/import_job.py`.

## Database

`core/database.py` giữ connection/transaction, đường dẫn data tương đối, kiểm tra version và backup. `core/migrations/` giữ SQL và chuyển đổi dữ liệu:

1. `base_schema.py`: bảng lõi và cột bổ sung cho dữ liệu cũ.
2. `sequence_schema.py`: sequence, temporal, video jobs, AI drafts và backfill metadata.
3. `face_schema.py`: bảng/index Face và marker ban đầu.
4. `face_metrics.py`: nâng cấp bảng Face cũ, giữ kết quả và tính lại metrics/count.
5. Runner ghi các marker còn lại theo `versions.MIGRATION_VERSIONS`.

Đây là việc tách initializer hiện có; không thay version ID, thứ tự SQL, quy tắc backup hoặc schema. Các bảng/cột dùng `IF NOT EXISTS` và kiểm tra cột để khởi tạo lặp lại an toàn. Khi sửa schema, bổ sung version và test nâng cấp dữ liệu thực tế; không chỉ kiểm tra DB trống.

Test compatibility đối chiếu fingerprint tables/indexes/constraints trước refactor và kiểm tra DB Face cũ, idempotency, backup và foreign keys.

## Kiểm tra

Chạy từ repository root trên Windows:

```powershell
backend\.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

Chạy trong `frontend/`:

```powershell
npm test
npm run build
```

Frontend tests dùng Node test runner và `stripTypeScriptTypes`, yêu cầu Node 22.13+ (đã chạy trên Node 24). Không cài thêm test framework. Test API/polling/session không thay thế việc kiểm tra giao diện tương tác trên trình duyệt.

## Các phiên và nhánh

| Phiên | Nhánh | Phạm vi commit |
|---|---|---|
| 1 | `refactor/hiep-01-frontend-shared` | API/types, session, feature consumers, tests |
| 2 | `refactor/hiep-02-human-pose` | Workspace hook/types, review canvas |
| 3 | `refactor/hiep-03-face-temporal` | Face hook/helpers, Temporal hook, polling và tests |
| 4 | `refactor/hiep-04-backend-services` | Dataset, Pose review/export, Temporal, Face review services |
| 5 | `refactor/hiep-05-backend-dependencies` | Parser dispatch/import job, xóa shim, boundary/format tests |
| 6 | `refactor/hiep-06-database-css-docs` | Migration và compatibility tests, CSS, tài liệu |

Các nhánh được tạo nối tiếp sau khi merge phiên trước vào `main` local, nên diff của từng phiên là chênh lệch với phiên ngay trước nó. Commit không thêm tác giả phụ.
