# Kế hoạch tích hợp Face Landmark VF50

## 1. Mục tiêu và nguyên tắc

- Tích hợp có chọn lọc chức năng Face Landmark VF50 từ `origin/dev/dung` vào kiến trúc hiện tại.
- HumanPose và Face Landmark là hai feature độc lập, không import logic nghiệp vụ của nhau.
- Chỉ đưa vào `shared` những thành phần thực sự dùng chung như database, xác thực, project, storage, upload và vòng đời run.
- Giữ nguyên hành vi, API và dữ liệu HumanPose hiện tại.
- Dùng chung frontend cổng `5173`, backend cổng `8000`, Project, recovery key và SQLite.
- Nhãn gốc luôn bất biến. AI chỉ đưa ra đề xuất; người dùng duyệt từng điểm hoặc xác nhận batch trước khi export.
- Không mang nguyên app `face_vf50`, database, uploads, demo giả, port riêng hoặc model binary từ nhánh Dũng.
- Codex không tự commit hoặc push; người dùng tự thực hiện các thao tác Git đó.

## 2. Nhánh làm việc và trình tự triển khai

- Nhánh làm việc: `dev/hiep-facelandmarkVF`.
- Nhánh được tạo từ `dev/humanpose-new` khi worktree sạch.
- Thực hiện theo từng giai đoạn nhỏ và chạy regression test sau mỗi lần di chuyển module.
- Không di chuyển toàn bộ HumanPose trong một lần.
- Trong thời gian chuyển cấu trúc, giữ compatibility re-export tại các đường dẫn Python cũ nếu cần; chỉ xóa sau khi toàn bộ import và test đã được cập nhật.

## 3. Cấu trúc mã nguồn đích

### Backend

```text
backend/app/
|-- features/
|   |-- human_pose/
|   |   |-- adapters/
|   |   |-- rules/
|   |   |-- services/
|   |   |-- schemas/
|   |   `-- api.py
|   `-- face_landmark_vf50/
|       |-- adapters/
|       |-- parsers/
|       |-- rules/
|       |-- scoring/
|       |-- exporters/
|       |-- services/
|       |-- schemas/
|       `-- api.py
|-- shared/
|   |-- auth/
|   |-- database/
|   |-- datasets/
|   |-- storage/
|   |-- jobs/
|   `-- imaging/
`-- main.py
```

### Frontend

```text
frontend/src/
|-- app/
|   |-- router/
|   `-- shell/
|-- features/
|   |-- human-pose/
|   `-- face-landmark-vf50/
`-- shared/
    |-- api/
    |-- auth/
    |-- components/
    |-- hooks/
    `-- types/
```

### Quy tắc phụ thuộc

- `human_pose` không được import từ `face_landmark_vf50` và ngược lại.
- Hai feature chỉ phụ thuộc vào interface hoặc hạ tầng trong `shared`.
- `shared` không chứa schema điểm, rule, scoring, parser hay UI gắn riêng với một loại landmark.
- Composition root trong `main.py` được phép đăng ký handler/router của cả hai feature.
- Thêm kiểm tra import boundary để phát hiện phụ thuộc nghiệp vụ chéo.
- `dataset_service` hiện phụ thuộc `COCO_TO_VF17`, vì vậy không dùng nguyên module này làm shared. Chỉ tách phần giải nén, kiểm tra file và storage; mỗi feature có parser riêng.

## 4. Tách HumanPose an toàn

1. Viết hoặc bổ sung characterization test cho API, parser, scoring và run HumanPose hiện tại.
2. Tách hạ tầng database, security, archive extraction, storage và run lifecycle sang `shared`.
3. Di chuyển lần lượt adapter, scoring, dataset parser, run handler và router HumanPose vào `features/human_pose`.
4. Cập nhật import theo từng nhóm; dùng re-export tạm thời nếu một đường dẫn cũ vẫn còn consumer.
5. Chạy test sau từng nhóm di chuyển và so sánh kết quả với baseline.
6. Chỉ xóa compatibility wrapper sau khi không còn import cũ.

Việc chia folder chỉ thay đổi vị trí mã nguồn và import. Nó không được phép thay đổi endpoint, payload, thuật toán, database hoặc kết quả HumanPose.

## 5. Tích hợp backend Face Landmark

Chuyển có chọn lọc từ hai commit của `origin/dev/dung`:

- MediaPipe Face Landmarker.
- Ánh xạ 478 landmarks sang VF50 bằng arc-length resampling.
- 14 luật kiểm định Face.
- IOD, NME, scoring và hard-case detection.
- Parser ZIP ảnh, CVAT XML, COCO JSON, `landmark-qa/v1` và cleaned JSON.
- Export cleaned JSON, CVAT XML, CSV và checklist.

Yêu cầu triển khai:

- Tạo run mode `face_mediapipe_vf50` và chạy qua runner hiện tại.
- Không nuốt lỗi parser và không tự fallback sang mock.
- Lỗi cấu hình, parser hoặc inference phải được lưu trong trạng thái run và trả về UI.
- Hỗ trợ nhiều khuôn mặt trong một ảnh bằng cách ghép annotation người dùng với prediction phù hợp.
- Cho phép ảnh chưa có annotation vẫn lưu prediction để xem xét.
- Giữ đúng width, height và frame index; không hard-code kích thước khi export CVAT.
- Giữ đủ dữ liệu thứ tự frame cho luật temporal.
- ZIP upload phải chặn path traversal, đường dẫn trùng, định dạng không cho phép, vượt số file và vượt tổng dung lượng giải nén.

## 6. Database và migration

Tiếp tục sử dụng các bảng lõi hiện tại:

- `projects`
- `datasets`
- `images`
- `annotations`
- `runs`
- `warnings`

Bổ sung:

- `schema_migrations` để quản lý migration idempotent.
- Cột `images.frame_index` nullable.
- Bảng `face_frame_results` liên kết run, image và annotation nguồn; lưu prediction, IOD/NME, violations, severity, hard-case và trạng thái review.
- Bảng `face_point_reviews` lưu lựa chọn Human/AI theo từng điểm cùng audit timestamp.
- Foreign key và index theo `run_id`, `image_id`, `status` và `severity`.

Nguyên tắc dữ liệu:

- Không chỉnh sửa keypoints trong annotation nguồn.
- Bản cleaned được dựng từ annotation gốc cộng với các lựa chọn review.
- Migration chỉ thêm bảng, cột và index; không xóa hoặc đổi nghĩa dữ liệu cũ.
- Tạo backup SQLite có timestamp trước migration đầu tiên.
- Đối chiếu số lượng project, dataset, image, annotation và run trước/sau migration.
- Không import `face_vf50.sqlite3` và không giữ mô hình `sessions/frames` độc lập của nhánh Dũng.

## 7. API

- Giữ tương thích toàn bộ API HumanPose hiện tại.
- Mở rộng upload dataset bằng `schema_id=face_vf50`.
- Tất cả API Face nằm dưới project hiện tại và bắt buộc kiểm tra recovery key.
- Bổ sung API để:
  - Tạo Face run.
  - Lấy danh sách và chi tiết từng frame/face.
  - Lưu lựa chọn Human/AI theo điểm.
  - Preview và xác nhận batch fix.
  - Export cleaned JSON, CVAT XML, CSV hoặc checklist.
- Shared run service chỉ quản lý vòng đời job; việc thực thi HumanPose hoặc Face được ủy quyền cho handler của feature tương ứng.

## 8. Frontend

- Thêm routing trong cùng SPA:
  - `/`: HumanPose hiện tại.
  - `/face`: Face Landmark VF50.
- Trích UI HumanPose khỏi `main.tsx` bằng thay đổi cơ học, xác minh hoạt động rồi mới thêm Face.
- Dùng app shell, project session, recovery key và API client chung.
- Giữ visual style hiện tại; không bê nguyên frontend riêng của nhánh Dũng.
- Face UI gồm:
  - Dataset/run selection.
  - Canvas zoom/pan.
  - Layer Human/Model, skeleton và vectors.
  - Danh sách rule violations.
  - Point triage và lựa chọn nguồn từng điểm.
  - Hard-case và guideline.
  - Preview, xác nhận batch fix và export.
- Batch fix phải hiển thị số điểm sẽ thay đổi và yêu cầu xác nhận.
- Component chỉ được đưa vào `shared` nếu không chứa giả định về HumanPose hoặc VF50.
- Không tạo frontend `5174` hoặc backend `8001`.

## 9. Quản lý pretrained model

- Không commit file MediaPipe `.task`.
- Thêm `FACE_LANDMARKER_MODEL_PATH`.
- Thêm script tải model theo URL phiên bản cố định và xác minh checksum.
- Model mặc định được lưu trong thư mục đã Git ignore.
- Backend vẫn khởi động khi thiếu model; chỉ Face run tương ứng chuyển sang trạng thái lỗi với hướng dẫn cấu hình rõ ràng.

## 10. Tài liệu

- Chuyển và hiệu chỉnh 5 tài liệu Face từ nhánh Dũng vào `backend/tailieu/face_vf50/`.
- Giữ tài liệu về quy trình, 14 luật, giải phẫu VF50 và công thức scoring sau khi đối chiếu với code tích hợp.
- Viết lại tài liệu kiến trúc/API theo mô hình Project-Dataset-Run và cấu trúc feature mới.
- Cập nhật README với:
  - Cách tải/cấu hình model.
  - Backend `8000`, frontend `5173`.
  - URL `/face`.
  - Luồng import -> run -> review -> export.

## 11. Kiểm thử

- Chạy regression test HumanPose sau mỗi nhóm di chuyển file/import.
- Unit test đủ 14 luật Face, resampling 478 -> 50, IOD/NME, scoring, hard-case và temporal ordering.
- Test parser cho tất cả định dạng được hỗ trợ.
- Test ZIP độc hại, file trùng, annotation thiếu và dữ liệu nhiều khuôn mặt.
- Test round-trip import -> review -> export -> import, bảo toàn tọa độ đã duyệt, kích thước ảnh và frame index.
- Test migration trên bản sao DB có dữ liệu HumanPose.
- Test recovery key sai, truy cập chéo project, model thiếu, inference lỗi và concurrent runs.
- Chạy backend test, TypeScript build và Vite build.
- Kiểm tra trực quan `/` và `/face`, gồm loading, error, empty state, canvas và responsive.

## 12. Tiêu chí hoàn thành

- Không có import nghiệp vụ chéo giữa HumanPose và Face Landmark.
- HumanPose giữ nguyên API, thuật toán và kết quả.
- `/face` hoàn thành được luồng import, run, review và export.
- Chỉ cần frontend `5173`, backend `8000` và một SQLite.
- Dữ liệu HumanPose cũ được bảo toàn.
- Nhãn gốc không bị chỉnh sửa.
- Không còn app, database hoặc port Face độc lập.
- Không commit hoặc push tự động.
