# Landmark QA Web Application

Hệ thống kiểm định chất lượng nhãn Landmark/Keypoint bằng cách so sánh nhãn người gán với dự đoán AI.

Hai feature nghiệp vụ được tách độc lập:

- `vf_humanpose17_v1`: HumanPose 17 điểm, ensemble YOLO/RTMPose.
- `vf_face_landmark50_v1`: Face Landmark VF50, MediaPipe Face Landmarker và 14 luật guideline.

## Kiến trúc

```text
backend/app/
|-- features/
|   |-- human_pose/                 # Adapter, ensemble, scoring và parser HumanPose
|   `-- face_landmark_vf50/        # Adapter, parser, 14 rules, scoring, review và export VF50
|-- application/                    # Điều phối dataset, video import, review và export
|-- shared/
|   |-- database/                   # Biên truy cập SQLite dùng chung
|   |-- datasets/                   # Safe ZIP extraction dùng chung
|   `-- jobs/                       # Registry và dispatch background run
|-- core/                           # Cấu hình, SQLite, recovery-key security
|   `-- migrations/                 # Schema và migration có thứ tự, giữ tương thích DB cũ
|-- routers/                        # Project, dataset, run và HumanPose warning API
`-- main.py                         # Composition root đăng ký hai feature

frontend/src/
|-- app/                            # Router và app shell
|-- features/
|   |-- human-pose/
|   |-- face-landmark-vf50/
|   |-- temporal/
|   `-- video/
|-- shared/                         # API, session, types và polling dùng chung
`-- styles/                         # CSS theo phạm vi, import theo thứ tự
```

Hai feature không được import logic nghiệp vụ của nhau. Tầng `application` phối hợp các feature; `shared` không phụ thuộc ngược vào feature/application. Các compatibility re-export cũ đã được xóa.

Xem [bản đồ kiến trúc và hướng dẫn refactor](docs/ARCHITECTURE.md).

## Chạy local

Triển khai mới bằng Docker Compose: xem [hướng dẫn deploy](docs/DOCKER_DEPLOY.md).

### Backend — cổng 8000

Yêu cầu Python 3.10+.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_face_landmarker.py
uvicorn app.main:app --reload --port 8000
```

Có thể đặt model ở vị trí khác bằng biến môi trường `FACE_LANDMARKER_MODEL_PATH`. Script download kiểm tra SHA-256 của model đã ghim phiên bản; file `.task` không được commit.

API docs: http://localhost:8000/docs

### Frontend — cổng 5173

```powershell
cd frontend
npm install
npm run dev -- --port 5173
```

- HumanPose: http://localhost:5173/
- Face Landmark VF50: http://localhost:5173/face

## Luồng sử dụng

1. Tạo hoặc mở lại Project bằng Recovery key.
2. Upload ZIP ảnh và annotation tương ứng.
3. Chọn dataset và tạo run.
4. Xem overlay Human/Model, rule violations và các ca khó.
5. Chọn Human hoặc Model theo từng điểm; batch fix luôn yêu cầu preview và xác nhận.
6. Export JSON sạch, CVAT XML, CSV hoặc checklist.

Face chấp nhận CVAT XML, COCO JSON, `landmark-qa/v1`, cleaned JSON hoặc ZIP chỉ có ảnh. HumanPose tiếp tục dùng định dạng hiện có.

## Database và bảo toàn nhãn

- Dùng một SQLite tại `backend/data/landmark_qa.sqlite3`.
- Các bảng lõi Project–Dataset–Image–Annotation–Run dùng chung.
- `face_frame_results` và `face_point_reviews` chỉ thuộc Face.
- Annotation nguồn không bị chỉnh sửa; export được dựng từ nhãn gốc và quyết định review.
- Migration tạo backup có timestamp trước lần nâng schema đầu tiên và không xóa dữ liệu cũ.

## Bảo mật

- Mọi API thuộc Project yêu cầu `Authorization: Bearer <recovery-key>`.
- URL ảnh có thể dùng `?token=<recovery-key>` để trình duyệt tải trực tiếp.
- ZIP được kiểm tra path traversal, symlink, đường dẫn trùng, số file và tổng kích thước giải nén.
- Thư mục dữ liệu, database, model và artifact build đều nằm ngoài Git.

## Kiểm thử

```powershell
cd backend
python -m unittest discover -s tests -v

cd ..\frontend
npm run build
```

Tài liệu nghiệp vụ Face nằm tại `backend/tailieu/face_vf50/`.
