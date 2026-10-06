# Landmark QA Web Application

Hệ thống thẩm định chất lượng nhãn Landmark/Keypoint (Landmark Quality Assurance) — Tự động phát hiện nhãn gán lệch, hoán đổi trái/phải và các điểm nghi ngờ thông qua so sánh giữa **Nhãn người gán** và **Dự đoán từ Model AI**.

Hỗ trợ chuẩn nhãn VinFast: `vf_humanpose17_v1` (tích hợp model **YOLO26s-pose**) và `vf_face_landmark50_v1`.

---

## Cấu trúc dự án Backend (Modular Layered Architecture)

Backend được tổ chức theo kiến trúc chuẩn của FastAPI:

```text
backend/app/
├── main.py                     # Entry point: Khởi tạo FastAPI, CORS & gắn Routers
│
├── core/                       # Cấu hình hệ thống, Bảo mật & Database
│   ├── config.py               # Hằng số, biến môi trường (DATA_DIR, SCHEMAS_DIR, ...)
│   ├── database.py             # Kết nối SQLite & migration bảng
│   └── security.py             # Xác thực project_guard, băm SHA-256 recovery key
│
├── schemas/                    # Pydantic Schemas (Data Transfer Objects)
│   ├── project.py              # CreateProject DTO
│   └── warning.py              # ReviewWarning DTO
│
├── services/                   # Xử lý nghiệp vụ (Business Logic)
│   ├── dataset_service.py      # Giải nén an toàn (anti-Zip Slip), validate schema
│   ├── ensemble.py             # Thuật toán Ensemble: p*, delta, e, R, score, IoU consensus
│   ├── scoring.py              # Dịch vụ chấm điểm nghi ngờ đa mô hình (K=1, K=2, K=3)
│   └── mock_service.py         # Trình điều phối chạy nền & fallback kết quả giả lập
│
├── routers/                    # Tầng API Endpoints (APIRouter)
│   ├── projects.py             # /api/projects, /api/projects/{id}
│   ├── datasets.py             # /api/projects/{id}/datasets, get_image
│   ├── runs.py                 # /api/projects/{id}/runs, get_run
│   └── warnings.py             # /api/projects/{id}/runs/{id}/warnings, review, export
│
└── adapters/                   # Lớp cắm Model AI tự do (Model-Agnostic)
    ├── base.py                 # BasePosePredictor interface
    ├── yolo_pose.py            # YoloPoseAdapter (YOLO26s-pose)
    ├── rtm_pose.py             # RtmPosePredictor (RTMPose-m qua rtmlib/ONNX)
    └── yolo8m_pose.py          # Yolo8mPosePredictor (YOLOv8m-pose cho K=3)
```

---

## Chạy Local

### 1. Backend (Python 3.10+)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend (Node 20+)

```powershell
cd frontend
npm install
npm run dev
```
Giao diện ứng dụng: [http://localhost:5173](http://localhost:5173)

---

## Luồng hoạt động chính

1. **Tạo Project:** Sinh mã `Recovery Key` để xác thực cho toàn bộ thao tác sau này.
2. **Upload Dataset:** Nhận file `.zip` chứa ảnh và file `.json` annotation (COCO Keypoints từ CVAT hoặc `landmark-qa/v1`).
3. **Quét lỗi (Run QA):** Background task tự động đưa ảnh vào model AI (**`yolo26s-pose.pt`**) để dự đoán lại, tính khoảng cách lệch chuẩn hóa và bắt lỗi hoán đổi trái/phải (`swap_error`).
4. **Review trực quan:** Khung Canvas hiển thị overlay so sánh giữa điểm người gán và điểm model gợi ý kèm các phím tắt nhanh:
   - <kbd>K</kbd>: **Giữ nhãn** (xác nhận người gán đúng, model sai).
   - <kbd>M</kbd>: **Lấy Model** (thay thế điểm sai bằng gợi ý của model).
   - <kbd>S</kbd>: **Bỏ qua** (trường hợp khó/chưa quyết định).
5. **Export dữ liệu:** Xuất file annotation JSON sạch đã được tự động sửa các điểm sai theo quyết định của người review.

---

## Bảo mật

- Ảnh/ZIP được lưu trong `backend/data/projects/<project_id>/`, không public trực tiếp qua static hosting.
- Mọi API thuộc project yêu cầu `Authorization: Bearer <recovery-key>`.
- Cơ chế giải nén an toàn ngăn chặn tấn công Path Traversal / Zip Slip.
- Bản production cần cấu hình thêm HTTPS/TLS, Object Storage (S3/MinIO) có mã hóa và chính sách tự hủy dữ liệu (TTL).
