<p align="center">
  <img src="frontend/public/favicon.svg" alt="Landmark QA" width="64" height="64">
</p>

<h1 align="center">Landmark QA</h1>

<p align="center">
  Công cụ kiểm định và review chất lượng nhãn keypoint cho Human Pose và Face Landmark.
</p>

<p align="center">
  <a href="#tính-năng">Tính năng</a> ·
  <a href="#bắt-đầu">Bắt đầu</a> ·
  <a href="docs/DOCKER_DEPLOY.md">Triển khai Docker</a> ·
  <a href="docs/ARCHITECTURE.md">Kiến trúc</a> ·
  <a href="#tài-liệu">Tài liệu</a>
</p>

---

Landmark QA đối chiếu annotation với dự đoán AI, luật hình học và thông tin theo thời gian để tìm các điểm cần kiểm tra. Reviewer xem trực tiếp trên ảnh, chọn tọa độ phù hợp và xuất annotation từ những quyết định đã lưu.

Ứng dụng hỗ trợ cả **dữ liệu đã gán nhãn** và **dữ liệu chưa có nhãn**. Với dữ liệu chưa có nhãn, hệ thống tạo AI draft và đánh giá prediction; kết quả này cần được con người kiểm tra trước khi dùng làm ground truth.

## Tính năng

| Phân hệ | Khả năng chính |
|---|---|
| **Human Pose · 17 điểm** | Ensemble YOLO26s-pose và RTMPose; phát hiện điểm lệch, nghi hoán đổi trái/phải và ca khó; review trên overlay khung xương. |
| **Face Landmark · VF50** | Ánh xạ MediaPipe về 50 điểm; kiểm tra 14 luật guideline và NME; review từng điểm, batch preview và áp dụng sau xác nhận. |
| **Temporal QA** | Phân tích chuỗi ảnh/video, tracking và ước lượng theo thời gian; review điểm Human/Temporal, tải audit JSON/CSV. Đây là tính năng thử nghiệm. |
| **AI draft** | Sinh nhãn khởi tạo từ ZIP ảnh hoặc video chưa gán nhãn; export `landmark-qa/v1`, COCO Keypoints và CVAT XML. |

Các chức năng dùng chung:

- Tạo và mở lại project bằng Project ID và Recovery Key.
- Nhập ZIP ảnh, annotation riêng hoặc video; theo dõi tiến độ và thử lại video import thất bại.
- Lọc, xem overlay và lưu quyết định review mà không ghi đè annotation nguồn.
- Xuất annotation đã review; Face hỗ trợ thêm báo cáo CSV và checklist.

**Công nghệ:** React · TypeScript · Vite · FastAPI · SQLite · Ultralytics · ONNX Runtime · MediaPipe · Docker Compose.

## Bắt đầu

### Yêu cầu môi trường

| Thành phần | Yêu cầu |
|---|---|
| Python | Khuyến nghị **3.11**, cùng phiên bản với Docker image của dự án. |
| Node.js | **22.13+**; Node 24 đã được dùng để chạy frontend tests và build. |
| Model | YOLO Pose, MediaPipe Face Landmarker; RTMPose cho chế độ ensemble `K2`. |
| Kết nối mạng | Cần khi cài dependency và tải model chưa có trong máy/cache. |

### 1. Lấy mã nguồn

```sh
git clone https://github.com/vth2004/Vin-BuildPhase.git
cd Vin-BuildPhase
```

### 2. Chạy backend

Các lệnh dưới đây dùng **PowerShell trên Windows**, chạy từ thư mục gốc repository:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_face_landmarker.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.

Script Face tải model về `backend/models/face_landmarker.task` và kiểm tra SHA-256. YOLO/RTMPose dùng file model được cấu hình hoặc cơ chế tải/cache của thư viện khi chưa có model.

- **Health check:** [localhost:8000/api/health](http://localhost:8000/api/health)
- **Swagger UI:** [localhost:8000/docs](http://localhost:8000/docs)

### 3. Chạy frontend

Mở terminal thứ hai tại thư mục gốc repository:

```powershell
cd frontend
npm ci
npm run dev -- --port 5173
```

| Trang | Địa chỉ local |
|---|---|
| Trang chủ | [localhost:5173](http://localhost:5173) |
| Human Pose | [localhost:5173/pose](http://localhost:5173/pose) |
| Face Landmark | [localhost:5173/face](http://localhost:5173/face) |
| Pose Temporal | [localhost:5173/pose/temporal](http://localhost:5173/pose/temporal) |
| Face Temporal | [localhost:5173/face/temporal](http://localhost:5173/face/temporal) |

### 4. Thực hiện một lượt kiểm định

1. Vào **Human Pose** để tạo hoặc mở project. Lưu Recovery Key để truy cập lại.
2. Chọn phân hệ phù hợp và tải ZIP ảnh cùng annotation, hoặc dùng trang Temporal để nhập video/chuỗi ảnh.
3. Chọn dataset và khởi chạy kiểm định. Với Temporal, kiểm tra thứ tự frame và xác nhận mapping khi được yêu cầu.
4. Xem cảnh báo, đối chiếu overlay và lưu lựa chọn review.
5. Tải annotation đã review, AI draft hoặc báo cáo tương ứng.

Pose sử dụng schema `vf_humanpose17_v1`; Face sử dụng `vf_face_landmark50_v1`. Định dạng đầu vào và export phụ thuộc phân hệ; xem [hướng dẫn dự án](docs/AI_PROJECT_GUIDE.md) để biết cấu trúc dữ liệu chi tiết.

## Triển khai bằng Docker

Chuẩn bị các model trong `deploy-models/` theo [hướng dẫn Docker](docs/DOCKER_DEPLOY.md), sau đó chạy tại thư mục gốc:

```sh
docker compose up --build -d
docker compose ps
```

Ứng dụng được phục vụ tại [localhost:8080](http://localhost:8080). Nginx chuyển tiếp `/api/` tới backend; dữ liệu được lưu bền vững trong `deploy-data/`.

Cấu hình hiện tại dùng **một backend worker** với SQLite và background jobs trong process. Hướng dẫn deploy mô tả đầy đủ model cần có, domain, giới hạn video và cách sao lưu/cập nhật.

## Cấu hình

| Biến môi trường | Ý nghĩa / mặc định |
|---|---|
| `LANDMARK_DATA_DIR` | Thư mục dữ liệu backend; mặc định `backend/data/`. |
| `CORS_ORIGINS` | Các origin được phép, phân cách bằng dấu phẩy; mặc định `http://localhost:5173`. |
| `QA_ENSEMBLE_MODE` | Chế độ Pose: `K1`, `K2` hoặc `K3`; mặc định `K2`. |
| `POSE_MODEL_PATH` | Đường dẫn model YOLO26s-pose tùy chỉnh. |
| `FACE_LANDMARKER_MODEL_PATH` | Đường dẫn model Face; mặc định `backend/models/face_landmarker.task`. |
| `RTMPOSE_DET_MODEL_PATH` / `RTMPOSE_POSE_MODEL_PATH` | Đường dẫn hai model ONNX của RTMPose; cần đặt cả hai khi cấu hình thủ công. |
| `MAX_VIDEO_FRAMES` | Giới hạn số frame giải mã; mặc định `10000`. |
| `MAX_DECODED_VIDEO_BYTES` | Giới hạn dung lượng frame giải mã; mặc định 5 GiB. |
| `VITE_API_URL` | Base URL API frontend; mặc định `http://localhost:8000/api`, Docker dùng `/api`. |

Ví dụ đặt biến backend trước khi chạy Uvicorn trong PowerShell:

```powershell
$env:QA_ENSEMBLE_MODE = "K2"
$env:CORS_ORIGINS = "http://localhost:5173"
```

Backend đọc biến từ môi trường process. File [backend/.env.example](backend/.env.example) là mẫu tham khảo; backend không tự nạp file này. Với frontend, có thể đặt `VITE_API_URL` trong `frontend/.env.local` theo cơ chế của Vite.

## Kiến trúc

```text
backend/app/
├── application/       # Điều phối dataset, video import, review và export
├── features/          # Nghiệp vụ Human Pose và Face VF50
├── core/              # Cấu hình, SQLite, migration và xác thực
├── shared/            # ZIP, schema metadata, temporal engine và job dispatcher
├── routers/           # HTTP endpoints, validation và kiểm tra quyền
└── main.py            # Khởi tạo ứng dụng và đăng ký các phân hệ

frontend/src/
├── app/               # Điều hướng và bố cục ứng dụng
├── features/          # Page, workspace hook và component của từng phân hệ
├── shared/            # API client, session, types và polling
└── styles/            # CSS dùng chung, chia theo phạm vi
```

Pose và Face không import logic nghiệp vụ của nhau. Tầng application phối hợp các feature; tầng shared chứa hạ tầng dùng chung. Xem [tài liệu kiến trúc](docs/ARCHITECTURE.md) để biết ranh giới module và cách mở rộng.

## Kiểm thử

**Backend** — chạy từ thư mục gốc, sau khi cài dependency backend:

```powershell
backend\.venv\Scripts\python.exe -m pip install httpx
backend\.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

`httpx` phục vụ FastAPI TestClient. Bộ test kiểm tra API, parser, review/export, temporal, an toàn ZIP và tương thích migration với database cũ.

**Frontend** — chạy trong `frontend/`:

```sh
npm test
npm run build
```

Tests kiểm tra API client, session và polling. Build kiểm tra TypeScript và tạo bundle production; các kiểm tra này chưa thay thế việc thử luồng tương tác trên trình duyệt.

## Dữ liệu và quyền truy cập

- Project được truy cập bằng Recovery Key; các API thuộc project kiểm tra quyền qua key này.
- Annotation nguồn được giữ nguyên. File export được dựng từ dữ liệu nguồn và các quyết định review đã lưu.
- SQLite mặc định nằm tại `backend/data/landmark_qa.sqlite3`; migration tạo backup trước khi nâng cấp database cần cập nhật.
- Database, dữ liệu người dùng, model, secrets và build artifacts được loại khỏi Git. Sao lưu cả thư mục dữ liệu để giữ database và file tương ứng.
- AI draft có `visibility_confirmed=false`; reviewer cần xác nhận visibility trước khi sử dụng làm nhãn chuẩn.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [Kiến trúc và phát triển](docs/ARCHITECTURE.md) | Bản đồ module, ranh giới phụ thuộc, migration và quy trình kiểm tra. |
| [Hướng dẫn dự án](docs/AI_PROJECT_GUIDE.md) | Nghiệp vụ, API, dữ liệu và công thức Pose/Face. |
| [Triển khai Docker](docs/DOCKER_DEPLOY.md) | Chuẩn bị model, Compose, cấu hình và sao lưu. |
| [Định dạng annotation](backend/ANNOTATION_FORMAT.md) | Tham khảo định dạng nhãn đầu vào. |
| [Tài liệu Face VF50](backend/tailieu/face_vf50/) | Guideline và giải thích nghiệp vụ Face; đối chiếu source/config khi dùng tài liệu lịch sử. |
| [Danh mục tài liệu](docs/README.md) | Các tài liệu tham khảo và thứ tự ưu tiên khi đối chiếu. |
