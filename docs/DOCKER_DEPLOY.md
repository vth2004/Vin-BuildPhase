# Deploy mới bằng Docker Compose

Compose chạy một backend FastAPI và một frontend Nginx trên một máy. Backend dùng một worker vì run và video import hiện xử lý trong process web. Cần máy Linux x86-64 có Docker Engine và Docker Compose, hoặc Docker Desktop ở chế độ Linux containers.

Backend cài PyTorch bản CPU, khớp với code inference hiện tại. Lần build đầu cần Internet để tải dependency; các lần sau Docker có thể dùng cache layer.

## 1. Chuẩn bị model

Tạo thư mục `deploy-models/` tại gốc repository và đặt đúng bốn file sau:

```text
deploy-models/
├── yolo26s-pose.pt
├── face_landmarker.task
├── yolox_m_8xb8-300e_humanart-c2c7a14a.onnx
└── rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx
```

Trên máy phát triển hiện tại, file YOLO có ở `backend/yolo26s-pose.pt`, file Face ở `backend/models/face_landmarker.task`; hai file ONNX có trong cache `~/.cache/rtmlib/hub/checkpoints/`. Chép chúng sang server bằng kênh truyền file của bạn. Model và thư mục `deploy-models/` không được đưa vào Git. Backend sẽ báo tên file còn thiếu và dừng trước khi phục vụ request.

## 2. Chạy

Ở thư mục gốc repository:

```sh
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Mở `http://localhost:8080` trên máy chạy Docker. Khi truy cập từ máy khác, dùng IP hoặc domain của server. Nginx phục vụ frontend và chuyển `/api/` tới backend, nên trình duyệt gọi API cùng origin.

Các biến tùy chọn có thể đặt trong `.env` tại gốc repository:

```dotenv
APP_PORT=8080
PUBLIC_ORIGIN=http://localhost:8080
MAX_VIDEO_FRAMES=10000
MAX_DECODED_VIDEO_BYTES=5368709120
```

Nếu đặt domain và HTTPS qua reverse proxy phía trước, đổi `PUBLIC_ORIGIN` thành địa chỉ công khai; HTTPS/TLS do reverse proxy đó xử lý. Cần đủ disk cho image, model, source video và toàn bộ frame đã giải mã.

## 3. Dữ liệu và cập nhật

`deploy-data/` được mount vào `/app/data`. SQLite, ảnh, annotation, video, các bản backup migration và kết quả review đều ở đây. Dữ liệu mới ghi các path trong SQLite theo dạng tương đối với `LANDMARK_DATA_DIR`, ví dụ `projects/<project-id>/datasets/<dataset-id>/images`.

Để cập nhật code:

```sh
docker compose up --build -d
```

Sao lưu cả `deploy-data/` sau khi dừng backend để có bản database và file khớp nhau. Không chạy nhiều backend worker hoặc nhiều replica trên cùng SQLite trong cấu hình hiện tại.

## 4. Kiểm tra sau deploy

1. Mở trang chủ và tạo project mới; giữ recovery key ở nơi riêng.
2. Upload ZIP Face/Pose nhỏ và chạy QA thường.
3. Upload video ngắn ở trang Temporal, đợi dataset `ready`, chạy Temporal và mở frame.
4. Tải một file annotation/export để xác nhận dữ liệu và model hoạt động.

Deployment này khởi tạo database mới. Database local cũ không được mount vào `deploy-data/`; các path tuyệt đối Windows trong database đó không thể tự chuyển sang Linux container.
