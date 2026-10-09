# KIẾN TRÚC HỆ THỐNG VÀ TÀI LIỆU RESTful API

> **Dự án:** VinFast Face Landmark VF-50 QA Auditor  
> **Backend Port:** `8001`  
> **Frontend Port:** `5174`  
> **Cơ sở dữ liệu:** SQLite (`face_vf50.sqlite3`)

---

## 1. Sơ Đồ Kiến Trúc Hệ Thống (Architecture Diagram)

```
+---------------------------------------------------------------------------------------+
|                                    TRÌNH DUYỆT WEB                                     |
|                                (React + Vite + TypeScript)                            |
|                                       Port: 5174                                      |
|                                                                                       |
|  +--------------------+   +------------------------------------+   +---------------+  |
|  |     FrameList      |   |            CanvasViewer            |   |   Inspector   |  |
|  |  (Bộ lọc, sắp xếp  |   |    (Zoom 400%, Pan, SVG Vector,    |   |  (NME, IOD,   |  |
|  |   mức độ lỗi)      |   |     Overlay AI vs Nhãn gán tay)    |   |  14 Quy tắc)  |  |
|  +--------------------+   +------------------------------------+   +---------------+  |
+-------------------------------------------|-------------------------------------------+
                                            | Proxy: /api/* -> :8001
                                            v
+---------------------------------------------------------------------------------------+
|                                    BACKEND FASTAPI                                    |
|                                       Port: 8001                                      |
|                                                                                       |
|  +---------------------+   +---------------------+   +-----------------------------+  |
|  |     io / parser     |   |    model / inference|   |        qa / rules engine    |  |
|  | - Safe ZIP Extractor|   | - MediaPipe Tasks   |   | - 14 Luật Guideline VF-50   |  |
|  | - CVAT XML / JSON   |   | - 478 -> 50 Mapper  |   | - Tính sai số NME & IOD     |  |
|  +---------------------+   +---------------------+   +-----------------------------+  |
|                                           |                                           |
|                                           v                                           |
|                     +-------------------------------------------+                     |
|                     |             SQLite Database               |                     |
|                     |    sessions, frames, rule_violations      |                     |
|                     +-------------------------------------------+                     |
+---------------------------------------------------------------------------------------+
```

---

## 2. Chi Tiết Danh Sách RESTful API Endpoints

### 2.1. Kiểm Tra Sức Khỏe Hệ Thống
- **Endpoint:** `GET /api/health`
- **Mô tả:** Kiểm tra trạng thái hoạt động của Backend và model MediaPipe.
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "vf50_face_qa_backend",
  "version": "1.3.0",
  "port": 8001,
  "model": "MediaPipe Face Landmarker (CPU)"
}
```

---

### 2.2. Tải Lên Dataset Kiểm Định
- **Endpoint:** `POST /api/upload`
- **Content-Type:** `multipart/form-data`
- **Tham số Request:**
  - `dataset_zip` *(bắt buộc)*: File ZIP chứa ảnh các khung hình (`.zip`).
  - `annotation_file` *(tùy chọn)*: File nhãn xuất từ CVAT (XML "CVAT for images 1.1" hoặc JSON).
  - `session_name` *(tùy chọn)*: Tên định danh cho đợt kiểm định.
  - `run_model` *(boolean, mặc định true)*: Có chạy model MediaPipe để lấy tham chiếu tính NME không.
- **Response (200 OK):**
```json
{
  "session_id": "9b17d735",
  "name": "Batch_Face_Task1",
  "total_frames": 120,
  "total_issues": 14,
  "avg_nme": 0.0245
}
```

---

### 2.3. Tạo Phiên Dữ Liệu Bản Mẫu (Demo)
- **Endpoint:** `POST /api/demo/create-sample`
- **Mô tả:** Tự động sinh ra 5 khung hình mẫu minh họa các lỗi tiêu biểu (Lật mí R03, Mũi méo R05, Tràn môi R07, v.v.) để học viên trải nghiệm giao diện ngay lập tức mà không cần upload dữ liệu lớn.
- **Response (200 OK):**
```json
{
  "session_id": "demo_a1b2c3",
  "name": "Interactive Demo (5 Frames, 7 Issues)",
  "total_frames": 5,
  "total_issues": 7,
  "message": "Demo session created successfully!"
}
```

---

### 2.4. Lấy Danh Sách Các Phiên Kiểm Định
- **Endpoint:** `GET /api/sessions`
- **Response (200 OK):** Danh sách các session được sắp xếp theo thời gian mới nhất.

---

### 2.5. Lấy Danh Sách Khung Hình Của Phiên (Có Phân Trang & Lọc)
- **Endpoint:** `GET /api/sessions/{session_id}/frames`
- **Query Parameters:**
  - `page`: Số trang (mặc định 1).
  - `page_size`: Kích thước trang (mặc định 50, tối đa 200).
  - `sort_by`: Tiêu chí sắp xếp:
    - `severity`: Ưu tiên khung hình có lỗi nặng nhất lên đầu (Mặc định).
    - `frame_index`: Thứ tự khung hình từ 0 đến N.
    - `nme`: Giá trị NME từ cao xuống thấp.
  - `filter_rule`: Lọc theo mã lỗi (Ví dụ: `R03`, `R05`, `R07`, hoặc `all`).
- **Response (200 OK):**
```json
{
  "total": 5,
  "page": 1,
  "page_size": 50,
  "items": [
    {
      "id": 1,
      "session_id": "demo_a1b2c3",
      "frame_index": 1,
      "image_filename": "frame_0001.jpg",
      "iod": 100.0,
      "nme": 0.042,
      "severity_score": 8.0,
      "error_count": 3,
      "status": "pending",
      "rule_violations": [
        {
          "code": "R03",
          "rule_name": "Mí trên thấp hơn mí dưới (Lật mí)",
          "level": "ERROR",
          "severity": "critical",
          "message": "Điểm mí trên 16 có vị trí thấp hơn mí dưới 20.",
          "points": [16, 20],
          "gl_section": "Mục 5.1 & 5.2"
        }
      ]
    }
  ]
}
```

---

### 2.6. Lấy Chi Tiết Khung Hình (Kèm Toạ Độ 50 Điểm)
- **Endpoint:** `GET /api/sessions/{session_id}/frames/{frame_index}`
- **Response (200 OK):** Chứa toạ độ đầy đủ của nhãn người gán `human_keypoints`, toạ độ AI tham chiếu `model_keypoints`, chỉ số IOD, NME, và toàn bộ lỗi vi phạm.

---

### 2.7. Cập Nhật Trạng Thái Rà Soát Của Khung Hình
- **Endpoint:** `POST /api/sessions/{session_id}/frames/{frame_index}/status`
- **Form Data:**
  - `status`: Giá trị nhận vào là `pending` (Chưa sửa), `reviewed` (Đã rà soát), hoặc `fixed` (Đã sửa).

---

### 2.8. Xuất Báo Cáo Checklist Dạng Markdown
- **Endpoint:** `GET /api/sessions/{session_id}/export/checklist`
- **Mô tả:** Trả về file text dạng Markdown (`checklist_{session_id}.md`) tổng hợp toàn bộ các khung hình bị lỗi, độ lệch từng điểm và hướng dẫn thao tác để học viên mở song song đối chiếu sửa tay trên CVAT.

---

### 2.9. Xuất Báo Cáo Bảng Dạng CSV
- **Endpoint:** `GET /api/sessions/{session_id}/export/csv`
- **Mô tả:** Trả về file bảng tính CSV (`qa_report_{session_id}.csv`) để phân công công việc rà soát cho nhóm học viên.
