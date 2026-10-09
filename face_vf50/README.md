# VinFast Face Landmark VF-50 — Hand-Labeling Copilot & QA Auditor

> **Mã Chuẩn (Schema):** `vf_face_landmark50_v1`  
> **Căn cứ:** Guideline Face Landmark VF-50 v1.3 (Ban Tổ Chức VinFast AI Thực Chiến)  
> **Backend:** FastAPI (Python 3.13) — Port `8001`  
> **Frontend:** React + Vite + TypeScript — Port `5174`  
> **Mô hình AI:** Google MediaPipe Face Landmarker (CPU, ~15-20ms/frame)

---

## 1. Giới Thiệu Dự Án

Hệ thống **VinFast Face Landmark VF-50 QA Auditor** là công cụ trợ lý kiểm định nhãn chất lượng cao dành cho học viên và thí sinh tham gia cuộc thi VinFast AI Thực Chiến:
- **Tuân thủ tuyệt đối quy định:** Ban tổ chức cấm tải nhãn tự động đè lên CVAT (gây mất dữ liệu các job con). Hệ thống hoạt động theo phương thức **Pre-submission Auditor & Copilot**, quét tự động các lỗi giải phẫu và cung cấp bảng hướng dẫn sửa tay chi tiết cho học viên trước khi nộp bài.
- **Tốc độ siêu nhanh với 1 model duy nhất:** Sử dụng Google MediaPipe Face Landmarker trích xuất 478 điểm mốc dày đặc và thuật toán Arc-Length Polyline Resampling để chuẩn hóa về 50 điểm VinFast, chạy hoàn toàn trên CPU với tốc độ $\approx 15\text{ms/khung hình}$.
- **Bộ kiểm định 14 quy tắc Guideline:** Tự động phát hiện các lỗi nghiêm trọng như lật mí mắt (R03), tự cắt chéo đa giác (R04), chia sai sống mũi (R05), nhầm lẫn góc nhìn trái/phải (R06), tràn bờ môi (R07), gán sai kính mắt (R08), lệch điểm neo (R11), và chỉ số NME vượt ngưỡng $0.035$ (R13).
- **Giao diện Web Viewer chuyên nghiệp:** Phóng to $200\% - 400\%$, rê chuột xem toạ độ, bật/tắt các lớp véc-tơ lệch, xếp hạng khung hình theo mức độ lỗi, và xuất báo cáo Checklist Markdown / CSV.

---

## 2. Cấu Trúc Thư Mục Độc Lập (`face_vf50/`)

Mã nguồn và tài liệu của dự án được đóng gói hoàn toàn độc lập trong thư mục `face_vf50/`, không gây ảnh hưởng đến phần code khác của workspace:

```
face_vf50/
├── backend/
│   ├── app/
│   │   ├── io/                # Safe ZIP Extractor (chống Zip Slip) & CVAT Parser (XML/JSON)
│   │   ├── model/             # MediaPipe Tasks API & Arc-length Polyline Resampler (478 -> 50)
│   │   ├── qa/                # Động cơ 14 quy tắc, hình học, scoring, checklist export
│   │   ├── database.py        # Quản trị SQLite (sessions, frames, rule_violations)
│   │   └── main.py            # FastAPI RESTful API server (Port 8001)
│   ├── config/                # vf50_spec.yaml (Đặc tả 50 điểm, 7 skeletons, 12 điểm neo)
│   ├── models/                # face_landmarker.task (MediaPipe Vision task, 3.58MB)
│   ├── scripts/               # Script tải tự động model
│   ├── tests/                 # Unit tests tự động cho Rules và Mapper (11 tests)
│   └── requirements.txt       # Dependencies Python
├── frontend/                  # Web App React + Vite + TypeScript (Port 5174)
│   ├── src/
│   │   ├── components/        # Header, CanvasViewer, FrameList, Inspector, UploadModal
│   │   ├── api.ts             # API client giao tiếp với Backend
│   │   ├── types.ts           # Khai báo TypeScript types
│   │   ├── vf50_constants.ts  # Cấu hình màu sắc 7 skeletons, 12 điểm neo
│   │   └── index.css          # Hệ thống Design Tokens hiện đại, dark mode
│   └── vite.config.ts         # Proxy API tới Port 8001
├── docs/                      # 6 Tài liệu kỹ thuật chuyên sâu độc lập
│   ├── 01_TONG_QUAN_VA_QUY_TRINH.md
│   ├── 02_QUY_TAC_14_LUAT_KIEM_DINH.md
│   ├── 03_GIAI_PHAU_50_DIEM_VA_7_SKELETONS.md
│   ├── 04_CONG_THUC_TOAN_HOC_VA_SCORING.md
│   ├── 05_KIEN_TRUC_HE_THONG_VA_API.md
│   └── 06_HUONG_DAN_THAO_TAC_CHO_HOC_VIEN.md
└── README.md                  # Tài liệu tổng quan này
```

---

## 3. Hướng Dẫn Cài Đặt & Chạy Nhanh

### Bước 1: Khởi Động Backend (Port 8001)
```powershell
# Tại thư mục gốc của repository
python -m uvicorn face_vf50.backend.app.main:app --host 127.0.0.1 --port 8001
```
Kiểm tra sức khỏe Backend: `http://127.0.0.1:8001/api/health`

### Bước 2: Khởi Động Frontend (Port 5174)
```powershell
cd face_vf50\frontend
npm run dev -- --host 127.0.0.1 --port 5174
```
Truy cập giao diện Web: **`http://localhost:5174`**

### Bước 3: Trải Nghiệm Bản Mẫu (Demo Ngay Lập Tức)
Trên giao diện web, bấm nút **`⚡ Thử Bản Mẫu (Demo)`**. Hệ thống sẽ tự động tạo một phiên kiểm định với 5 khung hình minh họa các loại lỗi điển hình (Lật mí R03, Sống mũi méo R05, Tràn môi R07, v.v.).

---

## 4. Chạy Kiểm Thử Tự Động (Unit Tests)

```powershell
python -m unittest discover -s face_vf50/backend/tests
```
Tất cả 11 unit tests kiểm tra 14 quy tắc giải phẫu và thuật toán resample điểm đều đạt chuẩn $100\%$.

---

## 5. Tài Liệu Kỹ Thuật Đính Kèm
Học viên vui lòng tham khảo trọn bộ tài liệu chi tiết tại thư mục [`face_vf50/docs/`](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs):
- [01. Tổng quan & Quy trình làm việc](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/01_TONG_QUAN_VA_QUY_TRINH.md)
- [02. Chi tiết 14 Quy tắc kiểm định](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/02_QUY_TAC_14_LUAT_KIEM_DINH.md)
- [03. Giải phẫu 50 điểm mốc & 7 Skeletons](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/03_GIAI_PHAU_50_DIEM_VA_7_SKELETONS.md)
- [04. Công thức toán học & NME Scoring](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/04_CONG_THUC_TOAN_HOC_VA_SCORING.md)
- [05. Kiến trúc hệ thống & RESTful API](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/05_KIEN_TRUC_HE_THONG_VA_API.md)
- [06. Cẩm nang thao tác thực chiến cho học viên](file:///d:/AI-Thuc%20Chien-Vin/Vin-BuildPhase/face_vf50/docs/06_HUONG_DAN_THAO_TAC_CHO_HOC_VIEN.md)
