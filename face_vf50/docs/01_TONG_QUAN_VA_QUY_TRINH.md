# TỔNG QUAN HỆ THỐNG VÀ QUY TRÌNH KIỂM ĐỊNH NHÃN KHUÔN MẶT VINFAST VF-50

> **Dự án:** VinFast Face Landmark VF-50 Hand-Labeling Copilot & QA Auditor  
> **Schema ID:** `vf_face_landmark50_v1`  
> **Tài liệu căn cứ:** `Week2_Guideline_Face_Landmark_VF50_HocVien_v1.3.docx` (Ban Tổ Chức VinFast)  
> **Phạm vi lưu trữ:** Độc lập hoàn toàn trong thư mục `face_vf50/`  

---

## 1. Bối Cảnh Và Yêu Cầu Cốt Lõi Từ Ban Tổ Chức

Trong tuần 2 của cuộc thi VinFast AI Thực Chiến, bài toán gán nhãn mốc khuôn mặt (Face Landmark) sử dụng bộ tiêu chuẩn **50 điểm mốc** phân bổ trên **7 skeleton** của khuôn mặt người tài xế trong cabin ô tô:
- Độ phân giải ảnh tiêu chuẩn: **$1280 \times 720$ pixels** (tỉ lệ 16:9).
- Khoảng cách hai mắt (Inter-Ocular Distance - IOD): Dao động trung bình khoảng **$96$ đến $105$ pixels**.
- Môi trường cabin: Chụp trong điều kiện ánh sáng thay đổi, ban ngày, ban đêm (hồng ngoại), tài xế đeo kính, nhắm mắt khi buồn ngủ, ngáp hoặc cười nói.

### ⚠️ Ràng Buộc Sống Còn: TUYỆT ĐỐI CẤM UPLOAD FILE NHÃN ĐÈ LÊN CVAT
Tại **Mục 1.2** của Guideline Ban Tổ Chức quy định rõ:
> *Học viên bắt buộc phải gắn nhãn hoàn toàn bằng tay trên giao diện web của CVAT. Khi tải file annotation lên task CVAT, toàn bộ các job con sẽ bị ghi đè dữ liệu và mất hết công sức đã làm trước đó của học viên.*

Do đó, **không một công cụ nào được phép tự động upload nhãn đè lên CVAT**.

---

## 2. Giải Pháp: Hand-Labeling Copilot & Pre-submission QA Auditor

Thay vì can thiệp ghi đè dữ liệu, hệ thống **Face VF-50 QA Auditor** được thiết kế làm **trợ lý đắc lực kiểm định trước khi nộp bài (Pre-submission Auditor)**:

```
+------------------------------------------------------------------------------------+
|                                 QUY TRÌNH HOẠT ĐỘNG                                 |
+------------------------------------------------------------------------------------+
| 1. Học viên gán nhãn thủ công trên CVAT (Hand-labeling)                            |
|    |                                                                               |
|    v                                                                               |
| 2. Xuất dữ liệu từ CVAT: "CVAT for images 1.1" (XML) hoặc JSON                     |
|    |                                                                               |
|    v                                                                               |
| 3. Nạp vào Web Auditor (Port 5174 / Backend Port 8001):                           |
|    +-- Bộ lọc quét tự động 14 Quy tắc Guideline (R01 -> R14)                       |
|    +-- Model duy nhất MediaPipe Face Landmarker (CPU ~15ms) đối chiếu sai số       |
|    +-- Tính sai số chuẩn hóa toàn ảnh NME (Yêu cầu <= 0.035 ~ 3.4px)               |
|    |                                                                               |
|    v                                                                               |
| 4. Xếp hạng các khung hình lỗi nặng nhất lên đầu:                                  |
|    +-- Phóng to 200% - 400% soi trực quan điểm gán nhãn vs đường giải phẫu        |
|    +-- Xem giải thích mã lỗi và hướng dẫn sửa tay ngay trên giao diện              |
|    +-- Xuất bảng Checklist Markdown (.md) và CSV để đối chiếu sửa trên CVAT        |
+------------------------------------------------------------------------------------+
```

---

## 3. Kiến Trúc Cô Lập (Zero-Pollution Architecture)

Toàn bộ mã nguồn, cấu hình, dữ liệu và tài liệu của MVP Face VF-50 được đặt hoàn toàn trong thư mục `face_vf50/`, không làm ảnh hưởng hay trộn lẫn với mã nguồn HumanPose-17 ở thư mục gốc:

```
face_vf50/
├── backend/
│   ├── app/
│   │   ├── io/          # Bộ giải nén an toàn ZIP & Parser nhãn CVAT (XML/JSON)
│   │   ├── model/       # MediaPipe Tasks API & Arc-length Polyline Resampler
│   │   ├── qa/          # Động cơ 14 luật kiểm định, hình học, scoring, checklist
│   │   ├── database.py  # SQLite lưu trữ session và frame evaluations
│   │   └── main.py      # FastAPI server chạy trên port 8001
│   ├── config/          # vf50_spec.yaml (chuẩn 50 điểm, tỉ lệ cung, dung sai)
│   ├── models/          # face_landmarker.task (Google MediaPipe chính thức, 3.58MB)
│   ├── tests/           # Kiểm thử tự động (Unit tests cho rules & mapper)
│   └── requirements.txt # Thư viện phụ thuộc Python
├── frontend/            # Web Viewer React + Vite + TypeScript chạy trên port 5174
├── docs/                # 6 tài liệu kỹ thuật chuyên sâu độc lập
└── README.md            # Hướng dẫn chạy nhanh dự án
```

---

## 4. Bảng Chỉ Tiêu Chất Lượng Cần Đạt (KPI)

| Hạng mục | Ngưỡng yêu cầu trong Guideline | Cách Auditor kiểm soát |
| :--- | :--- | :--- |
| **NME Toàn Ảnh** | $\le 0.035$ ($\approx 3.4$ px với IOD = 96px) | Tự động tính trung bình sai lệch $\frac{1}{50 \cdot \text{IOD}} \sum e_i$ |
| **12 Điểm Neo (Anchor)** | Sai lệch $\le 3\% \times \text{IOD}$ ($\approx 2.9$ px) | Bắt lỗi R11, đánh dấu viền vàng 2 vòng |
| **38 Điểm Đường Viền** | Sai lệch $\le 5\% \times \text{IOD}$ ($\approx 4.8$ px) | Bắt lỗi R12, cảnh báo véc-tơ màu đỏ |
| **Lật Mí Mắt (Eyelid Inversion)** | Tuyệt đối cấm mí trên thấp hơn mí dưới | Bắt lỗi R03, kiểm tra toạ độ xoay theo góc nghiêng mắt |
| **Đường Viền Cắt Chéo (Chữ X)** | Tuyệt đối không tự cắt chéo | Bắt lỗi R04 bằng giao điểm đoạn thẳng |
| **Chia Đều Sống Mũi** | Tỉ lệ đoạn dài nhất / ngắn nhất $\le 1.45$ | Bắt lỗi R05, phân tích 3 đoạn giữa 4 điểm 10-13 |
| **Tràn Môi Trong** | Môi trong phải nằm trọn trong môi ngoài | Bắt lỗi R07 bằng thuật toán Ray-Casting Polygon |
| **Kính Mắt & Điểm Khuất** | Gán lên mí mắt thật, trạng thái `occluded` | Bắt lỗi R08 |
