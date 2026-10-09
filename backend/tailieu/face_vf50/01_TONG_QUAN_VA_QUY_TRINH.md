# TỔNG QUAN HỆ THỐNG VÀ QUY TRÌNH KIỂM ĐỊNH NHÃN KHUÔN MẶT VINFAST VF-50

> **Dự án:** VinFast Face Landmark VF-50 Hand-Labeling Copilot & QA Auditor  
> **Schema ID:** `vf_face_landmark50_v1`  
> **Tài liệu căn cứ:** `Week2_Guideline_Face_Landmark_VF50_HocVien_v1.3.docx` (Ban Tổ Chức VinFast)  
> **Phạm vi lưu trữ:** Feature độc lập trong `backend/app/features/face_landmark_vf50/`  

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
| 3. Nạp vào Web Auditor (`/face` Port 5173 / Backend Port 8000):                   |
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

## 3. Kiến trúc feature cô lập

Face VF50 và HumanPose chạy trong cùng ứng dụng nhưng không import logic nghiệp vụ của nhau. Chúng chỉ dùng chung hạ tầng Project, Recovery key, database, storage và run dispatcher:

```
backend/app/
├── features/
│   ├── human_pose/                 # Nghiệp vụ HumanPose độc lập
│   └── face_landmark_vf50/        # Parser, model, rules, scoring, API và export Face
├── shared/                         # Database boundary, safe ZIP và run dispatcher
└── main.py                         # Composition root đăng ký cả hai feature

frontend/src/
├── features/human-pose/            # Route `/`
└── features/face-landmark-vf50/   # Route `/face`
```

---

## 4. Bảng Chỉ Tiêu Chất Lượng Cần Đạt (KPI)

| Hạng mục | Ngưỡng yêu cầu trong Guideline | Cách Auditor kiểm soát |
| :--- | :--- | :--- |
| **NME Toàn Ảnh** | $\le 0.035$ ($\approx 3.4$ px với IOD = 96px) | Tự động tính trung bình sai lệch $\frac{1}{50 \cdot \text{IOD}} \sum e_i$ |
| **12 Điểm Neo (Anchor)** | Sai lệch $\le 3\% \times \text{IOD}$ ($\approx 2.9$ px) | Scoring đánh dấu điểm vượt dung sai |
| **38 Điểm Đường Viền** | Sai lệch $\le 5\% \times \text{IOD}$ ($\approx 4.8$ px) | Scoring và vector Human–Model |
| **Lật Mí Mắt (Eyelid Inversion)** | Tuyệt đối cấm mí trên thấp hơn mí dưới | R06, bù góc nghiêng đầu |
| **Đường Viền Cắt Chéo (Chữ X)** | Tuyệt đối không tự cắt chéo | R07, giao điểm đoạn thẳng |
| **Chia Đều Sống Mũi** | Tỉ lệ đoạn dài nhất / ngắn nhất $\le 1.45$ | R08, phân tích 3 đoạn giữa điểm 10–13 |
| **Tràn Môi Trong** | Môi trong phải nằm trọn trong môi ngoài | R09, Ray-Casting Polygon |
| **Kính Mắt & Điểm Khuất** | Gán lên mí mắt thật, trạng thái `occluded` | R10 và R14 |
