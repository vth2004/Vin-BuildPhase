# PROJECT CONTEXT: Landmark QA Web Application

## 1. Tổng quan Dự án (Project Overview)
Hệ thống **Landmark Quality Assurance (QA)** tự động thẩm định chất lượng gán nhãn cho các tập dữ liệu thị giác máy tính dạng Landmark/Keypoint.
- Mục tiêu chính: Phát hiện nhãn người gán bị lệch tọa độ, gán nhầm đối tượng, hoặc hoán đổi đối xứng trái/phải (`swap_error`).
- Hỗ trợ các chuẩn nhãn:
  - `vf_humanpose17_v1`: Chuẩn VinFast HumanPose 17 khớp (COCO-17 đối xứng với quy ước chẵn: right, lẻ: left).
  - `vf_face_landmark50_v1`: Chuẩn khuôn mặt 50 điểm (mock).

---

## 2. Kiến trúc Hệ thống (System Architecture)
```text
backend/app/
├── main.py                     # Entry point FastAPI, CORS & Routers
├── core/
│   ├── config.py               # Biến môi trường & cấu hình hệ thống
│   ├── database.py             # SQLite connection & lightweight migration
│   └── security.py             # Xác thực project_guard qua Recovery Key
├── adapters/                   # Lớp cắm mô hình AI (Model-Agnostic Interface)
│   ├── base.py                 # BasePosePredictor interface
│   ├── yolo_pose.py            # YoloPoseAdapter (YOLO26s-pose)
│   ├── rtm_pose.py             # RtmPosePredictor (RTMPose-m qua rtmlib/ONNX)
│   └── yolo8m_pose.py          # Yolo8mPosePredictor (tùy chọn K=3)
├── services/
│   ├── ensemble.py             # Toán học Ensemble: p*, delta, e, R, score, IoU consensus
│   ├── scoring.py              # Dịch vụ chấm điểm nghi ngờ đa mô hình
│   ├── dataset_service.py      # Giải nén an toàn & validate schema
│   └── mock_service.py         # Mock pipeline cho Face-50
└── routers/
    ├── projects.py             # API dự án
    ├── datasets.py             # API dataset & ảnh
    ├── runs.py                 # API khởi chạy & tiến độ Run
    └── warnings.py             # API danh sách cảnh báo, review & xuất JSON
```

---

## 3. Thiết kế Cấu hình Ensemble trong Production

### 3.1. Các Chế độ Hoạt động (Operating Modes)
Cấu hình qua biến môi trường `QA_ENSEMBLE_MODE`:
1. **$K=2$ (Mặc định - `QA_ENSEMBLE_MODE=K2`):**
   - Kết hợp **$M_0$: YOLO26s-pose** + **$M_3$: RTMPose-m** (ONNXRuntime CPU).
   - Điểm xếp hạng kết hợp: $Score_i = e_i \cdot R_i^\alpha$.
   - Giảm tới gần $50\%$ báo động oan ở ca khó, tăng Average Precision (AP) $+13.3\%$.
2. **$K=1$ (Chế độ Nhanh - `QA_ENSEMBLE_MODE=K1`):**
   - Chỉ chạy **YOLO26s-pose** đơn lẻ với công thức OKS $\sigma_i$.
   - Dùng khi cần phản hồi tức thì với tài nguyên CPU tối thiểu.
3. **$K=3$ (Tùy chọn Mở rộng - `QA_ENSEMBLE_MODE=K3`):**
   - Kết hợp cả 3 mô hình ($M_0$ YOLO26s + $M_3$ RTMPose-m + $M_2$ YOLOv8m).

### 3.2. Cơ chế An toàn & Tự phục hồi (Fail-safe Fallback)
- Nếu thư viện `rtmlib` hoặc ONNX runtime của RTMPose gặp sự cố lúc khởi tạo hoặc dự đoán:
  - Hệ thống tự động ghi log cảnh báo và **fallback về chế độ `K1_fallback`**.
  - Không bao giờ làm gián đoạn hoặc sập tiến trình chạy nền.
  - Cột `mode` trong bảng `runs` được cập nhật thành `K1_fallback` để minh bạch thông tin.

---

## 4. Công thức Toán học Cốt lõi (Ensemble Formulations)

### 4.1. Vị trí Tham chiếu Đồng thuận ($p^*_i$)
$$p^*_i = \frac{\sum_{k=1}^K w_{k,i} c_{k,i} p_{k,i}}{\sum_{k=1}^K w_{k,i} c_{k,i}}$$
- Trọng số nghịch bình phương MPE đo trên Dev: $w(M_0) = 2026.5$, $w(M_3) = 2854.5$, $w(M_2) = 1827.1$.

### 4.2. Độ Bất đồng Không gian ($\delta_i$)
$$\delta_i = \frac{1}{s}\sqrt{\frac{\sum_{k=1}^K w_{k,i} c_{k,i} \|p_{k,i} - p^*_i\|^2}{\sum_{k=1}^K w_{k,i} c_{k,i}}}$$
- $s = \sqrt{W \cdot H}$ của bounding box người để chuẩn hóa tỉ lệ khoảng cách.
- Khi $K=1$ hoặc các model trùng nhau: $\delta_i = 0$.

### 4.3. Điểm Nghi ngờ Chuẩn hóa COCO OKS ($e_i$)
$$e_i = 1 - \exp\left(-\frac{(d_i/s)^2}{2\,(k_i^2 + \lambda\,\delta_i^2)}\right)$$
- $d_i = \|h_i - p^*_i\|$: khoảng cách giữa điểm người gán ($h_i$) và điểm tham chiếu ($p^*_i$).
- $k_i = 2\sigma_i$: dung sai giải phẫu chuẩn COCO (mắt: $\sigma=0.025$, hông: $\sigma=0.107$).
- $\lambda = 0.5$: hệ số điều hòa bất đồng.

### 4.4. Độ Tin cậy Đánh giá ($R_i$) & Phạt Thiếu Giám khảo ($\rho$)
$$R_i = \rho(K_{\text{eff}}) \cdot \bar{c}_i \cdot \exp\left(-\frac{\delta_i^2}{2\tau^2}\right)$$
- $\tau = 0.08$: dung sai bất đồng.
- $\rho(K_{\text{eff}}) = K_{\text{eff}} / K_{\text{target}}$: phạt khi có model bỏ sót người hoặc không đồng thuận bbox ($\text{IoU} < 0.30$).

### 4.5. Điểm Xếp hạng Cảnh báo (Ranking Score)
$$\text{Score}_i = e_i \cdot R_i^\alpha \qquad (\alpha = 0.5)$$

---

## 5. Cơ sở Dữ liệu & Lightweight Migration
Bảng `warnings`:
- Bổ sung cột `reliability REAL` (độ tin cậy $R \in [0, 1]$).
- Bổ sung cột `delta REAL` (độ bất đồng $\delta$).

Bảng `runs`:
- Bổ sung cột `mode TEXT DEFAULT 'K2'`.

Cơ chế migration qua `PRAGMA table_info` tự động kiểm tra và thêm cột khi thiếu mà không làm gián đoạn hay mất dữ liệu SQLite hiện có.

---

## 6. Kết quả Thực nghiệm trên `humanpose1` (20 ảnh VinFast)
- **Chế độ K=1 (Nhanh):** Phát hiện 19 cảnh báo ($\ge 20\%$).
- **Chế độ K=2 (Ensemble):** Phát hiện 10 cảnh báo ($\ge 20\%$).
- **Mức giảm cảnh báo oan:** Giảm **$47.4\%$** các điểm nghi ngờ gây nhiễu, tập trung chính xác vào các lỗi có độ tin cậy đồng thuận cao.
- **Thời gian xử lý:** ~409 ms/ảnh trên CPU tuần tự.
