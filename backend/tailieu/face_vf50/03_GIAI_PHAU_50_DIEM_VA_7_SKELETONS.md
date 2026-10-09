# GIẢI PHẪU 50 ĐIỂM MỐC VÀ 7 SKELETON (SCHEMA VF-50)

> **Căn cứ:** Guideline Face Landmark VF-50 v1.3 — VinFast AI Thực Chiến  
> **Schema ID:** `vf_face_landmark50_v1`  
> **Độ phân giải tiêu chuẩn:** $1280 \times 720$ pixels

---

## 1. Phân Bổ 7 Skeletons Khuôn Mặt

| Tên Skeleton | CVAT ID | Số Lượng Điểm | Dải Index | Dạng Đường | Màu CVAT Constructor | Mô Tả Giải Phẫu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`longmaytrai`** | 116 | 5 | $0 \to 4$ | Mở | **Đỏ cờ** (`#FF0000`) | Bờ trên lông mày trái: 0 đuôi ngoài $\to$ 4 đầu trong |
| **`longmayphai`** | 122 | 5 | $5 \to 9$ | Mở | **Xanh lam nhạt** (`#3399FF`) | Bờ trên lông mày phải: 5 đầu trong $\to$ 9 đuôi ngoài |
| **`songmui`** | 128 | 4 | $10 \to 13$ | Mở | **Vàng tươi** (`#FFFF00`) | Sống mũi: 10 đỉnh $\to$ 11-12 thân $\to$ 13 chân sống mũi |
| **`mattrai`** | 133 | 8 | $14 \to 21$ | Khép kín | **Hồng cánh sen** (`#FF00FF`) | Mắt trái: 14 khoé ngoài $\to$ 15-17 mí trên $\to$ 18 khoé trong $\to$ 19-21 mí dưới |
| **`matphai`** | 142 | 8 | $22 \to 29$ | Khép kín | **Xanh lơ / Cyan** (`#00FFFF`) | Mắt phải: 22 khoé trong $\to$ 23-25 mí trên $\to$ 26 khoé ngoài $\to$ 27-29 mí dưới |
| **`moingoai`** | 151 | 12 | $30 \to 41$ | Khép kín | **Xanh lá chuối** (`#00FF00`) | Viền môi ngoài: 30 khoé trái $\to$ 31-35 môi trên $\to$ 36 khoé phải $\to$ 37-41 môi dưới |
| **`moitrong`** | 164 | 8 | $42 \to 49$ | Khép kín | **Cam tươi** (`#FF9900`) | Viền môi trong: 42 khoé trái $\to$ 43-45 bờ trên $\to$ 46 khoé phải $\to$ 47-49 bờ dưới |

---

## 2. Bảng Tra Cứu Toàn Bộ 50 Điểm Mốc (ID 0 Đến 49)

| ID | Tên Skeleton | Vị Trí Giải Phẫu Cụ Thể | Điểm Neo (Anchor) | Dung Sai Cho Phép |
| :---: | :--- | :--- | :---: | :---: |
| **0** | `longmaytrai` | Đuôi ngoài lông mày trái (nhìn trên ảnh) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **1** | `longmaytrai` | Bờ trên thân ngoài lông mày trái | Không | $\le 5\% \times \text{IOD}$ |
| **2** | `longmaytrai` | Đỉnh vòm bờ trên lông mày trái | Không | $\le 5\% \times \text{IOD}$ |
| **3** | `longmaytrai` | Bờ trên thân trong lông mày trái | Không | $\le 5\% \times \text{IOD}$ |
| **4** | `longmaytrai` | Đầu trong lông mày trái (giáp sống mũi) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **5** | `longmayphai` | Đầu trong lông mày phải (giáp sống mũi) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **6** | `longmayphai` | Bờ trên thân trong lông mày phải | Không | $\le 5\% \times \text{IOD}$ |
| **7** | `longmayphai` | Đỉnh vòm bờ trên lông mày phải | Không | $\le 5\% \times \text{IOD}$ |
| **8** | `longmayphai` | Bờ trên thân ngoài lông mày phải | Không | $\le 5\% \times \text{IOD}$ |
| **9** | `longmayphai` | Đuôi ngoài lông mày phải (nhìn trên ảnh) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **10** | `songmui` | Đỉnh sống mũi (Glabella - giữa hai đầu lông mày) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **11** | `songmui` | Thân sống mũi trên (chia 1/3 phía trên) | Không | $\le 5\% \times \text{IOD}$ |
| **12** | `songmui` | Thân sống mũi dưới (chia 1/3 phía dưới) | Không | $\le 5\% \times \text{IOD}$ |
| **13** | `songmui` | **Chân sống mũi** (ngay trên hai cánh mũi, KHÔNG phải chóp mũi) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **14** | `mattrai` | Khoé ngoài mắt trái (Outer canthus) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **15** | `mattrai` | Mí trên mắt trái - đoạn ngoài | Không | $\le 5\% \times \text{IOD}$ |
| **16** | `mattrai` | Mí trên mắt trái - điểm cao nhất giữa mí | Không | $\le 5\% \times \text{IOD}$ |
| **17** | `mattrai` | Mí trên mắt trái - đoạn trong | Không | $\le 5\% \times \text{IOD}$ |
| **18** | `mattrai` | Khoé trong mắt trái (Inner canthus) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **19** | `mattrai` | Mí dưới mắt trái - đoạn trong | Không | $\le 5\% \times \text{IOD}$ |
| **20** | `mattrai` | Mí dưới mắt trái - điểm thấp nhất giữa mí | Không | $\le 5\% \times \text{IOD}$ |
| **21** | `mattrai` | Mí dưới mắt trái - đoạn ngoài | Không | $\le 5\% \times \text{IOD}$ |
| **22** | `matphai` | Khoé trong mắt phải (Inner canthus) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **23** | `matphai` | Mí trên mắt phải - đoạn trong | Không | $\le 5\% \times \text{IOD}$ |
| **24** | `matphai` | Mí trên mắt phải - điểm cao nhất giữa mí | Không | $\le 5\% \times \text{IOD}$ |
| **25** | `matphai` | Mí trên mắt phải - đoạn ngoài | Không | $\le 5\% \times \text{IOD}$ |
| **26** | `matphai` | Khoé ngoài mắt phải (Outer canthus) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **27** | `matphai` | Mí dưới mắt phải - đoạn ngoài | Không | $\le 5\% \times \text{IOD}$ |
| **28** | `matphai` | Mí dưới mắt phải - điểm thấp nhất giữa mí | Không | $\le 5\% \times \text{IOD}$ |
| **29** | `matphai` | Mí dưới mắt phải - đoạn trong | Không | $\le 5\% \times \text{IOD}$ |
| **30** | `moingoai` | Khoé môi ngoài bên trái (nhìn trên ảnh) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **31** | `moingoai` | Viền bờ môi trên bên trái | Không | $\le 5\% \times \text{IOD}$ |
| **32** | `moingoai` | Đỉnh cánh môi trên trái (Cupid's bow left) | Không | $\le 5\% \times \text{IOD}$ |
| **33** | `moingoai` | Đáy nhân trung (Cupid's bow center) | Không | $\le 5\% \times \text{IOD}$ |
| **34** | `moingoai` | Đỉnh cánh môi trên phải (Cupid's bow right) | Không | $\le 5\% \times \text{IOD}$ |
| **35** | `moingoai` | Viền bờ môi trên bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **36** | `moingoai` | Khoé môi ngoài bên phải (nhìn trên ảnh) | ⭐ Có | $\le 3\% \times \text{IOD}$ |
| **37** | `moingoai` | Viền bờ môi dưới bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **38** | `moingoai` | Đoạn giữa bờ môi dưới bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **39** | `moingoai` | Điểm thấp nhất bờ môi dưới (Center bottom) | Không | $\le 5\% \times \text{IOD}$ |
| **40** | `moingoai` | Đoạn giữa bờ môi dưới bên trái | Không | $\le 5\% \times \text{IOD}$ |
| **41** | `moingoai` | Viền bờ môi dưới bên trái | Không | $\le 5\% \times \text{IOD}$ |
| **42** | `moitrong` | Khoé môi trong bên trái | Không | $\le 5\% \times \text{IOD}$ |
| **43** | `moitrong` | Mép trong môi trên bên trái | Không | $\le 5\% \times \text{IOD}$ |
| **44** | `moitrong` | Mép trong môi trên chính giữa | Không | $\le 5\% \times \text{IOD}$ |
| **45** | `moitrong` | Mép trong môi trên bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **46** | `moitrong` | Khoé môi trong bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **47** | `moitrong` | Mép trong môi dưới bên phải | Không | $\le 5\% \times \text{IOD}$ |
| **48** | `moitrong` | Mép trong môi dưới chính giữa | Không | $\le 5\% \times \text{IOD}$ |
| **49** | `moitrong` | Mép trong môi dưới bên trái | Không | $\le 5\% \times \text{IOD}$ |

---

## 3. Các Điểm Giải Phẫu Đặc Thù & Cạm Bẫy Cần Lưu Ý

### 1. Điểm 13: Chân Sống Mũi (Nose Bridge Foot)
- **Sai lầm phổ biến nhất:** Học viên hay kéo điểm 13 xuống chóp mũi (Nose Tip).
- **Quy chuẩn đúng (Mục 5.5):** Điểm 13 nằm tại vị trí **chân sống mũi ngay phía trên hai cánh mũi** (Alar base line), cách đỉnh sống mũi (điểm 10) khoảng $\approx 72\%$ quãng đường xuống đường chân cánh mũi. Tuyệt đối không đặt vào chóp mũi hay lỗ mũi.

### 2. Mắt Nhắm (Chiếm $\approx 80\%$ Dữ Liệu Thực Tế)
- Khi tài xế nhắm mắt: Mí trên và mí dưới khép sát lại với nhau tại khe mí (Palpebral fissure).
- Cả 8 điểm mốc vẫn phải được gán đầy đủ: điểm 14 và 18 (hoặc 22 và 26) ở hai góc khoé; các điểm mí trên 15-17 và mí dưới 19-21 nằm sát mép ranh giới nhưng **mí trên không được có toạ độ $y$ tụt xuống dưới mí dưới**.
- Trạng thái điểm: Vẫn là `visible` nếu nhìn thấy rõ đường khe mí nhắm.

### 3. Miệng Ngậm (Chiếm $\approx 80\%$ Dữ Liệu Thực Tế)
- Khi ngậm chặt miệng: Khe môi khép lại, đường viền môi trong (42-49) ép sát nhau.
- Bắt buộc phải gán đủ 8 điểm môi trong, xếp đều dọc theo ranh giới tiếp giáp hai môi và nằm gọn bên trong lòng của 12 điểm viền môi ngoài (30-41). Tuyệt đối không xóa điểm môi trong!

### 4. Người Mẫu Đeo Kính Mắt (Glasses Wearer)
- Kính mắt là phụ kiện ngoại cảnh, không phải bộ phận giải phẫu người.
- Khi gọng kính che ngang mí mắt hoặc lông mày: Phải ước lượng giải phẫu thực của mí mắt nằm sau tròng kính và bật cờ `occluded = 1`. Tuyệt đối không đặt điểm mốc lên gọng kính.
