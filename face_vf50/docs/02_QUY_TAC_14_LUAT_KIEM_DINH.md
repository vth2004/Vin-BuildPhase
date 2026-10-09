# CHI TIẾT 14 QUY TẮC KIỂM ĐỊNH (RULES AUDITING SYSTEM)

> **Căn cứ:** Guideline Face Landmark VF-50 v1.3 — VinFast AI Thực Chiến  
> **Mã Schema:** `vf_face_landmark50_v1`  
> **Mục tiêu:** Tự động phát hiện lỗi sai nhãn và hướng dẫn học viên sửa tay trên CVAT.

---

## Danh Mục Tổng Quan 14 Quy Tắc

| Mã | Tên Quy Tắc | Mức Độ | Căn Cứ Guideline | Hậu Quả Nếu Không Sửa |
| :---: | :--- | :---: | :---: | :--- |
| **R01** | Thiếu điểm hoặc sai số lượng điểm | `CRITICAL` | Mục 1.3, Mục 3 | Bài nộp bị lỗi schema, loại trực tiếp |
| **R02** | Tọa độ nằm ngoài phạm vi ảnh $1280 \times 720$ | `CRITICAL` | Mục 1.3 | Lỗi toạ độ âm hoặc tràn kích thước |
| **R03** | Mí trên thấp hơn mí dưới (Lật mí mắt) | `CRITICAL` | Mục 5.1, Mục 5.2 | Sai cấu trúc giải phẫu mắt cơ bản |
| **R04** | Đường viền tự cắt chéo (Tạo hình chữ X) | `CRITICAL` | Mục 4.1 | Topology bị xoắn gập hình học |
| **R05** | Sống mũi chia không đều (Tỉ lệ $> 1.45$) | `MAJOR` | Mục 5.3, Hình 5 | Mất điểm độ thẩm mỹ và tính đều đặn |
| **R06** | Nhầm lẫn Trái / Phải theo góc nhìn ảnh | `CRITICAL` | Mục 1.3 (Quy ước) | Đổi ngược hoàn toàn mốc khuôn mặt |
| **R07** | Môi trong tràn ra ngoài lòng môi ngoài | `CRITICAL` | Mục 5.4, Mục 6.4 | Lỗi giải phẫu miệng nghiêm trọng |
| **R08** | Sai trạng thái điểm (Kính mắt, tóc che) | `MAJOR` | Mục 4.2, Mục 6.1 | Đánh giá sai tỉ lệ che khuất thực tế |
| **R09** | Tọa độ nhảy vọt giữa 2 frame liên tiếp ($> 15$px) | `MAJOR` | Mục 6.9 (Video) | Mất tính mượt mà thời gian (Jitter) |
| **R10** | Góc xoay nghiêng mắt bất thường ($> 45^\circ$) | `WARNING` | Mục 5.2 | Nghi ngờ gán nhầm điểm khoé mắt |
| **R11** | Sai lệch điểm neo vượt quá $3\% \times \text{IOD}$ | `CRITICAL` | Mục 5.5, Mục 8 | Mất điểm độ chính xác điểm chuẩn |
| **R12** | Sai lệch đường viền vượt quá $5\% \times \text{IOD}$ | `MAJOR` | Mục 8 | Mất điểm độ trơn tru cung mốc |
| **R13** | Chỉ số sai số NME toàn ảnh vượt quá $0.035$ | `CRITICAL` | Mục 8 (KPI chính) | Không đạt chỉ tiêu chất lượng nộp bài |
| **R14** | Mắt/Môi nhắm nhưng xóa hoặc làm mất điểm | `CRITICAL` | Mục 6.3, Mục 6.4 | Vi phạm quy tắc giữ nguyên đủ 50 điểm |

---

## Phân Tích Kỹ Thuật Chi Tiết Từng Quy Tắc

### Quy Tắc R01: Thiếu Điểm Hoặc Sai Số Lượng Điểm
- **Nghiệp vụ:** Khuôn mặt bắt buộc phải có đầy đủ 50 điểm với đúng ID từ 0 đến 49 phân bố trên 7 skeleton:
  - `longmaytrai`: điểm 0 đến 4 (5 điểm)
  - `longmayphai`: điểm 5 đến 9 (5 điểm)
  - `songmui`: điểm 10 đến 13 (4 điểm)
  - `mattrai`: điểm 14 đến 21 (8 điểm)
  - `matphai`: điểm 22 đến 29 (8 điểm)
  - `moingoai`: điểm 30 đến 41 (12 điểm)
  - `moitrong`: điểm 42 đến 49 (8 điểm)
- **Kiểm tra toán học:**
  $$\{i \in [0, 49]\} \subseteq \text{ID danh sách}$$
  Nếu bất kỳ điểm nào bị thiếu không có trong mảng nhãn, kích hoạt `CRITICAL`.
- **Cách sửa trên CVAT:** Tìm skeleton bị thiếu điểm, thêm lại điểm với đúng ID quy định. Tuyệt đối không xóa bất kỳ điểm nào ngay cả khi bị che khuất (dùng trạng thái `occluded` hoặc `outside` thay vì xóa điểm).

---

### Quy Tắc R02: Tọa Độ Nằm Ngoài Khung Ảnh ($1280 \times 720$)
- **Nghiệp vụ:** Mọi điểm ở trạng thái `visible` hoặc `occluded` đều phải có tọa độ thực nằm trong giới hạn màn hình:
  $$0 \le x_i \le 1280, \quad 0 \le y_i \le 720$$
- **Kiểm tra toán học:** Nếu điểm có $x_i < 0$, $y_i < 0$, $x_i > 1280$, hoặc $y_i > 720$: kích hoạt `CRITICAL`.
- **Cách sửa trên CVAT:** Kéo điểm trở lại bên trong khung hình. Nếu bộ phận khuôn mặt thực sự bị cắt ra ngoài khung ảnh, chuyển trạng thái điểm sang `outside`.

---

### Quy Tắc R03: Mí Trên Thấp Hơn Mí Dưới (Lật Mí Mắt)
- **Nghiệp vụ:** Theo giải phẫu học, bờ mí trên luôn luôn nằm ở vị trí cao hơn (hoặc bằng khi nhắm mắt) so với bờ mí dưới tương ứng.
- **Các cặp mí đối ứng:**
  - Mắt trái (`mattrai`): Cặp `(15, 21)`, `(16, 20)`, `(17, 19)`.
  - Mắt phải (`matphai`): Cặp `(23, 29)`, `(24, 28)`, `(25, 27)`.
- **Kiểm tra toán học (Có bù góc nghiêng đầu $\theta$):**
  Khi đầu nghiêng một góc $\theta$, toạ độ $y'$ sau khi xoay về phương ngang:
  $$y' = -(x - x_c) \sin\theta + (y - y_c) \cos\theta$$
  Điều kiện vi phạm: $y'_{\text{mí trên}} > y'_{\text{mí dưới}} + \epsilon$ (trục $y$ hướng xuống dưới, tức mí trên có toạ độ lớn hơn mí dưới).
- **Cách sửa trên CVAT:** Kéo điểm mí trên (15-17 hoặc 23-25) lên phía trên con ngươi. Đảm bảo mí trên uốn vòm lên trên và mí dưới uốn võng xuống dưới.

---

### Quy Tắc R04: Đường Viền Tự Cắt Chéo (Tạo Hình Chữ X)
- **Nghiệp vụ:** Các đường viền khép kín (`mattrai`, `matphai`, `moingoai`, `moitrong`) và đường hở (`longmay`, `songmui`) phải tạo thành đường cong đơn giản không tự cắt nhau.
- **Kiểm tra toán học:** Với mỗi cặp đoạn thẳng không liền kề $S_1 = (P_i, P_{i+1})$ và $S_2 = (P_j, P_{j+1})$, kiểm tra phương trình giao điểm:
  $$P_i + t(P_{i+1} - P_i) = P_j + u(P_{j+1} - P_j), \quad t, u \in (0, 1)$$
  Nếu tồn tại giao điểm, đa giác tạo thành hình số 8 hoặc chữ X $\implies$ Kích hoạt `CRITICAL`.
- **Cách sửa trên CVAT:** Đổi lại vị trí các điểm bị bắt chéo nhau sao cho đường viền chạy theo đúng vòng chu vi giải phẫu thuận chiều.

---

### Quy Tắc R05: Sống Mũi Chia Không Đều (Tỉ Lệ $> 1.45$)
- **Nghiệp vụ (Mục 5.3 & Hình 5):** Sống mũi gồm 4 điểm: 10 (đỉnh), 11 (thân trên), 12 (thân dưới), 13 (chân sống mũi trên cánh mũi). Ba đoạn thẳng $d_1 = |P_{10}P_{11}|$, $d_2 = |P_{11}P_{12}|$, $d_3 = |P_{12}P_{13}|$ phải tương đối đều nhau.
- **Kiểm tra toán học:**
  $$\frac{\max(d_1, d_2, d_3)}{\min(d_1, d_2, d_3)} \le 1.45$$
  Nếu tỉ lệ vượt quá $1.45 \implies$ Kích hoạt `MAJOR`.
- **Cách sửa trên CVAT:** Dịch chuyển nhẹ điểm 11 và điểm 12 sao cho chúng chia đều đoạn sống mũi từ điểm 10 đến điểm 13 thành 3 phần bằng nhau.

---

### Quy Tắc R06: Nhầm Lẫn Trái / Phải Theo Góc Nhìn Người Quan Sát
- **Nghiệp vụ (Mục 1.3):** Quy ước bên TRÁI và PHẢI luôn theo góc nhìn người quan sát trên bức ảnh:
  - Bên trái ảnh (trục $x$ nhỏ hơn): `longmaytrai` (0-4), `mattrai` (14-21).
  - Bên phải ảnh (trục $x$ lớn hơn): `longmayphai` (5-9), `matphai` (22-29).
- **Kiểm tra toán học:**
  $$\bar{x}_{\text{longmaytrai}} < \bar{x}_{\text{longmayphai}} \quad \text{và} \quad \bar{x}_{\text{mattrai}} < \bar{x}_{\text{matphai}}$$
  Nếu điều kiện bị đảo ngược $\implies$ Kích hoạt `CRITICAL`.
- **Cách sửa trên CVAT:** Chuyển đổi tên nhãn hoặc đổi toạ độ giữa 2 cụm skeleton trái và phải. Không được gán theo cơ thể người mẫu.

---

### Quy Tắc R07: Môi Trong Tràn Ra Ngoài Lòng Môi Ngoài
- **Nghiệp vụ (Mục 5.4 & Mục 6.4):** Đường viền môi trong (42-49) mô tả khe hở hoặc bờ trong của miệng, do đó toàn bộ 8 điểm môi trong bắt buộc phải nằm bên trong vùng bao của 12 điểm môi ngoài (30-41).
- **Kiểm tra toán học:** Thuật toán Ray-Casting (Bắn tia cắt đa giác): Với mỗi điểm $P \in \text{moitrong}$, số lần tia cắt đa giác $\text{moingoai}$ phải là số lẻ.
- **Cách sửa trên CVAT:** Kéo điểm môi trong vào sâu trong lòng môi. Nếu miệng ngậm chặt (Closed mouth), đặt 8 điểm môi trong nằm đè sát lên đường tiếp giáp giữa môi trên và môi dưới nhưng không được vượt ra viền ngoài.

---

### Quy Tắc R08: Sai Trạng Thái Điểm (Kính Mắt, Tóc Che)
- **Nghiệp vụ (Mục 4.2 & Mục 6.1):**
  - Khi người mẫu đeo kính mắt, các điểm mí mắt hoặc lông mày bị tròng kính/gọng kính đè lên: Học viên **phải ước lượng vị trí giải phẫu thật sau kính** và chuyển trạng thái sang `occluded`.
  - Tuyệt đối không được kéo điểm đặt đè lên gọng kính.
- **Cách sửa trên CVAT:** Mở thuộc tính của điểm, chuyển `occluded = 1 (true)` và đặt điểm vào vị trí mắt thật.

---

### Quy Tắc R09: Tọa Độ Nhảy Vọt Giữa 2 Frame Liên Tiếp ($> 15$px)
- **Nghiệp vụ (Mục 6.9):** Khi gán nhãn chuỗi video với chuyển động đầu bình thường, vị trí của cùng một điểm mốc giữa 2 frame liên tiếp $t$ và $t+1$ không được lệch đột ngột quá 15 pixels.
- **Kiểm tra toán học:**
  $$\|P_i^{(t+1)} - P_i^{(t)}\|_2 \le 15.0 \text{ px}$$
- **Cách sửa trên CVAT:** Kiểm tra lại frame $t$ và $t+1$, căn chỉnh điểm mốc mượt mà theo chuyển động tự nhiên của khuôn mặt.

---

### Quy Tắc R10: Góc Xoay Nghiêng Mắt Bất Thường ($> 45^\circ$)
- **Nghiệp vụ:** Đường nối hai khoé mắt (14-18 hoặc 22-26) có góc nghiêng so với phương ngang không được vượt quá $45^\circ$ trừ khi người mẫu nghiêng hẳn đầu.
- **Cách sửa trên CVAT:** Kiểm tra xem khoé trong (18, 22) và khoé ngoài (14, 26) có bị gán nhầm vị trí cho nhau không.

---

### Quy Tắc R11: Sai Lệch Điểm Neo Vượt Quá $3\% \times \text{IOD}$
- **Nghiệp vụ (Mục 5.5 & Mục 8):** 12 Điểm neo là các mốc giải phẫu cốt lõi cố định:
  $$\text{Anchors} = \{0, 4, 5, 9, 10, 13, 14, 18, 22, 26, 30, 36\}$$
  Dung sai cho phép tối đa là $3\% \times \text{IOD}$ (khoảng $\approx 2.9$ pixels).
- **Cách sửa trên CVAT:** Dựa vào véc-tơ lệch hiển thị trên Web Auditor, dịch chuyển điểm neo sát đúng vào mốc giải phẫu chuẩn (khoé mắt thật, đỉnh sống mũi, hai góc khoé môi).

---

### Quy Tắc R12: Sai Lệch Đường Viền Vượt Quá $5\% \times \text{IOD}$
- **Nghiệp vụ (Mục 8):** 38 Điểm còn lại phân bố dọc các đường cong cung mí mắt, cung lông mày, sống mũi, viền môi có dung sai tối đa là $5\% \times \text{IOD}$ (khoảng $\approx 4.8$ pixels).
- **Cách sửa trên CVAT:** Uốn nắn đường cong các điểm sao cho phân bố đều đặn và mượt mà theo cung bờ môi/mí mắt.

---

### Quy Tắc R13: Chỉ Số Sai Số Chuẩn Hóa NME Vượt Quá $0.035$
- **Nghiệp vụ (Mục 8 - Tiêu chuẩn chấm bài):** Sai số trung bình toàn ảnh chuẩn hóa theo IOD phải đạt:
  $$\text{NME} = \frac{1}{50 \cdot \text{IOD}} \sum_{i=0}^{49} e_i \le 0.035$$
  (tương đương sai số trung bình $\le 3.4$ pixels trên ảnh $1280 \times 720$).
- **Cách sửa trên CVAT:** Rà soát và sửa các điểm có độ lệch lớn nhất (hiển thị trên Inspector) cho đến khi NME giảm xuống dưới ngưỡng $0.035$.

---

### Quy Tắc R14: Mắt/Môi Nhắm Nhưng Xóa Hoặc Bỏ Điểm
- **Nghiệp vụ (Mục 6.3 & Mục 6.4):** Trong dữ liệu cabin xe, có tới 80% ảnh tài xế nhắm mắt hoặc ngậm chặt môi. Ban Tổ Chức quy định rõ:
  > *Dù mắt nhắm hay miệng ngậm, BẮT BUỘC vẫn phải gán đủ 8 điểm cho mỗi mắt và 8 điểm cho môi trong.*
- **Cách sửa trên CVAT:** Tuyệt đối không xóa điểm. Khi mắt nhắm, kéo mí trên và mí dưới áp sát vào khe mí (Palpebral fissure). Khi miệng ngậm, xếp 8 điểm môi trong nằm dọc theo đường tiếp giáp hai bờ môi.
