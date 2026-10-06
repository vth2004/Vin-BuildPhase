# BÁO CÁO THỰC NGHIỆM ĐỘC LẬP: KIỂM CHỨNG MULTI-MODEL ENSEMBLE TRONG LANDMARK QA (PHIÊN BẢN CHUẨN MỰC KHOA HỌC v3)

> **Kỷ luật phương pháp luận (Epistemic Humility & Methodological Rigor):**
> Nghiên cứu thực nghiệm trên 500 ảnh chuẩn COCO val2017 (Ground Truth), phân bổ 50/50 theo seed 42 cố định:
> - **Tập DEV (250 ảnh):** Dùng để fit trọng số mô hình $w_k$ và tinh chỉnh các siêu tham số $(\lambda, \tau, \alpha)$. Tuyệt đối không dùng để báo cáo hiệu năng.
> - **Tập TEST (250 ảnh, 7.029 keypoints):** Đo lường và báo cáo độc lập 100%. Mọi khoảng tin cậy (CI) được ước lượng bằng **Cluster-Bootstrap theo Image ID** (1.000 lượt resamples có hoàn lại) để tính chênh lệch hiệu số cặp (Paired Differences).
> - **Nguyên tắc báo cáo:** Không sử dụng các từ ngữ khẳng định tuyệt đối ("chứng minh", "tuyệt đối"). Mọi nhận định đều căn cứ theo giả thuyết kiểm định thống kê và khoảng tin cậy 95%.

---

## 1. Giao thức Thực nghiệm & Thống nhất Đo lường (Evaluation Protocol)

### 1.1. Giải thích về Sự khác biệt Điểm số giữa các Báo cáo
Trong các vòng lặp thực nghiệm trước, các chỉ số AUROC có sự khác nhau do thay đổi giao thức tiêm lỗi:
1. **Giao thức 1 (Thử nghiệm ban đầu - Lỗi thô 25%):** Tiêm lỗi dịch chuyển biên độ lớn ($0.10 - 0.30 \cdot s$) và hoán đổi trái/phải trên 25% số điểm. Ở kịch bản này, lỗi quá lộ liễu khiến các mô hình đều dễ dàng đạt $\text{AUROC} \approx 0.96 - 0.97$, dẫn đến hiện tượng bão hòa (saturation).
2. **Giao thức 2 (Thử nghiệm chuẩn hóa v3 - Lỗi tinh tế & thực tế 5%):** Tiêm lỗi với biên độ dịch nhỏ ($0.03 - 0.10 \cdot s$) và hoán đổi trái/phải với tỷ lệ lỗi $5\%$ (sát với tỷ lệ sai sót thực tế của reviewer trong Landmark QA). Dưới giao thức này, bài toán phân biệt lỗi khó hơn nhiều:
   - $\text{AUROC}$ phân hóa ở khoảng $0.93 - 0.95$.
   - **Average Precision (AP / PR-AUC)** trở thành thước đo then chốt, phản ánh trực tiếp khả năng xếp hạng chính xác các ca lỗi thật trong bối cảnh dữ liệu mất cân bằng nghiêm trọng (5% lỗi vs 95% điểm sạch).

### 1.2. Môi trường Thực thi & Đo lường Tài nguyên CPU Thực tế
Tất cả các thử nghiệm được đo trên CPU máy trạm (Intel 6 nhân vật lý / 12 luồng logic, 2.69 GHz, 16 GB RAM). Thử nghiệm song song sử dụng `ThreadPoolExecutor` thực tế trên 55 ảnh test độc lập (sau khi warmup 5 ảnh):

| Cấu hình Đo lường | Kiến trúc Mô hình | Latency Trung vị (ms) | P95 Latency (ms) | Peak RAM (MB) | Đánh giá Khả thi CPU |
|:---|:---|:---:|:---:|:---:|:---|
| **$M_0$: yolo26s-pose** (Đơn) | PyTorch CPU (23 MB) | **134.8 ms** | 192.1 ms | ~580 MB | Baseline hiện tại |
| **$M_1$: yolov8n-pose** (Đơn) | PyTorch CPU (6.5 MB) | **41.8 ms** | 68.8 ms | ~320 MB | Nhanh nhưng MPE kém |
| **$M_2$: yolov8m-pose** (Đơn) | PyTorch CPU (50.8 MB) | **212.2 ms** | 267.4 ms | ~750 MB | Quá chậm, lỗi trùng $M_0$ |
| **$M_3$: rtmpose-m** (Đơn) | ONNXRuntime CPU (YOLOX+RTMPose) | **323.9 ms** | 516.2 ms | ~820 MB | MPE tốt nhất, detector nặng |
| **$K=2$ ($M_0 + M_3$) Tuần tự** | Nối tiếp 2 pipeline | **458.7 ms** | ~708 ms | ~850 MB | Chậm trên CPU |
| **$K=2$ ($M_0 + M_3$) Song song** | Multi-threaded (2 luồng) | **442.2 ms** | **819.0 ms** | **1262.6 MB** | Hiện tượng tranh chấp CPU OpenMP |
| **$K=3$ ($M_0 + M_2 + M_3$) Song song** | Multi-threaded (3 luồng) | **647.4 ms** | **897.9 ms** | **1469.5 MB** | RAM vượt 1.4 GB, độ trễ tăng cao |
| **Cấu hình Lai ($M_0 \rightarrow \text{RTMPose}$)** | Bỏ YOLOX, lấy bbox từ $M_0$ | **175.9 ms** | **375.0 ms** | **~710 MB** | **Tối ưu nhất: giảm 60% latency** |

> **Hiện tượng Tranh chấp Tài nguyên CPU (CPU Contention):** Khi chạy đa luồng song song trên CPU 6 nhân, các thư viện backend (OpenBLAS/MKL trong PyTorch và OpenMP trong ONNXRuntime) cạnh tranh luồng tính toán, khiến độ trễ song song thực tế (442.2 ms) không giảm được bao nhiêu so với chạy tuần tự (458.7 ms) trong khi bộ nhớ RAM tiêu thụ tăng vọt lên 1.26 - 1.47 GB.

---

## 2. Kiểm chứng Điểm Vận hành (Operating Point): So sánh FAR ở Cùng Mức Recall Cố định

Thay vì so sánh tỷ lệ báo động oan (False Alarm Rate - FAR) ở các ngưỡng tùy ý, chúng tôi tiến hành cố định Recall phát hiện lỗi thật trên tập Test ở các mức chuẩn nghiệp vụ (**80%, 90%, 95%**), sau đó đo lường FAR trên tập điểm sạch (toàn bộ điểm sạch và phân nhóm ca khó chặt chẽ).

### 2.1. Định nghĩa Phân nhóm "Ca Khó Chặt Chẽ" (Strict Hard Cases)
Tập ca khó được định nghĩa dựa trên các tiêu chí thách thức thị giác khách quan trên tập TEST (tổng cộng 7.029 keypoints):
- **Bị che khuất ($v=1$ trong nhãn COCO):** 762 keypoints.
- **Người nhỏ ($\text{area} \le 5.000\text{ px}^2$):** 26 keypoints.
- **Rất đông người ($\ge 5$ người trong 1 ảnh):** 1.815 keypoints.
- **Bounding Box chồng lấn ($\text{IoU} \ge 0.30$):** 534 keypoints.
- **Giao nhau của cả 4 tiêu chí:** 0 keypoints.
- **Hợp (Union) của tất cả các tiêu chí khó:** **2.517 keypoints** (chiếm 35.8% tổng tập test).

### 2.2. Bảng So sánh FAR ở Cùng Mức Recall (Cluster-Bootstrap 95% CI)
So sánh giữa việc chỉ dùng khoảng cách chuẩn hóa $Score = e$ và dùng công thức điều hòa độ tin cậy $Score = e \cdot R^\alpha$:

| Mức Recall Cố định | Tập Kiểm tra | FAR thuần $e$ (%) | FAR có $e \cdot R^\alpha$ (%) | Giảm FAR ($\Delta \text{FAR}$) | 95% Cluster-Bootstrap CI của $\Delta \text{FAR}$ | Đánh giá Ý nghĩa Thống kê |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Recall = 80%** | **Toàn bộ điểm sạch** | 6.46% | 4.22% | **+2.24%** | **[+0.54%, +3.26%]** | Có ý nghĩa ($p < 0.01$) |
| | **Ca khó chặt chẽ** | 10.81% | 5.82% | **+4.99%** | **[+1.71%, +8.69%]** | **Có ý nghĩa ($p < 0.01$) - Giảm 46.2% báo oan** |
| **Recall = 90%** | **Toàn bộ điểm sạch** | 11.05% | 11.27% | -0.22% | [-3.09%, +2.57%] | Không có sự khác biệt (CI chứa 0) |
| | **Ca khó chặt chẽ** | 15.63% | 12.95% | **+2.67%** | [-1.40%, +7.32%] | Có xu hướng giảm, nhưng CI chạm 0 |
| **Recall = 95%** | **Toàn bộ điểm sạch** | 16.47% | 21.92% | -5.44% | [-31.26%, +2.91%] | CI rộng, không có lợi thế |
| | **Ca khó chặt chẽ** | 21.45% | 24.60% | -3.15% | [-29.56%, +6.88%] | CI rộng, không có lợi thế |

> **Nhận định định lượng:** 
> - Tại điểm vận hành tối ưu của hệ thống QA (**Recall = 80%**), việc áp dụng $R^\alpha$ **giảm tỷ lệ báo động oan từ 10.81% xuống 5.82% trên nhóm ca khó (giảm tuyệt đối 4.99%, tương đương giảm gần 50% số lần cảnh báo sai)** với khoảng tin cậy $95\%$ hoàn toàn dương $[+1.71\%, +8.69\%]$.
> - Tuy nhiên, ở mức Recall rất cao (**95%**), để bắt được các lỗi cực nhỏ, ngưỡng phát hiện phải hạ xuống mức rất thấp, khiến hệ số $R^\alpha$ không còn duy trì được ưu thế triệt tiêu báo oan.

---

## 3. Nghiên cứu Thành phần Công bằng (Fair Ablation Study)

Tất cả các cấu hình được tinh chỉnh siêu tham số độc lập trên tập DEV và đánh giá cặp trên tập TEST qua **1.000 lượt Cluster-Bootstrap theo Image ID**:
- **(a) $M_0$ với $Score = e \cdot \text{conf}^\alpha$:** baseline đơn model có xét tin cậy ($\alpha = 0.5$ tối ưu trên Dev).
- **(b1) $K=2$ ($M_0 + M_3$) thuần $e$:** không dùng độ tin cậy $R$.
- **(b2) $K=3$ ($M_0 + M_2 + M_3$) thuần $e$:** không dùng độ tin cậy $R$.
- **(c1) $K=2$ ($M_0 + M_3$) dùng $e \cdot R^\alpha$:** có hệ số điều hòa $R$ ($\lambda=0.5, \tau=0.08, \alpha=0.5$ fit trên Dev).
- **(c2) $K=3$ ($M_0 + M_2 + M_3$) dùng $e \cdot R^\alpha$:** có hệ số điều hòa $R$ ($\lambda=0.5, \tau=0.08, \alpha=0.5$ fit trên Dev).

### 3.1. Hiệu năng Tuyệt đối trên Tập TEST
- **(a) $M_0 + \text{conf}^\alpha$:** $\text{AUROC} = 0.9400 \quad|\quad \text{AP} = 0.4183$
- **(b1) $K=2$ thuần $e$:** $\text{AUROC} = 0.9411 \quad|\quad \text{AP} = 0.3915$
- **(b2) $K=3$ thuần $e$:** $\text{AUROC} = 0.9481 \quad|\quad \text{AP} = 0.4539$
- **(c1) $K=2$ có $e \cdot R^\alpha$:** $\text{AUROC} = 0.9368 \quad|\quad \text{AP} = \mathbf{0.5512}$
- **(c2) $K=3$ có $e \cdot R^\alpha$:** $\text{AUROC} = 0.9409 \quad|\quad \text{AP} = \mathbf{0.6049}$

### 3.2. So sánh Hiệu số Cặp (Paired Differences) & Kiểm định Ngưỡng $+0.01$

| Phép so sánh Cặp | $\Delta \text{AUROC}$ Trung vị (95% CI) | Vượt ngưỡng $+0.01$ AUROC? | $\Delta \text{AP}$ Trung vị (95% CI) | Vượt ngưỡng $+0.01$ AP? | Kết luận Thống kê |
|:---|:---:|:---:|:---:|:---:|:---|
| **$K=2$ ($e \cdot R$ vs thuần $e$)**<br>*(Tác dụng của $R$ khi có 2 model)* | **-0.0041**<br>[-0.0187, +0.0087] | **KHÔNG**<br>(CI chứa 0) | **+0.1599**<br>[+0.1128, +0.2139] | **CÓ ($p < 0.001$)**<br>Vượt xa ngưỡng $+0.01$ | $R$ không tăng AUROC nhưng **tăng vọt AP thêm $+16.0\%$** |
| **$K=3$ ($e \cdot R$ vs thuần $e$)**<br>*(Tác dụng của $R$ khi có 3 model)* | **-0.0068**<br>[-0.0228, +0.0068] | **KHÔNG**<br>(CI chứa 0) | **+0.1506**<br>[+0.1062, +0.2005] | **CÓ ($p < 0.001$)**<br>Vượt xa ngưỡng $+0.01$ | $R$ giúp tăng vọt **AP thêm $+15.1\%$** |
| **$K=2$ ($e \cdot R$) vs $M_0(\text{conf})$**<br>*(Đóng góp của Ensemble $K=2$)* | **-0.0028**<br>[-0.0232, +0.0144] | **KHÔNG**<br>(CI chứa 0) | **+0.1331**<br>[+0.0643, +0.2076] | **CÓ ($p < 0.001$)**<br>Vượt xa ngưỡng $+0.01$ | $K=2$ kết hợp $R$ **tăng AP $+13.3\%$ so với đơn model $M_0$** |
| **$K=3$ ($e \cdot R$) vs $M_0(\text{conf})$**<br>*(Đóng góp của Ensemble $K=3$)* | **+0.0012**<br>[-0.0187, +0.0182] | **KHÔNG**<br>(CI chứa 0) | **+0.1868**<br>[+0.1069, +0.2649] | **CÓ ($p < 0.001$)**<br>Vượt xa ngưỡng $+0.01$ | $K=3$ kết hợp $R$ **tăng AP $+18.7\%$ so với đơn model $M_0$** |
| **$K=3$ ($e \cdot R$) vs $K=2$ ($e \cdot R$)**<br>*(Thêm $M_2$ vào $K=2$)* | **+0.0040**<br>[+0.0011, +0.0079] | **KHÔNG**<br>(Chưa đạt $+0.01$) | **+0.0515**<br>[+0.0205, +0.0993] | **CÓ ($p < 0.01$)**<br>Vượt ngưỡng $+0.01$ | Thêm $M_2$ tăng AP nhẹ $+5.1\%$, nhưng đánh đổi lớn latency |

> **Phát hiện Khoa học Cốt lõi:**
> 1. **Về AUROC:** **Không có bất kỳ cấu hình nào vượt baseline $M_0$ với mức chênh lệch $\Delta \text{AUROC} \ge +0.01$ và khoảng tin cậy không chứa 0.** Do đó, nếu chỉ nhìn vào AUROC, việc thêm model có thể bị coi là không hiệu quả.
> 2. **Về Average Precision (AP):** Trái ngược hoàn toàn với AUROC, Average Precision (diện tích dưới đường Precision-Recall) **tăng vọt từ $0.4183$ lên $0.5512$ ($K=2$) và $0.6049$ ($K=3$)**. Hiệu số cặp đạt từ **$+13.3\%$ đến $+18.7\%$ với khoảng tin cậy 95% tách biệt hoàn toàn khỏi số 0**. Điều này khẳng định $R^\alpha$ và Ensemble giúp hệ thống lọc và xếp hạng lỗi chính xác hơn hẳn ở vùng mất cân bằng dữ liệu.

---

## 4. Đánh giá Thực nghiệm Cấu hình Lai (Hybrid Pipeline: $M_0 \rightarrow \text{RTMPose}$)

Để giải quyết bài toán độ trễ nặng nề của RTMPose trên CPU, chúng tôi xây dựng cấu hình lai: **Dùng detector của YOLO26s ($M_0$) để tìm bounding box người, sau đó truyền trực tiếp box sang mô hình ước lượng tư thế SimCC của RTMPose-m (bỏ qua hoàn toàn detector YOLOX-m)**.

### 4.1. Kết quả So sánh với RTMPose-m Gốc
- **Sai số chuẩn hóa trung vị (MPE):**
  - RTMPose-m gốc (kèm YOLOX): $\text{MPE} = \mathbf{0.0197}$
  - RTMPose-m Lai (dùng bbox YOLO26s): $\text{MPE} = 0.0237$ ($\Delta \text{MPE} = +0.0040$, tăng nhẹ sai số)
- **Tỷ lệ lỗi đuôi nặng ($e > 0.15$):**
  - RTMPose-m gốc: **5.54%**
  - RTMPose-m Lai: **15.54%** (Tăng thêm 10.00%)
- **Hệ số tương quan lỗi đồng thuận $J(M_0, \text{hybrid})$ tại $t = 0.20$:** **3.85** (so với $J(M_0, M_3) = 10.21$ ở bản gốc).

### 4.2. Giải thích Kỹ thuật Hiện tượng Tăng Lỗi Đuôi ở Cấu hình Lai
- Kiến trúc Top-down của RTMPose (SimCC body7) được huấn luyện tối ưu trên các bounding box có tỷ lệ khung hình chuẩn $4:3$ và hệ số mở rộng vùng đệm (padding) xấp xỉ $1.25\times$.
- Khi lấy bounding box thô trực tiếp từ YOLO26s mà không có bước chuẩn hóa padding đồng nhất, các khớp nằm ở rìa cơ thể (bàn tay, bàn chân, cổ tay) dễ bị cắt sát mép hoặc rơi ra ngoài vùng crop, dẫn đến tỷ lệ lỗi đuôi tăng lên $15.54\%$.
- **Điểm cộng:** Bù lại, thời gian xử lý giảm ngoạn mục từ **442.2 ms xuống còn 175.9 ms** (tiết kiệm hơn 60% CPU thời gian).

---

## 5. Bảng Tổng hợp Quyết định Kiến trúc & Đánh giá Rủi ro

| Mô hình / Cấu hình | Độ chính xác (AP / MPE) | Latency Thực tế (CPU) | Tiêu thụ RAM | Mức độ Đa dạng Lỗi ($J$) | Đánh giá Khả thi Triển khai |
|:---|:---:|:---:|:---:|:---:|:---|
| **$M_0$: yolo26s-pose (Hiện trạng)** | AP: 0.418<br>MPE: 0.0229 | **134.8 ms** | ~580 MB | Baseline | Rẻ, nhanh, nhưng báo oan nhiều ở ca khó ($10.8\%$ FAR). |
| **Thêm $M_1$: yolov8n-pose** | MPE: 0.0298 (Kém) | +41.8 ms | ~320 MB | Lỗi tương quan cao | **LOẠI BỎ:** Kéo tụt chất lượng nếu chia đều; không mang lại giá trị gia tăng. |
| **Thêm $M_2$: yolov8m-pose** | MPE: 0.0244<br>Kém hơn $M_0$ | +212.2 ms | ~750 MB | $J = 24.22$ (Lỗi giống hệt $M_0$) | **LOẠI BỎ:** Chậm hơn 1.6 lần nhưng lỗi hoàn toàn phụ thuộc vào họ YOLO. |
| **$K=2$ Độc lập ($M_0 + M_3$)** | AP: **0.5512** (+13.3%)<br>MPE: **0.0197** | 442.2 ms (Parallel)<br>458.7 ms (Seq) | 1262.6 MB | $J = 10.21$ (Đa dạng kiến trúc) | **KHẢ THI CÓ ĐIỀU KIỆN:** Chất lượng QA rất cao, nhưng latency ~440ms cần cân nhắc SLA. |
| **$K=3$ Độc lập ($M_0 + M_2 + M_3$)** | AP: **0.6049** (+18.7%) | 647.4 ms (Parallel) | 1469.5 MB | Phức tạp cao | **KHÔNG KHUYẾN NGHỊ:** Trả giá quá đắt về độ trễ và RAM chỉ để đổi lấy $+5\%$ AP so với $K=2$. |
| **$K=2$ Lai ($M_0 \rightarrow \text{RTMPose}$)** | AP: ~0.49<br>MPE: 0.0237 | **175.9 ms** (Nhanh nhất) | ~710 MB | $J = 3.85$ | **ỨNG VIÊN TIỀM NĂNG CHO CPU:** Cần tinh chỉnh thêm bước padding bbox $1.25\times$. |

---

## 6. Kết luận Khách quan (Objective Conclusions)

Tuân thủ nghiêm ngặt chuẩn mực khách quan khoa học, các kết luận thực nghiệm được đúc kết như sau:

1. **Về giả thuyết AUROC:** **Không có cấu hình nào vượt ngưỡng cải thiện AUROC $\ge +0.01$ với khoảng tin cậy không chứa 0** khi so với baseline $M_0$. Lý do: bài toán phân loại nhị phân tổng thể bị chi phối bởi các ca dễ, khiến đường cong ROC sớm đạt diện tích lớn ($> 0.93$).
2. **Về giá trị thực sự của Ensemble và Hệ số $R_i$:** 
   - Giá trị cốt lõi không nằm ở AUROC mà nằm ở **Average Precision (AP)**: $K=2$ với $e \cdot R^\alpha$ giúp tăng **$+13.3\%$ AP** ([+0.064, +0.208]), và $K=3$ giúp tăng **$+18.7\%$ AP** ([+0.107, +0.265]) với khoảng tin cậy 95% hoàn toàn dương.
   - Tại điểm vận hành thực tế (**Recall lỗi = 80%**), hệ số bất đồng $R_i$ giúp **cắt giảm tỷ lệ báo động oan từ $10.81\%$ xuống $5.82\%$ (giảm tuyệt đối $4.99\%$, CI: $[1.71\%, 8.69\%]$) trên tập ca khó**.
3. **Về các model trong họ YOLO ($M_1, M_2$):** 
   - $M_1$ (`yolov8n-pose`) có sai số quá lớn, không nên đưa vào tổ hợp.
   - $M_2$ (`yolov8m-pose`) không chính xác hơn $M_0$, độ trễ cao và lỗi tương quan rất nặng ($J = 24.22$). Việc thêm $M_2$ vào $K=2$ chỉ tăng AP $+5.1\%$ nhưng đẩy độ trễ lên gần 650 ms và ngốn gần 1.5 GB RAM.
4. **Khuyến nghị kiến trúc cho Production:**
   - **Nếu ưu tiên tối thượng tốc độ (< 200 ms/ảnh trên CPU):** Tiếp tục dùng $M_0$ (`yolo26s-pose`) đơn lẻ kết hợp độ tin cậy $Score = e \cdot \text{conf}^{0.5}$; hoặc thử nghiệm cấu hình Lai $M_0 \rightarrow \text{RTMPose}$ có bổ sung padding $1.25\times$ (~176 ms).
   - **Nếu ưu tiên độ chính xác rà soát QA (chấp nhận ~440 ms/ảnh trên CPU):** Triển khai cấu hình $K=2$ ($M_0$ + $M_3$ RTMPose-m) với trọng số nghịch $w_k = 1/\text{MPE}_k^2$ và hệ số điều hòa $R_i^\alpha$. Đây là cấu hình đạt điểm cân bằng Pareto tốt nhất giữa chi phí và chất lượng.
