# KẾT QUẢ ĐỐI CHIẾU TOP CẢNH BÁO TRÊN TẬP HUMANPOSE1 (K=1 vs K=2)

- **Tập dữ liệu:** `humanpose1` (20 ảnh VinFast)
- **Thời gian K=1:** 575.7 ms/ảnh (Tổng cảnh báo: 19)
- **Thời gian K=2:** 409.5 ms/ảnh (Tổng cảnh báo: 10)

## 1. Top Cảnh báo Chế độ K=2 (Ensemble YOLO26s + RTMPose-m)

| STT | Tên ảnh | Khớp | Điểm nghi ngờ | Độ tin cậy ($R$) | Bất đồng ($\delta$) | Loại cảnh báo | Người gán | Tham chiếu $p^*$ |
|:---:|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|
| 1 | `images/default/train_20.jpg` | `r_knee` | **33.3%** | 80.2% | 0.0059 | `suspected_error` | (194.3, 247.2) | (198.3, 218.6) |
| 2 | `images/default/train_20.jpg` | `r_wrist` | **32.7%** | 74.5% | 0.0140 | `suspected_error` | (199.1, 178.3) | (179.8, 170.3) |
| 3 | `images/default/train_01.jpg` | `l_wrist` | **30.4%** | 47.3% | 0.0415 | `suspected_error` | (333.0, 316.1) | (319.7, 335.6) |
| 4 | `images/default/train_14.jpg` | `l_eye` | **30.3%** | 86.8% | 0.0065 | `suspected_error` | (268.8, 199.8) | (275.6, 201.2) |
| 5 | `images/default/train_01.jpg` | `l_hip` | **27.0%** | 78.6% | 0.0241 | `suspected_error` | (454.0, 344.9) | (429.7, 367.4) |
| 6 | `images/default/train_07.jpg` | `l_elbow` | **22.5%** | 86.0% | 0.0051 | `suspected_error` | (337.6, 364.6) | (313.8, 387.8) |
| 7 | `images/default/train_01.jpg` | `r_wrist` | **21.2%** | 65.3% | 0.0304 | `suspected_error` | (354.6, 313.4) | (359.6, 296.4) |
| 8 | `images/default/train_14.jpg` | `r_eye` | **20.9%** | 79.9% | 0.0077 | `suspected_error` | (261.6, 196.9) | (266.2, 200.3) |
| 9 | `images/default/train_19.jpg` | `l_knee` | **20.8%** | 90.9% | 0.0049 | `suspected_error` | (108.0, 239.6) | (107.1, 225.8) |
| 10 | `images/default/train_01.jpg` | `r_hip` | **20.1%** | 66.8% | 0.0480 | `suspected_error` | (236.7, 348.2) | (250.0, 372.8) |

## 2. Top Cảnh báo Chế độ K=1 (Đơn model YOLO26s + OKS)

| STT | Tên ảnh | Khớp | Điểm nghi ngờ | Loại cảnh báo | Người gán | Model gợi ý |
|:---:|:---|:---|:---:|:---|:---:|:---:|
| 1 | `images/default/train_15.jpg` | `r_wrist` | **57.7%** | `suspected_error` | (135.6, 259.0) | (91.1, 254.8) |
| 2 | `images/default/train_15.jpg` | `r_elbow` | **50.5%** | `suspected_error` | (79.8, 260.1) | (58.3, 224.6) |
| 3 | `images/default/train_01.jpg` | `l_wrist` | **49.4%** | `suspected_error` | (333.0, 316.1) | (313.0, 336.7) |
| 4 | `images/default/train_20.jpg` | `r_wrist` | **41.5%** | `suspected_error` | (199.1, 178.3) | (177.9, 171.5) |
| 5 | `images/default/train_19.jpg` | `r_wrist` | **38.3%** | `suspected_error` | (290.8, 190.9) | (293.5, 173.4) |
| 6 | `images/default/train_15.jpg` | `l_hip` | **37.2%** | `suspected_error` | (136.0, 312.9) | (128.1, 266.7) |
| 7 | `images/default/train_20.jpg` | `r_knee` | **36.3%** | `suspected_error` | (194.3, 247.2) | (197.2, 218.5) |
| 8 | `images/default/train_15.jpg` | `r_ankle` | **35.5%** | `suspected_error` | (66.2, 504.9) | (65.1, 458.4) |
| 9 | `images/default/train_03.jpg` | `r_ankle` | **34.0%** | `suspected_error` | (253.1, 470.9) | (303.1, 465.9) |
| 10 | `images/default/train_15.jpg` | `r_hip` | **33.7%** | `suspected_error` | (49.0, 315.2) | (75.5, 280.1) |
| 11 | `images/default/train_12.jpg` | `r_ankle` | **32.7%** | `suspected_error` | (168.0, 439.4) | (173.0, 495.6) |
| 12 | `images/default/train_01.jpg` | `r_wrist` | **31.7%** | `suspected_error` | (354.6, 313.4) | (364.5, 296.2) |
| 13 | `images/default/train_14.jpg` | `l_eye` | **28.0%** | `suspected_error` | (268.8, 199.8) | (274.8, 201.9) |
| 14 | `images/default/train_01.jpg` | `l_hip` | **25.5%** | `suspected_error` | (454.0, 344.9) | (430.3, 363.1) |
| 15 | `images/default/train_07.jpg` | `r_knee` | **22.2%** | `suspected_error` | (166.9, 625.8) | (178.5, 588.4) |
| 16 | `images/default/train_07.jpg` | `l_elbow` | **22.1%** | `suspected_error` | (337.6, 364.6) | (314.7, 386.5) |
| 17 | `images/default/train_14.jpg` | `r_eye` | **21.6%** | `suspected_error` | (261.6, 196.9) | (265.2, 201.3) |
| 18 | `images/default/train_10.jpg` | `nose` | **20.4%** | `suspected_error` | (293.7, 232.1) | (301.4, 223.5) |
| 19 | `images/default/train_19.jpg` | `l_knee` | **20.3%** | `suspected_error` | (108.0, 239.6) | (107.4, 226.3) |
