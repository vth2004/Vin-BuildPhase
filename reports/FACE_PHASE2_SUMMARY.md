# Benchmark Face VF50 — Pha 2

## Phạm vi và cách đọc

Chạy ngày 2026-10-13, seed 42, trên 1.300 khuôn mặt TEST độc lập (1.000 WFLW và 300 ảnh 300W-private), tổng cộng 39.000 scenario. Mỗi lỗi được tiêm vào **bản sao** annotation; annotation sạch được dùng làm model reference để cô lập hành vi của `check_all_rules()` và `evaluate_frame_quality()` production. Vì vậy đây là benchmark logic QA trong điều kiện model lý tưởng, không thay thế baseline MediaPipe ở [FACE_PHASE1_SUMMARY.md](FACE_PHASE1_SUMMARY.md).

Raw nguồn: `.bench_workspace/results/face_phase2_20261013/scenario_results.csv`, `rule_metrics.json`, `rule_metrics_bootstrap.json`, `point_scores.csv`, `point_metrics_by_noise.json`. Checksum nằm trong `SHA256SUMS.txt` cùng thư mục.

## Kết quả theo luật trên TEST

CI 95% là bootstrap 1.000 lần theo cụm ảnh. Positive của mỗi luật là scenario được thiết kế cho chính luật đó; các scenario còn lại (gồm mẫu sạch và lỗi khác) là negative, nên precision/FPR phản ánh cả cảnh báo chéo.

| Luật | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| R01 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| R02 | 0.037 [0.037, 0.037] | 1.000 [1.000, 1.000] | 0.071 [0.071, 0.071] | 0.897 [0.897, 0.897] |
| R03 | 0.839 [0.823, 0.853] | 1.000 [1.000, 1.000] | 0.912 [0.903, 0.921] | 0.007 [0.006, 0.007] |
| R04 | 0.432 [0.418, 0.443] | 0.990 [0.984, 0.995] | 0.601 [0.588, 0.612] | 0.045 [0.043, 0.047] |
| R05 | 0.452 [0.408, 0.502] | 1.000 [1.000, 1.000] | 0.622 [0.580, 0.668] | 0.042 [0.034, 0.050] |
| R06 | 0.386 [0.377, 0.392] | 0.995 [0.991, 0.998] | 0.556 [0.547, 0.563] | 0.055 [0.053, 0.057] |
| R07 | 0.233 [0.218, 0.248] | 0.998 [0.995, 1.000] | 0.378 [0.358, 0.397] | 0.113 [0.104, 0.123] |
| R08 | 0.629 [0.568, 0.698] | 0.980 [0.972, 0.987] | 0.767 [0.718, 0.816] | 0.020 [0.015, 0.026] |
| R09 | 0.219 [0.209, 0.228] | 1.000 [1.000, 1.000] | 0.359 [0.346, 0.371] | 0.123 [0.117, 0.130] |
| R10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| R11 | 0.880 [0.789, 0.957] | 1.000 [1.000, 1.000] | 0.936 [0.882, 0.978] | 0.005 [0.002, 0.009] |
| R12 | 0.267 [0.239, 0.299] | 1.000 [1.000, 1.000] | 0.422 [0.386, 0.460] | 0.095 [0.081, 0.110] |
| R13 | 1.000 [1.000, 1.000] | 0.986 [0.979, 0.992] | 0.993 [0.990, 0.996] | 0.000 [0.000, 0.000] |
| R14 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |

Recall riêng từng injection có đủ `n=1.300` trong `injection_recall.json`. Các lỗi chưa đạt recall tuyệt đối: R04 0,990; R06 0,995; R07 0,998; R08 0,980; R13 0,986. Các luật hình học R07/R09/R12 có precision thấp chủ yếu vì chúng cũng bắt các bất thường sẵn có hoặc hệ quả hình học của lỗi khác; cần xem confusion theo injection trước khi coi tất cả là false alarm ngoài đời.

Trên riêng 1.300 control ảnh tĩnh không tiêm lỗi, FPR lần lượt là: R01 0%; R02 100%; R03 0%; R04 0,15%; R05 2,69%; R06 0,08%; R07 3,62%; R08 1,23%; R09 1,31%; R10 0%; R11 0,46%; R12 8,31%; R13 0%; R14 0%. Raw: `clean_control_fpr.json`. R02 phải đọc cùng giới hạn visibility bên dưới; R11 clean-positive cho thấy một số GT public nằm ngoài kích thước ảnh khai báo/mapping và cần audit riêng.

## R13 trên chuỗi 300VW thật

Đã đọc trực tiếp annotation 68 điểm từ archive người dùng cung cấp, dùng mapping iBUG68→VF50 chung và không giải nén/sửa dữ liệu nguồn. Tập con cố định gồm 10 video × 300 frame liên tiếp: train `001`; Category 1 `114,124,125`; Category 2 `203,208,211`; Category 3 `410,411,516`.

- 2.990 cặp frame liên tiếp, R13 cảnh báo sai 316 cặp: **FPR 10,57%**, CI 95% bootstrap 1.000 lần theo video **[0,97%; 21,81%]**.
- Category 1: 0/897; Category 2: 10/897 (1,11%); train 001: 4/299 (1,34%).
- Category 3 khó: 302/897 (33,67%); riêng sequence 410 là 56,19%, 411 là 26,42%, 516 là 18,39%.

Độ phân tán rất lớn theo video cho thấy ngưỡng R13 hiện tại nhạy với pose/chuyển động thật ở Category 3; chưa nên suy diễn một FPR duy nhất cho mọi điều kiện. Raw: `r13_300vw_frames.csv`; tổng hợp, danh sách sequence, checksum archive và CI: `r13_300vw_summary.json`.

## Nhiễu tọa độ

Đã đo riêng Gaussian độc lập, dịch tương quan cả contour và lệch hệ thống theo skeleton ở 1/3/5/10/20% IOD. Với reference đúng bằng GT sạch, `norm_error` và suspicion dạng liên tục phân hạng hoàn hảo (AUROC/AUPRC 1,000) vì mọi điểm sạch có error 0; con số này chỉ là sanity check scoring. Khi lượng tử hóa thành severity NORMAL/WARNING/CRITICAL, hiệu năng ở mức nhỏ có ý nghĩa hơn:

| Nhiễu | Mức | Severity AUROC | Severity AUPRC |
|---|---:|---:|---:|
| Gaussian độc lập | 1% / 3% / 5% / 10% / 20% | 0.737 / 0.760 / 0.854 / 0.958 / 0.985 | 0.480 / 0.525 / 0.712 / 0.917 / 0.971 |
| Dịch tương quan contour | 1% / 3% / 5% / 10% / 20% | 0.718 / 0.730 / 0.874 / 1.000 / 1.000 | 0.481 / 0.505 / 0.773 / 1.000 / 1.000 |
| Lệch hệ thống skeleton | 1% / 3% / 5% / 10% / 20% | 0.748 / 0.781 / 1.000 / 1.000 / 1.000 | 0.530 / 0.593 / 1.000 / 1.000 / 1.000 |

Nguồn: `point_metrics_by_noise.json`; raw 1.040.000 dòng ở `point_scores.csv`. File JSON cũng chứa macro-AUROC/AUPRC và CI 95% bootstrap 1.000 lần theo khuôn mặt cho từng loại/mức nhiễu (ví dụ severity Gaussian 1%: AUROC CI [0,724; 0,752], AUPRC CI [0,454; 0,508]).

## Giả định và giới hạn

- WFLW/300W sau ánh xạ không có visibility từng điểm. Tất cả mẫu sạch có 50 điểm `visible`, đúng điều kiện production R02 cảnh báo; vì vậy R02 có FPR 0,897 trên ma trận scenario và **không phải** ước lượng false positive ngoài đời. Chỉ xác nhận unit-level rằng điều kiện all-visible được bắt.
- Recall R13 dùng chuỗi tổng hợp với bước nhảy biết trước; false-positive dùng chuỗi GT 300VW thật. Tập con 10 video chưa đại diện đầy đủ 114 video và chưa loại các frame pose xa chính diện được `extra.zip` đánh dấu.
- R02/R10/R14 dùng state tổng hợp, không chứng minh chất lượng trên che khuất thật. COFW-68/MERL-RAV chưa được cung cấp nên chưa thể bổ sung visibility thật.
- AUROC/AUPRC liên tục có CI suy biến ở 1,000 vì reference lý tưởng; CI severity theo mặt vẫn được lưu. Pha 3 cần đánh giá lại với MediaPipe/reference có nhiễu thực tế.
- Không thay đổi threshold hay source production.

## Tái lập

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_phase2.py --output .bench_workspace/results/face_phase2_20261013 --max-wflw 1000 --max-300w 300 --bootstrap 1000
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_summarize_phase2.py .bench_workspace/results/face_phase2_20261013
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_r13_300vw.py --zip .bench_workspace/datasets/300vw/300VW_Dataset_2015_12_14.zip --output .bench_workspace/results/face_phase2_20261013 --frames-per-video 300 --bootstrap 1000
```
