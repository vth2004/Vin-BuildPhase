# Pha 4 — Benchmark Pose (COCO + CrowdPose)

## Phạm vi và dữ liệu

- COCO val2017: 500 ảnh được chọn cố định với seed 42, cân bằng ảnh một/người và nhiều người; 250 ảnh DEV để chọn ngưỡng, 250 ảnh TEST khóa để báo cáo.
- TEST có 7.029 keypoint được gán nhãn; DEV có 6.657. Tỷ lệ tiêm lỗi là 5%, seed TEST 123.
- Lỗi gồm dịch nhỏ 3–6% scale, dịch lớn 10–20%, đổi trái–phải và sai state occluded. Mẫu sạch có jitter Gaussian 1,2% scale.
- K1 = YOLO26s-pose; K2 = K1 + RTMPose-m; K3 = K2 + YOLOv8m-pose. Tất cả inference là thật và cache theo ảnh; mock fallback bằng 0. K3 được giữ như kết quả thí nghiệm theo yêu cầu ban đầu, nhưng YOLOv8m không thuộc phạm vi triển khai thực tế nên không chạy sanity check AP và không dùng làm khuyến nghị production.

Raw chính: `.bench_workspace/pose/results/pose_qa_results.json`; từng điểm TEST: `.bench_workspace/pose/results/pose_qa_test_rows.csv`; tốc độ: `speed_benchmark.json`; tài nguyên: `resource_K1.json`–`resource_K3.json`; audit mock: `mock_fallback_audit.json`.

## Kết quả QA trên TEST

Khoảng tin cậy 95% được bootstrap 1.000 lần theo ảnh (cluster bootstrap). Ngưỡng warning được chọn chỉ trên DEV.

| Mode | Ngưỡng DEV | AUROC (95% CI) | AUPRC (95% CI) | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 0,30 | 0,6952 (0,6608–0,7330) | 0,1895 (0,1486–0,2535) | 0,2436 | 0,3553 | 0,2890 | 0,0576 |
| K2 | 0,30 | 0,7020 (0,6669–0,7391) | 0,2831 (0,2226–0,3626) | 0,3893 | 0,3324 | 0,3586 | 0,0272 |
| K3 | 0,30 | 0,6985 (0,6641–0,7350) | 0,2962 (0,2330–0,3759) | 0,4260 | 0,3381 | 0,3770 | 0,0238 |

| Mode | Median latency | P95 latency | Peak RSS |
|---|---:|---:|---:|
| K1 | 85,81 ms | 119,59 ms | 549,61 MiB |
| K2 | 505,01 ms | 685,37 ms | 793,64 MiB |
| K3 | 920,13 ms | 1.292,18 ms | 1.100,50 MiB |

K2 tăng AUPRC 0,0936 so với K1, đổi lại khoảng +419 ms median và +244 MiB. K3 chỉ tăng thêm 0,0131 AUPRC so với K2 nhưng tốn thêm khoảng +415 ms và +307 MiB. Vì vậy K2 chỉ đáng dùng khi ưu tiên chất lượng QA hơn độ trễ; K3 chưa cho thấy lợi ích đủ lớn.

## Theo loại lỗi

Các ô là AUROC / AUPRC / F1 tại ngưỡng đã khóa trên DEV.

| Lỗi | K1 | K2 | K3 |
|---|---:|---:|---:|
| Dịch nhỏ | 0,7000 / 0,0322 / 0,0672 | 0,7320 / 0,0460 / 0,0997 | 0,7226 / 0,0467 / 0,1079 |
| Dịch lớn | 0,9467 / 0,1333 / 0,1922 | 0,9589 / 0,2781 / 0,2970 | 0,9624 / 0,2961 / 0,3333 |
| Đổi trái–phải | 0,9420 / 0,2123 / 0,2179 | 0,9579 / 0,3783 / 0,3601 | 0,9571 / 0,4120 / 0,3889 |
| Sai state occluded | 0,3031 / 0,0094 / 0,0083 | 0,2713 / 0,0089 / 0,0000 | 0,2669 / 0,0088 / 0,0000 |

Sai state occluded không được pipeline hiện tại phát hiện: logic production giảm một nửa suspicion khi state là occluded, nên kết quả còn thấp hơn ngẫu nhiên. Đây là giới hạn đo được, không phải lỗi benchmark được che đi.

## Ngưỡng

- Warning hiện tại 0,20 so với DEV chọn 0,30: TEST F1 lần lượt K1 0,2852→0,2890; K2 0,3628→0,3586; K3 0,3632→0,3770. Do K2 giảm trên TEST, chưa đủ cơ sở áp một ngưỡng 0,30 chung; không sửa config production.
- Grid swap trên DEV chọn đúng giá trị hiện hành `ratio=0,6`, `min_scale=0,15` cho cả ba mode. Khuyến nghị giữ nguyên.

## Kiểm tra tổng quát trên CrowdPose

CrowdPose có 14 điểm: 12 điểm chi + `top_head` + `neck`, không phải COCO-17 và không có nose/eyes/ears. Mapper chỉ chuyển 12 điểm chi sang VF17; để trống VF17 ID 1–5 và không suy diễn `top_head` thành nose. Ba unit test xác nhận quy ước right=ID chẵn, left=ID lẻ và không tạo swap giả.

Đã tải từ hai link Google Drive trong repository chính thức: 20.000 ảnh và bốn annotation `train/val/trainval/test`. License dataset **chưa xác minh** vì repository không công bố license và issue hỏi license vẫn đang mở; chỉ dùng nội bộ cho nghiên cứu, lưu trong `.bench_workspace/` và không commit. Từ 2.000 ảnh val, chọn cố định 500 ảnh (250 DEV/250 TEST, seed 42). TEST có 7.897 keypoint VF17 hợp lệ.

| Mode | Ngưỡng DEV | AUROC (95% CI) | AUPRC (95% CI) | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 0,20 | 0,6820 (0,6501–0,7120) | 0,1032 (0,0864–0,1221) | 0,1304 | 0,4335 | 0,2005 | 0,1445 |
| K2 | 0,20 | 0,7054 (0,6710–0,7382) | 0,1734 (0,1392–0,2266) | 0,2477 | 0,3511 | 0,2904 | 0,0533 |
| K3 | 0,20 | 0,7087 (0,6732–0,7409) | 0,2028 (0,1598–0,2621) | 0,2797 | 0,3564 | 0,3135 | 0,0459 |

Kết quả giữ cùng xu hướng với COCO: K2 cải thiện rõ so với K1; K3 chỉ cải thiện vừa phải so với K2. Ngưỡng DEV chọn 0,20 cho cả ba mode, cho thấy đề xuất 0,30 từ COCO không tổng quát sang CrowdPose. Mock fallback CrowdPose bằng 0.

Raw: `.bench_workspace/pose/crowdpose/benchmark/results/pose_qa_results.json`, `pose_qa_test_rows.csv`, `speed_benchmark.json`, `mock_fallback_audit.json`. Nguồn: [CrowdPose chính thức](https://github.com/jeffffffli/CrowdPose), [schema CrowdPose của MMPose](https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/crowdpose.py).

## Sanity check AP công bố

| Model | Số công bố | Điều kiện nguồn | Tự đo đúng điều kiện |
|---|---:|---|---|
| YOLO26s-pose | AP 63,0; AP50 86,6 | COCO, 640, single-model/single-scale | **AP 62,56; AP50 85,71; AP75 68,56** |
| YOLOv8m-pose | AP 65,0; AP50 88,8 | COCO, 640 | Không đo: không dùng trong dự án |
| RTMPose-m chính thức | AP 74,6; AP50 89,9; AP75 81,7 | COCO, 256×192, flip-test, detector boxes AP-H=56,4 | **AP 74,58; AP50 89,85; AP75 81,65** |

Nguồn: [Ultralytics COCO pose/YOLO26](https://docs.ultralytics.com/datasets/pose/coco/), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md), [RTMPose official model zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md), [RTMPose-m official config](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py).

YOLO26s được tự đo bằng Ultralytics 8.4.174 trên đúng danh sách COCO-pose val 2.346 ảnh, input 640, single-model/single-scale, không flip, CPU. AP thấp hơn công bố 0,44 điểm phần trăm và AP50 thấp hơn 0,89 điểm phần trăm; chênh lệch nhỏ, không cho thấy lỗi inference nghiêm trọng. Raw: `.bench_workspace/pose/full_coco/results/yolo26s-pose_official_ap.json`.

RTMPose-m chính thức được đo bằng checkpoint/config MMPose COCO, flip-test và 104.125 person proposals từ detector AP-H=56,4. Kết quả khớp công bố trong 0,021 điểm phần trăm AP. Sanity phụ dùng GT boxes đạt AP 76,61, xác nhận bbox là nguyên nhân làm kết quả cao hơn nếu đánh giá không đúng điều kiện. Raw summary: `.bench_workspace/pose/rtmpose_official/results/rtmpose_official_ap.json`; raw predictions detected-box và GT-box nằm cùng thư mục. Artifact production body7 + YOLOX-HumanArt vẫn là biến thể khác và được đánh giá riêng trong benchmark QA, không đánh tráo với checkpoint chính thức.

## Tái lập

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_pose/bench_pose_qa.py --output .bench_workspace/pose/results --bootstrap 1000
backend/.venv/Scripts/python.exe experiments/benchmark_pose/bench_resource_mode.py --mode K1 --output .bench_workspace/pose/results/resource_K1.json
backend/.venv/Scripts/python.exe -m unittest experiments.benchmark_pose.bench_test_pose_mapping -v
```

Benchmark QA chính của Pha 4 đã hoàn tất trên COCO và CrowdPose. Sanity check AP của YOLO26s và RTMPose-m chính thức đều hoàn tất; YOLOv8m được loại theo phạm vi dự án. RTMPose chính thức và artifact RTMLib production được giữ thành hai hàng/ý nghĩa riêng biệt.
