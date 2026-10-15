# Báo cáo Benchmark Landmark QA

Ngày tổng hợp: 2026-10-14  
Commit benchmark: `c09804e0ea732330f64f002f7367bc7a68d9f677`  
Seed chính: `42` (`123` cho injection TEST Pose)

Evidence sau cleanup: `D:\Vin\VinPrj\landmark_qa_evidence\bench_evidence_2026-10-14.zip`  
SHA-256: `0d05e96007f415a7a8db7060740a3752fe8e8d7907f7e5d2de1e63eb67e6bade`. Các đường dẫn `.bench_workspace/...` trong báo cáo là vị trí raw lúc chạy; bản đóng gói giữ cùng nội dung theo các thư mục `face_phase*`, `pose_coco_qa`, `pose_crowdpose_qa`, `pose_yolo26_ap` và `pose_rtmpose_ap`.

## 1. Mục tiêu và câu hỏi nghiên cứu

Benchmark này đo khả năng **phát hiện nhãn keypoint sai** của Landmark QA, không chỉ đo độ chính xác của model pose/face. Các câu hỏi chính:

1. Rules Face R01–R14 bắt từng loại lỗi với precision, recall, F1 và FPR nào?
2. Điểm suspicion/severity phân hạng điểm lỗi tốt đến đâu?
3. Ngưỡng hiện tại có nên đổi khi chỉ được chọn trên DEV và đánh giá một lần trên TEST?
4. Với Pose, K2 có đáng chi phí so với K1 và K3 có đem lại đủ lợi ích bổ sung không?
5. Kết quả có giữ xu hướng trên một dataset ngoài COCO hay không?

Kết luận ngắn: Face bắt rất tốt các lỗi xác định và swap, nhưng R02 chưa đo được ngoài đời, một số luật hình học có precision thấp, và R13 có FPR cao trên video khó. Với Pose, K2 cải thiện AUPRC rõ rệt so với K1 nhưng chậm hơn khoảng 419 ms/ảnh trên CPU; K3 chỉ tăng thêm 0,0131 AUPRC trên COCO nên không đáng chi phí trong phạm vi triển khai hiện tại. Chưa có cơ sở để đổi đồng loạt các ngưỡng production.

## 2. Dữ liệu

Không sử dụng dữ liệu VinFast. Dataset, ảnh, model tải thêm và cache chỉ nằm trong `.bench_workspace/`.

| Dataset | Vai trò | Quy mô đã dùng | License/quyền truy cập |
|---|---|---:|---|
| [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) | Face chính, 98 điểm, sáu lát thuộc tính | 10.000 face; DEV/TEST theo ảnh | Chưa thấy license minh thị; xử lý research-only |
| [300W](https://ibug.doc.ic.ac.uk/resources/300-W/) | Face ngoài miền, iBUG-68 | 600 face; DEV/TEST 300/300 | Đã xác minh research-only, không dùng để huấn luyện thương mại |
| [300VW](https://www.ibug.doc.ic.ac.uk/resources/300-VW/) | FPR thời gian R13 | 10 video × 300 frame | Chưa thấy license minh thị; archive do người dùng cung cấp, research-only |
| [COFW-68](https://github.com/g0lnaz/cofw68-benchmark) | Visibility theo điểm dự kiến | Chưa đo | Chưa xác minh license minh thị; dữ liệu chưa được cung cấp |
| [MERL-RAV](https://github.com/abhi1kumar/MERL-RAV_dataset) | Ba lớp occlusion dự kiến | Chưa đo | Chưa có quyền ảnh AFLW; không tải và không thay bằng dữ liệu giả |
| [COCO val2017](https://cocodataset.org/#download) | Pose chính và sanity AP | 500 ảnh QA; full val cho AP | Theo điều khoản COCO; benchmark nghiên cứu |
| [CrowdPose val](https://github.com/jeffffffli/CrowdPose) | Kiểm tra tổng quát Pose | 500/2.000 ảnh val | Repository không công bố license rõ; research-only, không phân phối lại |

Nguồn và trạng thái xác minh Face chi tiết: [`FACE_DATASET_LICENSES.md`](FACE_DATASET_LICENSES.md). Nguồn CrowdPose và điều kiện sử dụng được ghi cùng raw tại `.bench_workspace/pose/crowdpose/benchmark/results/`.

### Ánh xạ Face về VF50

Hai mapper độc lập được dùng: WFLW-98→VF50 và iBUG-68→VF50. Mười hai anchor `{0,4,5,9,10,13,14,18,22,26,30,36}` được ánh xạ trực tiếp theo ngữ nghĩa. Contour được resample theo chiều dài cung; riêng mắt WFLW có đủ tám điểm nên không nội suy. Quy ước trái/phải theo khung nhìn ảnh, `outside` có `x=y=null`. Bảng duyệt từng điểm: [`FACE_MAPPING_TABLES.md`](FACE_MAPPING_TABLES.md); cấu hình tái lập: `experiments/benchmark_face/configs/`.

Noise floor thuật toán của WFLW bằng 0 trên cả bảy skeleton. Với iBUG-68, P95 của mắt trái/phải lần lượt là 1,212% và 1,193% IOD; các skeleton khác bằng 0. Mức này thấp hơn tolerance contour 5%, nhưng không chứng minh protocol annotation của dataset và VF50 hoàn toàn tương đương. Nguồn: `.bench_workspace/results/face_phase1_20261013/phase1_metrics.json` và `face_baseline_raw.csv`.

### Ánh xạ Pose về VF17

COCO được ánh xạ qua `COCO_TO_VF17`, giữ quy ước right ở ID chẵn và left ở ID lẻ. CrowdPose chỉ có 12 điểm chi dùng được; `top_head` và `neck` không bị suy diễn thành nose/eyes/ears, VF17 ID 1–5 để trống. Ba unit test kiểm tra parity và không tạo swap giả. Nguồn: `experiments/benchmark_pose/bench_pose_mapping.py` và `bench_test_pose_mapping.py`.

## 3. Phương pháp

- Chia DEV/TEST theo ảnh với seed cố định; chọn ngưỡng chỉ trên DEV, sau đó khóa và đánh giá TEST.
- Face Phase 2 dùng 1.300 face TEST độc lập, tạo 39.000 scenario trên bản sao annotation. Có control sạch và các lỗi R01–R14, Gaussian độc lập, dịch tương quan contour, lệch hệ thống skeleton và đảo nhãn.
- Face model reference trong Phase 2–3 là GT sạch để cô lập rules/scoring. Baseline MediaPipe thật được đo riêng ở Phase 1 để không đánh tráo điều kiện lý tưởng với inference thực.
- Pose dùng 500 ảnh COCO (250 DEV/250 TEST), 5% injection: dịch nhỏ, dịch lớn, swap trái–phải và sai state occluded. CrowdPose dùng cách chia tương tự.
- Metric: precision, recall, F1, FPR, AUROC và AUPRC. CI 95% dùng cluster bootstrap 1.000 lần theo ảnh/face; R13 bootstrap theo video.
- Pose monkeypatch bộ đếm `make_mock_results()` ở script benchmark; mọi run có mock > 0 phải loại. Cả COCO và CrowdPose đều ghi nhận 0 lần.
- Raw lưu commit hash, cấu hình, `pip freeze`, CSV/JSON và checksum. Không sửa rules, scoring, ensemble hoặc config production.

Nguồn cấu hình: `.bench_workspace/results/face_phase2_20261013/run_config.json`, `.bench_workspace/results/face_phase3_20261013/phase3_threshold_results.json`, `.bench_workspace/pose/results/split_manifest.json`, `.bench_workspace/pose/results/mock_fallback_audit.json`.

## 4. Kết quả Pose

### 4.1. COCO TEST: chất lượng QA và chi phí

Ngưỡng warning được chọn trên DEV. CI là bootstrap 1.000 lần theo ảnh.

| Mode | Ngưỡng | AUROC (95% CI) | AUPRC (95% CI) | Precision | Recall | F1 | FPR | Median / P95 | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 | 0,30 | 0,6952 (0,6608–0,7330) | 0,1895 (0,1486–0,2535) | 0,2436 | 0,3553 | 0,2890 | 0,0576 | 85,81 / 119,59 ms | 549,61 MiB |
| K2 | 0,30 | 0,7020 (0,6669–0,7391) | 0,2831 (0,2226–0,3626) | 0,3893 | 0,3324 | 0,3586 | 0,0272 | 505,01 / 685,37 ms | 793,64 MiB |
| K3 | 0,30 | 0,6985 (0,6641–0,7350) | 0,2962 (0,2330–0,3759) | 0,4260 | 0,3381 | 0,3770 | 0,0238 | 920,13 / 1.292,18 ms | 1.100,50 MiB |

Nguồn raw của toàn bộ bảng: `.bench_workspace/pose/results/pose_qa_results.json`, `pose_qa_test_rows.csv`, `speed_benchmark.json`, `resource_K1.json`, `resource_K2.json`, `resource_K3.json`.

K2 tăng AUPRC 0,0936 so với K1, đổi lại +419,20 ms median và +244,03 MiB. K3 chỉ tăng 0,0131 AUPRC so với K2 nhưng thêm khoảng 415,12 ms và 306,86 MiB. K2 phù hợp khi ưu tiên chất lượng QA hơn latency; K3 không đáng chi phí và YOLOv8m không thuộc phạm vi triển khai của dự án.

### 4.2. Theo loại lỗi

Mỗi ô là AUROC / AUPRC / F1 tại ngưỡng khóa từ DEV.

| Lỗi | K1 | K2 | K3 |
|---|---:|---:|---:|
| Dịch nhỏ | 0,7000 / 0,0322 / 0,0672 | 0,7320 / 0,0460 / 0,0997 | 0,7226 / 0,0467 / 0,1079 |
| Dịch lớn | 0,9467 / 0,1333 / 0,1922 | 0,9589 / 0,2781 / 0,2970 | 0,9624 / 0,2961 / 0,3333 |
| Swap trái–phải | 0,9420 / 0,2123 / 0,2179 | 0,9579 / 0,3783 / 0,3601 | 0,9571 / 0,4120 / 0,3889 |
| Sai state occluded | 0,3031 / 0,0094 / 0,0083 | 0,2713 / 0,0089 / 0,0000 | 0,2669 / 0,0088 / 0,0000 |

Nguồn: `.bench_workspace/pose/results/pose_qa_results.json` và `pose_qa_test_rows.csv`. Sai state occluded hiện không được phát hiện: production giảm một nửa suspicion cho điểm occluded, nên metric thấp hơn ngẫu nhiên.

### 4.3. Khả năng tổng quát trên CrowdPose

| Mode | Ngưỡng DEV | AUROC (95% CI) | AUPRC (95% CI) | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 0,20 | 0,6820 (0,6501–0,7120) | 0,1032 (0,0864–0,1221) | 0,1304 | 0,4335 | 0,2005 | 0,1445 |
| K2 | 0,20 | 0,7054 (0,6710–0,7382) | 0,1734 (0,1392–0,2266) | 0,2477 | 0,3511 | 0,2904 | 0,0533 |
| K3 | 0,20 | 0,7087 (0,6732–0,7409) | 0,2028 (0,1598–0,2621) | 0,2797 | 0,3564 | 0,3135 | 0,0459 |

Nguồn: `.bench_workspace/pose/crowdpose/benchmark/results/pose_qa_results.json`, `pose_qa_test_rows.csv`, `speed_benchmark.json`, `mock_fallback_audit.json`. Xu hướng K2>K1 được giữ, nhưng ngưỡng DEV quay về 0,20; điều này phản bác việc áp 0,30 chung cho mọi miền dữ liệu.

### 4.4. Sanity check AP

| Model | Công bố | Tự đo đúng điều kiện |
|---|---:|---:|
| YOLO26s-pose | AP 63,0; AP50 86,6 | AP 62,56; AP50 85,71; AP75 68,56 |
| RTMPose-m 256×192 | AP 74,6; AP50 89,9; AP75 81,7 | AP 74,58; AP50 89,85; AP75 81,65 |

Nguồn YOLO26: `.bench_workspace/pose/full_coco/results/yolo26s-pose_official_ap.json`. Nguồn RTMPose: `.bench_workspace/pose/rtmpose_official/results/rtmpose_official_ap.json` và `SHA256SUMS.txt`. RTMPose dùng flip-test và 104.125 person proposals của detector AP-H=56,4; sai khác AP so với công bố chỉ 0,021 điểm phần trăm. YOLOv8m không đo theo quyết định phạm vi của người dùng.

Số công bố được đối chiếu từ [Ultralytics COCO Pose](https://docs.ultralytics.com/datasets/pose/coco/) và [RTMPose official model zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md), không lấy từ trí nhớ.

## 5. Kết quả Face VF50

### 5.1. MediaPipe làm tham chiếu: coverage và NME

NME chỉ tính trên face phát hiện và ghép thành công; phải đọc cùng coverage để tránh selection bias.

| Dataset/lát | GT face | Matched | Recall detection | NME mean | Median | P90 |
|---|---:|---:|---:|---:|---:|---:|
| 300W toàn bộ | 600 | 384 | 64,00% | 8,54% | 7,05% | 11,96% |
| WFLW toàn bộ | 10.000 | 4.019 | 40,19% | 9,14% | 7,04% | 13,75% |
| WFLW pose | 588 | 147 | 25,00% | 21,85% | 18,52% | 26,77% |
| WFLW expression | 496 | 181 | 36,49% | 10,67% | 8,08% | 17,77% |
| WFLW illumination | 2.225 | 786 | 35,33% | 9,28% | 7,20% | 13,65% |
| WFLW make-up | 285 | 101 | 35,44% | 8,17% | 6,56% | 12,31% |
| WFLW occlusion | 2.077 | 611 | 29,42% | 10,40% | 8,11% | 15,17% |
| WFLW blur | 2.183 | 208 | 9,53% | 14,50% | 9,35% | 21,24% |

Nguồn: `.bench_workspace/results/face_phase1_20261013/face_baseline_summary.json`, `phase1_metrics.json`, `face_baseline_raw.csv`. Miss rate tổng thể tương ứng là 36,00% trên 300W và 59,81% trên WFLW. Pose và blur là hai lát rủi ro lớn nhất.

### 5.2. Rules R01–R14 trên TEST

| Rule | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | FPR (95% CI) |
|---|---:|---:|---:|---:|
| R01 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 0,000 [0,000–0,000] |
| R02 | 0,037 [0,037–0,037] | 1,000 [1,000–1,000] | 0,071 [0,071–0,071] | 0,897 [0,897–0,897] |
| R03 | 0,839 [0,823–0,853] | 1,000 [1,000–1,000] | 0,912 [0,903–0,921] | 0,007 [0,006–0,007] |
| R04 | 0,432 [0,418–0,443] | 0,990 [0,984–0,995] | 0,601 [0,588–0,612] | 0,045 [0,043–0,047] |
| R05 | 0,452 [0,408–0,502] | 1,000 [1,000–1,000] | 0,622 [0,580–0,668] | 0,042 [0,034–0,050] |
| R06 | 0,386 [0,377–0,392] | 0,995 [0,991–0,998] | 0,556 [0,547–0,563] | 0,055 [0,053–0,057] |
| R07 | 0,233 [0,218–0,248] | 0,998 [0,995–1,000] | 0,378 [0,358–0,397] | 0,113 [0,104–0,123] |
| R08 | 0,629 [0,568–0,698] | 0,980 [0,972–0,987] | 0,767 [0,718–0,816] | 0,020 [0,015–0,026] |
| R09 | 0,219 [0,209–0,228] | 1,000 [1,000–1,000] | 0,359 [0,346–0,371] | 0,123 [0,117–0,130] |
| R10 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 0,000 [0,000–0,000] |
| R11 | 0,880 [0,789–0,957] | 1,000 [1,000–1,000] | 0,936 [0,882–0,978] | 0,005 [0,002–0,009] |
| R12 | 0,267 [0,239–0,299] | 1,000 [1,000–1,000] | 0,422 [0,386–0,460] | 0,095 [0,081–0,110] |
| R13 | 1,000 [1,000–1,000] | 0,986 [0,979–0,992] | 0,993 [0,990–0,996] | 0,000 [0,000–0,000] |
| R14 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 0,000 [0,000–0,000] |

Nguồn: `.bench_workspace/results/face_phase2_20261013/rule_metrics.json`, `rule_metrics_bootstrap.json`, `scenario_results.csv`. R02 không phải ước lượng thực địa: WFLW/300W sau mapping không có visibility từng điểm nên control đều all-visible. R10/R14 cũng chỉ được xác nhận bằng state tổng hợp.

Trên 1.300 control sạch, FPR nổi bật là R12 8,31%, R07 3,62%, R05 2,69%, R09 1,31% và R08 1,23%. Nguồn: `.bench_workspace/results/face_phase2_20261013/clean_control_fpr.json`.

### 5.3. Điểm suspicion/severity

Vì reference đúng bằng GT sạch, score liên tục có AUROC/AUPRC 1,000 và chỉ là sanity check. Severity lượng tử hóa cho kết quả thực tế hơn:

| Kiểu nhiễu | 1% IOD AUROC/AUPRC | 3% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|---:|
| Gaussian độc lập | 0,737/0,480 | 0,760/0,525 | 0,854/0,712 | 0,958/0,917 | 0,985/0,971 |
| Dịch tương quan contour | 0,718/0,481 | 0,730/0,505 | 0,874/0,773 | 1,000/1,000 | 1,000/1,000 |
| Lệch hệ thống skeleton | 0,748/0,530 | 0,781/0,593 | 1,000/1,000 | 1,000/1,000 | 1,000/1,000 |

Nguồn: `.bench_workspace/results/face_phase2_20261013/point_metrics_by_noise.json`; 1.040.000 dòng raw: `point_scores.csv`.

### 5.4. R13 trên video thật

Trong 2.990 cặp frame liên tiếp của 300VW, R13 cảnh báo sai 316 cặp: FPR 10,57%, CI 95% theo video [0,97%; 21,81%]. Category 3 đạt 33,67%; sequence 410 đạt 56,19%. Nguồn: `.bench_workspace/results/face_phase2_20261013/r13_300vw_summary.json` và `r13_300vw_frames.csv`. Recall 0,986 ở bảng rules dùng chuỗi tổng hợp có jump biết trước; hai con số trả lời hai câu hỏi khác nhau.

## 6. Đề xuất ngưỡng

| Thành phần | Hiện tại | DEV chọn | TEST trước → sau | Kết luận |
|---|---:|---:|---:|---|
| Face R12 IOD ratio | 0,012 | 0,006 | F1 0,9610→0,9668; FPR 8,13%→6,88% | Chỉ đề xuất thử nghiệm |
| Face R04/R06/R13/R14 | hiện hành | không đổi | Không cải thiện đủ | Giữ |
| Face scoring anchor/contour/NME/severity/q | 0,03/0,05/0,035/1,8/2,5 | 0,015/0,02/0,015/1,2/1,5 | Clean reference có error 0 | Không đề xuất áp dụng |
| Pose warning | 0,20 | COCO 0,30; CrowdPose 0,20 | K2 COCO F1 0,3628→0,3586 | Giữ 0,20 |
| Pose swap | ratio 0,6; min-scale 0,15 | trùng hiện tại | DEV chọn giá trị hiện hành | Giữ |

Nguồn Face: `.bench_workspace/results/face_phase3_20261013/phase3_threshold_results.json`, `test_comparison.csv`, `proposed_vf50_thresholds.yaml`. Nguồn Pose: `.bench_workspace/pose/results/pose_qa_results.json` và `.bench_workspace/pose/crowdpose/benchmark/results/pose_qa_results.json`. File đề xuất benchmark-only: `experiments/benchmark_face/proposed_vf50_thresholds.yaml`; production config không bị sửa.

Hai lệch config–code ảnh hưởng việc tune:

- R09 runtime đọc `per_point_tolerance_px`, không đọc `contour_tolerance_multiplier`; đổi YAML multiplier 0,1→100 không đổi số cảnh báo 3→3 trên 100 mẫu.
- R10 runtime chỉ kiểm mắt/mày; bắt 0/100 lỗi thiếu một phần môi ngoài dù YAML có threshold cho môi và mũi.

Nguồn: `.bench_workspace/results/face_phase3_20261013/config_code_audit.json`.

## 7. Giới hạn trung thực

- Public dataset có domain gap với dữ liệu triển khai; không dùng dữ liệu VinFast nên chưa đo trực tiếp miền mục tiêu.
- Lỗi tiêm có ground truth chính xác nhưng không đại diện đầy đủ lỗi người gán thật.
- Mapping 68/98→VF50 tạo sai số và khác protocol; mắt iBUG-68 cần nội suy 6→8.
- Face scoring Phase 2–3 dùng model reference lý tưởng; AUROC/AUPRC liên tục 1,000 không được coi là hiệu năng thực địa.
- MediaPipe NME chỉ trên detection đã match, nên chịu selection bias; WFLW coverage chỉ 40,19%.
- COFW-68 và MERL-RAV chưa đo; do đó visibility/occlusion thật của Face còn thiếu.
- CrowdPose license chưa xác minh minh thị và schema thiếu năm điểm mặt VF17.
- R13 mới dùng 10/114 video, CI theo video rộng.
- Chưa có load test API, concurrency/throughput, lỗi I/O hoặc benchmark end-to-end qua DB.
- Chưa đo downstream task nên chưa biết việc làm sạch nhãn cải thiện model huấn luyện bao nhiêu.
- Latency/RAM được đo trên CPU và môi trường hiện tại; không suy rộng trực tiếp sang máy production.

## 8. Hướng tiếp theo

1. Thu thập một tập lỗi người gán thật, được adjudicate độc lập; khóa làm TEST thực địa.
2. Khi có quyền, bổ sung COFW-68/MERL-RAV để hiệu chỉnh R02/R10/R14 với visibility thật.
3. Chạy pilot downstream: cùng kiến trúc và seed trên ba tập nhiễu/chạy QA/làm sạch thủ công; so sánh AP/NME.
4. Sửa lệch config–code R09/R10 trong một nhánh riêng rồi benchmark lại, không trộn vào benchmark hiện tại.
5. Load test API riêng: throughput, P95/P99, RAM, timeout, recovery và nhiều job đồng thời.
6. Nếu đưa K2 vào production, cân nhắc chạy bất đồng bộ hoặc chỉ kích hoạt cho mẫu K1 có uncertainty cao.

## 9. Phụ lục tái lập và raw evidence

### Face mapping và baseline

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_test_mapping.py
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_face_baseline.py --output .bench_workspace/results/face_phase1_20261013
```

Raw: `.bench_workspace/results/face_phase1_20261013/`; checksum: `RESULT_SHA256.csv`.

### Face rules và R13

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_phase2.py --output .bench_workspace/results/face_phase2_20261013 --max-wflw 1000 --max-300w 300 --bootstrap 1000
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_r13_300vw.py --zip .bench_workspace/datasets/300vw/300VW_Dataset_2015_12_14.zip --output .bench_workspace/results/face_phase2_20261013 --frames-per-video 300 --bootstrap 1000
```

Raw: `.bench_workspace/results/face_phase2_20261013/`; checksum: `SHA256SUMS.txt`.

### Face threshold

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_phase3_thresholds.py --output .bench_workspace/results/face_phase3_20261013 --zip-300vw .bench_workspace/datasets/300vw/300VW_Dataset_2015_12_14.zip --limit 500 --seed 42
backend/.venv/Scripts/python.exe experiments/benchmark_face/bench_phase3_config_code_audit.py .bench_workspace/results/face_phase3_20261013/config_code_audit.json
```

Raw: `.bench_workspace/results/face_phase3_20261013/`; checksum: `SHA256SUMS.txt`. PR/ROC DEV: `reports/assets/face_phase3_dev_pr_curves.png` và `face_phase3_dev_roc_curves.png`.

### Pose QA

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_pose/bench_pose_qa.py --output .bench_workspace/pose/results --bootstrap 1000
backend/.venv/Scripts/python.exe experiments/benchmark_pose/bench_resource_mode.py --mode K1 --output .bench_workspace/pose/results/resource_K1.json
backend/.venv/Scripts/python.exe -m unittest experiments.benchmark_pose.bench_test_pose_mapping -v
```

Raw COCO QA: `.bench_workspace/pose/results/`; raw CrowdPose: `.bench_workspace/pose/crowdpose/benchmark/results/`; mỗi thư mục có `SHA256SUMS.txt`.

### Pose AP sanity

```powershell
backend/.venv/Scripts/python.exe experiments/benchmark_pose/bench_coco_official_ap.py
```

Raw YOLO26: `.bench_workspace/pose/full_coco/results/`; raw RTMPose: `.bench_workspace/pose/rtmpose_official/results/`; checksum nằm cùng thư mục.

Các báo cáo trung gian có diễn giải bổ sung: [`FACE_PHASE1_SUMMARY.md`](FACE_PHASE1_SUMMARY.md), [`FACE_PHASE2_SUMMARY.md`](FACE_PHASE2_SUMMARY.md), [`FACE_PHASE3_SUMMARY.md`](FACE_PHASE3_SUMMARY.md), [`POSE_PHASE4_SUMMARY.md`](POSE_PHASE4_SUMMARY.md).
