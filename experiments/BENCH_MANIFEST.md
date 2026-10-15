# Benchmark Manifest

Mọi đường dẫn tương đối tính từ repository root. `DELETE` là dữ liệu tạm; `KEEP` là mã, cấu hình hoặc bằng chứng tóm tắt cần giữ lại.

| Path | Mục đích | Dung lượng lúc ghi | Nhãn |
|---|---|---:|---|
| `.gitignore` | Cô lập toàn bộ workspace benchmark | N/A | KEEP |
| `experiments/BENCH_MANIFEST.md` | Sổ theo dõi file và thư mục benchmark | N/A | KEEP |
| `.bench_workspace/` | Gốc duy nhất cho dữ liệu nặng và tạm | 0 B khi tạo | DELETE |
| `.bench_workspace/baseline/` | Snapshot trạng thái repo trước benchmark | 0 B khi tạo | DELETE |
| `.bench_workspace/tmp_data/` | `LANDMARK_DATA_DIR` cô lập cho benchmark | 0 B khi tạo | DELETE |
| `.bench_workspace/models/` | Model tải riêng cho benchmark | 0 B khi tạo | DELETE |
| `.bench_workspace/baseline/repo_state.txt` | Nhánh, commit và worktree gốc | 155 B | DELETE |
| `.bench_workspace/baseline/backend_data_inventory.csv` | Danh sách, dung lượng, mtime và SHA-256 của dữ liệu hệ thống gốc | 20,512 B | DELETE |
| `.bench_workspace/baseline/processes.csv` | Snapshot tiến trình trước benchmark | 17,997 B | DELETE |
| `.bench_workspace/baseline/listening_ports.csv` | Snapshot cổng TCP đang lắng nghe trước benchmark | 858 B | DELETE |
| `.bench_workspace/baseline/backend_tests_before.txt` | Kết quả regression test nền: 54 test pass | 9,269 B | DELETE |
| `.bench_workspace/baseline/pip_freeze_before.txt` | Phiên bản thư viện của venv benchmark ban đầu | 1,281 B | DELETE |
| `.bench_workspace/requirements_backend_resolved.txt` | Bản requirements tạm, ghim `rtmlib==0.0.16` do constraint production không tồn tại trên index | 243 B | DELETE |
| `.bench_workspace/venv/` | Venv benchmark Python 3.11.9 | 1,417,465,575 B sau cài nền | DELETE |
| `.bench_workspace/venv_failed_py312/` | Venv Python 3.12 cài lỗi, giữ cô lập để cleanup | 11,790,584 B | DELETE |
| `experiments/benchmark_face/` | Code benchmark Face độc lập với production | 30,479 B (source/config) | KEEP |
| `experiments/benchmark_face/configs/` | Cấu hình mapping public datasets → VF50 | 2,840 B | KEEP |
| `experiments/benchmark_face/__init__.py` | Package benchmark Face | 75 B | KEEP |
| `experiments/benchmark_face/bench_mapping_core.py` | Mapper cấu hình và phép đo noise floor theo skeleton | 5,292 B | KEEP |
| `experiments/benchmark_face/bench_mapping_wflw98_to_vf50.py` | Entry point mapping WFLW-98 → VF50 | 310 B | KEEP |
| `experiments/benchmark_face/bench_mapping_68_to_vf50.py` | Entry point mapping iBUG-68 → VF50 dùng chung | 313 B | KEEP |
| `experiments/benchmark_face/bench_generate_mapping_table.py` | Sinh bảng mapping Markdown có thể kiểm toán | 1,329 B | KEEP |
| `experiments/benchmark_face/bench_test_mapping.py` | Regression test mapping và anchor | 1,608 B | KEEP |
| `experiments/benchmark_face/configs/wflw98_to_vf50.yaml` | Mapping WFLW-98 đã duyệt | 1,184 B | KEEP |
| `experiments/benchmark_face/configs/ibug68_to_vf50.yaml` | Mapping dùng chung 300W/COFW-68/MERL-RAV/300VW | 1,656 B | KEEP |
| `reports/FACE_MAPPING_TABLES.md` | Bảng ánh xạ sinh trực tiếp từ YAML | 5,190 B | KEEP |
| `reports/DATASET_ACCESS.md` | Hướng dẫn tải thủ công từ nguồn chính thức | 1,733 B | KEEP |
| `experiments/benchmark_face/bench_prepare_datasets.py` | Parse WFLW/300W, map VF50 và chia DEV/TEST theo ảnh | 6,618 B | KEEP |
| `experiments/benchmark_face/bench_face_baseline.py` | Chạy MediaPipe, coverage, NME và mapping noise floor | 7,307 B | KEEP |
| `experiments/benchmark_face/bench_summarize_phase1.py` | Tổng hợp recall/NME theo lát từ raw per-face | ~5.6 KiB | KEEP |
| `reports/FACE_PHASE1_SUMMARY.md` | Bảng kết quả tóm tắt Pha 1 | 3,344 B | KEEP |
| `reports/FACE_DATASET_LICENSES.md` | Trạng thái license/access đã xác minh và chưa xác minh | 1,769 B | KEEP |
| `.bench_workspace/datasets/` | Archive nguồn, checksum, bản ghép và dữ liệu giải nén | 7,961,422,160 B; 7,770 files | DELETE |
| `.bench_workspace/datasets/300w/` | Bốn volume nguồn, ZIP ghép và 600 ảnh/annotation giải nén | 6,417,915,404 B; 1,205 files | DELETE |
| `.bench_workspace/datasets/wflw/` | Hai archive nguồn và 6,551 ảnh/annotation giải nén | 1,543,505,846 B; 6,564 files | DELETE |
| `.bench_workspace/prepared/` | Hai bản sao landmark-qa/v1 và danh sách DEV/TEST | 66,083,965 B; 6 files | DELETE |
| `.bench_workspace/results/face_phase1_20261013/` | Raw per-face, aggregate, run config, configs, commit và pip freeze | ~2.3 MiB | DELETE |
| `.bench_workspace/datasets/` | Thư mục gốc chứa toàn bộ dataset benchmark | 0 B khi tạo | DELETE |
| `experiments/benchmark_face/bench_phase2.py` | Tiêm lỗi có seed và chạy trực tiếp rules/scoring production | ~15 KiB | KEEP |
| `experiments/benchmark_face/bench_summarize_phase2.py` | Tổng hợp recall và AUROC/AUPRC theo họ/mức nhiễu | ~3 KiB | KEEP |
| `reports/FACE_PHASE2_SUMMARY.md` | Báo cáo tóm tắt Pha 2 và lệnh tái lập | ~7 KiB | KEEP |
| `.bench_workspace/results/face_phase2_20261013/` | Raw 39.000 scenario, 1.040.000 point, bootstrap, môi trường và checksum | ~112 MiB | DELETE |
| `.bench_workspace/datasets/300vw/300VW_Dataset_2015_12_14.zip` | Archive 300VW do người dùng cung cấp; chỉ đọc annotation R13 | 1,853,231,339 B | DELETE |
| `experiments/benchmark_face/bench_r13_300vw.py` | Đo false-positive R13 trên chuỗi GT 300VW liên tiếp | ~5 KiB | KEEP |
| `.bench_workspace/results/face_phase2_20261013/r13_300vw_frames.csv` | Raw R13 của 2.990 cặp frame 300VW | ~150 KiB | DELETE |
| `.bench_workspace/results/face_phase2_20261013/r13_300vw_summary.json` | FPR/CI và checksum nguồn 300VW | ~3 KiB | DELETE |
| `experiments/benchmark_face/bench_phase3_thresholds.py` | Quét DEV, khóa ngưỡng và đánh giá TEST bằng monkeypatch benchmark | ~14 KiB | KEEP |
| `experiments/benchmark_face/bench_summarize_phase3.py` | Xuất bảng so sánh và đồ thị PR/ROC DEV | ~3 KiB | KEEP |
| `experiments/benchmark_face/bench_phase3_config_code_audit.py` | Đo thực nghiệm lệch config–code R09/R10 | ~2 KiB | KEEP |
| `experiments/benchmark_face/proposed_vf50_thresholds.yaml` | Đề xuất benchmark-only, không tự áp production | ~1.5 KiB | KEEP |
| `reports/FACE_PHASE3_SUMMARY.md` | Báo cáo chọn ngưỡng DEV/TEST Pha 3 | ~6 KiB | KEEP |
| `reports/assets/face_phase3_dev_threshold_curves.png` | Đồ thị precision/recall/F1 theo grid DEV | ~250 KiB | KEEP |
| `reports/assets/face_phase3_dev_pr_curves.png` | Đường PR từ grid DEV | ~200 KiB | KEEP |
| `reports/assets/face_phase3_dev_roc_curves.png` | Đường ROC từ grid DEV | ~200 KiB | KEEP |
| `.bench_workspace/results/face_phase3_20261013/` | Raw grid, TEST comparison, audit và biểu đồ Pha 3 | ~1 MiB | DELETE |
| `experiments/benchmark_pose/` | Code chuẩn bị COCO, inference thật, QA K1/K2/K3, resource profiling và mapping | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_prepare_coco.py` | Chuẩn bị split COCO 500 ảnh trong workspace | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_run_inference.py` | Cache inference thật và audit mock fallback | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_pose_qa.py` | Error injection, DEV sweep, TEST metrics và bootstrap theo ảnh | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_resource_mode.py` | Đo latency và peak RSS riêng từng mode | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_pose_mapping.py` | Mapping COCO/CrowdPose sang VF17 không suy diễn điểm thiếu | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_test_pose_mapping.py` | Unit test parity trái/phải và chống swap giả | source nhỏ | KEEP |
| `reports/POSE_PHASE4_SUMMARY.md` | Báo cáo Pha 4, kết quả và giới hạn trung thực | source nhỏ | KEEP |
| `.bench_workspace/pose/` | COCO 500 ảnh, model bổ sung, prediction cache và raw Pha 4 | 322,541,936 B; 2,013 files | DELETE |
| `.bench_workspace/pose/crowdpose/` | 20.000 ảnh + annotation chính thức, subset/cache 500 ảnh và raw cross-dataset | 4,954,839,665 B; 21,512 files | DELETE |
| `experiments/benchmark_pose/bench_prepare_crowdpose.py` | Chia CrowdPose val thành DEV/TEST cố định | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_run_crowdpose_inference.py` | Inference ba model trên CrowdPose, cache riêng và audit mock | source nhỏ | KEEP |
| `experiments/benchmark_pose/bench_coco_official_ap.py` | Đo AP model Pose trên danh sách COCO-pose val chính thức | source nhỏ | KEEP |
| `.bench_workspace/pose/full_coco/` | COCO val2017, pose labels, raw AP và artifact tạm đã gom | 1,833,747,712 B; 63,959 files | DELETE |
| `.bench_workspace/pose/rtmpose_official/` | Dependency cô lập, source/config, checkpoint, raw AP và checksum RTMPose chính thức | 212,029,594 B; 10,938 files | DELETE |
| `reports/BENCHMARK_REPORT.md` | Báo cáo cuối Pha 5, liên kết toàn bộ số liệu với raw evidence | source nhỏ | KEEP |
| `experiments/bench_cleanup.py` | Cleanup chỉ đọc các dòng DELETE trong manifest; mặc định dry-run | source nhỏ | KEEP |
| `D:\Vin\VinPrj\landmark_qa_evidence\bench_evidence_2026-10-14.zip` | Gói raw JSON/CSV, cấu hình, split, pip freeze, commit hash và báo cáo đã duyệt; lưu ngoài repo | 6,114,199 B | KEEP |
| `D:\Vin\VinPrj\landmark_qa_evidence\SHA256SUMS.txt` | SHA-256 của gói evidence ngoài repo | 96 B | KEEP |
