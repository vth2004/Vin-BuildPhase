# Face Benchmark — Phase 1 Summary

> NME chỉ được tính trên face mà MediaPipe phát hiện và ghép thành công. Detection recall dùng toàn bộ GT face, vì vậy phải đọc hai chỉ số cùng nhau.

Raw per-face: `.bench_workspace/results/face_phase1_20261013/face_baseline_raw.csv`  
Raw aggregate: `.bench_workspace/results/face_phase1_20261013/face_baseline_summary.json`

## Detection coverage tổng thể

| Dataset | Images | GT faces | Images có detection | Image coverage | Matched faces | Face coverage | Miss rate | Latency median | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300W-private | 600 | 600 | 454 | 75.67% | 384 | 64.00% | 36.00% | 16.71 ms | 43.64 ms |
| WFLW | 6551 | 10000 | 3593 | 54.85% | 4019 | 40.19% | 59.81% | 12.68 ms | 31.30 ms |

## MediaPipe coverage và NME theo lát

| Dataset | Slice | GT face | Matched | Detection recall | NME mean | NME median | NME P90 |
|---|---|---:|---:|---:|---:|---:|---:|
| 300W-private | all | 600 | 384 | 64.00% | 8.54% | 7.05% | 11.96% |
| 300W-private | dev | 300 | 180 | 60.00% | 8.80% | 7.21% | 12.05% |
| 300W-private | indoor | 300 | 219 | 73.00% | 8.51% | 7.00% | 12.01% |
| 300W-private | outdoor | 300 | 165 | 55.00% | 8.58% | 7.10% | 11.48% |
| 300W-private | test | 300 | 204 | 68.00% | 8.32% | 6.89% | 11.63% |
| WFLW | all | 10000 | 4019 | 40.19% | 9.14% | 7.04% | 13.75% |
| WFLW | blur | 2183 | 208 | 9.53% | 14.50% | 9.35% | 21.24% |
| WFLW | dev | 4992 | 2052 | 41.11% | 9.23% | 7.12% | 13.49% |
| WFLW | expression | 496 | 181 | 36.49% | 10.67% | 8.08% | 17.77% |
| WFLW | illumination | 2225 | 786 | 35.33% | 9.28% | 7.20% | 13.65% |
| WFLW | makeup | 285 | 101 | 35.44% | 8.17% | 6.56% | 12.31% |
| WFLW | occlusion | 2077 | 611 | 29.42% | 10.40% | 8.11% | 15.17% |
| WFLW | pose | 588 | 147 | 25.00% | 21.85% | 18.52% | 26.77% |
| WFLW | test | 5008 | 1967 | 39.28% | 9.05% | 6.99% | 14.37% |

## Mapping noise floor

Đơn vị dưới đây là phần trăm IOD; direct mapping có algorithmic floor bằng 0.

| Dataset | Skeleton | Median | P95 | Max |
|---|---|---:|---:|---:|
| 300W-private | longmaytrai | 0.000% | 0.000% | 0.000% |
| 300W-private | longmayphai | 0.000% | 0.000% | 0.000% |
| 300W-private | songmui | 0.000% | 0.000% | 0.000% |
| 300W-private | mattrai | 0.939% | 1.212% | 1.511% |
| 300W-private | matphai | 0.932% | 1.193% | 1.440% |
| 300W-private | moingoai | 0.000% | 0.000% | 0.000% |
| 300W-private | moitrong | 0.000% | 0.000% | 0.000% |
| WFLW | longmaytrai | 0.000% | 0.000% | 0.000% |
| WFLW | longmayphai | 0.000% | 0.000% | 0.000% |
| WFLW | songmui | 0.000% | 0.000% | 0.000% |
| WFLW | mattrai | 0.000% | 0.000% | 0.000% |
| WFLW | matphai | 0.000% | 0.000% | 0.000% |
| WFLW | moingoai | 0.000% | 0.000% | 0.000% |
| WFLW | moitrong | 0.000% | 0.000% | 0.000% |

## Giới hạn

- WFLW attributes là nhãn cấp face, không phải visibility từng điểm.
- NME có selection bias do chỉ tính được khi detector tìm và ghép đúng face.
- WFLW direct 98→50 không nội suy; noise floor 0 chỉ nói về thuật toán mapping, không nói annotation protocol giữa WFLW và VF50 hoàn toàn tương đương.
- 300W-68 cần nội suy mí mắt 6→8; đây là nguồn mapping noise đo được.
