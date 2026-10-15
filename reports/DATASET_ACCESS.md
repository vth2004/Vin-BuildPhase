# Dataset Access for Face Benchmark

Không commit dữ liệu. Tất cả archive và nội dung giải nén phải nằm dưới `.bench_workspace/datasets/`.

## 300W — cần thao tác thủ công

Nguồn chính thức: <https://ibug.doc.ic.ac.uk/resources/300-W/>

Mỗi phần tải chuyển tới form yêu cầu first name, last name, email và affiliation. Sau khi tự điền form và tải đủ bốn phần, đặt nguyên tên file tại:

```text
.bench_workspace/datasets/300w/300w.zip.001
.bench_workspace/datasets/300w/300w.zip.002
.bench_workspace/datasets/300w/300w.zip.003
.bench_workspace/datasets/300w/300w.zip.004
```

Không tự đổi tên, nối hoặc giải nén trước khi script kiểm checksum và ghi manifest.

## WFLW — tải từ trang tác giả

Nguồn chính thức: <https://wywu.github.io/projects/LAB/WFLW.html>

Tải cả hai gói `WFLW Training and Testing images` và `WFLW Face Annotations`, rồi đặt vào:

```text
.bench_workspace/datasets/wflw/downloads/
```

Giữ nguyên tên file do nguồn cung cấp. Script chuẩn bị dữ liệu sẽ nhận diện archive sau khi kiểm tra cấu trúc, không dựa vào tên mirror.

## Dataset bổ sung

- COFW-68 annotation: <https://github.com/g0lnaz/cofw68-benchmark>
- MERL-RAV label: <https://github.com/abhi1kumar/MERL-RAV_dataset>
- AFLW request form phải do người dùng tự điền; không có quyền ảnh thì bỏ MERL-RAV.
- 300VW cũng dùng form iBUG; chỉ cần ở bước benchmark R13 sau này.

## Quy tắc xác nhận

Sau khi file được đặt đúng chỗ, agent chỉ đọc metadata/checksum trước, ghi `BENCH_MANIFEST.md`, rồi mới giải nén vào `.bench_workspace/datasets/`.
