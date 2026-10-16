# Temporal QA/QC quickstart

Temporal QA là tính năng thử nghiệm, tùy chọn cho Pose 17 điểm và Face VF50. Khi tắt, run tiếp tục dùng toàn bộ luồng QA/QC hiện tại và cho kết quả baseline như trước.

## Chạy bản demo ZIP có annotation

1. Mở project bằng Recovery key, vào trang Pose hoặc Face.
2. Upload ZIP ảnh và annotation như hiện tại. Có thể thêm `sequence_manifest.json` riêng; manifest nhúng trong ZIP vẫn được hỗ trợ.
3. Chọn dataset. Kiểm tra thumbnail đầu/giữa/cuối, frame delta và nguồn thứ tự.
4. Nếu mapping chưa được tin cậy, nhập FPS của **chuỗi đã lấy mẫu** nếu biết rồi bấm **Xác nhận mapping**. Không nhập FPS gốc của video khi ảnh đã được lấy mẫu ở nhịp khác.
5. Bật **Temporal QA**, chọn cửa sổ lẻ 5–29 frame và tạo run.
6. Khi run hoàn tất, panel Temporal hiển thị coverage, warning, unavailable reason, overlay Human/Model/Temporal, quỹ đạo và timeline.
7. Chọn điểm cần dùng, tải annotation đã sửa và tải audit JSON/CSV để đối chiếu quyết định.

Trong panel review, chọn **Giữ điểm Human** hoặc **Dùng điểm Temporal** cho từng keypoint. Có thể đổi quyết định hoặc bấm **Bỏ lựa chọn**; trạng thái và tọa độ snapshot vẫn còn sau khi tải lại trang. Bộ lọc Review cho phép xem các điểm chưa review hoặc đã chọn theo từng nguồn.

Với Pose, nút export annotation hiện có áp dụng quyết định Temporal vào đúng annotation và keypoint. Tọa độ thay đổi nhưng trạng thái visibility được giữ nguyên; điểm `outside` bị từ chối. Hai nút **Audit JSON** và **Audit CSV** tải raw prediction, Human, Temporal, tọa độ được chọn và provenance của review.

Với Face, **cleaned JSON** và **CVAT XML** cũng áp dụng decision vào đúng annotation/khuôn mặt/keypoint, giữ nguyên `visible`, `occluded`, `outside`. CSV/checklist tiếp tục là báo cáo QA; audit Temporal dùng hai nút JSON/CSV trong panel.

## Manifest

Manifest phải phủ mọi ảnh trong ZIP. `frame_index` là chỉ số frame gốc; `timestamp_us` là tùy chọn.

```json
{
  "images": [
    {"file_name": "seq-a/frame_0000.jpg", "sequence": "seq-a", "frame_index": 0, "timestamp_us": 0},
    {"file_name": "seq-a/frame_0003.jpg", "sequence": "seq-a", "frame_index": 3, "timestamp_us": 100000}
  ]
}
```

Khoảng trống được tính theo `frame_index` gốc hoặc timestamp tin cậy, không theo vị trí phần tử trong ZIP.

## Giới hạn hiện tại

- Ngưỡng scene cut, gap và temporal quality chưa được benchmark/hiệu chỉnh; không diễn giải quality như xác suất.
- Pose ngắt track khi mất primary detection để tránh ID switch; coverage có thể giảm ở che khuất nặng.
- Face không có confidence landmark đáng tin cậy từ MediaPipe; quality dựa trên matching, geometry và residual.
- Dataset có mapping mơ hồ phải import lại cùng manifest. Upload video, chấp nhận điểm và luồng chưa nhãn nằm ở các phase sau.
