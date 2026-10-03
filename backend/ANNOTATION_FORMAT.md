# Landmark QA annotation JSON v1

MVP nhận một ZIP ảnh và một JSON annotation. `file_name` phải khớp đường dẫn tương đối trong ZIP. Mỗi ảnh chứa một hoặc nhiều annotation; bộ VF hiện tại mặc định chỉ có một người lái/khuôn mặt.

```json
{
  "format": "landmark-qa/v1",
  "schema_id": "vf_humanpose17_v1",
  "images": [{
    "file_name": "frames/0001.jpg",
    "width": 960,
    "height": 540,
    "annotations": [{
      "label": "person",
      "keypoints": [
        { "id": 1, "x": 120.5, "y": 240.2, "state": "visible" },
        { "id": 16, "x": null, "y": null, "state": "outside" }
      ]
    }]
  }]
}
```

`state` chỉ nhận `visible`, `occluded`, `outside`. Hai trạng thái đầu phải có `x`, `y`; `outside` phải dùng `null`. JSON phải có đủ ID của schema: 1–17 cho HumanPose và 0–49 cho VF50.
