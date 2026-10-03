from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

# Ánh xạ chỉ số chuẩn COCO-17 (0..16) sang chuẩn VinFast vf_humanpose17_v1 (1..17)
# Lưu ý: Chuẩn COCO mặc định left trước right; VinFast vf_humanpose17_v1 quy định right trước left (chẵn: right, lẻ: left)
COCO_TO_VF17: dict[int, tuple[int, str]] = {
    0: (1, "nose"),
    1: (3, "l_eye"),
    2: (2, "r_eye"),
    3: (5, "l_ear"),
    4: (4, "r_ear"),
    5: (7, "l_shoulder"),
    6: (6, "r_shoulder"),
    7: (9, "l_elbow"),
    8: (8, "r_elbow"),
    9: (11, "l_wrist"),
    10: (10, "r_wrist"),
    11: (13, "l_hip"),
    12: (12, "r_hip"),
    13: (15, "l_knee"),
    14: (14, "r_knee"),
    15: (17, "l_ankle"),
    16: (16, "r_ankle"),
}


class YoloPoseAdapter:
    def __init__(self, model_name: str = "yolo26s-pose.pt") -> None:
        from ultralytics import YOLO

        backend_dir = Path(__file__).resolve().parents[2]
        candidate_path = backend_dir / model_name
        target = str(candidate_path) if candidate_path.is_file() else model_name

        self.model = YOLO(target)

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """
        Chạy YOLO-pose trên ảnh và trích xuất danh sách người kèm keypoints theo schema vf_humanpose17_v1.
        Trả về:
            list[dict]:
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": float,
                    "keypoints": dict[int, dict]  # truy cập theo keypoint id (1..17): {id, name, x, y, conf}
                }
        """
        results = self.model.predict(
            source=str(image_path),
            verbose=False,
            device="cpu",
        )
        if not results:
            return []

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []

        persons: list[dict[str, Any]] = []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        kpts_xy = result.keypoints.xy.cpu().numpy()  # shape (N, 17, 2)
        kpts_conf = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones((len(kpts_xy), 17))
        )

        for i in range(len(kpts_xy)):
            box = tuple(boxes_xyxy[i].tolist()) if i < len(boxes_xyxy) else None
            score = float(boxes_conf[i]) if i < len(boxes_conf) else 1.0

            xy = kpts_xy[i]
            conf = kpts_conf[i]

            keypoints_by_id: dict[int, dict[str, Any]] = {}
            for coco_idx in range(min(17, len(xy))):
                vf_id, name = COCO_TO_VF17[coco_idx]
                x_val = float(xy[coco_idx][0])
                y_val = float(xy[coco_idx][1])
                c_val = float(conf[coco_idx])
                keypoints_by_id[vf_id] = {
                    "id": vf_id,
                    "name": name,
                    "x": x_val,
                    "y": y_val,
                    "conf": c_val,
                }

            persons.append({
                "bbox": box,
                "score": score,
                "keypoints": keypoints_by_id,
            })

        return persons
