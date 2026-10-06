from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from app.adapters.base import BasePosePredictor
from app.adapters.yolo_pose import COCO_TO_VF17


class Yolo8mPosePredictor(BasePosePredictor):
    """Adapter cho model tùy chọn YOLOv8m-pose (K=3)."""

    def __init__(self, model_name: str = "yolov8m-pose.pt") -> None:
        super().__init__(name="yolov8m-pose")
        from ultralytics import YOLO

        backend_dir = Path(__file__).resolve().parents[2]
        candidate_path = backend_dir / model_name
        target = str(candidate_path) if candidate_path.is_file() else model_name

        self.model = YOLO(target)

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
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
        kpts_xy = result.keypoints.xy.cpu().numpy()
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
                keypoints_by_id[vf_id] = {
                    "id": vf_id,
                    "name": name,
                    "x": float(xy[coco_idx][0]),
                    "y": float(xy[coco_idx][1]),
                    "conf": float(conf[coco_idx]),
                }

            persons.append({
                "bbox": box,
                "score": score,
                "keypoints": keypoints_by_id,
            })

        return persons
