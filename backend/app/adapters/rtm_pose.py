from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from app.adapters.base import BasePosePredictor
from app.adapters.yolo_pose import COCO_TO_VF17


class RtmPosePredictor(BasePosePredictor):
    """Adapter tích hợp RTMPose-m (ONNXRuntime CPU) qua thư viện rtmlib."""

    def __init__(self, mode: str = "balanced", device: str = "cpu") -> None:
        super().__init__(name="rtmpose-m")
        import cv2
        self.cv2 = cv2
        self.mode = mode
        self.device = device

        from rtmlib import Body
        self.body = Body(
            mode=mode,
            to_openpose=False,
            backend="onnxruntime",
            device=device,
        )

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """
        Dự đoán tư thế người bằng RTMPose.
        
        Trả về:
            list[dict[str, Any]]: Danh sách người kèm keypoints theo chuẩn vf_humanpose17_v1.
        """
        img = self.cv2.imread(str(image_path))
        if img is None:
            return []

        try:
            # Lấy bboxes từ detector (YOLOX) và đưa vào pose model
            if hasattr(self.body, "det_model") and hasattr(self.body, "pose_model") and not getattr(self.body, "one_stage", False):
                bboxes = self.body.det_model(img)
                if bboxes is None or len(bboxes) == 0:
                    return []
                keypoints, scores = self.body.pose_model(img, bboxes=bboxes)
            else:
                keypoints, scores = self.body(img)
                bboxes = None
        except Exception:
            keypoints, scores = self.body(img)
            bboxes = None

        if keypoints is None or len(keypoints) == 0:
            return []

        persons: list[dict[str, Any]] = []
        for i in range(len(keypoints)):
            kpt = keypoints[i]  # shape (17, 2)
            sc = scores[i] if scores is not None else None  # shape (17,)

            # Xác định bbox
            box = None
            if bboxes is not None and i < len(bboxes):
                raw_b = bboxes[i]
                if len(raw_b) >= 4:
                    box = (float(raw_b[0]), float(raw_b[1]), float(raw_b[2]), float(raw_b[3]))
            
            if box is None:
                valid_x = [float(pt[0]) for pt in kpt if pt[0] > 0]
                valid_y = [float(pt[1]) for pt in kpt if pt[1] > 0]
                if valid_x and valid_y:
                    min_x, max_x = min(valid_x), max(valid_x)
                    min_y, max_y = min(valid_y), max(valid_y)
                    w, h = max(10.0, max_x - min_x), max(10.0, max_y - min_y)
                    box = (
                        max(0.0, min_x - 0.1 * w),
                        max(0.0, min_y - 0.1 * h),
                        max_x + 0.1 * w,
                        max_y + 0.1 * h,
                    )

            keypoints_by_id: dict[int, dict[str, Any]] = {}
            for coco_idx in range(min(17, len(kpt))):
                vf_id, k_name = COCO_TO_VF17[coco_idx]
                x_val = float(kpt[coco_idx][0])
                y_val = float(kpt[coco_idx][1])
                c_val = float(sc[coco_idx]) if sc is not None else 1.0

                keypoints_by_id[vf_id] = {
                    "id": vf_id,
                    "name": k_name,
                    "x": x_val,
                    "y": y_val,
                    "conf": c_val,
                }

            person_score = float(np.mean(sc)) if sc is not None and len(sc) > 0 else 1.0
            persons.append({
                "bbox": box,
                "score": person_score,
                "keypoints": keypoints_by_id,
            })

        return persons
