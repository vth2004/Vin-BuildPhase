import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import json
from pathlib import Path
from typing import Any
import numpy as np

SANDBOX_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SANDBOX_DIR.parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = SANDBOX_DIR / "models"
LOGS_DIR = SANDBOX_DIR / "logs"

# Ánh xạ chỉ số COCO-17 (0..16) sang chuẩn VinFast vf_humanpose17_v1 (1..17)
# COCO: 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear, 5:l_shoulder, 6:r_shoulder,
#       7:l_elbow, 8:r_elbow, 9:l_wrist, 10:r_wrist, 11:l_hip, 12:r_hip,
#       13:l_knee, 14:r_knee, 15:l_ankle, 16:r_ankle
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


class BasePoseModel:
    def __init__(self, name: str, weight_path: str | Path | None = None) -> None:
        self.name = name
        self.weight_path = Path(weight_path) if weight_path else None

    def get_model_size_mb(self) -> float:
        if self.weight_path and self.weight_path.is_file():
            return round(self.weight_path.stat().st_size / (1024 * 1024), 2)
        return 0.0

    def predict(self, image_path: str | Path) -> tuple[list[dict[str, Any]], float]:
        """Trả về: (danh_sách_người, latency_ms)"""
        raise NotImplementedError


class YoloPoseModel(BasePoseModel):
    def __init__(self, name: str, weight_path: str | Path) -> None:
        super().__init__(name, weight_path)
        from ultralytics import YOLO

        print(f"[*] Đang tải model YOLO: {name} từ {weight_path}...")
        self.model = YOLO(str(weight_path))

    def predict(self, image_path: str | Path) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        results = self.model.predict(
            source=str(image_path),
            verbose=False,
            device="cpu",
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return [], latency_ms

        res = results[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return [], latency_ms

        persons: list[dict[str, Any]] = []
        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        boxes_conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        kpts_xy = res.keypoints.xy.cpu().numpy()
        kpts_conf = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones((len(kpts_xy), 17))
        )

        for i in range(len(kpts_xy)):
            box = boxes_xyxy[i].tolist() if i < len(boxes_xyxy) else None
            score = float(boxes_conf[i]) if i < len(boxes_conf) else 1.0
            xy = kpts_xy[i]
            conf = kpts_conf[i]

            kpts_dict: dict[int, dict[str, Any]] = {}
            for coco_idx in range(min(17, len(xy))):
                vf_id, k_name = COCO_TO_VF17[coco_idx]
                kpts_dict[vf_id] = {
                    "id": vf_id,
                    "name": k_name,
                    "x": float(xy[coco_idx][0]),
                    "y": float(xy[coco_idx][1]),
                    "conf": float(conf[coco_idx]),
                }

            persons.append({
                "bbox": box,
                "score": score,
                "keypoints": kpts_dict,
            })

        return persons, latency_ms


class RTMPoseModel(BasePoseModel):
    def __init__(self, mode: str = "balanced") -> None:
        super().__init__(name="rtmpose-m", weight_path=None)
        import cv2
        self.cv2 = cv2

        print(f"[*] Khởi tạo RTMPose (mode={mode}, engine=onnxruntime CPU) qua rtmlib...")
        try:
            from rtmlib import Body
            # mode='balanced' tự động sử dụng RTMDet-m và RTMPose-m (COCO 17 keypoints)
            self.body = Body(
                mode=mode,
                to_openpose=False,
                backend="onnxruntime",
                device="cpu",
            )
        except Exception as e:
            print(f"[LỖI] Khởi tạo rtmlib Body thất bại: {e}")
            raise e

    def predict(self, image_path: str | Path) -> tuple[list[dict[str, Any]], float]:
        img = self.cv2.imread(str(image_path))
        if img is None:
            return [], 0.0

        t0 = time.perf_counter()
        keypoints, scores = self.body(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if keypoints is None or len(keypoints) == 0:
            return [], latency_ms

        persons: list[dict[str, Any]] = []
        for i in range(len(keypoints)):
            kpt = keypoints[i]  # shape (17, 2)
            sc = scores[i]      # shape (17,)

            # Tính ước lượng bbox từ keypoints
            valid_x = [pt[0] for pt in kpt if pt[0] > 0]
            valid_y = [pt[1] for pt in kpt if pt[1] > 0]
            if valid_x and valid_y:
                bbox = [min(valid_x), min(valid_y), max(valid_x), max(valid_y)]
            else:
                bbox = None

            kpts_dict: dict[int, dict[str, Any]] = {}
            for coco_idx in range(min(17, len(kpt))):
                vf_id, k_name = COCO_TO_VF17[coco_idx]
                kpts_dict[vf_id] = {
                    "id": vf_id,
                    "name": k_name,
                    "x": float(kpt[coco_idx][0]),
                    "y": float(kpt[coco_idx][1]),
                    "conf": float(sc[coco_idx]) if sc is not None else 1.0,
                }

            persons.append({
                "bbox": bbox,
                "score": float(np.mean(sc)) if sc is not None else 1.0,
                "keypoints": kpts_dict,
            })

        return persons, latency_ms


def setup_all_models() -> dict[str, BasePoseModel]:
    """
    Chuẩn bị 4 model:
    M0: yolo26s-pose (có sẵn ở backend/yolo26s-pose.pt)
    M1: yolov8n-pose (có sẵn ở backend/yolov8n-pose.pt)
    M2: yolov8m-pose (tải qua ultralytics về models/)
    M3: rtmpose-m (qua rtmlib/onnxruntime)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    prov_file = LOGS_DIR / "model_provenance.log"

    provenance_lines = ["=== THÔNG TIN NGUỒN GỐC & LICENSE CỦA CÁC MÔ HÌNH ==="]
    models: dict[str, BasePoseModel] = {}

    # M0: yolo26s-pose
    m0_path = BACKEND_DIR / "yolo26s-pose.pt"
    if m0_path.is_file():
        models["yolo26s-pose"] = YoloPoseModel("yolo26s-pose", m0_path)
        provenance_lines.append(f"M0 [yolo26s-pose]: {m0_path} (Size: {m0_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy M0 tại {m0_path}")

    # M1: yolov8n-pose
    m1_path = BACKEND_DIR / "yolov8n-pose.pt"
    if m1_path.is_file():
        models["yolov8n-pose"] = YoloPoseModel("yolov8n-pose", m1_path)
        provenance_lines.append(f"M1 [yolov8n-pose]: {m1_path} (Size: {m1_path.stat().st_size / 1e6:.1f} MB, License: AGPL-3.0)")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy M1 tại {m1_path}")

    # M2: yolov8m-pose
    m2_path = MODELS_DIR / "yolov8m-pose.pt"
    if not m2_path.is_file():
        print("[*] Đang tải yolov8m-pose.pt từ Ultralytics...")
        from ultralytics import YOLO
        temp_yolo = YOLO("yolov8m-pose.pt")
        # Copy/move về MODELS_DIR
        import shutil
        if Path("yolov8m-pose.pt").is_file():
            shutil.move("yolov8m-pose.pt", str(m2_path))
    
    if m2_path.is_file():
        models["yolov8m-pose"] = YoloPoseModel("yolov8m-pose", m2_path)
        provenance_lines.append(f"M2 [yolov8m-pose]: {m2_path} (Size: {m2_path.stat().st_size / 1e6:.1f} MB, License: AGPL-3.0)")

    # M3: rtmpose-m
    try:
        import rtmlib
        import onnxruntime
        import importlib.metadata
        try:
            rtm_ver = importlib.metadata.version("rtmlib")
        except Exception:
            rtm_ver = "0.0.16"
        provenance_lines.append(f"M3 [rtmpose-m]: rtmlib {rtm_ver} (License: Apache 2.0), ONNXRuntime {onnxruntime.__version__}")
        provenance_lines.append("    Weights: OpenMMLab MMPose RTMPose-m (ONNX) + RTMDet-m (ONNX)")
        m3_model = RTMPoseModel(mode="balanced")
        models["rtmpose-m"] = m3_model
        print("[OK] Đã tải và khởi tạo thành công M3: RTMPose-m")
    except Exception as e:
        provenance_lines.append(f"M3 [rtmpose-m]: LỖI KHỞI TẠO THƯ VIỆN ({e}).")
        print(f"[CẢNH BÁO M3] Lỗi khởi tạo RTMPose: {e}")

    with open(prov_file, "w", encoding="utf-8") as f:
        f.write("\n".join(provenance_lines) + "\n")
    print(f"[OK] Đã ghi nhật ký nguồn gốc model vào: {prov_file}")

    return models
