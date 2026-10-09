"""MediaPipe Face Landmarker inference engine for VinFast VF-50.

Uses Google MediaPipe Tasks API (Tasks Vision) with CPU-optimized delegate.
Infers 478 dense landmarks in ~15-20ms per frame and resamples them to
VinFast VF-50 landmarks.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from app.model.mapper import map_mediapipe_to_vf50
except ImportError:
    from .mapper import map_mediapipe_to_vf50



DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"


class MediaPipeFaceDetector:
    """Wrapper around MediaPipe Tasks Vision FaceLandmarker."""

    def __init__(self, model_path: Optional[str] = None):
        path = str(model_path or DEFAULT_MODEL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(f"MediaPipe face model not found at: {path}")

        base_options = python.BaseOptions(model_asset_path=path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(
        self,
        image_bgr: np.ndarray,
        target_keypoints: Optional[Dict[int, Dict[str, Any]] | np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect face landmarks on a BGR image with smart face focusing.
        
        Args:
            image_bgr: OpenCV BGR image (H, W, 3).
            target_keypoints: Optional annotator keypoints to focus on the exact annotated face
                              when multiple people appear in the image (e.g. driver vs passenger).
            
        Returns:
            Tuple of:
            - vf50: (50, 2) array of pixel coordinates for VinFast landmarks.
            - raw_478: (478, 2) array of pixel coordinates for MediaPipe landmarks.
            Or None if no face detected.
        """
        all_faces = self.detect_all_faces(image_bgr)
        if not all_faces:
            return None

        # If only 1 face detected, return it directly
        if len(all_faces) == 1:
            return all_faces[0][0], all_faces[0][1]

        # Multi-face scenario: focus on the annotated face if target_keypoints is provided
        if target_keypoints:
            target_pts = []
            if isinstance(target_keypoints, dict):
                for kp in target_keypoints.values():
                    if kp.get("state") != "outside" and kp.get("x") is not None and kp.get("y") is not None:
                        target_pts.append((float(kp["x"]), float(kp["y"])))
            elif isinstance(target_keypoints, np.ndarray):
                target_pts = [(float(pt[0]), float(pt[1])) for pt in target_keypoints]

            if target_pts:
                t_xs = [p[0] for p in target_pts]
                t_ys = [p[1] for p in target_pts]
                t_cx = sum(t_xs) / len(t_xs)
                t_cy = sum(t_ys) / len(t_ys)

                # Find face whose centroid is closest to annotator target centroid
                best_face = None
                min_dist = float("inf")
                for vf50, raw_478_px, (bx, by, bw, bh), area in all_faces:
                    f_cx = bx + bw / 2.0
                    f_cy = by + bh / 2.0
                    dist = ((f_cx - t_cx) ** 2 + (f_cy - t_cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_face = (vf50, raw_478_px)
                if best_face:
                    return best_face

        # Default fallback: select the largest face (the driver in in-cabin DMS context)
        all_faces_sorted = sorted(all_faces, key=lambda f: f[3], reverse=True)
        return all_faces_sorted[0][0], all_faces_sorted[0][1]

    def detect_all_faces(
        self,
        image_bgr: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], float]]:
        """Detect all faces in the image.
        
        Returns:
            List of tuples: (vf50, raw_478_px, (bbox_x, bbox_y, bbox_w, bbox_h), area)
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self.detector.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return []

        faces = []
        for face_landmarks in result.face_landmarks:
            raw_478_norm = np.array([[lm.x, lm.y] for lm in face_landmarks], dtype=np.float32)

            raw_478_px = raw_478_norm.copy()
            raw_478_px[:, 0] *= w
            raw_478_px[:, 1] *= h

            vf50 = map_mediapipe_to_vf50(raw_478_norm, image_width=float(w), image_height=float(h), normalized=True)

            min_x, max_x = float(np.min(raw_478_px[:, 0])), float(np.max(raw_478_px[:, 0]))
            min_y, max_y = float(np.min(raw_478_px[:, 1])), float(np.max(raw_478_px[:, 1]))
            bw = max_x - min_x
            bh = max_y - min_y
            area = bw * bh

            faces.append((vf50, raw_478_px, (min_x, min_y, bw, bh), area))

        return faces

    def detect_from_file(
        self,
        file_path: str,
        target_keypoints: Optional[Dict[int, Dict[str, Any]] | np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convenience method to detect landmarks directly from an image file."""
        img = cv2.imread(file_path)
        if img is None:
            return None
        return self.detect_landmarks(img, target_keypoints=target_keypoints)


# Global singleton instance for high performance
_GLOBAL_DETECTOR: Optional[MediaPipeFaceDetector] = None


def get_detector() -> MediaPipeFaceDetector:
    """Retrieve or initialize the global FaceDetector singleton."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = MediaPipeFaceDetector()
    return _GLOBAL_DETECTOR

