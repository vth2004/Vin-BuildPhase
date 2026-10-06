from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePosePredictor(ABC):
    """Giao diện chuẩn cho các mô hình Pose Estimation trong Landmark QA."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        """
        Dự đoán keypoints trên ảnh.
        
        Trả về:
            list[dict[str, Any]]: Danh sách người tìm thấy:
                [
                    {
                        "bbox": (x1, y1, x2, y2) | list | None,
                        "score": float,
                        "keypoints": {
                            vf_id (int): {
                                "id": int,
                                "name": str,
                                "x": float,
                                "y": float,
                                "conf": float
                            }
                        }
                    }
                ]
        """
        pass
