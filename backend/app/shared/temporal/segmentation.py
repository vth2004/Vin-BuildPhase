from __future__ import annotations

from typing import Any

import numpy as np


def luminance_histogram(pixels: Any, bins: int = 32) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float64).mean(axis=2)
    histogram, _ = np.histogram(array, bins=bins, range=(0, 256))
    total = histogram.sum()
    return histogram.astype(np.float64) / total if total else histogram.astype(np.float64)


def is_scene_cut(previous: np.ndarray | None, current: np.ndarray, threshold: float = 0.65) -> bool:
    """Histogram-distance scene-cut heuristic; threshold is deliberately uncalibrated."""
    if previous is None or previous.shape != current.shape:
        return False
    return float(np.abs(previous - current).sum() / 2.0) >= threshold
