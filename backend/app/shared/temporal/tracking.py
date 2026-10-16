from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from scipy.optimize import linear_sum_assignment


def bbox_iou(a: Any, b: Any) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def item_center(item: dict[str, Any]) -> tuple[float, float] | None:
    bbox = item.get("bbox")
    if bbox and len(bbox) >= 4:
        return ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)
    points = item.get("keypoints", {})
    values = points.values() if isinstance(points, dict) else points
    xy = [(float(p["x"]), float(p["y"])) for p in values if p.get("x") is not None and p.get("y") is not None]
    return (sum(x for x, _ in xy) / len(xy), sum(y for _, y in xy) / len(xy)) if xy else None


def item_scale(item: dict[str, Any]) -> float:
    bbox = item.get("bbox")
    if bbox and len(bbox) >= 4:
        return max(10.0, math.sqrt(max(1.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))))
    return 100.0


def _keypoint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ak, bk = a.get("keypoints", {}), b.get("keypoints", {})
    common = set(ak) & set(bk) if isinstance(ak, dict) and isinstance(bk, dict) else set()
    distances = [math.dist((ak[k]["x"], ak[k]["y"]), (bk[k]["x"], bk[k]["y"])) for k in common
                 if ak[k].get("x") is not None and bk[k].get("x") is not None]
    return sum(distances) / len(distances) / max(item_scale(a), item_scale(b)) if distances else 1.0


def continuity_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    ca, cb = item_center(a), item_center(b)
    if ca is None or cb is None:
        return float("inf")
    center = math.dist(ca, cb) / max(item_scale(a), item_scale(b))
    return 0.45 * (1.0 - bbox_iou(a.get("bbox"), b.get("bbox"))) + 0.35 * center + 0.20 * _keypoint_distance(a, b)


@dataclass
class Match:
    left_index: int
    right_index: int
    cost: float


@dataclass
class Assignment:
    matches: list[Match] = field(default_factory=list)
    unmatched_left: list[int] = field(default_factory=list)
    unmatched_right: list[int] = field(default_factory=list)
    uncertain_left: list[int] = field(default_factory=list)


def assign_one_to_one(costs: list[list[float]], cutoffs: list[float] | float, ambiguity_margin: float = 0.08) -> Assignment:
    if not costs:
        return Assignment()
    width = len(costs[0]) if costs[0] else 0
    if not width:
        return Assignment(unmatched_left=list(range(len(costs))))
    rows, cols = linear_sum_assignment(costs)
    result = Assignment()
    used_left: set[int] = set()
    used_right: set[int] = set()
    for left, right in zip(rows.tolist(), cols.tolist()):
        cutoff = cutoffs[right] if isinstance(cutoffs, list) else cutoffs
        row_candidates = sorted(value for value in costs[left] if math.isfinite(value))
        col_candidates = sorted(costs[i][right] for i in range(len(costs)) if math.isfinite(costs[i][right]))
        ambiguous = (
            len(row_candidates) > 1 and row_candidates[1] - row_candidates[0] < ambiguity_margin
        ) or (
            len(col_candidates) > 1 and col_candidates[1] - col_candidates[0] < ambiguity_margin
        )
        if costs[left][right] <= cutoff and not ambiguous:
            result.matches.append(Match(left, right, float(costs[left][right])))
            used_left.add(left)
            used_right.add(right)
        elif ambiguous:
            result.uncertain_left.append(left)
    result.unmatched_left = [i for i in range(len(costs)) if i not in used_left]
    result.unmatched_right = [i for i in range(width) if i not in used_right]
    return result


class DetectionTracker:
    """Conservative consecutive-frame tracker; no re-identification or secondary bridge."""

    def __init__(self, max_cost: float = 0.95, ambiguity_margin: float = 0.08) -> None:
        self.max_cost = max_cost
        self.ambiguity_margin = ambiguity_margin
        self.previous: list[dict[str, Any]] = []
        self.segment_id = 0

    def reset(self) -> None:
        self.previous = []
        self.segment_id += 1

    def update(self, detections: list[dict[str, Any]], *, discontinuity: bool = False) -> list[dict[str, Any]]:
        if discontinuity:
            self.reset()
        if not detections:
            if self.previous:
                self.reset()
            return []
        if not self.previous:
            current = [dict(item, internal_track_id=str(uuid.uuid4()), segment_id=self.segment_id) for item in detections]
            self.previous = current
            return current
        costs = [[continuity_cost(old, new) for new in detections] for old in self.previous]
        assignment = assign_one_to_one(costs, self.max_cost, self.ambiguity_margin)
        match_ratio = len(assignment.matches) / max(len(self.previous), len(detections), 1)
        if match_ratio < 0.25:
            self.reset()
            current = [dict(item, internal_track_id=str(uuid.uuid4()), segment_id=self.segment_id,
                            tracking_uncertain=True) for item in detections]
            self.previous = current
            return current
        by_new = {match.right_index: self.previous[match.left_index]["internal_track_id"] for match in assignment.matches}
        current = [dict(item, internal_track_id=by_new.get(i, str(uuid.uuid4())), segment_id=self.segment_id,
                        tracking_uncertain=i in assignment.unmatched_right and bool(self.previous))
                   for i, item in enumerate(detections)]
        self.previous = current
        return current
