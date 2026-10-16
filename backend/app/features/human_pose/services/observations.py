from __future__ import annotations

from typing import Any

from app.shared.temporal.tracking import assign_one_to_one, bbox_iou

from .ensemble import DEFAULT_MODEL_WEIGHTS, compute_disagreement, compute_person_scale, compute_reference_point

PRIMARY_POSE_MODEL = "yolo26s-pose"


def build_pose_observations(predictions: dict[str, list[dict[str, Any]]], *, primary_model: str = PRIMARY_POSE_MODEL,
                            iou_threshold: float = 0.30) -> list[dict[str, Any]]:
    """Fuse model predictions around primary detections without consulting annotations."""
    primary = predictions.get(primary_model, [])
    if not primary:
        return []
    members: list[dict[str, dict[str, Any]]] = [{primary_model: detection} for detection in primary]
    for model_name, detections in predictions.items():
        if model_name == primary_model or not detections:
            continue
        costs = [[1.0 - bbox_iou(p.get("bbox"), d.get("bbox")) for d in detections] for p in primary]
        assignment = assign_one_to_one(costs, 1.0 - iou_threshold, ambiguity_margin=0.05)
        for match in assignment.matches:
            members[match.left_index][model_name] = detections[match.right_index]

    output: list[dict[str, Any]] = []
    for primary_index, support in enumerate(members):
        scale = compute_person_scale(primary[primary_index].get("bbox"))
        fused: dict[int, dict[str, Any]] = {}
        point_ids = set().union(*(detection.get("keypoints", {}).keys() for detection in support.values()))
        disagreements: list[float] = []
        for point_id in point_ids:
            values = []
            for model_name, detection in support.items():
                point = detection.get("keypoints", {}).get(point_id)
                if point and point.get("x") is not None and point.get("y") is not None:
                    values.append((float(point["x"]), float(point["y"]), float(point.get("conf", 1.0)),
                                   DEFAULT_MODEL_WEIGHTS.get(model_name, 1000.0)))
            if not values:
                continue
            fused_xy = compute_reference_point(values)
            disagreement = compute_disagreement(values, fused_xy, scale)
            disagreements.append(disagreement)
            fused[point_id] = {"id": point_id, "x": fused_xy[0], "y": fused_xy[1],
                               "conf": sum(value[2] for value in values) / len(values), "disagreement": disagreement}
        models = sorted(support)
        output.append({"bbox": primary[primary_index].get("bbox"), "score": primary[primary_index].get("score"),
                       "keypoints": fused, "observation_source": "ensemble" if len(models) > 1 else "primary",
                       "supporting_model_count": len(models), "supporting_models": models,
                       "model_matches": support,
                       "ensemble_disagreement": sum(disagreements) / len(disagreements) if disagreements else 0.0,
                       "primary_detection_index": primary_index})
    return output
