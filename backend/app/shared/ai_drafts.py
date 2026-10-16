from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Callable


def apply_temporal_prediction_qc(run_id: str, db_factory: Callable) -> None:
    """Flag unstable AI predictions without comparing them to human annotation."""
    with db_factory() as conn:
        rows = conn.execute(
            """SELECT d.id, d.internal_track_id, d.keypoints_json, d.qc_json, d.provenance_json,
                      i.original_frame_index, i.width, i.height
                 FROM ai_draft_results d JOIN images i ON i.id = d.image_id
                WHERE d.run_id = ? AND d.internal_track_id IS NOT NULL
                ORDER BY d.internal_track_id, i.original_frame_index""", (run_id,),
        ).fetchall()
    tracks: dict[str, list] = defaultdict(list)
    for row in rows:
        tracks[row["internal_track_id"]].append(row)
    updates = []
    for track_rows in tracks.values():
        for previous, current, following in zip(track_rows, track_rows[1:], track_rows[2:]):
            indices = [item["original_frame_index"] for item in (previous, current, following)]
            if any(index is None for index in indices) or indices[1] - indices[0] != 1 or indices[2] - indices[1] != 1:
                continue
            mappings = [
                {int(point["id"]): point for point in json.loads(item["keypoints_json"])}
                for item in (previous, current, following)
            ]
            scale = math.hypot(current["width"] or 1, current["height"] or 1)
            unstable = []
            for point_id in set(mappings[0]) & set(mappings[1]) & set(mappings[2]):
                points = [mapping[point_id] for mapping in mappings]
                if any(point.get("x") is None or point.get("y") is None for point in points):
                    continue
                acceleration = math.hypot(
                    float(points[2]["x"]) - 2 * float(points[1]["x"]) + float(points[0]["x"]),
                    float(points[2]["y"]) - 2 * float(points[1]["y"]) + float(points[0]["y"]),
                ) / max(scale, 1.0)
                if acceleration > 0.08:
                    unstable.append({"keypoint_id": point_id, "normalized_acceleration": round(acceleration, 6)})
            qc = json.loads(current["qc_json"] or "{}")
            flags = list(qc.get("flags", []))
            if unstable and "temporal_prediction_instability" not in flags:
                flags.append("temporal_prediction_instability")
            qc.update({"flags": flags, "temporal_prediction_instability": unstable})
            provenance = json.loads(current["provenance_json"] or "{}")
            provenance["temporal_qc_version"] = "ai-draft-second-derivative-v1"
            updates.append((json.dumps(qc), json.dumps(provenance), current["id"]))
    if updates:
        with db_factory() as conn:
            conn.executemany(
                "UPDATE ai_draft_results SET qc_json = ?, provenance_json = ? WHERE id = ?", updates,
            )
