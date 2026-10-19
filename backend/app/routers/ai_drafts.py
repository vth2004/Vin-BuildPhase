from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response

from app.core.database import db
from app.core.security import project_guard

router = APIRouter(prefix="/api/projects/{project_id}/runs/{run_id}/ai-drafts", tags=["ai-drafts"])

POSE_KEYPOINTS = [
    "nose", "r_eye", "l_eye", "r_ear", "l_ear", "r_shoulder", "l_shoulder",
    "r_elbow", "l_elbow", "r_wrist", "l_wrist", "r_hip", "l_hip", "r_knee",
    "l_knee", "r_ankle", "l_ankle",
]
POSE_SKELETON = [[1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7],
                 [6, 12], [7, 13], [12, 13], [6, 8], [8, 10], [7, 9], [9, 11],
                 [12, 14], [14, 16], [13, 15], [15, 17]]
FACE_KEYPOINTS = [str(index) for index in range(50)]
FACE_SKELETON = [[index + 1, index + 2] for start, end in (
    (0, 4), (5, 9), (10, 13), (14, 21), (22, 29), (30, 41), (42, 49)
) for index in range(start, end)] + [[22, 15], [30, 23], [42, 31], [50, 43]]
FACE_GROUPS = [("longmaytrai", range(0, 5)), ("longmayphai", range(5, 10)),
               ("songmui", range(10, 14)), ("mattrai", range(14, 22)),
               ("matphai", range(22, 30)), ("moingoai", range(30, 42)),
               ("moitrong", range(42, 50))]


def _rows(project_id: str, run_id: str) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
    with db() as conn:
        run = conn.execute(
            """SELECT r.*, d.name AS dataset_name, d.schema_id
                 FROM runs r JOIN datasets d ON d.id = r.dataset_id
                WHERE r.id = ? AND r.project_id = ?""", (run_id, project_id),
        ).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        if run["input_mode"] != "ai_qc":
            raise HTTPException(409, "AI drafts are available only for runs without source annotations")
        rows = conn.execute(
            """SELECT d.*, i.file_name, i.width, i.height, i.original_frame_index,
                      i.timestamp_us, i.sequence_id
                 FROM ai_draft_results d JOIN images i ON i.id = d.image_id
                WHERE d.run_id = ?
                ORDER BY COALESCE(i.sequence_id, ''), COALESCE(i.original_frame_index, 2147483647),
                         i.file_name, d.instance_index""", (run_id,),
        ).fetchall()
    return run, rows


def _item(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "image_id": row["image_id"], "file_name": row["file_name"],
        "frame_index": row["original_frame_index"], "timestamp_us": row["timestamp_us"],
        "sequence_id": row["sequence_id"], "instance_index": row["instance_index"],
        "track_id": row["internal_track_id"], "keypoints": json.loads(row["keypoints_json"]),
        "provenance": json.loads(row["provenance_json"]), "qc": json.loads(row["qc_json"]),
        "visibility_confirmed": bool(row["visibility_confirmed"]),
    }


@router.get("")
def list_ai_drafts(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> dict:
    run, rows = _rows(project_id, run_id)
    items = [_item(row) for row in rows]
    return {
        "run_id": run_id, "schema_id": run["schema_id"], "label": "AI draft",
        "visibility_confirmed": False, "count": len(items), "items": items,
    }


@router.get("/export")
def export_ai_drafts(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> JSONResponse:
    run, rows = _rows(project_id, run_id)
    images: dict[str, dict] = {}
    for row in rows:
        image = images.setdefault(row["image_id"], {
            "file_name": row["file_name"], "width": row["width"], "height": row["height"],
            "sequence_id": row["sequence_id"], "frame_index": row["original_frame_index"],
            "timestamp_us": row["timestamp_us"], "annotations": [],
        })
        provenance = json.loads(row["provenance_json"])
        image["annotations"].append({
            "label": "AI draft", "instance_id": f"ai-{row['instance_index']}",
            "track_id": row["internal_track_id"], "visibility_confirmed": False,
            "keypoints": json.loads(row["keypoints_json"]), "provenance": provenance,
            "qc": json.loads(row["qc_json"]),
        })
    payload = {
        "format": "landmark-qa/v1", "schema_id": run["schema_id"],
        "dataset_name": run["dataset_name"], "label": "AI draft",
        "visibility_confirmed": False, "source_run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(), "images": list(images.values()),
    }
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="ai_draft_{run_id}.json"',
    })


@router.get("/export/coco")
def export_ai_drafts_coco(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> JSONResponse:
    """Export AI drafts as COCO Keypoints, which CVAT can import into an image task."""
    run, rows = _rows(project_id, run_id)
    is_pose = run["schema_id"] == "vf_humanpose17_v1"
    names = POSE_KEYPOINTS if is_pose else FACE_KEYPOINTS
    skeleton = POSE_SKELETON if is_pose else FACE_SKELETON
    image_ids: dict[str, int] = {}
    images: list[dict] = []
    annotations: list[dict] = []
    for row in rows:
        if row["image_id"] not in image_ids:
            image_ids[row["image_id"]] = len(image_ids) + 1
            images.append({
                "id": image_ids[row["image_id"]], "file_name": row["file_name"],
                "width": row["width"], "height": row["height"],
                "frame_index": row["original_frame_index"],
            })
        by_id = {int(point["id"]): point for point in json.loads(row["keypoints_json"])}
        ordered_ids = range(1, 18) if is_pose else range(50)
        flat: list[float | int] = []
        valid: list[tuple[float, float]] = []
        for point_id in ordered_ids:
            point = by_id.get(point_id, {})
            x, y = point.get("x"), point.get("y")
            present = x is not None and y is not None and point.get("state") != "outside"
            flat.extend([float(x) if present else 0, float(y) if present else 0, 2 if present else 0])
            if present:
                valid.append((float(x), float(y)))
        if valid:
            min_x, max_x = min(x for x, _ in valid), max(x for x, _ in valid)
            min_y, max_y = min(y for _, y in valid), max(y for _, y in valid)
            bbox = [min_x, min_y, max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)]
        else:
            bbox = [0, 0, 1, 1]
        annotations.append({
            "id": len(annotations) + 1, "image_id": image_ids[row["image_id"]],
            "category_id": 1, "keypoints": flat, "num_keypoints": len(valid),
            "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
            "track_id": row["internal_track_id"],
        })
    payload = {
        "info": {"description": f"AI draft from {run['dataset_name']}",
                 "source_run_id": run_id, "visibility_confirmed": False,
                 "exported_at": datetime.now(timezone.utc).isoformat()},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "person" if is_pose else "face",
                        "supercategory": "person", "keypoints": names, "skeleton": skeleton}],
    }
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="ai_draft_cvat_{run_id}.json"',
    })


@router.get("/export/cvat")
def export_ai_drafts_cvat(
    project_id: str, run_id: str, _: sqlite3.Row = Depends(project_guard),
) -> Response:
    """Export frame-indexed skeleton tracks for direct import into a CVAT video task."""
    run, rows = _rows(project_id, run_id)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = run["dataset_name"]
    ET.SubElement(task, "source").text = f"Landmark QA AI draft run {run_id}"
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        identity = row["internal_track_id"] or f"{row['image_id']}:{row['instance_index']}"
        grouped.setdefault((row["sequence_id"] or "default", identity), []).append(row)
    track_id = 0
    is_pose = run["schema_id"] == "vf_humanpose17_v1"
    groups = [("person", range(1, 18))] if is_pose else FACE_GROUPS
    names = {index + 1: name for index, name in enumerate(POSE_KEYPOINTS)} if is_pose else {index: str(index) for index in range(50)}
    for (_, _identity), track_rows in grouped.items():
        for label, point_ids in groups:
            track = ET.SubElement(root, "track", id=str(track_id), label=label, source="auto")
            track_id += 1
            for row in sorted(track_rows, key=lambda item: item["original_frame_index"] if item["original_frame_index"] is not None else 2147483647):
                if row["original_frame_index"] is None:
                    continue
                points = {int(point["id"]): point for point in json.loads(row["keypoints_json"])}
                available = [point_id for point_id in point_ids if points.get(point_id, {}).get("x") is not None and points.get(point_id, {}).get("y") is not None]
                if not available:
                    continue
                skeleton = ET.SubElement(track, "skeleton", frame=str(row["original_frame_index"]), keyframe="1")
                for point_id in point_ids:
                    point = points.get(point_id, {})
                    present = point.get("x") is not None and point.get("y") is not None and point.get("state") != "outside"
                    ET.SubElement(skeleton, "points", label=names[point_id],
                                  points=f"{float(point.get('x') or 0):.3f},{float(point.get('y') or 0):.3f}",
                                  outside="0" if present else "1", occluded="0", keyframe="1")
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return Response(content=content, media_type="application/xml", headers={
        "Content-Disposition": f'attachment; filename="ai_draft_cvat_{run_id}.xml"',
    })
