from __future__ import annotations

import json
import sqlite3
import time
from fastapi import HTTPException
from app.core.database import db
from app.schemas.dataset import SequenceConfirmation

def _sequence_summary(conn: sqlite3.Connection, dataset_id: str) -> dict:
    sequences = conn.execute(
        "SELECT * FROM sequences WHERE dataset_id = ? ORDER BY name", (dataset_id,)
    ).fetchall()
    items = []
    for sequence in sequences:
        frames = conn.execute(
            """SELECT file_name, original_frame_index, timestamp_us, timestamp_source, frame_index_source
                 FROM images WHERE sequence_id = ?
                 ORDER BY original_frame_index, file_name""",
            (sequence["id"],),
        ).fetchall()
        indices = [row["original_frame_index"] for row in frames if row["original_frame_index"] is not None]
        deltas = [b - a for a, b in zip(indices, indices[1:])]
        items.append({
            "id": sequence["id"], "name": sequence["name"], "source": sequence["source"],
            "frame_order_source": sequence["frame_order_source"], "frame_count": len(frames),
            "first_frame": indices[0] if indices else None, "last_frame": indices[-1] if indices else None,
            "frame_deltas": deltas, "confirmed_at": sequence["confirmed_at"],
            "fps": sequence["fps"], "fps_source": sequence["fps_source"],
            "sampling_confirmed": bool(sequence["sampling_confirmed"]),
            "requires_manifest": bool(sequence["requires_manifest"]),
            "video_codec": sequence["video_codec"], "video_width": sequence["video_width"],
            "video_height": sequence["video_height"], "video_frame_count": sequence["video_frame_count"],
            "rotation_degrees": sequence["rotation_degrees"],
            "reason": json.loads(sequence["mapping_snapshot_json"] or "{}").get("reason"),
            "frames": [dict(row) for row in frames],
        })
    return {
        "dataset_id": dataset_id,
        "temporal_ready": bool(items) and all(item["confirmed_at"] and not item["requires_manifest"] for item in items),
        "sequences": items,
    }

def get_sequence_summary(project_id: str, dataset_id: str) -> dict:
    with db() as conn:
        dataset = conn.execute(
            "SELECT id FROM datasets WHERE id = ? AND project_id = ?", (dataset_id, project_id)
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        return _sequence_summary(conn, dataset_id)

def confirm_sequences(project_id: str, dataset_id: str, payload: SequenceConfirmation) -> dict:
    if payload.sampling_confirmed and payload.fps is None:
        raise HTTPException(400, "fps is required when sampling_confirmed is true")
    with db() as conn:
        dataset = conn.execute(
            "SELECT id FROM datasets WHERE id = ? AND project_id = ?", (dataset_id, project_id)
        ).fetchone()
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        sequences = conn.execute(
            "SELECT id, requires_manifest FROM sequences WHERE dataset_id = ?", (dataset_id,)
        ).fetchall()
        if not sequences:
            raise HTTPException(409, "Dataset has no sequence candidate")
        if any(row["requires_manifest"] for row in sequences):
            raise HTTPException(409, "Sequence mapping is ambiguous; upload a sequence manifest")

        now = int(time.time())
        for sequence in sequences:
            conn.execute(
                """UPDATE sequences
                      SET confirmed_at = ?, confirmed_project_id = ?, confirmation_source = ?,
                          fps = ?, fps_source = ?, sampling_confirmed = ?
                    WHERE id = ?""",
                (
                    now, project_id, payload.confirmation_source, payload.fps,
                    "user_input" if payload.fps is not None else None,
                    int(payload.sampling_confirmed), sequence["id"],
                ),
            )
            if payload.sampling_confirmed and payload.fps:
                frames = conn.execute(
                    """SELECT id FROM images WHERE sequence_id = ?
                         ORDER BY original_frame_index, file_name""", (sequence["id"],)
                ).fetchall()
                for sampled_index, frame in enumerate(frames):
                    conn.execute(
                        """UPDATE images SET timestamp_us = ?, timestamp_source = 'fps_derived'
                             WHERE id = ? AND timestamp_us IS NULL""",
                        (round(sampled_index * 1_000_000 / payload.fps), frame["id"]),
                    )
        return _sequence_summary(conn, dataset_id)
