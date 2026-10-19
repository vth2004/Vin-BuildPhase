from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from app.core.database import db, resolve_data_path, store_data_path
from .formats import parse_annotations
from app.shared.video import VideoAnnotationError, VideoDecodeError, decode_video, prepare_video_annotation

logger = logging.getLogger(__name__)


def execute_video_import(job_id: str) -> None:
    with db() as conn:
        job = conn.execute(
            """SELECT j.*, d.project_id, d.name, d.schema_id, d.storage_path, d.annotation_path
                 FROM video_import_jobs j JOIN datasets d ON d.id = j.dataset_id
                WHERE j.id = ?""", (job_id,),
        ).fetchone()
        if not job or job["status"] not in {"pending", "processing"}:
            return
        conn.execute(
            "UPDATE video_import_jobs SET status = 'processing', progress = 5, started_at = ? WHERE id = ?",
            (int(time.time()), job_id),
        )
        conn.execute("UPDATE datasets SET status = 'processing' WHERE id = ?", (job["dataset_id"],))

    staging_root = resolve_data_path(job["staging_path"])
    staging_frames = staging_root / "frames"
    final_images = resolve_data_path(job["storage_path"])
    dataset_root = final_images.parent
    try:
        source_candidates = sorted(dataset_root.glob("source_video.*"))
        if len(source_candidates) != 1:
            raise VideoDecodeError("Stored source video is missing or ambiguous")
        shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=False)
        decoded = decode_video(
            source_candidates[0], staging_frames,
            max_frames=job["max_frames"], max_decoded_bytes=job["max_decoded_bytes"],
        )
        with db() as conn:
            conn.execute("UPDATE video_import_jobs SET progress = 75 WHERE id = ?", (job_id,))

        validated: list[dict] = []
        if job["annotation_path"]:
            raw_annotation = resolve_data_path(job["annotation_path"]).read_bytes()
            normalized = prepare_video_annotation(raw_annotation, len(decoded.frames))
            validated = parse_annotations(
                job["schema_id"], normalized, [frame.file_name for frame in decoded.frames], source="video",
            )

        shutil.rmtree(final_images, ignore_errors=True)
        staging_frames.replace(final_images)
        sequence_id = str(uuid.uuid4())
        now = int(time.time())
        mapping = [{
            "file_name": frame.file_name, "frame_index": frame.frame_index,
            "timestamp_us": frame.timestamp_us, "timestamp_source": frame.timestamp_source,
        } for frame in decoded.frames]
        snapshot = {
            "source": "video_decode", "mapping": mapping, "codec": decoded.codec,
            "width": decoded.width, "height": decoded.height, "frame_count": len(decoded.frames),
            "fps": decoded.fps, "rotation_degrees": decoded.rotation_degrees,
        }
        records_by_name = {record["file_name"]: record for record in validated}
        with db() as conn:
            old_images = [row[0] for row in conn.execute(
                "SELECT id FROM images WHERE dataset_id = ?", (job["dataset_id"],),
            )]
            if old_images:
                placeholders = ",".join("?" for _ in old_images)
                conn.execute(f"DELETE FROM annotations WHERE image_id IN ({placeholders})", old_images)
            conn.execute("DELETE FROM images WHERE dataset_id = ?", (job["dataset_id"],))
            conn.execute("DELETE FROM sequences WHERE dataset_id = ?", (job["dataset_id"],))
            conn.execute(
                """INSERT INTO sequences (
                     id, dataset_id, name, source, fps, fps_source, frame_order_source, confirmed_at,
                     confirmed_project_id, confirmation_source, mapping_snapshot_json, sampling_confirmed,
                     requires_manifest, import_status, media_path, video_codec, video_width, video_height,
                     video_frame_count, rotation_degrees, created_at
                   ) VALUES (?, ?, ?, 'video', ?, ?, 'video_decode', ?, ?, 'video_decode', ?, 0, 0,
                             'ready', ?, ?, ?, ?, ?, ?, ?)""",
                (sequence_id, job["dataset_id"], job["name"], decoded.fps,
                 "video_metadata" if decoded.fps else None, now, job["project_id"],
                 json.dumps(snapshot, ensure_ascii=False), store_data_path(source_candidates[0]), decoded.codec,
                 decoded.width, decoded.height, len(decoded.frames), decoded.rotation_degrees, now),
            )
            for frame in decoded.frames:
                image_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO images (
                         id, dataset_id, file_name, width, height, frame_index, sequence_id,
                         original_frame_index, timestamp_us, timestamp_source, frame_index_source
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'video_decode')""",
                    (image_id, job["dataset_id"], frame.file_name, frame.width, frame.height,
                     frame.frame_index, sequence_id, frame.frame_index, frame.timestamp_us,
                     frame.timestamp_source),
                )
                for annotation in records_by_name.get(frame.file_name, {}).get("annotations", []):
                    conn.execute(
                        """INSERT INTO annotations (
                             id, image_id, label, keypoints_json, source_instance_id, source_track_id
                           ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), image_id, annotation["label"], json.dumps(annotation["keypoints"]),
                         annotation.get("source_instance_id"), annotation.get("source_track_id")),
                    )
            conn.execute(
                "UPDATE datasets SET status = 'ready', image_count = ? WHERE id = ?",
                (len(decoded.frames), job["dataset_id"]),
            )
            conn.execute(
                """UPDATE video_import_jobs SET status = 'ready', progress = 100,
                          error_message = NULL, finished_at = ? WHERE id = ?""",
                (int(time.time()), job_id),
            )
        shutil.rmtree(staging_root, ignore_errors=True)
    except Exception as error:
        logger.exception("Video import job %s failed", job_id)
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(final_images, ignore_errors=True)
        with db() as conn:
            conn.execute(
                "UPDATE datasets SET status = 'failed', image_count = 0 WHERE id = ?", (job["dataset_id"],),
            )
            conn.execute(
                """UPDATE video_import_jobs SET status = 'failed', error_message = ?, finished_at = ?
                    WHERE id = ?""", (str(error)[:2000], int(time.time()), job_id),
            )


def recover_interrupted_video_imports() -> int:
    now = int(time.time())
    with db() as conn:
        rows = conn.execute(
            "SELECT id, dataset_id, staging_path FROM video_import_jobs WHERE status IN ('pending', 'processing')"
        ).fetchall()
        for row in rows:
            shutil.rmtree(resolve_data_path(row["staging_path"]), ignore_errors=True)
            conn.execute(
                """UPDATE video_import_jobs SET status = 'interrupted',
                          error_message = 'Backend restarted during video import', finished_at = ? WHERE id = ?""",
                (now, row["id"]),
            )
            conn.execute(
                "UPDATE datasets SET status = 'interrupted', image_count = 0 WHERE id = ?", (row["dataset_id"],),
            )
    return len(rows)
