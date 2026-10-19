from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.core.config import MAX_UPLOAD_BYTES
from app.core.database import db, store_data_path
from app.features.face_landmark_vf50 import SCHEMA_ID as FACE_SCHEMA_ID
from app.features.face_landmark_vf50.services.ingestion import parse_face_annotations
from app.features.human_pose.services.dataset import load_schema, validate_annotations
from app.shared.datasets import safe_extract_dataset, validate_exif_orientation
from app.shared.temporal import infer_sequences, parse_sequence_manifest
from .sequences import _sequence_summary

def upload_dataset(project_id: str, file: UploadFile, schema_id: UploadFile, annotation_file: UploadFile | None, sequence_manifest_file: UploadFile | None, data_dir: Path) -> dict:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file")

    schema = load_schema(schema_id)
    dataset_id = str(uuid.uuid4())
    location = data_dir / "projects" / project_id / "datasets" / dataset_id / "images"
    location.mkdir(parents=True, exist_ok=False)

    try:
        images, embedded_annotation, embedded_manifest = safe_extract_dataset(file, location)
        if not images:
            raise HTTPException(400, "ZIP contains no supported images")

        raw_annotation = None
        if annotation_file and annotation_file.filename:
            raw_annotation = annotation_file.file.read(MAX_UPLOAD_BYTES + 1)
        elif embedded_annotation:
            raw_annotation = embedded_annotation

        if raw_annotation and len(raw_annotation) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Annotation JSON exceeds 500 MB limit")

        if schema_id == FACE_SCHEMA_ID:
            validated = parse_face_annotations(raw_annotation, set(images)) if raw_annotation else []
            records_by_name = {record["file_name"]: record for record in validated}
            validated = [
                records_by_name.get(name, {
                    "file_name": name,
                    "width": None,
                    "height": None,
                    "annotations": [],
                })
                for name in images
            ]
        else:
            if not raw_annotation:
                validated = [
                    {"file_name": name, "width": None, "height": None, "annotations": []}
                    for name in images
                ]
            elif raw_annotation.lstrip().startswith(b"<"):
                raise HTTPException(
                    400,
                    "File annotation XML này thuộc định dạng Face Landmark (CVAT XML). "
                    "Phân hệ Human Pose chỉ hỗ trợ file JSON COCO 17 điểm. Vui lòng chuyển sang phân hệ Face Landmark.",
                )
            else:
                try:
                    annotation_payload = json.loads(raw_annotation)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise HTTPException(400, "Annotation file is not valid JSON") from error
                validated = validate_annotations(annotation_payload, schema, set(images))

        validate_exif_orientation(location, validated)

        manifest_raw = embedded_manifest
        if sequence_manifest_file and sequence_manifest_file.filename:
            manifest_raw = sequence_manifest_file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(manifest_raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Sequence manifest exceeds 500 MB limit")
        inference = (
            parse_sequence_manifest(manifest_raw, set(images))
            if manifest_raw else infer_sequences(validated)
        )

        annotation_path = None
        if raw_annotation:
            annotation_suffix = ".xml" if raw_annotation.lstrip().startswith(b"<") else ".json"
            annotation_path = location.parent / f"source_annotation{annotation_suffix}"
            annotation_path.write_bytes(raw_annotation)

        with db() as conn:
            conn.execute(
                "INSERT INTO datasets (id, project_id, name, status, image_count, storage_path, schema_id, annotation_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    project_id,
                    Path(file.filename).stem,
                    "ready",
                    len(images),
                    store_data_path(location),
                    schema_id,
                    store_data_path(annotation_path) if annotation_path else None,
                    int(time.time()),
                ),
            )
            mapping_by_name = {mapping.file_name: mapping for mapping in inference.mappings}
            sequence_ids: dict[str, str] = {}
            sequence_names = sorted({mapping.sequence_name for mapping in inference.mappings}) or ["unresolved"]
            trusted = inference.source in {"manifest", "annotation"} and not inference.requires_manifest
            now = int(time.time())
            for sequence_name in sequence_names:
                sequence_id = str(uuid.uuid4())
                sequence_ids[sequence_name] = sequence_id
                sequence_mapping = [
                    {"file_name": m.file_name, "frame_index": m.original_frame_index, "timestamp_us": m.timestamp_us}
                    for m in inference.mappings if m.sequence_name == sequence_name
                ]
                snapshot = {"source": inference.source, "reason": inference.reason, "mapping": sequence_mapping}
                conn.execute(
                    """INSERT INTO sequences (
                         id, dataset_id, name, source, frame_order_source, confirmed_at,
                         confirmed_project_id, confirmation_source, mapping_snapshot_json,
                         requires_manifest, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sequence_id, dataset_id, sequence_name, "zip", inference.source,
                        now if trusted else None, project_id if trusted else None,
                        inference.source if trusted else None, json.dumps(snapshot, ensure_ascii=False),
                        int(inference.requires_manifest), now,
                    ),
                )
            for frame_index, image in enumerate(validated):
                image_id = str(uuid.uuid4())
                mapping = mapping_by_name.get(image["file_name"])
                conn.execute(
                    """INSERT INTO images (
                         id, dataset_id, file_name, width, height, frame_index, sequence_id,
                         original_frame_index, timestamp_us, timestamp_source, frame_index_source
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        image_id, dataset_id, image["file_name"], image["width"], image["height"], frame_index,
                        sequence_ids.get(mapping.sequence_name) if mapping else sequence_ids.get("unresolved"),
                        mapping.original_frame_index if mapping else frame_index,
                        mapping.timestamp_us if mapping else None,
                        inference.source if mapping and mapping.timestamp_us is not None else "unavailable",
                        inference.source if mapping else "import_order",
                    ),
                )
                for annotation in image["annotations"]:
                    conn.execute(
                        """INSERT INTO annotations (
                             id, image_id, label, keypoints_json, source_instance_id, source_track_id
                           ) VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()), image_id, annotation["label"], json.dumps(annotation["keypoints"]),
                            annotation.get("source_instance_id"), annotation.get("source_track_id"),
                        ),
                    )

        with db() as conn:
            summary = _sequence_summary(conn, dataset_id)
        return {
            "id": dataset_id,
            "name": Path(file.filename).stem,
            "status": "ready",
            "image_count": len(images),
            "schema_id": schema_id,
            "sequence_summary": summary,
        }
    except Exception:
        shutil.rmtree(location.parent, ignore_errors=True)
        raise
