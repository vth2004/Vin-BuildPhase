from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from fastapi import HTTPException, UploadFile

from app.core.config import MAX_UPLOAD_BYTES, MAX_FILES_PER_ZIP, IMAGE_EXTENSIONS
from app.shared.datasets.schemas import load_schema
from ..adapters.yolo_pose import COCO_TO_VF17


FACE_CATEGORY_HINTS = {
    "longmaytrai", "longmayphai", "songmui", "mattrai", "matphai", "moingoai", "moitrong",
    "lông mày trái", "lông mày phải", "sống mũi", "mắt trái", "mắt phải", "môi ngoài", "môi trong",
    "face", "face_landmark", "face_landmarks", "vf50", "landmark50", "face_vf50",
}


def validate_annotations(payload: object, schema: dict, image_names: set[str]) -> list[dict]:
    if not isinstance(payload, dict):
        raise HTTPException(400, "Annotation file is not valid JSON")

    # Hỗ trợ 1: Định dạng COCO Keypoints 1.0 xuất trực tiếp từ CVAT
    if "annotations" in payload and "categories" in payload and isinstance(payload.get("images"), list):
        categories = payload.get("categories", [])
        cat_map = {
            c.get("id"): str(c.get("name", "")).strip().lower()
            for c in categories
            if isinstance(c, dict) and c.get("id") is not None
        }

        # 1. Phát hiện nếu danh mục chứa các bộ phận của Face Landmark VF50
        face_detected = [name for name in cat_map.values() if name in FACE_CATEGORY_HINTS]
        if face_detected:
            raise HTTPException(
                400,
                f"File annotation/ZIP này chứa nhãn Face Landmark ({', '.join(face_detected)}), "
                f"không phải Human Pose (COCO 17 điểm). Vui lòng tải lên tại phân hệ Face Landmark.",
            )

        # 2. Kiểm tra nếu có category nào chỉ định 50 keypoints của Face
        for c in categories:
            if isinstance(c, dict):
                kpts_meta = c.get("keypoints", [])
                if len(kpts_meta) == 50:
                    raise HTTPException(
                        400,
                        "File annotation/ZIP này chứa cấu hình 50 điểm của Face Landmark VF50, "
                        "không phải 17 điểm của Human Pose. Vui lòng tải lên tại phân hệ Face Landmark.",
                    )

        coco_images = {img["id"]: img for img in payload["images"] if isinstance(img, dict) and "id" in img}
        annots_by_img: dict[int, list[dict]] = {}
        for annot in payload.get("annotations", []):
            img_id = annot.get("image_id")
            if img_id is not None:
                annots_by_img.setdefault(img_id, []).append(annot)

        output: list[dict] = []
        for coco_position, (img_id, img_info) in enumerate(coco_images.items()):
            raw_fn = img_info.get("file_name", "").replace("\\", "/")
            matched_name = None
            if raw_fn in image_names:
                matched_name = raw_fn
            elif Path(raw_fn).name in image_names:
                matched_name = Path(raw_fn).name
            else:
                for iname in image_names:
                    if iname.endswith(Path(raw_fn).name):
                        matched_name = iname
                        break

            if not matched_name:
                continue

            img_annots = annots_by_img.get(img_id, [])
            checked_annots: list[dict] = []
            for annot in img_annots:
                kpts = annot.get("keypoints", [])

                # Kiểm tra nếu annotation chứa 150 phần tử (50 điểm * 3) -> Face VF50
                if len(kpts) == 150:
                    raise HTTPException(
                        400,
                        "Dữ liệu keypoints chứa 50 điểm (định dạng Face Landmark VF50). "
                        "Phân hệ Human Pose chỉ hỗ trợ 17 điểm COCO toàn thân.",
                    )

                points: list[dict] = []
                for coco_idx in range(min(17, len(kpts) // 3)):
                    x = kpts[coco_idx * 3]
                    y = kpts[coco_idx * 3 + 1]
                    v = kpts[coco_idx * 3 + 2]
                    vf_id, vf_name = COCO_TO_VF17[coco_idx]

                    if v == 0:
                        state = "outside"
                        px, py = None, None
                    elif v == 1:
                        state = "occluded"
                        px, py = float(x), float(y)
                    else:
                        state = "visible"
                        px, py = float(x), float(y)

                    points.append({"id": vf_id, "name": vf_name, "x": px, "y": py, "state": state})

                points.sort(key=lambda p: p["id"])
                attributes = annot.get("attributes") if isinstance(annot.get("attributes"), dict) else {}
                source_track_id = next((annot.get(key, attributes.get(key)) for key in ("track_id", "group_id", "instance_id") if annot.get(key, attributes.get(key)) is not None), None)
                checked_annots.append({
                    "label": schema.get("annotation_label", "person"),
                    "keypoints": points,
                    "source_instance_id": str(annot.get("id")) if annot.get("id") is not None else None,
                    "source_track_id": str(source_track_id) if source_track_id is not None else None,
                })

            if checked_annots:
                output.append({
                    "file_name": matched_name,
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                    "annotations": checked_annots,
                    "coco_array_position": coco_position,
                    "coco_image_id": img_id,
                    "original_frame_index": img_info.get("frame_index", img_info.get("frame")),
                    "sequence_name": img_info.get("sequence", img_info.get("sequence_name")),
                    "timestamp_us": img_info.get("timestamp_us"),
                })
        if not output:
            raise HTTPException(400, "No matching images found between CVAT COCO JSON and ZIP")
        return output

    # Hỗ trợ 2: Định dạng landmark-qa/v1
    if payload.get("format") != "landmark-qa/v1":
        raise HTTPException(400, "Annotation JSON must be either COCO Keypoints 1.0 (from CVAT) or landmark-qa/v1")
    if payload.get("schema_id") == "vf_face_landmark50_v1":
        raise HTTPException(
            400,
            "File annotation này thuộc schema vf_face_landmark50_v1 (Face Landmark). "
            "Vui lòng tải lên tại phân hệ Face Landmark.",
        )
    if payload.get("schema_id") != schema["id"]:
        raise HTTPException(400, "schema_id in annotation does not match selected schema")
    records = payload.get("images")
    if not isinstance(records, list) or not records:
        raise HTTPException(400, "annotation JSON requires a non-empty images array")
    expected_ids = {point["id"] for point in schema["keypoints"]}
    output = []
    seen_names: set[str] = set()
    for image in records:
        if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
            raise HTTPException(400, "each annotation image requires file_name")
        file_name = image["file_name"].replace("\\", "/")
        if file_name not in image_names:
            raise HTTPException(400, f"annotation refers to image missing from ZIP: {file_name}")
        if file_name in seen_names:
            raise HTTPException(400, f"duplicate annotation image: {file_name}")
        seen_names.add(file_name)
        annotations = image.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise HTTPException(400, f"{file_name} has no annotations")
        checked: list[dict] = []
        for annotation in annotations:
            points = annotation.get("keypoints") if isinstance(annotation, dict) else None
            if not isinstance(points, list):
                raise HTTPException(400, f"{file_name} annotation requires keypoints")
            ids = {point.get("id") for point in points if isinstance(point, dict)}
            if ids != expected_ids or len(points) != len(expected_ids):
                raise HTTPException(400, f"{file_name} must contain exactly the schema point IDs")
            for point in points:
                state = point.get("state")
                x, y = point.get("x"), point.get("y")
                if state not in {"visible", "occluded", "outside"}:
                    raise HTTPException(400, f"{file_name} has an invalid point state")
                if state == "outside" and (x is not None or y is not None):
                    raise HTTPException(400, f"{file_name}: outside points must have null coordinates")
                if state != "outside" and (not isinstance(x, (int, float)) or not isinstance(y, (int, float))):
                    raise HTTPException(400, f"{file_name}: visible/occluded points require numeric coordinates")
            source_instance_id = annotation.get("instance_id", annotation.get("id"))
            checked.append({
                "label": annotation.get("label", schema["annotation_label"]),
                "keypoints": points,
                "source_instance_id": str(source_instance_id) if source_instance_id is not None else None,
                "source_track_id": str(annotation["track_id"]) if annotation.get("track_id") is not None else None,
            })
        output.append({
            "file_name": file_name,
            "width": image.get("width"),
            "height": image.get("height"),
            "annotations": checked,
            "original_frame_index": image.get("frame_index", image.get("original_frame_index")),
            "sequence_name": image.get("sequence", image.get("sequence_name")),
            "timestamp_us": image.get("timestamp_us"),
        })
    return output


def safe_extract_images(upload: UploadFile, destination: Path) -> tuple[list[str], bytes | None]:
    temp = destination.parent / "source.zip"
    total = 0
    with temp.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "ZIP exceeds 500 MB limit")
            out.write(chunk)
    embedded_annotation: bytes | None = None
    try:
        with zipfile.ZipFile(temp) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES_PER_ZIP:
                raise HTTPException(400, "ZIP has too many files")
            images: list[str] = []
            for info in infos:
                path = PurePosixPath(info.filename)
                if info.is_dir() or path.is_absolute() or ".." in path.parts:
                    continue
                if path.suffix.lower() == ".json" and ("annotation" in path.as_posix().lower() or "person_keypoints" in path.as_posix().lower()):
                    with archive.open(info) as f:
                        embedded_annotation = f.read()
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = destination / Path(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                images.append(path.as_posix())
            return images, embedded_annotation
    except zipfile.BadZipFile as error:
        raise HTTPException(400, "Invalid ZIP file") from error
    finally:
        temp.unlink(missing_ok=True)
