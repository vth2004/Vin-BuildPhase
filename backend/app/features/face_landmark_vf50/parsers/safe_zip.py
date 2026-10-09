from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath
from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_FILES_PER_ZIP = 10_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ANNOTATION_EXTENSIONS = {".xml", ".json"}


def safe_extract_images(upload: UploadFile, destination: Path) -> tuple[list[str], bytes | None]:
    """
    Giải nén an toàn file ZIP chứa ảnh và tự động nhận diện file annotation (CVAT XML / JSON) bên trong.
    Chống tấn công Zip Slip (Path Traversal) và Zip Bomb.
    
    Returns:
        tuple[list[str], bytes | None]: (danh_sách_tên_ảnh, nội_dung_file_nhãn_nếu_có)
    """
    temp_zip = destination.parent / "temp_upload.zip"
    total_size = 0

    try:
        upload.file.seek(0)
    except Exception:
        pass

    with temp_zip.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                temp_zip.unlink(missing_ok=True)
                raise HTTPException(413, "File ZIP vượt quá giới hạn 500 MB")
            out.write(chunk)

    extracted_images: list[str] = []
    annotation_bytes: bytes | None = None
    best_ann_score = -1

    try:
        with zipfile.ZipFile(temp_zip) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES_PER_ZIP:
                raise HTTPException(400, "File ZIP chứa quá nhiều tệp (> 10.000 tệp)")

            for info in infos:
                path = PurePosixPath(info.filename)
                # Bỏ qua thư mục hoặc đường dẫn chứa .. (chống Path Traversal)
                if info.is_dir() or path.is_absolute() or ".." in path.parts:
                    continue
                # Bỏ qua file rác hệ điều hành macOS / Windows
                if (
                    path.name.startswith(".")
                    or "__MACOSX" in path.parts
                    or path.name.startswith("._")
                    or "Thumbs.db" in path.name
                ):
                    continue

                suffix = path.suffix.lower()

                # Tự động bắt file annotation trong ZIP (annotations.xml, default.xml, *.json, person_keypoints_default.json)
                if suffix in ANNOTATION_EXTENSIONS:
                    score = 0
                    fn_lower = path.name.lower()
                    full_lower = str(path).lower()

                    # Bỏ qua các file cấu hình / metadata không phải nhãn
                    if fn_lower in {"package.json", "tsconfig.json", "manifest.json", "project.json"}:
                        continue

                    if "keypoint" in fn_lower or "keypoints" in fn_lower:
                        score += 30
                    if "coco" in fn_lower or "coco" in full_lower:
                        score += 25
                    if "annot" in fn_lower or "annot" in full_lower:
                        score += 20
                    if "person" in fn_lower:
                        score += 15
                    if "default" in fn_lower:
                        score += 10
                    if suffix == ".xml":
                        score += 8
                    if suffix == ".json":
                        score += 5

                    if score > best_ann_score or annotation_bytes is None:
                        try:
                            with archive.open(info) as ann_src:
                                content = ann_src.read()
                                if len(content) > 5:
                                    annotation_bytes = content
                                    best_ann_score = score
                                    # Lưu dự phòng vào thư mục session
                                    ann_target = destination.parent / path.name
                                    with ann_target.open("wb") as ann_dst:
                                        ann_dst.write(annotation_bytes)
                        except Exception:
                            pass
                    continue

                if suffix not in IMAGE_EXTENSIONS:
                    continue

                # Lưu ảnh trực tiếp vào destination theo tên file phẳng
                target = destination / path.name
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_images.append(path.name)

        return sorted(list(set(extracted_images))), annotation_bytes
    except zipfile.BadZipFile as err:
        raise HTTPException(400, "Tệp ZIP không hợp lệ hoặc bị hỏng") from err
    finally:
        temp_zip.unlink(missing_ok=True)

