from __future__ import annotations

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, UploadFile

from app.core.config import IMAGE_EXTENSIONS, MAX_FILES_PER_ZIP, MAX_UPLOAD_BYTES

ANNOTATION_EXTENSIONS = {".json", ".xml"}


def _annotation_score(path: PurePosixPath) -> int:
    name = path.name.lower()
    full = path.as_posix().lower()
    if name in {"package.json", "tsconfig.json", "manifest.json", "project.json"}:
        return -1
    score = 8 if path.suffix.lower() == ".xml" else 5
    for token, weight in (("keypoint", 30), ("coco", 25), ("annot", 20), ("person", 15), ("default", 10)):
        if token in full:
            score += weight
    return score


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_extract_images(upload: UploadFile, destination: Path) -> tuple[list[str], bytes | None]:
    """Extract images while preserving relative paths and return the best embedded annotation."""
    destination.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / "source.zip"
    compressed_size = 0
    try:
        upload.file.seek(0)
    except Exception:
        pass
    with temp.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            compressed_size += len(chunk)
            if compressed_size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "ZIP exceeds 500 MB limit")
            output.write(chunk)

    annotation: bytes | None = None
    annotation_score = -1
    try:
        with zipfile.ZipFile(temp) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES_PER_ZIP:
                raise HTTPException(400, f"ZIP has too many files (>{MAX_FILES_PER_ZIP})")
            declared_size = sum(max(0, info.file_size) for info in infos)
            if declared_size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Extracted ZIP content exceeds 500 MB limit")

            seen: set[str] = set()
            images: list[str] = []
            extracted_size = 0
            for info in infos:
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if info.is_dir():
                    continue
                has_windows_drive = bool(path.parts and len(path.parts[0]) >= 2 and path.parts[0][1] == ":")
                if path.is_absolute() or has_windows_drive or not path.parts or ".." in path.parts or _is_symlink(info):
                    raise HTTPException(400, f"Unsafe ZIP member: {info.filename}")
                normalized = path.as_posix()
                duplicate_key = normalized.casefold()
                if duplicate_key in seen:
                    raise HTTPException(400, f"Duplicate ZIP path: {normalized}")
                seen.add(duplicate_key)
                if any(part == "__MACOSX" for part in path.parts) or path.name.startswith((".", "._")):
                    continue

                suffix = path.suffix.lower()
                if suffix in ANNOTATION_EXTENSIONS:
                    score = _annotation_score(path)
                    if score > annotation_score:
                        with archive.open(info) as source:
                            candidate = source.read(MAX_UPLOAD_BYTES + 1)
                        if len(candidate) > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Embedded annotation exceeds 500 MB limit")
                        annotation = candidate
                        annotation_score = score
                    continue
                if suffix not in IMAGE_EXTENSIONS:
                    continue

                target = destination.joinpath(*path.parts).resolve()
                root = destination.resolve()
                if root != target.parent and root not in target.parents:
                    raise HTTPException(400, f"Unsafe ZIP member: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        extracted_size += len(chunk)
                        if extracted_size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Extracted images exceed 500 MB limit")
                        output.write(chunk)
                images.append(normalized)
            return sorted(images), annotation
    except zipfile.BadZipFile as error:
        raise HTTPException(400, "Invalid ZIP file") from error
    finally:
        temp.unlink(missing_ok=True)
