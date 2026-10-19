from __future__ import annotations

from pathlib import Path
from fastapi import HTTPException
from app.core.config import IMAGE_EXTENSIONS
from app.core.database import db, resolve_data_path

def get_image(project_id: str, dataset_id: str, image_path: str) -> Path:
    with db() as conn:
        dataset = conn.execute(
            "SELECT storage_path FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()

    if not dataset:
        raise HTTPException(404, "Dataset not found")

    root = resolve_data_path(dataset["storage_path"])
    target = (root / image_path).resolve()

    if root not in target.parents or not target.is_file() or target.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(404, "Image not found")

    return target
