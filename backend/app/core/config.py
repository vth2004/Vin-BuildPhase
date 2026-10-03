from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("LANDMARK_DATA_DIR", ROOT / "data")).resolve()
SCHEMAS_DIR = ROOT / "configs" / "schemas"
DB_PATH = DATA_DIR / "landmark_qa.sqlite3"

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_FILES_PER_ZIP = 10_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
