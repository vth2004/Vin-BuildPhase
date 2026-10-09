"""Download the pinned MediaPipe Face Landmarker model and verify its digest."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> Path:
    target = Path(os.getenv("FACE_LANDMARKER_MODEL_PATH", DEFAULT_TARGET)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == MODEL_SHA256:
        print(f"Model is ready: {target}")
        return target

    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".download") as temp:
        temp_path = Path(temp.name)
    try:
        urllib.request.urlretrieve(MODEL_URL, temp_path)
        actual = sha256(temp_path)
        if actual != MODEL_SHA256:
            raise RuntimeError(f"Model checksum mismatch: expected {MODEL_SHA256}, got {actual}")
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Downloaded verified model to: {target}")
    return target


if __name__ == "__main__":
    download()
