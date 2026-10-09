import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TARGET_DIR = Path(__file__).resolve().parents[1] / "models"
TARGET_FILE = TARGET_DIR / "face_landmarker.task"


def download_face_landmarker():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.is_file() and TARGET_FILE.stat().st_size > 1000:
        print(f"Model already exists at: {TARGET_FILE}")
        return

    print(f"Downloading MediaPipe Face Landmarker from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, TARGET_FILE)
    print(f"Downloaded successfully to {TARGET_FILE} ({TARGET_FILE.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    download_face_landmarker()
