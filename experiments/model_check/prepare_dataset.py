from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import random
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

# Duong dan sandbox
SANDBOX_DIR = Path(__file__).resolve().parent
DATA_DIR = SANDBOX_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
LOGS_DIR = SANDBOX_DIR / "logs"

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"

MIN_BBOX_AREA = 3200.0  # Toi thieu ~ 56x56 px, tranh nguoi qua nho o xa
MIN_KEYPOINTS = 10
TARGET_TOTAL_IMAGES = 500
RANDOM_SEED = 42


def download_with_progress(url: str, dest_path: Path, desc: str = "") -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"[*] Dang tai {desc}: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=40) as response, open(temp_path, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            total_bytes = int(total_size) if total_size else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(f"\r    Tien do: {downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r    Da tai: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
            print()

        if temp_path.exists():
            temp_path.replace(dest_path)
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"\n[LOI MANG] Khong the tai {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def get_coco_annotations() -> Path | None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    target_json = ANNOTATIONS_DIR / "person_keypoints_val2017.json"
    if target_json.exists() and target_json.stat().st_size > 10 * 1024 * 1024:
        print(f"[OK] Da tim thay annotation file: {target_json}")
        return target_json

    zip_dest = DATA_DIR / "annotations_trainval2017.zip"
    if not zip_dest.exists():
        success = download_with_progress(ANNOTATIONS_ZIP_URL, zip_dest, "COCO 2017 Annotations (trainval)")
        if not success:
            print("[DUNG] Mang sandbox bi chan hoac loi ket noi khi tai COCO annotations.")
            print("Vui long tai 'person_keypoints_val2017.json' va dat vao thu muc:")
            print(f"    {target_json}")
            return None

    print("[*] Dang giai nen person_keypoints_val2017.json...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as z:
            match_name = None
            for name in z.namelist():
                if name.endswith("person_keypoints_val2017.json"):
                    match_name = name
                    break
            if not match_name:
                print("[LOI] Khong tim thay person_keypoints_val2017.json trong file zip!")
                return None
            
            with z.open(match_name) as source, open(target_json, "wb") as target:
                target.write(source.read())
        
        print(f"[OK] Da giai nen thanh cong: {target_json} ({target_json.stat().st_size / (1024*1024):.1f} MB)")
        try:
            zip_dest.unlink()
            print("[*] Da don dep file zip tam.")
        except Exception:
            pass
        return target_json
    except Exception as e:
        print(f"[LOI] Loi khi giai nen zip annotations: {e}")
        return None


def filter_and_sample_dataset(ann_path: Path) -> dict[str, Any]:
    print(f"[*] Dang doc va phan tich annotations tu {ann_path.name}...")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images_map = {img["id"]: img for img in coco_data.get("images", [])}
    
    valid_annos_by_img: dict[int, list[dict]] = {}
    for ann in coco_data.get("annotations", []):
        if ann.get("category_id") != 1:
            continue
        if ann.get("iscrowd", 0) != 0:
            continue
        if ann.get("num_keypoints", 0) < MIN_KEYPOINTS:
            continue
        
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        w, h = bbox[2], bbox[3]
        area = ann.get("area", w * h)
        if area < MIN_BBOX_AREA:
            continue
        
        img_id = ann["image_id"]
        valid_annos_by_img.setdefault(img_id, []).append(ann)

    print(f"[*] Tong so anh val co it nhat 1 person dat chuan: {len(valid_annos_by_img)}")

    single_person_imgs = [img_id for img_id, anns in valid_annos_by_img.items() if len(anns) == 1]
    multi_person_imgs = [img_id for img_id, anns in valid_annos_by_img.items() if len(anns) >= 2]

    print(f"    - Single-person images: {len(single_person_imgs)}")
    print(f"    - Multi-person images: {len(multi_person_imgs)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(single_person_imgs)
    rng.shuffle(multi_person_imgs)

    target_per_group = TARGET_TOTAL_IMAGES // 2
    selected_single = single_person_imgs[:target_per_group]
    selected_multi = multi_person_imgs[:target_per_group]

    all_selected_ids = selected_single + selected_multi
    rng.shuffle(all_selected_ids)

    half = len(all_selected_ids) // 2
    dev_ids = all_selected_ids[:half]
    test_ids = all_selected_ids[half:]

    print(f"[OK] Da chon {len(all_selected_ids)} anh: Dev={len(dev_ids)}, Test={len(test_ids)}")

    manifest = {
        "metadata": {
            "seed": RANDOM_SEED,
            "min_bbox_area": MIN_BBOX_AREA,
            "min_keypoints": MIN_KEYPOINTS,
            "total_images": len(all_selected_ids),
            "dev_count": len(dev_ids),
            "test_count": len(test_ids),
            "single_person_count": len(selected_single),
            "multi_person_count": len(selected_multi),
        },
        "dev": [
            {
                "image_id": img_id,
                "file_name": images_map[img_id]["file_name"],
                "width": images_map[img_id]["width"],
                "height": images_map[img_id]["height"],
                "annotations": valid_annos_by_img[img_id],
            }
            for img_id in dev_ids
        ],
        "test": [
            {
                "image_id": img_id,
                "file_name": images_map[img_id]["file_name"],
                "width": images_map[img_id]["width"],
                "height": images_map[img_id]["height"],
                "annotations": valid_annos_by_img[img_id],
            }
            for img_id in test_ids
        ],
    }

    manifest_file = DATA_DIR / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Da luu manifest phan chia vao: {manifest_file}")

    return manifest


def download_selected_images(manifest: dict[str, Any]) -> bool:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    all_items = manifest["dev"] + manifest["test"]
    total = len(all_items)

    print(f"[*] Bat dau kiem tra va tai {total} anh COCO val...")
    downloaded_count = 0
    skipped_count = 0

    for idx, item in enumerate(all_items):
        file_name = item["file_name"]
        local_path = IMAGES_DIR / file_name
        if local_path.exists() and local_path.stat().st_size > 1024:
            skipped_count += 1
            continue

        url = f"{COCO_IMAGE_BASE_URL}/{file_name}"
        success = download_with_progress(url, local_path, f"anh {idx+1}/{total} ({file_name})")
        if not success:
            print(f"[DUNG] Tai anh that bai tai {file_name}. Dung lai bao cao.")
            return False
        downloaded_count += 1

    print(f"[HOAN THANH] Tong anh san sang: {total} (Moi tai: {downloaded_count}, Da co: {skipped_count})")
    return True


def main():
    print("=== BUOC 1: CHUAN BI DATASET COCO VAL2017 SANDBOX ===")
    ann_path = get_coco_annotations()
    if not ann_path:
        sys.exit(1)

    manifest = filter_and_sample_dataset(ann_path)
    success = download_selected_images(manifest)
    if not success:
        sys.exit(2)

    print("[THANH CONG] Du lieu benchmark COCO val2017 da san sang 100% trong sandbox.")


if __name__ == "__main__":
    main()
