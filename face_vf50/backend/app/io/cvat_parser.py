from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote
from fastapi import HTTPException

# Bảng ánh xạ ID sublabel từ JSON cấu hình CVAT constructor người dùng cung cấp -> VF-50 Point ID
CVAT_SUBLABEL_ID_TO_VF50: dict[int, int] = {
    # longmaytrai (skeleton id: 116): 0 -> 4
    117: 0, 118: 1, 119: 2, 120: 3, 121: 4,
    # longmayphai (skeleton id: 122): 5 -> 9
    123: 5, 124: 6, 125: 7, 126: 8, 127: 9,
    # songmui (skeleton id: 128): 10 -> 13
    129: 10, 130: 11, 131: 12, 132: 13,
    # mattrai (skeleton id: 133): 14 -> 21
    134: 14, 135: 15, 136: 16, 137: 17, 138: 18, 139: 19, 140: 20, 141: 21,
    # matphai (skeleton id: 142): 22 -> 29
    143: 22, 144: 23, 145: 24, 146: 25, 147: 26, 148: 27, 149: 28, 150: 29,
    # moingoai (skeleton id: 151): 30 -> 41
    152: 30, 153: 31, 154: 32, 155: 33, 156: 34, 157: 35,
    158: 36, 159: 37, 160: 38, 161: 39, 162: 40, 163: 41,
    # moitrong (skeleton id: 164): 42 -> 49
    165: 42, 166: 43, 167: 44, 168: 45, 169: 46, 170: 47, 171: 48, 172: 49,
}

# Cấu hình 7 skeleton chuẩn VinFast VF-50
SKELETON_SPECS: dict[str, dict[str, int]] = {
    "longmaytrai": {"start": 0, "count": 5},
    "longmayphai": {"start": 5, "count": 5},
    "songmui": {"start": 10, "count": 4},
    "mattrai": {"start": 14, "count": 8},
    "matphai": {"start": 22, "count": 8},
    "moingoai": {"start": 30, "count": 12},
    "moitrong": {"start": 42, "count": 8},
}

# Ánh xạ tên / alias skeleton hỗ trợ cả tiếng Việt không dấu, có dấu và tiếng Anh
SKELETON_ALIASES: dict[str, str] = {
    "longmaytrai": "longmaytrai",
    "lông mày trái": "longmaytrai",
    "left_eyebrow": "longmaytrai",
    "eyebrow_left": "longmaytrai",
    "long_may_trai": "longmaytrai",

    "longmayphai": "longmayphai",
    "lông mày phải": "longmayphai",
    "right_eyebrow": "longmayphai",
    "eyebrow_right": "longmayphai",
    "long_may_phai": "longmayphai",

    "songmui": "songmui",
    "sống mũi": "songmui",
    "nose_bridge": "songmui",
    "nose": "songmui",
    "song_mui": "songmui",

    "mattrai": "mattrai",
    "mắt trái": "mattrai",
    "left_eye": "mattrai",
    "eye_left": "mattrai",
    "mat_trai": "mattrai",

    "matphai": "matphai",
    "mắt phải": "matphai",
    "right_eye": "matphai",
    "eye_right": "matphai",
    "mat_phai": "matphai",

    "moingoai": "moingoai",
    "môi ngoài": "moingoai",
    "outer_lip": "moingoai",
    "lip_outer": "moingoai",
    "moi_ngoai": "moingoai",

    "moitrong": "moitrong",
    "môi trong": "moitrong",
    "inner_lip": "moitrong",
    "lip_inner": "moitrong",
    "moi_trong": "moitrong",
}


def resolve_point_id(
    skel_label: str = "",
    raw_label: str = "",
    label_id: str = "",
    index_in_skel: int = 0,
) -> Optional[int]:
    """
    Quy đổi thông minh ID điểm nhãn từ CVAT XML / JSON về định danh 0..49 của VinFast VF-50:
    1. Kiểm tra ID sublabel gốc của CVAT (117..172)
    2. Nếu thuộc skeleton cụ thể: tính theo offset skeleton (start + local_index) hoặc kiểm tra dải toàn cục
    3. Trích xuất số nguyên từ raw_label (ví dụ: 'pt_15', 'point_0', '15')
    4. Fallback theo chỉ số index_in_skel nếu hợp lệ (0..49)
    """
    # 1. Kiểm tra nếu có label_id hoặc raw_label trùng ID sublabel CVAT (117..172)
    for cand in (label_id, raw_label):
        if cand is not None and str(cand).strip():
            s = str(cand).strip()
            try:
                cand_int = int(s)
                if cand_int in CVAT_SUBLABEL_ID_TO_VF50:
                    return CVAT_SUBLABEL_ID_TO_VF50[cand_int]
            except ValueError:
                digits = re.findall(r"\d+", s)
                if digits:
                    val = int(digits[0])
                    if val in CVAT_SUBLABEL_ID_TO_VF50:
                        return CVAT_SUBLABEL_ID_TO_VF50[val]

    # 2. Xử lý theo skeleton
    norm_skel = SKELETON_ALIASES.get(skel_label.strip().lower(), skel_label.strip().lower())
    spec = SKELETON_SPECS.get(norm_skel)

    if spec is not None:
        start = spec["start"]
        count = spec["count"]

        # Parse raw_label nếu có số
        if raw_label is not None and str(raw_label).strip():
            digits = re.findall(r"\d+", str(raw_label))
            if digits:
                val = int(digits[0])
                # Trường hợp A: Đã là ID toàn cục chính xác cho skeleton này (vd: 5..9 cho longmayphai)
                if start <= val < start + count:
                    return val
                # Trường hợp B: Là chỉ số cục bộ trong skeleton (0..count-1) (vd: 0..4 cho longmayphai)
                if 0 <= val < count:
                    return start + val
                # Trường hợp C: Số nguyên 0..49 hợp lệ
                if 0 <= val < 50:
                    return val

        # Nếu không có raw_label hoặc không parse được số, dùng thứ tự trong skeleton
        if 0 <= index_in_skel < count:
            return start + index_in_skel

    # 3. Khi không có skeleton hoặc skeleton không xác định (ví dụ category là "face", "person", "vf50")
    if raw_label is not None and str(raw_label).strip():
        digits = re.findall(r"\d+", str(raw_label))
        if digits:
            val = int(digits[0])
            if 0 <= val < 50:
                return val

    # 4. Fallback theo index_in_skel nếu trong khoảng 0..49
    if 0 <= index_in_skel < 50:
        return index_in_skel

    return None


def parse_cvat_xml(xml_content: str | bytes) -> list[dict[str, Any]]:
    """
    Parse định dạng xuất chuẩn 'CVAT for images 1.1' hoặc 'CVAT for video 1.1' (XML).
    Hỗ trợ cả thẻ <image> với <skeleton>, thẻ <track>, và thẻ <points> phẳng.
    """
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        raise HTTPException(400, f"File XML CVAT không hợp lệ: {e}")

    parsed_images: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Format CVAT for images (Chứa các thẻ <image>)
    # -------------------------------------------------------------
    image_elements = root.findall("image")
    if image_elements:
        for img_el in image_elements:
            file_name = img_el.get("name", "")
            width = int(img_el.get("width", 1280))
            height = int(img_el.get("height", 720))

            keypoints_by_id: dict[int, dict[str, Any]] = {}

            # A. Skeletons
            for skel_el in img_el.findall("skeleton"):
                skel_label = skel_el.get("label", "")
                pts_list = skel_el.findall("points")
                for idx_in_skel, pt_el in enumerate(pts_list):
                    raw_label = pt_el.get("label", "")
                    label_id = pt_el.get("label_id", "")
                    raw_pts = pt_el.get("points", "")
                    is_outside = pt_el.get("outside", "0") in ("1", "true", "True")
                    is_occluded = pt_el.get("occluded", "0") in ("1", "true", "True")

                    pt_id = resolve_point_id(
                        skel_label=skel_label,
                        raw_label=raw_label,
                        label_id=label_id,
                        index_in_skel=idx_in_skel,
                    )
                    if pt_id is None:
                        continue

                    state = "outside" if is_outside else ("occluded" if is_occluded else "visible")
                    px, py = None, None
                    if raw_pts:
                        parts = raw_pts.replace(";", ",").split(",")
                        if len(parts) >= 2:
                            try:
                                px = float(parts[0].strip())
                                py = float(parts[1].strip())
                            except ValueError:
                                pass

                    if state == "outside":
                        px, py = None, None

                    keypoints_by_id[pt_id] = {
                        "id": pt_id,
                        "x": px,
                        "y": py,
                        "state": state,
                        "skeleton": skel_label,
                    }

            # B. Direct <points> (nằm trực tiếp dưới <image> không qua <skeleton>)
            for pt_el in img_el.findall("points"):
                raw_label = pt_el.get("label", "")
                label_id = pt_el.get("label_id", "")
                raw_pts = pt_el.get("points", "")
                is_outside = pt_el.get("outside", "0") in ("1", "true", "True")
                is_occluded = pt_el.get("occluded", "0") in ("1", "true", "True")

                pt_id = resolve_point_id(
                    skel_label="",
                    raw_label=raw_label,
                    label_id=label_id,
                )
                if pt_id is None or pt_id in keypoints_by_id:
                    continue

                state = "outside" if is_outside else ("occluded" if is_occluded else "visible")
                px, py = None, None
                if raw_pts:
                    parts = raw_pts.replace(";", ",").split(",")
                    if len(parts) >= 2:
                        try:
                            px = float(parts[0].strip())
                            py = float(parts[1].strip())
                        except ValueError:
                            pass

                if state == "outside":
                    px, py = None, None

                keypoints_by_id[pt_id] = {
                    "id": pt_id,
                    "x": px,
                    "y": py,
                    "state": state,
                }

            parsed_images.append({
                "file_name": file_name.replace("\\", "/"),
                "width": width,
                "height": height,
                "keypoints": keypoints_by_id,
            })

        return parsed_images

    # -------------------------------------------------------------
    # 2. Format CVAT for video (Chứa các thẻ <track>)
    # -------------------------------------------------------------
    track_elements = root.findall("track")
    if track_elements:
        frames_map: dict[int, dict[int, dict[str, Any]]] = {}

        for track_el in track_elements:
            track_label = track_el.get("label", "")
            # Các điểm points theo từng frame trong track
            for idx_pt, pt_el in enumerate(track_el.findall("points")):
                frame_idx = int(pt_el.get("frame", 0))
                raw_label = pt_el.get("label", "")
                label_id = pt_el.get("label_id", "")
                raw_pts = pt_el.get("points", "")
                is_outside = pt_el.get("outside", "0") in ("1", "true", "True")
                is_occluded = pt_el.get("occluded", "0") in ("1", "true", "True")

                pt_id = resolve_point_id(
                    skel_label=track_label,
                    raw_label=raw_label,
                    label_id=label_id,
                    index_in_skel=idx_pt,
                )
                if pt_id is None:
                    continue

                state = "outside" if is_outside else ("occluded" if is_occluded else "visible")
                px, py = None, None
                if raw_pts:
                    parts = raw_pts.replace(";", ",").split(",")
                    if len(parts) >= 2:
                        try:
                            px = float(parts[0].strip())
                            py = float(parts[1].strip())
                        except ValueError:
                            pass

                if state == "outside":
                    px, py = None, None

                if frame_idx not in frames_map:
                    frames_map[frame_idx] = {}

                frames_map[frame_idx][pt_id] = {
                    "id": pt_id,
                    "x": px,
                    "y": py,
                    "state": state,
                    "skeleton": track_label,
                }

        sorted_frame_indices = sorted(frames_map.keys())
        for f_idx in sorted_frame_indices:
            parsed_images.append({
                "file_name": f"frame_{f_idx:06d}.jpg",
                "frame_index": f_idx,
                "width": 1280,
                "height": 720,
                "keypoints": frames_map[f_idx],
            })

        return parsed_images

    return []


def parse_annotation_payload(content: bytes) -> list[dict[str, Any]]:
    """
    Tự động nhận diện định dạng XML (CVAT for images / video 1.1) hoặc JSON (COCO, Landmark QA, Cleaned Export).
    """
    if isinstance(content, bytes) and content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]

    text_sample = content[:400].decode("utf-8", errors="ignore").strip().lower()

    # Nhận diện XML
    if text_sample.startswith("<?xml") or "<annotations>" in text_sample or "<image" in text_sample:
        return parse_cvat_xml(content)

    # Nhận diện JSON
    try:
        data = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(400, "File annotation không phải là XML hoặc JSON hợp lệ")

    parsed_images: list[dict[str, Any]] = []

    # 1. Định dạng xuất sạch của chính hệ thống chúng ta (Cleaned Export)
    if isinstance(data, dict) and "frames" in data:
        for f in data["frames"]:
            fn = f.get("image_filename", "").replace("\\", "/")
            raw_kpts = f.get("keypoints", {})
            kpts_by_id: dict[int, dict[str, Any]] = {}
            for k, pt in raw_kpts.items():
                try:
                    pid = int(k)
                    kpts_by_id[pid] = {
                        "id": pid,
                        "x": pt.get("x"),
                        "y": pt.get("y"),
                        "state": pt.get("state", "visible"),
                    }
                except ValueError:
                    pass
            parsed_images.append({
                "file_name": fn,
                "width": 1280,
                "height": 720,
                "keypoints": kpts_by_id,
            })
        return parsed_images

    # 2. Định dạng COCO Keypoints
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        categories_by_id: dict[Any, dict[str, Any]] = {}
        for c in data.get("categories", []):
            cid = c.get("id")
            if cid is not None:
                categories_by_id[cid] = c
                categories_by_id[str(cid)] = c

        annots_by_img: dict[str, list[Any]] = {}
        for annot in data.get("annotations", []):
            i_id = annot.get("image_id")
            if i_id is not None:
                s_id = str(i_id)
                if s_id not in annots_by_img:
                    annots_by_img[s_id] = []
                annots_by_img[s_id].append(annot)

        for img_info in data["images"]:
            img_id = img_info.get("id")
            str_img_id = str(img_id) if img_id is not None else ""
            raw_fn = img_info.get("file_name", "")
            fn = unquote(raw_fn).replace("\\", "/")
            w = int(img_info.get("width", 1280))
            h = int(img_info.get("height", 720))
            kpts_by_id: dict[int, dict[str, Any]] = {}

            # Tra cứu annotations theo img_id (cả string, int, hoặc khớp tên file)
            matched_annots = annots_by_img.get(str_img_id, [])
            if not matched_annots and fn in annots_by_img:
                matched_annots = annots_by_img[fn]
            if not matched_annots and Path(fn).name in annots_by_img:
                matched_annots = annots_by_img[Path(fn).name]

            # Fallback nếu cả file chỉ có 1 ảnh
            if not matched_annots and len(data["images"]) == 1:
                matched_annots = data.get("annotations", [])

            for annot in matched_annots:
                cat_id = annot.get("category_id")
                cat = categories_by_id.get(cat_id) or categories_by_id.get(str(cat_id)) or {}
                if not cat and len(data.get("categories", [])) == 1:
                    cat = data["categories"][0]

                cat_name = cat.get("name", "")
                cat_kpts = cat.get("keypoints", [])
                raw_kpts = annot.get("keypoints", [])

                if not isinstance(raw_kpts, list) or not raw_kpts:
                    continue

                # Đọc attributes (ví dụ occluded cấp annotation từ CVAT)
                annot_attrs = annot.get("attributes", {})
                is_annot_occluded = annot_attrs.get("occluded", False) if isinstance(annot_attrs, dict) else False

                # Xử lý nếu raw_kpts là danh sách dict [{"x": ..., "y": ...}]
                if isinstance(raw_kpts[0], dict):
                    for idx, pt_dict in enumerate(raw_kpts):
                        pt_label = pt_dict.get("id", idx)
                        pt_id = resolve_point_id(
                            skel_label=cat_name,
                            raw_label=str(pt_label),
                            index_in_skel=idx,
                        )
                        if pt_id is not None and 0 <= pt_id < 50:
                            state = pt_dict.get("state", "occluded" if is_annot_occluded else "visible")
                            kpts_by_id[pt_id] = {
                                "id": pt_id,
                                "x": pt_dict.get("x"),
                                "y": pt_dict.get("y"),
                                "state": state,
                                "skeleton": cat_name,
                            }
                    continue

                # Xử lý nếu raw_kpts là danh sách tọa độ [[x, y], [x, y, v]]
                if isinstance(raw_kpts[0], (list, tuple)):
                    for idx, pt_arr in enumerate(raw_kpts):
                        if len(pt_arr) >= 2:
                            pt_label = cat_kpts[idx] if idx < len(cat_kpts) else str(idx)
                            pt_id = resolve_point_id(
                                skel_label=cat_name,
                                raw_label=str(pt_label),
                                index_in_skel=idx,
                            )
                            if pt_id is not None and 0 <= pt_id < 50:
                                vis = pt_arr[2] if len(pt_arr) >= 3 else (1 if is_annot_occluded else 2)
                                st = "outside" if vis == 0 else ("occluded" if vis == 1 else "visible")
                                kpts_by_id[pt_id] = {
                                    "id": pt_id,
                                    "x": float(pt_arr[0]) if st != "outside" else None,
                                    "y": float(pt_arr[1]) if st != "outside" else None,
                                    "state": st,
                                    "skeleton": cat_name,
                                }
                    continue

                # Chuẩn COCO phẳng: bộ 3 [x0, y0, v0, x1, y1, v1, ...]
                step = 3
                if len(cat_kpts) > 0 and len(raw_kpts) == len(cat_kpts) * 2:
                    step = 2

                num_pts = len(raw_kpts) // step
                for pt_idx in range(num_pts):
                    base_i = pt_idx * step
                    if base_i + 1 >= len(raw_kpts):
                        break

                    try:
                        px = float(raw_kpts[base_i])
                        py = float(raw_kpts[base_i + 1])
                        vis = int(raw_kpts[base_i + 2]) if (step == 3 and base_i + 2 < len(raw_kpts)) else (1 if is_annot_occluded else 2)
                    except (ValueError, TypeError, IndexError):
                        continue

                    pt_label = cat_kpts[pt_idx] if pt_idx < len(cat_kpts) else str(pt_idx)
                    pt_id = resolve_point_id(
                        skel_label=cat_name,
                        raw_label=str(pt_label),
                        index_in_skel=pt_idx,
                    )
                    if pt_id is None or not (0 <= pt_id < 50):
                        continue

                    state = "outside" if vis == 0 else ("occluded" if vis == 1 else "visible")
                    if state == "outside":
                        px, py = None, None

                    kpts_by_id[pt_id] = {
                        "id": pt_id,
                        "x": px,
                        "y": py,
                        "state": state,
                        "skeleton": cat_name,
                    }

            parsed_images.append({
                "file_name": fn,
                "width": w,
                "height": h,
                "keypoints": kpts_by_id,
            })
        return parsed_images

    # 3. Định dạng landmark-qa/v1 lồng annotations bên trong images
    if isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            fn = img.get("file_name", "").replace("\\", "/")
            w = img.get("width", 1280)
            h = img.get("height", 720)
            keypoints_by_id = {}

            annots = img.get("annotations", [])
            for annot in annots:
                for pt in annot.get("keypoints", []):
                    pt_id = pt.get("id")
                    if pt_id is not None:
                        try:
                            pid = int(pt_id)
                            keypoints_by_id[pid] = {
                                "id": pid,
                                "x": pt.get("x"),
                                "y": pt.get("y"),
                                "state": pt.get("state", "visible"),
                            }
                        except ValueError:
                            pass

            parsed_images.append({
                "file_name": fn,
                "width": w,
                "height": h,
                "keypoints": keypoints_by_id,
            })
        return parsed_images

    return parsed_images
