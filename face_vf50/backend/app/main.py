"""FastAPI backend application for VinFast VF-50 Face QA.

Runs on port 8001. Provides endpoints for:
- Uploading image ZIPs and CVAT XML/JSON annotations.
- Single-model MediaPipe inference (CPU-optimized, ~15-20ms/frame).
- Running the 14 VinFast Face Landmark guideline rules.
- Frame audit ranking and interactive inspection.
- Generating manual-fix Markdown checklists and CSV reports.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.database import (
        get_all_frame_evaluations,
        get_all_frames_for_export,
        get_all_sessions,
        get_frame_detail,
        get_session,
        get_session_frames,
        init_db,
        recalculate_session_stats,
        save_frames_batch,
        save_session,
        update_frame_full,
        update_frame_status,
    )
    from app.io.cvat_parser import parse_annotation_payload
    from app.io.safe_zip import safe_extract_images
    from app.model.mediapipe_face import get_detector
    from app.qa.report import generate_csv_report, generate_markdown_checklist
    from app.qa.rules import check_all_rules
    from app.qa.scoring import evaluate_frame_quality
except ImportError:
    from .database import (
        get_all_frame_evaluations,
        get_all_frames_for_export,
        get_all_sessions,
        get_frame_detail,
        get_session,
        get_session_frames,
        init_db,
        recalculate_session_stats,
        save_frames_batch,
        save_session,
        update_frame_full,
        update_frame_status,
    )
    from .io.cvat_parser import parse_annotation_payload
    from .io.safe_zip import safe_extract_images
    from .model.mediapipe_face import get_detector
    from .qa.report import generate_csv_report, generate_markdown_checklist
    from .qa.rules import check_all_rules
    from .qa.scoring import evaluate_frame_quality

UPLOADS_DIR = BASE_DIR / "data" / "uploads"

app = FastAPI(
    title="VinFast Face Landmark VF-50 Hand-Labeling Copilot & QA Auditor",
    version="1.3.0",
    description="Backend QA system for vf_face_landmark50_v1 hand-labeling validation."
)

app.add_middleware(
     CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Warm up detector
    try:
        get_detector()
    except Exception as exc:
        print(f"[WARNING] Model warmup failed: {exc}")


def violation_to_dict(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    level = getattr(v, "level", "ERROR")
    return {
        "code": getattr(v, "code", ""),
        "rule_name": getattr(v, "rule_name", ""),
        "level": level,
        "severity": "critical" if level == "ERROR" else "major",
        "message": getattr(v, "message", ""),
        "points": getattr(v, "points", []),
        "gl_section": getattr(v, "gl_section", ""),
    }


def _prepare_frames_report(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reports = []
    for f in evaluations:
        violations = f.get("rule_violations", [])
        err_c = sum(1 for v in violations if v.get("level") == "ERROR")
        warn_c = sum(1 for v in violations if v.get("level") == "WARNING")
        reports.append({
            "file_name": f.get("image_filename", f"frame_{f.get('frame_index')}.jpg"),
            "errors_count": err_c,
            "warnings_count": warn_c,
            "violations": violations,
            "quality": {
                "iod": f.get("iod", 100.0),
                "nme": f.get("nme", 0.0),
                "passed_nme": f.get("nme", 0.0) <= 0.035,
                "point_evaluations": {},
            }
        })
    return reports


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "vf50_face_qa_backend",
        "version": "1.3.0",
        "port": 8001,
        "model": "MediaPipe Face Landmarker (CPU)",
    }


@app.get("/api/images/{session_id}/{filename}")
def serve_image(session_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    img_path = UPLOADS_DIR / session_id / "images" / safe_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(img_path))


@app.post("/api/upload")
async def upload_dataset(
    dataset_zip: UploadFile = File(...),
    annotation_file: Optional[UploadFile] = File(None),
    session_name: Optional[str] = Form(None),
    run_model: bool = Form(True),
) -> Dict[str, Any]:
    """Upload image ZIP archive and optional CVAT XML / JSON annotations."""
    session_id = str(uuid.uuid4())[:8]
    session_folder = UPLOADS_DIR / session_id
    images_folder = session_folder / "images"
    session_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)

    # 1. Extract ZIP archive safely
    extracted_rel_paths, zip_ann_bytes = safe_extract_images(dataset_zip, images_folder)
    if not extracted_rel_paths:
        raise HTTPException(status_code=400, detail="No valid image files found in the ZIP archive")

    extracted_paths = [images_folder / p for p in extracted_rel_paths]
    # Sort images by filename natural order
    extracted_paths.sort(key=lambda p: p.name)

    # 2. Parse annotations if provided or found inside ZIP
    human_by_name: Dict[str, Dict[int, Dict[str, Any]]] = {}
    human_by_idx: Dict[int, Dict[int, Dict[str, Any]]] = {}
    
    ann_bytes: Optional[bytes] = None
    if annotation_file is not None:
        raw_ann = await annotation_file.read()
        if len(raw_ann) > 0:
            ann_bytes = raw_ann

    # Tự động dùng file nhãn tìm thấy bên trong file ZIP nếu người dùng không upload file riêng ở ô 2
    if ann_bytes is None and zip_ann_bytes is not None and len(zip_ann_bytes) > 0:
        ann_bytes = zip_ann_bytes

    if ann_bytes is not None and len(ann_bytes) > 0:
        parsed_list = parse_annotation_payload(ann_bytes)
        for p_idx, item in enumerate(parsed_list):
            kpts = item.get("keypoints", {})
            raw_fn = item.get("file_name", "")
            fn = Path(raw_fn).name
            stem = Path(raw_fn).stem
            if fn:
                human_by_name[fn] = kpts
                human_by_name[fn.lower()] = kpts
            if stem:
                human_by_name[stem] = kpts
                human_by_name[stem.lower()] = kpts
            human_by_idx[p_idx] = kpts
            if "frame_index" in item:
                human_by_idx[item["frame_index"]] = kpts

    # 3. Process each frame: Run MediaPipe & QA Rules
    detector = get_detector()
    frames_to_save: List[Dict[str, Any]] = []
    total_issues = 0
    total_nme = 0.0
    nme_count = 0
    prev_kpts: Optional[Dict[int, Dict[str, Any]]] = None

    for idx, img_path in enumerate(extracted_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        # Human keypoints for this frame
        human_kpts = (
            human_by_name.get(img_path.name)
            or human_by_name.get(img_path.name.lower())
            or human_by_name.get(img_path.stem)
            or human_by_name.get(img_path.stem.lower())
            or human_by_idx.get(idx, {})
        )

        # Fallback nếu chỉ có duy nhất 1 ảnh và 1 bộ nhãn trong session
        if not human_kpts and len(extracted_paths) == 1 and len(human_by_idx) >= 1:
            human_kpts = list(human_by_idx.values())[0]


        # Inference with single model (smartly focus on annotated face if human_kpts available)
        model_kpts: Dict[int, Dict[str, Any]] = {}
        if run_model:
            detection = detector.detect_landmarks(img_bgr, target_keypoints=human_kpts if human_kpts else None)
            if detection is not None:
                vf50_coords, _ = detection
                for p_id, (px, py) in enumerate(vf50_coords):
                    model_kpts[p_id] = {
                        "id": p_id,
                        "x": float(px),
                        "y": float(py),
                        "state": "visible",
                    }

        # Check guideline rules
        violations_raw = []
        active_kpts = human_kpts if human_kpts else model_kpts
        if active_kpts:
            violations_raw = check_all_rules(
                active_kpts,
                prev_keypoints_by_id=prev_kpts,
                model_keypoints_by_id=model_kpts if human_kpts else None,
                image_width=w,
                image_height=h,
            )
        violations = [violation_to_dict(v) for v in violations_raw]
        prev_kpts = active_kpts

        # Evaluate scoring & NME
        score_eval = evaluate_frame_quality(human_kpts or model_kpts, model_kpts if human_kpts else None)

        err_count = len(violations)
        total_issues += err_count

        severity_weight = sum(
            3.0 if v.get("severity") == "critical"
            else 2.0 if v.get("severity") == "major"
            else 1.0
            for v in violations
        )

        nme_val = float(score_eval.get("nme", 0.0))
        if human_kpts and model_kpts:
            total_nme += nme_val
            nme_count += 1

        frames_to_save.append({
            "session_id": session_id,
            "frame_index": idx,
            "image_filename": img_path.name,
            "image_path": str(img_path),
            "human_keypoints": human_kpts,
            "model_keypoints": model_kpts,
            "iod": float(score_eval.get("iod", 0.0)),
            "nme": nme_val,
            "rule_violations": violations,
            "severity_score": float(severity_weight),
            "error_count": err_count,
            "status": "pending",
            "case_type": score_eval.get("case_type", "normal"),
            "ai_reliability": float(score_eval.get("ai_reliability", 0.95)),
            "case_label": score_eval.get("case_label", "Bình thường"),
            "case_description": score_eval.get("case_description", ""),
        })

    # Save to database
    save_frames_batch(frames_to_save)
    avg_nme = (total_nme / nme_count) if nme_count > 0 else 0.0
    display_name = session_name or f"Session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_session(
        session_id=session_id,
        name=display_name,
        created_at=datetime.now().isoformat(),
        total_frames=len(frames_to_save),
        total_issues=total_issues,
        avg_nme=avg_nme,
    )

    return {
        "session_id": session_id,
        "name": display_name,
        "total_frames": len(frames_to_save),
        "total_issues": total_issues,
        "avg_nme": avg_nme,
    }


@app.get("/api/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    return get_all_sessions()


@app.get("/api/sessions/{session_id}")
def get_session_info(session_id: str) -> Dict[str, Any]:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


@app.get("/api/sessions/{session_id}/frames")
def list_session_frames(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("severity", pattern="^(severity|frame_index|nme)$"),
    filter_rule: Optional[str] = Query(None),
) -> Dict[str, Any]:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return get_session_frames(
        session_id=session_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        filter_rule=filter_rule,
    )


@app.get("/api/sessions/{session_id}/frames/{frame_index}")
def get_frame(session_id: str, frame_index: int) -> Dict[str, Any]:
    detail = get_frame_detail(session_id, frame_index)
    if not detail:
        raise HTTPException(status_code=404, detail="Frame not found")
    return detail


@app.post("/api/sessions/{session_id}/frames/{frame_index}/status")
def set_frame_status(
    session_id: str,
    frame_index: int,
    status: str = Form(..., pattern="^(pending|reviewed|fixed)$"),
) -> Dict[str, Any]:
    updated = update_frame_status(session_id, frame_index, status)
    if not updated:
        raise HTTPException(status_code=404, detail="Frame not found")
    return {"session_id": session_id, "frame_index": frame_index, "status": status}


@app.get("/api/sessions/{session_id}/export/checklist")
def export_checklist(session_id: str) -> PlainTextResponse:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    evaluations = get_all_frame_evaluations(session_id)
    reports = _prepare_frames_report(evaluations)
    md_text = generate_markdown_checklist(session_id, reports)
    return PlainTextResponse(
        content=md_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="checklist_{session_id}.md"'}
    )


@app.get("/api/sessions/{session_id}/export/csv")
def export_csv(session_id: str) -> Response:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    evaluations = get_all_frame_evaluations(session_id)
    reports = _prepare_frames_report(evaluations)
    csv_text = generate_csv_report(reports)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="qa_report_{session_id}.csv"'}
    )


class ApplyModelFixPayload(BaseModel):
    point_ids: Optional[List[int]] = None


class BatchApplyModelFixPayload(BaseModel):
    only_violated: bool = True
    min_nme: float = 0.035



def generate_cvat_xml(frames_data: List[Dict[str, Any]]) -> str:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "VinFast Face Landmark VF-50 (Cleaned)"
    ET.SubElement(task, "size").text = str(len(frames_data))

    skeletons_spec = [
        ("longmaytrai", list(range(0, 5))),
        ("longmayphai", list(range(5, 10))),
        ("songmui", list(range(10, 14))),
        ("mattrai", list(range(14, 22))),
        ("matphai", list(range(22, 30))),
        ("moingoai", list(range(30, 42))),
        ("moitrong", list(range(42, 50))),
    ]

    for f in frames_data:
        img_el = ET.SubElement(
            root,
            "image",
            {
                "id": str(f.get("frame_index", 0)),
                "name": f.get("image_filename", ""),
                "width": "1280",
                "height": "720",
            }
        )
        hkpts = f.get("human_keypoints", {})
        for skel_label, pt_ids in skeletons_spec:
            skel_el = ET.SubElement(img_el, "skeleton", {"label": skel_label})
            for pid in pt_ids:
                pt_data = hkpts.get(str(pid)) or hkpts.get(pid)
                if pt_data:
                    px = pt_data.get("x")
                    py = pt_data.get("y")
                    st = pt_data.get("state", "visible")
                    pts_str = f"{px:.2f},{py:.2f}" if px is not None and py is not None else ""
                    ET.SubElement(
                        skel_el,
                        "points",
                        {
                            "label": str(pid),
                            "points": pts_str,
                            "occluded": "1" if st == "occluded" else "0",
                            "outside": "1" if st == "outside" else "0",
                        }
                    )

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def generate_cleaned_json(session_info: Dict[str, Any], frames_data: List[Dict[str, Any]]) -> str:
    data = {
        "version": "1.3.0",
        "format": "vinfast_vf50_face_landmarks",
        "session_id": session_info.get("session_id", ""),
        "total_frames": len(frames_data),
        "exported_at": datetime.now().isoformat(),
        "frames": [
            {
                "frame_index": f.get("frame_index"),
                "image_filename": f.get("image_filename"),
                "status": f.get("status"),
                "nme": f.get("nme"),
                "iod": f.get("iod"),
                "keypoints": f.get("human_keypoints", {}),
            }
            for f in frames_data
        ]
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


@app.post("/api/sessions/{session_id}/frames/{frame_index}/apply-model-fix")
def apply_model_fix(
    session_id: str,
    frame_index: int,
    payload: Optional[ApplyModelFixPayload] = None,
) -> Dict[str, Any]:
    detail = get_frame_detail(session_id, frame_index)
    if not detail:
        raise HTTPException(status_code=404, detail="Frame not found")

    human_kpts = detail.get("human_keypoints", {})
    model_kpts = detail.get("model_keypoints", {})
    if not model_kpts:
        raise HTTPException(status_code=400, detail="Không có tọa độ Model AI để áp dụng")

    target_points: set[int] = set()
    if payload and payload.point_ids:
        target_points = set(payload.point_ids)
    else:
        for v in detail.get("rule_violations", []):
            for p in v.get("points", []):
                target_points.add(int(p))
        quality_curr = evaluate_frame_quality(human_kpts, model_kpts)
        for pid_str, p_eval in quality_curr.get("point_evaluations", {}).items():
            if p_eval.get("exceeded"):
                target_points.add(int(pid_str))

    if not target_points:
        target_points = set(range(50))

    for pid in target_points:
        m = model_kpts.get(str(pid)) or model_kpts.get(pid)
        if m and m.get("x") is not None and m.get("y") is not None:
            key = pid if pid in human_kpts else str(pid)
            if key not in human_kpts:
                human_kpts[key] = {"id": pid, "state": "visible"}
            human_kpts[key]["x"] = float(m["x"])
            human_kpts[key]["y"] = float(m["y"])
            if human_kpts[key].get("state") == "outside":
                human_kpts[key]["state"] = "visible"

    w, h = 1280, 720
    img_p = Path(detail.get("image_path", ""))
    if img_p.exists():
        im = cv2.imread(str(img_p))
        if im is not None:
            h, w = im.shape[:2]

    violations_raw = check_all_rules(human_kpts, model_keypoints_by_id=model_kpts, image_width=w, image_height=h)
    violations = [violation_to_dict(v) for v in violations_raw]
    score_eval = evaluate_frame_quality(human_kpts, model_kpts)

    severity_weight = sum(
        3.0 if v.get("severity") == "critical"
        else 2.0 if v.get("severity") == "major"
        else 1.0
        for v in violations
    )

    update_frame_full(session_id, frame_index, {
        "human_keypoints": human_kpts,
        "iod": float(score_eval.get("iod", detail.get("iod", 100.0))),
        "nme": float(score_eval.get("nme", 0.0)),
        "rule_violations": violations,
        "severity_score": float(severity_weight),
        "error_count": len(violations),
        "status": "fixed",
        "case_type": score_eval.get("case_type", "normal"),
        "ai_reliability": float(score_eval.get("ai_reliability", 0.95)),
        "case_label": score_eval.get("case_label", "Bình thường"),
        "case_description": score_eval.get("case_description", ""),
    })
    recalculate_session_stats(session_id)

    return get_frame_detail(session_id, frame_index)


@app.post("/api/sessions/{session_id}/batch-apply-model-fix")
def batch_apply_model_fix(
    session_id: str,
    payload: Optional[BatchApplyModelFixPayload] = None,
) -> Dict[str, Any]:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    only_violated = payload.only_violated if payload is not None else True
    min_nme = payload.min_nme if payload is not None else 0.035

    frames_all = get_all_frames_for_export(session_id)
    updated_indices: List[int] = []

    for f in frames_all:
        f_idx = f["frame_index"]
        detail = get_frame_detail(session_id, f_idx)
        if not detail:
            continue

        violations = detail.get("rule_violations", [])
        nme = float(detail.get("nme", 0.0))
        model_kpts = detail.get("model_keypoints", {})

        if not model_kpts:
            continue

        if only_violated:
            has_issues = len(violations) > 0 or nme > min_nme
            if not has_issues:
                continue

        human_kpts = detail.get("human_keypoints", {})
        target_points: set[int] = set()

        for v in violations:
            for p in v.get("points", []):
                target_points.add(int(p))

        quality_curr = evaluate_frame_quality(human_kpts, model_kpts)
        for pid_str, p_eval in quality_curr.get("point_evaluations", {}).items():
            if p_eval.get("exceeded"):
                target_points.add(int(pid_str))

        if not target_points:
            target_points = set(range(50))

        for pid in target_points:
            m = model_kpts.get(str(pid)) or model_kpts.get(pid)
            if m and m.get("x") is not None and m.get("y") is not None:
                key = pid if pid in human_kpts else str(pid)
                if key not in human_kpts:
                    human_kpts[key] = {"id": pid, "state": "visible"}
                human_kpts[key]["x"] = float(m["x"])
                human_kpts[key]["y"] = float(m["y"])
                if human_kpts[key].get("state") == "outside":
                    human_kpts[key]["state"] = "visible"

        w, h = 1280, 720
        img_p = Path(detail.get("image_path", ""))
        if img_p.exists():
            im = cv2.imread(str(img_p))
            if im is not None:
                h, w = im.shape[:2]

        violations_raw = check_all_rules(human_kpts, model_keypoints_by_id=model_kpts, image_width=w, image_height=h)
        new_violations = [violation_to_dict(v) for v in violations_raw]
        score_eval = evaluate_frame_quality(human_kpts, model_kpts)

        severity_weight = sum(
            3.0 if v.get("severity") == "critical"
            else 2.0 if v.get("severity") == "major"
            else 1.0
            for v in new_violations
        )

        update_frame_full(session_id, f_idx, {
            "human_keypoints": human_kpts,
            "iod": float(score_eval.get("iod", detail.get("iod", 100.0))),
            "nme": float(score_eval.get("nme", 0.0)),
            "rule_violations": new_violations,
            "severity_score": float(severity_weight),
            "error_count": len(new_violations),
            "status": "fixed",
            "case_type": score_eval.get("case_type", "normal"),
            "ai_reliability": float(score_eval.get("ai_reliability", 0.95)),
            "case_label": score_eval.get("case_label", "Bình thường"),
            "case_description": score_eval.get("case_description", ""),
        })
        updated_indices.append(f_idx)

    stats = recalculate_session_stats(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "updated_count": len(updated_indices),
        "updated_frame_indices": updated_indices,
        "total_frames": len(frames_all),
        "session_stats": stats,
    }


@app.get("/api/sessions/{session_id}/export/cvat-xml")
def export_cvat_xml(session_id: str) -> Response:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    frames = get_all_frames_for_export(session_id)
    xml_content = generate_cvat_xml(frames)
    return Response(
        content=xml_content,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="cvat_face50_cleaned_{session_id}.xml"'}
    )


@app.get("/api/sessions/{session_id}/export/json")
def export_json(session_id: str) -> Response:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    frames = get_all_frames_for_export(session_id)
    json_content = generate_cleaned_json(sess, frames)
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="face50_cleaned_{session_id}.json"'}
    )


@app.post("/api/demo")
@app.post("/api/demo/create-sample")
def create_sample_demo() -> Dict[str, Any]:
    """Generates a demo session with 5 synthetic face frames illustrating QA rules."""
    session_id = f"demo_{uuid.uuid4().hex[:6]}"
    session_folder = UPLOADS_DIR / session_id
    images_folder = session_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # Base valid coordinates template
    def make_valid_face():
        kpts = {}
        for i, x in enumerate([480.0, 495.0, 510.0, 525.0, 540.0]):
            kpts[i] = {"id": i, "x": x, "y": 260.0, "state": "visible"}
        for i, x in enumerate([560.0, 575.0, 590.0, 605.0, 620.0], start=5):
            kpts[i] = {"id": i, "x": x, "y": 260.0, "state": "visible"}
        for i, y in enumerate([280.0, 310.0, 340.0, 370.0], start=10):
            kpts[i] = {"id": i, "x": 550.0, "y": y, "state": "visible"}
        # Left eye
        kpts[14] = {"id": 14, "x": 470.0, "y": 300.0, "state": "visible"}
        kpts[15] = {"id": 15, "x": 485.0, "y": 292.0, "state": "visible"}
        kpts[16] = {"id": 16, "x": 500.0, "y": 290.0, "state": "visible"}
        kpts[17] = {"id": 17, "x": 515.0, "y": 293.0, "state": "visible"}
        kpts[18] = {"id": 18, "x": 530.0, "y": 300.0, "state": "visible"}
        kpts[19] = {"id": 19, "x": 515.0, "y": 307.0, "state": "visible"}
        kpts[20] = {"id": 20, "x": 500.0, "y": 310.0, "state": "visible"}
        kpts[21] = {"id": 21, "x": 485.0, "y": 308.0, "state": "visible"}
        # Right eye
        kpts[22] = {"id": 22, "x": 570.0, "y": 300.0, "state": "visible"}
        kpts[23] = {"id": 23, "x": 585.0, "y": 293.0, "state": "visible"}
        kpts[24] = {"id": 24, "x": 600.0, "y": 290.0, "state": "visible"}
        kpts[25] = {"id": 25, "x": 615.0, "y": 292.0, "state": "visible"}
        kpts[26] = {"id": 26, "x": 630.0, "y": 300.0, "state": "visible"}
        kpts[27] = {"id": 27, "x": 615.0, "y": 308.0, "state": "visible"}
        kpts[28] = {"id": 28, "x": 600.0, "y": 310.0, "state": "visible"}
        kpts[29] = {"id": 29, "x": 585.0, "y": 307.0, "state": "visible"}
        # Outer lips
        kpts[30] = {"id": 30, "x": 500.0, "y": 420.0, "state": "visible"}
        kpts[31] = {"id": 31, "x": 520.0, "y": 405.0, "state": "visible"}
        kpts[32] = {"id": 32, "x": 535.0, "y": 402.0, "state": "visible"}
        kpts[33] = {"id": 33, "x": 550.0, "y": 406.0, "state": "visible"}
        kpts[34] = {"id": 34, "x": 565.0, "y": 402.0, "state": "visible"}
        kpts[35] = {"id": 35, "x": 580.0, "y": 405.0, "state": "visible"}
        kpts[36] = {"id": 36, "x": 600.0, "y": 420.0, "state": "visible"}
        kpts[37] = {"id": 37, "x": 580.0, "y": 440.0, "state": "visible"}
        kpts[38] = {"id": 38, "x": 565.0, "y": 445.0, "state": "visible"}
        kpts[39] = {"id": 39, "x": 550.0, "y": 448.0, "state": "visible"}
        kpts[40] = {"id": 40, "x": 535.0, "y": 445.0, "state": "visible"}
        kpts[41] = {"id": 41, "x": 520.0, "y": 440.0, "state": "visible"}
        # Inner lips
        kpts[42] = {"id": 42, "x": 515.0, "y": 420.0, "state": "visible"}
        kpts[43] = {"id": 43, "x": 535.0, "y": 415.0, "state": "visible"}
        kpts[44] = {"id": 44, "x": 550.0, "y": 416.0, "state": "visible"}
        kpts[45] = {"id": 45, "x": 565.0, "y": 415.0, "state": "visible"}
        kpts[46] = {"id": 46, "x": 585.0, "y": 420.0, "state": "visible"}
        kpts[47] = {"id": 47, "x": 565.0, "y": 426.0, "state": "visible"}
        kpts[48] = {"id": 48, "x": 550.0, "y": 428.0, "state": "visible"}
        kpts[49] = {"id": 49, "x": 535.0, "y": 426.0, "state": "visible"}
        return kpts

    demo_samples_dir = BASE_DIR / "data" / "demo_samples"

    demo_cases = [
        (
            "frame_0000.jpg",
            "driver_2_passenger.png",
            "Frame 00: Chuẩn QA VinFast VF-50 (Pass - Khóa đúng tài xế)",
            {
                "modify": lambda h: [
                    h[0].update({"state": "occluded"}),
                ]
            },
        ),
        (
            "frame_0001.jpg",
            "driver_1_sleepy.png",
            "Frame 01: Lộn mí mắt (R06) & Mắt tự cắt (R07) - Tài xế ngủ gật",
            {
                "modify": lambda h: [
                    h.update({
                        16: {"id": 16, "x": h[20]["x"], "y": h[20]["y"] + 6.0, "state": "visible"},
                        20: {"id": 20, "x": h[16]["x"], "y": h[16]["y"] - 6.0, "state": "visible"},
                    })
                ]
            },
        ),
        (
            "frame_0002.jpg",
            "driver_0_shocked.png",
            "Frame 02: Bờ môi trong trồi ra ngoài bờ môi ngoài (R09) - Tài xế ngạc nhiên há miệng",
            {
                "modify": lambda h: [
                    h.update({
                        44: {"id": 44, "x": h[33]["x"], "y": h[33]["y"] - 12.0, "state": "visible"},
                        42: {"id": 42, "x": h[30]["x"] - 8.0, "y": h[30]["y"], "state": "visible"},
                    })
                ]
            },
        ),
        (
            "frame_0003.jpg",
            "driver_3_smiling.png",
            "Frame 03: Sống mũi chia không đều (R08) - Kéo điểm 13 xuống chóp mũi",
            {
                "modify": lambda h: [
                    h[9].update({"state": "occluded"}),
                    h.update({
                        13: {"id": 13, "x": h[13]["x"], "y": h[13]["y"] + 10.0, "state": "visible"},
                        11: {"id": 11, "x": h[11]["x"], "y": (h[10]["y"] + h[12]["y"]) / 2.0, "state": "visible"},
                    })
                ]
            },
        ),
        (
            "frame_0004.jpg",
            "driver_4_sunglasses.png",
            "Frame 04: Kính râm che mắt (R10 Ngưỡng ẩn bộ phận) & Ước lượng Occluded lệch (R14)",
            {
                "modify": lambda h: [
                    [h[i].update({"state": "occluded"}) for i in [16, 17, 18, 19, 20, 21]],
                    h[0].update({"x": h[0]["x"] - 35.0, "y": h[0]["y"] - 25.0, "state": "occluded"}),
                    h[1].update({"x": h[1]["x"] - 30.0, "y": h[1]["y"] - 25.0, "state": "occluded"}),
                ]
            },
        ),
    ]

    frames_to_save = []
    total_issues = 0
    detector = get_detector()

    for idx, (fname, sample_file, desc, mod) in enumerate(demo_cases):
        img_path = images_folder / fname
        src_sample = demo_samples_dir / sample_file

        if src_sample.exists():
            img = cv2.imread(str(src_sample))
            cv2.imwrite(str(img_path), img)
            det = detector.detect_landmarks(img)
            if det:
                vf50_det, _ = det
                human_kpts = {
                    i: {"id": i, "x": float(x), "y": float(y), "state": "visible"}
                    for i, (x, y) in enumerate(vf50_det)
                }
                model_kpts = copy.deepcopy(human_kpts)
                if "modify" in mod:
                    mod["modify"](human_kpts)
            else:
                human_kpts = make_valid_face()
                model_kpts = make_valid_face()
        else:
            img = np.full((720, 1280, 3), (245, 247, 250), dtype=np.uint8)
            cv2.putText(img, f"DEMO VF-50: {desc}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
            cv2.imwrite(str(img_path), img)
            human_kpts = make_valid_face()
            if "modify" in mod:
                mod["modify"](human_kpts)
            model_kpts = make_valid_face()

        violations_raw = check_all_rules(
            human_kpts,
            model_keypoints_by_id=model_kpts,
            image_width=img.shape[1],
            image_height=img.shape[0]
        )
        violations = [violation_to_dict(v) for v in violations_raw]
        score_eval = evaluate_frame_quality(human_kpts, model_kpts)

        err_count = len(violations)
        total_issues += err_count

        severity_weight = sum(
            3.0 if v.get("severity") == "critical"
            else 2.0 if v.get("severity") == "major"
            else 1.0
            for v in violations
        )

        frames_to_save.append({
            "session_id": session_id,
            "frame_index": idx,
            "image_filename": fname,
            "image_path": str(img_path),
            "human_keypoints": human_kpts,
            "model_keypoints": model_kpts,
            "iod": float(score_eval.get("iod", 100.0)),
            "nme": float(score_eval.get("nme", 0.0)),
            "rule_violations": violations,
            "severity_score": float(severity_weight),
            "error_count": err_count,
            "status": "pending",
            "case_type": score_eval.get("case_type", "normal"),
            "ai_reliability": float(score_eval.get("ai_reliability", 0.95)),
            "case_label": score_eval.get("case_label", "Bình thường"),
            "case_description": score_eval.get("case_description", ""),
        })

    save_frames_batch(frames_to_save)
    save_session(
        session_id=session_id,
        name=f"Interactive Demo (5 Frames, {total_issues} Issues)",
        created_at=datetime.now().isoformat(),
        total_frames=len(frames_to_save),
        total_issues=total_issues,
        avg_nme=0.018,
    )

    return {
        "session_id": session_id,
        "name": f"Interactive Demo (5 Frames, {total_issues} Issues)",
        "total_frames": 5,
        "total_issues": total_issues,
        "message": "Demo session created successfully! You can inspect frames immediately.",
    }
