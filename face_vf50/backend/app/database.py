"""SQLite database interface for VinFast VF-50 Face QA."""

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "face_vf50.sqlite3"


def init_db() -> None:
    """Initialize SQLite tables and indexes."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                total_frames INTEGER DEFAULT 0,
                total_issues INTEGER DEFAULT 0,
                avg_nme REAL DEFAULT 0.0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                has_human_labels INTEGER DEFAULT 0,
                has_model_prediction INTEGER DEFAULT 0,
                human_keypoints_json TEXT,
                model_keypoints_json TEXT,
                iod REAL DEFAULT 0.0,
                nme REAL DEFAULT 0.0,
                rule_violations_json TEXT,
                severity_score REAL DEFAULT 0.0,
                error_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_frame_idx ON frames(session_id, frame_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_severity ON frames(session_id, severity_score DESC)")

        # Migration for hard cases and AI reliability
        cursor.execute("PRAGMA table_info(frames)")
        cols = [r[1] for r in cursor.fetchall()]
        if "case_type" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN case_type TEXT DEFAULT 'normal'")
        if "ai_reliability" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN ai_reliability REAL DEFAULT 1.0")
        if "case_label" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN case_label TEXT DEFAULT ''")
        if "case_description" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN case_description TEXT DEFAULT ''")
        if "initial_human_keypoints_json" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN initial_human_keypoints_json TEXT")
        if "point_sources_json" not in cols:
            cursor.execute("ALTER TABLE frames ADD COLUMN point_sources_json TEXT DEFAULT '{}'")

        conn.commit()



# Automatically initialize schema on module import
init_db()


def save_session(
    session_id: str,
    name: str,
    created_at: str,
    total_frames: int,
    total_issues: int,
    avg_nme: float
) -> None:
    """Save or update session metadata."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, name, created_at, total_frames, total_issues, avg_nme)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, name, created_at, total_frames, total_issues, avg_nme))
        conn.commit()


def save_frames_batch(frames: List[Dict[str, Any]]) -> None:
    """Batch insert frame records."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO frames (
                session_id, frame_index, image_filename, image_path,
                has_human_labels, has_model_prediction,
                human_keypoints_json, model_keypoints_json,
                iod, nme, rule_violations_json,
                severity_score, error_count, status,
                case_type, ai_reliability, case_label, case_description,
                initial_human_keypoints_json, point_sources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                f["session_id"],
                f["frame_index"],
                f["image_filename"],
                f["image_path"],
                1 if f.get("human_keypoints") else 0,
                1 if f.get("model_keypoints") else 0,
                json.dumps(f.get("human_keypoints", {})),
                json.dumps(f.get("model_keypoints", {})),
                float(f.get("iod", 0.0)),
                float(f.get("nme", 0.0)),
                json.dumps(f.get("rule_violations", [])),
                float(f.get("severity_score", 0.0)),
                int(f.get("error_count", 0)),
                f.get("status", "pending"),
                f.get("case_type", "normal"),
                float(f.get("ai_reliability", 0.95)),
                f.get("case_label", "Bình thường"),
                f.get("case_description", ""),
                json.dumps(f.get("human_keypoints", {})),
                json.dumps({str(i): "human" for i in range(50)})
            )
            for f in frames
        ])
        conn.commit()


def get_all_sessions() -> List[Dict[str, Any]]:
    """Retrieve all sessions ordered by creation date."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_session_frames(
    session_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: str = "severity",
    filter_rule: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve paginated frames for a session."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build order clause
        if sort_by == "frame_index":
            order_clause = "frame_index ASC"
        elif sort_by == "nme":
            order_clause = "nme DESC"
        else:
            order_clause = "severity_score DESC, error_count DESC, frame_index ASC"

        where_clauses = ["session_id = ?"]
        params: List[Any] = [session_id]

        if filter_rule and filter_rule != "all":
            where_clauses.append("rule_violations_json LIKE ?")
            params.append(f"%{filter_rule}%")

        where_str = " AND ".join(where_clauses)

        # Count total
        count_query = f"SELECT COUNT(*) FROM frames WHERE {where_str}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        offset = (page - 1) * page_size
        query = f"""
            SELECT id, session_id, frame_index, image_filename, iod, nme,
                   severity_score, error_count, status, rule_violations_json,
                   case_type, ai_reliability, case_label
            FROM frames
            WHERE {where_str}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()

        items = []
        for r in rows:
            d = dict(r)
            d["rule_violations"] = json.loads(d["rule_violations_json"]) if d["rule_violations_json"] else []
            del d["rule_violations_json"]
            items.append(d)

        return {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "items": items
        }


def get_frame_detail(session_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific frame."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM frames WHERE session_id = ? AND frame_index = ?
        """, (session_id, frame_index))
        row = cursor.fetchone()
        if not row:
            return None

        d = dict(row)
        d["human_keypoints"] = json.loads(d["human_keypoints_json"]) if d.get("human_keypoints_json") else {}
        d["model_keypoints"] = json.loads(d["model_keypoints_json"]) if d.get("model_keypoints_json") else {}
        d["rule_violations"] = json.loads(d["rule_violations_json"]) if d.get("rule_violations_json") else []

        if d.get("initial_human_keypoints_json"):
            d["initial_human_keypoints"] = json.loads(d["initial_human_keypoints_json"])
        else:
            d["initial_human_keypoints"] = copy.deepcopy(d["human_keypoints"])

        if d.get("point_sources_json"):
            d["point_sources"] = json.loads(d["point_sources_json"])
        else:
            d["point_sources"] = {str(i): "human" for i in range(50)}

        del d["human_keypoints_json"]
        del d["model_keypoints_json"]
        del d["rule_violations_json"]
        if "initial_human_keypoints_json" in d:
            del d["initial_human_keypoints_json"]
        if "point_sources_json" in d:
            del d["point_sources_json"]
        return d


def update_frame_status(session_id: str, frame_index: int, status: str) -> bool:
    """Update review status of a frame (e.g. pending, reviewed, fixed)."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE frames SET status = ? WHERE session_id = ? AND frame_index = ?
        """, (status, session_id, frame_index))
        conn.commit()
        return cursor.rowcount > 0


def update_frame_full(session_id: str, frame_index: int, updates: Dict[str, Any]) -> bool:
    """Update full frame data including keypoints, evaluations, and status."""
    fields = []
    values = []

    for k, v in updates.items():
        if k in ("human_keypoints", "model_keypoints", "rule_violations", "initial_human_keypoints", "point_sources"):
            fields.append(f"{k}_json = ?")
            values.append(json.dumps(v))
        else:
            fields.append(f"{k} = ?")
            values.append(v)

    if not fields:
        return False

    values.extend([session_id, frame_index])
    sql = f"UPDATE frames SET {', '.join(fields)} WHERE session_id = ? AND frame_index = ?"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        return cursor.rowcount > 0


def get_all_frame_evaluations(session_id: str) -> List[Dict[str, Any]]:
    """Fetch all frame evaluations for report export."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT frame_index, image_filename, iod, nme, severity_score, error_count, rule_violations_json,
                   case_type, ai_reliability, case_label
            FROM frames
            WHERE session_id = ?
            ORDER BY severity_score DESC, frame_index ASC
        """, (session_id,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["rule_violations"] = json.loads(d["rule_violations_json"]) if d["rule_violations_json"] else []
            del d["rule_violations_json"]
            results.append(d)
        return results


def get_all_frames_for_export(session_id: str) -> List[Dict[str, Any]]:
    """Fetch all frame details including keypoints for XML/JSON annotation export."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT frame_index, image_filename, image_path, human_keypoints_json, model_keypoints_json, status, nme, iod
            FROM frames
            WHERE session_id = ?
            ORDER BY frame_index ASC
        """, (session_id,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["human_keypoints"] = json.loads(d["human_keypoints_json"]) if d["human_keypoints_json"] else {}
            d["model_keypoints"] = json.loads(d["model_keypoints_json"]) if d["model_keypoints_json"] else {}
            del d["human_keypoints_json"]
            del d["model_keypoints_json"]
            results.append(d)
        return results


def recalculate_session_stats(session_id: str) -> Dict[str, Any]:
    """Recalculate and update session summary statistics based on current frames."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*), SUM(error_count), AVG(nme)
            FROM frames
            WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        total_frames = int(row[0] or 0)
        total_issues = int(row[1] or 0)
        avg_nme = float(row[2] or 0.0)

        cursor.execute("""
            UPDATE sessions
            SET total_frames = ?, total_issues = ?, avg_nme = ?
            WHERE session_id = ?
        """, (total_frames, total_issues, avg_nme, session_id))
        conn.commit()

        return {
            "total_frames": total_frames,
            "total_issues": total_issues,
            "avg_nme": round(avg_nme, 4),
        }

