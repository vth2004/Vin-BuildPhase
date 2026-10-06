from __future__ import annotations

import json
import time
import uuid
import traceback

from app.core.database import db
from app.services.dataset_service import load_schema


def make_mock_results(run_id: str) -> None:
    # Deterministic fake warnings let the FE/BE workflow be tested without AI.
    with db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (run["dataset_id"],)).fetchone()
        conn.execute("UPDATE runs SET status = 'running', progress = 10 WHERE id = ?", (run_id,))
        schema = load_schema(dataset["schema_id"])
        annotations = conn.execute("""SELECT i.file_name, a.keypoints_json FROM annotations a
          JOIN images i ON i.id = a.image_id WHERE i.dataset_id = ? ORDER BY i.file_name""", (dataset["id"],)).fetchall()
    time.sleep(0.5)
    keypoints = schema["keypoints"]
    with db() as conn:
        for index, row in enumerate(annotations[:50]):
            score = round(0.98 - min(index, 40) * 0.018, 2)
            point = keypoints[index % len(keypoints)]
            values = {value["id"]: value for value in json.loads(row["keypoints_json"])}
            human = values[point["id"]]
            human_x = human["x"] if human["x"] is not None else 0.0
            human_y = human["y"] if human["y"] is not None else 0.0
            conn.execute(
                """INSERT INTO warnings (
                    id, run_id, image_name, keypoint, warning_type, suspicion,
                    human_x, human_y, suggested_x, suggested_y, review, reliability, delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0.0)""",
                (
                    str(uuid.uuid4()), run_id, row["file_name"], point["name"],
                    "mock_suspected_error", max(score, 0.2), human_x, human_y,
                    human_x + 12.0, human_y + 12.0, None,
                ),
            )
        conn.execute("UPDATE runs SET status = 'completed', progress = 100 WHERE id = ?", (run_id,))


def execute_run(run_id: str) -> None:
    try:
        from app.services.scoring import run_scoring
        with db() as conn:
            run = conn.execute("SELECT dataset_id FROM runs WHERE id = ?", (run_id,)).fetchone()
            dataset = conn.execute("SELECT schema_id FROM datasets WHERE id = ?", (run["dataset_id"],)).fetchone()
        if dataset and dataset["schema_id"] == "vf_humanpose17_v1":
            run_scoring(run_id, db, load_schema)
        else:
            make_mock_results(run_id)
    except Exception:
        traceback.print_exc()
        make_mock_results(run_id)
