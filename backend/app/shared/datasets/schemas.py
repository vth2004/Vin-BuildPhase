"""Schema metadata loading, independent of inference features."""
import json
from fastapi import HTTPException
from app.core.config import SCHEMAS_DIR


def load_schema(schema_id: str) -> dict:
    path = SCHEMAS_DIR / f"{schema_id}.json"
    if not path.is_file():
        raise HTTPException(400, f"Unsupported schema_id: {schema_id}")
    return json.loads(path.read_text(encoding="utf-8"))
