"""Ordered schema initialization; SQL order matches the original database initializer."""
import sqlite3
import time
from . import base_schema, sequence_schema, face_schema, face_metrics
from .versions import MIGRATION_VERSIONS


def apply_migrations(conn: sqlite3.Connection) -> None:
    base_schema.apply(conn)
    sequence_schema.apply(conn)
    applied_versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    face_schema.apply(conn, applied_versions)
    face_metrics.apply(conn, applied_versions)
    for version in MIGRATION_VERSIONS[2:]:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, int(time.time())),
        )
