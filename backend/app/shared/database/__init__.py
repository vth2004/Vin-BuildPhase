"""Stable database boundary used by feature packages."""

from app.core.database import db, init_db

__all__ = ["db", "init_db"]
