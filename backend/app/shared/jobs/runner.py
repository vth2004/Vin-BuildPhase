from __future__ import annotations

import logging
from collections.abc import Callable

from app.core.database import db

logger = logging.getLogger(__name__)
RunHandler = Callable[[str], None]
_handlers: dict[str, RunHandler] = {}


def register_run_handler(mode: str, handler: RunHandler) -> None:
    _handlers[mode] = handler


def dispatch_run(run_id: str) -> None:
    with db() as conn:
        run = conn.execute("SELECT mode FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return
    handler = _handlers.get(run["mode"])
    if handler is None:
        with db() as conn:
            conn.execute(
                "UPDATE runs SET status = 'failed', error_message = ? WHERE id = ?",
                (f"No run handler registered for mode: {run['mode']}", run_id),
            )
        return
    try:
        handler(run_id)
    except Exception as error:
        logger.exception("Run %s failed", run_id)
        with db() as conn:
            conn.execute(
                "UPDATE runs SET status = 'failed', error_message = ? WHERE id = ?",
                (str(error)[:2000], run_id),
            )
