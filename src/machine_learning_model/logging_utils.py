"""Structured logging helper.

Uses structlog if available; otherwise falls back to standard logging with
JSON-like formatting. Central place to get a logger to keep consistency.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

_USE_STRUCTLOG = False
try:  # pragma: no cover
    import structlog  # type: ignore

    _USE_STRUCTLOG = True
except ImportError:  # structlog optional
    _USE_STRUCTLOG = False


def _configure_std_logging():
    if logging.getLogger().handlers:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            payload = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "msg": record.getMessage(),
                "name": record.name,
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    for handler in logging.getLogger().handlers:
        handler.setFormatter(JsonFormatter())


def get_logger(name: str = "mlapp"):
    if _USE_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
            ),
        )
        return structlog.get_logger(name)
    _configure_std_logging()
    return logging.getLogger(name)


__all__ = ["get_logger"]
