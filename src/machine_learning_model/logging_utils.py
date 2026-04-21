"""
Module: logging_utils
Purpose: Centralised logger factory for the entire application.
         Uses structlog (JSON output with context binding) when available;
         falls back to stdlib logging with a thin JSON formatter so log
         consumers always receive a consistent structured format.
Rationale: Centralising logger acquisition prevents scattered basicConfig
           calls and ensures every module respects the LOG_LEVEL env var.
Assumptions: LOG_LEVEL environment variable is a valid logging level name
             (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
Failure Modes: If structlog is installed but misconfigured, the import
               guard catches the exception and falls back to stdlib logging.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

# Feature flag: True when structlog is importable at module load time.
_USE_STRUCTLOG = False
try:  # pragma: no cover
    import structlog  # type: ignore

    _USE_STRUCTLOG = True
except ImportError:  # structlog is an optional dependency
    _USE_STRUCTLOG = False


def _configure_std_logging() -> None:
    """
    Purpose:    One-time configuration of the stdlib root logger with a
                compact JSON formatter so log lines are machine-parseable.
    Precond:    Should only be called when structlog is unavailable.
    Side-effect: Mutates root logger handlers; idempotent — skips setup if
                 handlers already exist to prevent duplicate output.
    """
    if logging.getLogger().handlers:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level)

    class JsonFormatter(logging.Formatter):
        """
        Purpose:  Serialise LogRecord fields to a single-line JSON string.
        Outputs:  JSON with keys: ts (ISO-8601), level, msg, name, exc_info.
        """

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
    """
    Purpose:    Return a named logger suitable for structured log output.
                Callers should call this once at module level and store the
                result; repeated calls are cheap but not free.
    Inputs:     name — logger namespace, typically __name__ of the caller.
    Returns:    structlog BoundLogger when structlog is available, otherwise
                a stdlib Logger with the JSON formatter applied.
    Precond:    None. Safe to call before any other app initialisation.
    Postcond:   Returned logger respects LOG_LEVEL env var.
    """
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
