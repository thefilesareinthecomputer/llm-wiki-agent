"""Structured JSON debug logging for LLM Wiki Agent.

Writes JSONL (one JSON object per line) to /app/logs/ for easy parsing.
Each log entry has: timestamp, module, level, event, and context fields.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        # Merge any extra fields passed via log.info("...", extra={...})
        if hasattr(record, "event"):
            entry["event"] = record.event
        if hasattr(record, "data"):
            entry["data"] = record.data
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
        return json.dumps(entry, default=str, ensure_ascii=False)


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create or get a logger that writes JSONL to /app/logs/{name}.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"llm_wiki_agent.{name}")

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)

    handler = logging.FileHandler(LOG_DIR / f"{name}.log", mode="a")
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False  # Don't double-log to root

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing llm_wiki_agent logger (or create with defaults)."""
    return setup_logger(name)


# Convenience: log with structured data
def log_event(logger: logging.Logger, event: str, **kwargs):
    """Log a structured event with extra data fields."""
    logger.info(event, extra={"event": event, "data": kwargs})