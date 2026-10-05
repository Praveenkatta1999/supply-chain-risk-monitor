"""Structured JSON logging with a run ID that follows one brief end to end.

Usage:
    configure_logging()
    ctx = RunContext.new()
    log = get_logger(__name__)
    with ctx.bind():
        log.info("query.start", extra={"site_count": 3})

Every record emitted inside ``ctx.bind()`` carries ``run_id``, so all agent logs for one
brief can be filtered with a single field in Cloud Logging.
"""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)

# Attributes every LogRecord has; anything else came from `extra=` and is logged as a field.
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def new_run_id() -> str:
    """Return a short, unique run ID."""
    return uuid.uuid4().hex[:12]


def current_run_id() -> str | None:
    """Return the run ID bound to the current context, if any."""
    return _run_id.get()


@dataclass(frozen=True)
class RunContext:
    """Per-run state passed to every agent."""

    run_id: str = field(default_factory=new_run_id)

    @classmethod
    def new(cls) -> "RunContext":
        return cls()

    @contextmanager
    def bind(self) -> Iterator["RunContext"]:
        """Attach this run ID to all log records emitted inside the block."""
        token = _run_id.set(self.run_id)
        try:
            yield self
        finally:
            _run_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": current_run_id(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
