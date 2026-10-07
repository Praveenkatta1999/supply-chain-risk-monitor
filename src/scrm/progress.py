"""Command-line logging: full JSON logs to a file, one-line progress in the terminal.

The scripts call ``setup_cli_logging(run_id, verbose=...)``. Every log record, from this
project and its libraries, goes as JSON to ``data/logs/<run_id>.jsonl``. The terminal
(stderr) shows only short lines for the pipeline's milestones (sites chosen, triage
counts, verdicts, investigations, review rounds) and any errors; ``verbose=True`` shows
the full JSON log there instead.
"""

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from scrm.telemetry import JsonFormatter

LOG_DIR = Path("data/logs")

Fields = dict[str, Any]


def _sites_chosen(f: Fields) -> str | None:
    args = f.get("tool_args") or {}
    if f.get("tool") != "run_risk_brief" or not f.get("allowed") or "site_ids" not in args:
        return None
    sites = ", ".join(args["site_ids"])
    return f"Sites chosen: {sites}; {args.get('start_date')} to {args.get('end_date')}"


def _verdict(f: Fields) -> str:
    scope = f" ({f['scope']})" if f.get("verdict") == "yes" and f.get("scope") else ""
    story = f.get("title") or f.get("url")
    return f"{f.get('site_id', '?')}: verifier {f.get('verdict')}{scope}: {story}"


# Log message -> one-line rendering of its fields (None to stay quiet for that record).
PROGRESS: dict[str, Callable[[Fields], str | None]] = {
    "root_agent.start": lambda f: f"Question: {f['question']}",
    "root_agent.tool_call": _sites_chosen,
    "orchestrator.start": lambda f: (
        f"Running {len(f['site_ids'])} sites: {', '.join(f['site_ids'])}"
    ),
    "triage.done": lambda f: (
        f"{f['site_id']}: triage kept {f['selected']} of {f['considered']} stories"
    ),
    "verifier.done": _verdict,
    "report.drafted": lambda f: f"Drafted {f['items']} claims",
    "investigator.done": lambda f: (
        f"Investigated {f['subject_id']} ({f['subject'].replace('_', ' ')}): "
        f"{f['status']}, {f['confidence']} confidence, {f['tool_calls']} tool calls"
    ),
    "reviewer.round": lambda f: f"Review round {f['round']}: {f['corrections']} claims sent back",
    "orchestrator.done": lambda f: (
        f"Brief ready: {f['items']} claims, {f['investigations']} investigations"
    ),
}


def progress_line(record: logging.LogRecord) -> str | None:
    """The terminal line for a log record, or None if it should not be shown."""
    if record.levelno >= logging.ERROR:
        return f"Error: {record.getMessage()} (details in the log file)"
    render = PROGRESS.get(record.getMessage())
    if render is None:
        return None
    try:
        return render(record.__dict__)
    except (KeyError, TypeError):  # a malformed record must never break the run
        return None


class ProgressHandler(logging.Handler):
    """Writes one short line per pipeline milestone to a text stream."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self.stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        line = progress_line(record)
        if line:
            self.stream.write(line + "\n")
            self.stream.flush()


def setup_cli_logging(
    run_id: str,
    *,
    verbose: bool = False,
    level: str = "INFO",
    log_dir: Path = LOG_DIR,
    stream: TextIO | None = None,
) -> Path:
    """Send all logs to ``log_dir/<run_id>.jsonl`` and progress (or, if verbose, the full
    JSON log) to ``stream`` (stderr by default). Returns the log file path."""
    stream = stream or sys.stderr
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.jsonl"
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)
    to_file = logging.FileHandler(path, encoding="utf-8")
    to_file.setFormatter(JsonFormatter())
    root.addHandler(to_file)
    if verbose:
        to_terminal: logging.Handler = logging.StreamHandler(stream)
        to_terminal.setFormatter(JsonFormatter())
    else:
        to_terminal = ProgressHandler(stream)
    root.addHandler(to_terminal)
    return path
