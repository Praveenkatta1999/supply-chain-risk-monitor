import io
import json
import logging

import pytest

from scrm.progress import progress_line, setup_cli_logging
from scrm.telemetry import RunContext


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def record(message: str, level: int = logging.INFO, **fields) -> logging.LogRecord:
    return logging.makeLogRecord({"msg": message, "levelno": level, **fields})


def test_milestones_become_one_line_updates():
    assert (
        progress_line(
            record(
                "root_agent.tool_call",
                tool="run_risk_brief",
                allowed=True,
                tool_args={"site_ids": ["S01", "S07"], "start_date": "2026-10-01",
                           "end_date": "2026-10-07"},
            )
        )
        == "Sites chosen: S01, S07; 2026-10-01 to 2026-10-07"
    )  # fmt: skip
    assert (
        progress_line(record("triage.done", site_id="S01", selected=6, considered=50))
        == "S01: triage kept 6 of 50 stories"
    )
    assert (
        progress_line(
            record("verifier.done", site_id="P05", verdict="yes", scope="indirect",
                   title="rhine water levels fall", url="https://x.example/a")
        )
        == "P05: verifier yes (indirect): rhine water levels fall"
    )  # fmt: skip
    assert progress_line(
        record("investigator.done", subject_id="I1", subject="claim", status="ongoing",
               confidence="high", tool_calls=7)
    ) == "Investigated I1 (claim): ongoing, high confidence, 7 tool calls"  # fmt: skip


def test_other_records_stay_quiet_but_errors_are_shown():
    assert progress_line(record("bigquery.execute", estimated_bytes=10)) is None
    assert progress_line(record("root_agent.tool_call", tool="list_sites", allowed=True)) is None
    assert progress_line(record("triage.done")) is None  # malformed: no crash, no line
    assert progress_line(record("verifier.model_failed", logging.ERROR)).startswith("Error:")


def test_full_json_goes_to_the_file_and_progress_to_the_terminal(tmp_path):
    terminal = io.StringIO()
    ctx = RunContext.new()
    path = setup_cli_logging(ctx.run_id, log_dir=tmp_path, stream=terminal)
    log = logging.getLogger("scrm.test")
    with ctx.bind():
        log.info("triage.done", extra={"site_id": "S01", "selected": 2, "considered": 9})
        log.info("bigquery.execute", extra={"estimated_bytes": 10})
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert terminal.getvalue() == "S01: triage kept 2 of 9 stories\n"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["message"] for r in lines] == ["triage.done", "bigquery.execute"]
    assert {r["run_id"] for r in lines} == {ctx.run_id}
    assert path.name == f"{ctx.run_id}.jsonl"


def test_verbose_shows_the_full_json_log_in_the_terminal(tmp_path):
    terminal = io.StringIO()
    setup_cli_logging("r1", verbose=True, log_dir=tmp_path, stream=terminal)
    logging.getLogger("scrm.test").info("bigquery.execute", extra={"estimated_bytes": 10})
    assert json.loads(terminal.getvalue())["message"] == "bigquery.execute"
