import json
import logging

from scrm.telemetry import JsonFormatter, RunContext, current_run_id


def test_run_id_is_bound_and_logged():
    ctx = RunContext.new()
    record = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "site_count": 2})
    with ctx.bind():
        assert current_run_id() == ctx.run_id
        payload = json.loads(JsonFormatter().format(record))
    assert payload["run_id"] == ctx.run_id
    assert payload["site_count"] == 2
    assert current_run_id() is None
