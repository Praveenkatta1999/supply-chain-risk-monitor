"""The dry-run guard is the cost and safety backstop for the whole project."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from scrm.tools.bigquery_tool import (
    BigQueryTool,
    DatasetNotAllowedError,
    QueryTooExpensiveError,
    StatementNotAllowedError,
)

PROJECT = "dopl-507205"
SCRM_TABLE = SimpleNamespace(project=PROJECT, dataset_id="scrm", table_id="events_near_sites")
GDELT_TABLE = SimpleNamespace(project="gdelt-bq", dataset_id="gdeltv2", table_id="events")


@dataclass
class FakeClient:
    """Records every job config; dry runs report the configured estimate."""

    estimated_bytes: int
    tables: list = field(default_factory=lambda: [SCRM_TABLE])
    statement_type: str = "SELECT"
    calls: list = field(default_factory=list)
    page_sizes: list = field(default_factory=list)

    def query(self, sql, job_config, location):
        self.calls.append(job_config)
        if job_config.dry_run:
            return SimpleNamespace(
                total_bytes_processed=self.estimated_bytes,
                referenced_tables=self.tables,
                statement_type=self.statement_type,
            )

        def result(page_size=None):
            self.page_sizes.append(page_size)
            return [{"site_id": "SUP-01", "n": 3}]

        return SimpleNamespace(result=result, total_bytes_processed=self.estimated_bytes)

    @property
    def executed(self) -> list:
        return [c for c in self.calls if not c.dry_run]


def make_tool(client: FakeClient, limit: int = 1_000) -> BigQueryTool:
    return BigQueryTool(
        client, max_bytes_billed=limit, allowed_project=PROJECT, allowed_dataset="scrm"
    )


def test_runs_when_under_limit_after_dry_run():
    client = FakeClient(estimated_bytes=500)
    rows = make_tool(client).run_query("SELECT 1")
    assert rows == [{"site_id": "SUP-01", "n": 3}]
    assert [bool(c.dry_run) for c in client.calls] == [True, False]
    assert client.executed[0].maximum_bytes_billed == 1_000


def test_counts_bytes_processed():
    tool = make_tool(FakeClient(estimated_bytes=400))
    tool.run_query("SELECT 1")
    tool.run_query("SELECT 2")
    assert tool.bytes_processed == 800


def test_query_rows_is_guarded_and_passes_page_size():
    client = FakeClient(estimated_bytes=500)
    rows = make_tool(client).query_rows("SELECT 1", page_size=50)
    assert list(rows) == [{"site_id": "SUP-01", "n": 3}]
    assert client.page_sizes == [50]
    with pytest.raises(QueryTooExpensiveError):
        make_tool(FakeClient(estimated_bytes=1_001)).query_rows("SELECT * FROM big")


def test_refuses_and_never_executes_when_over_limit():
    client = FakeClient(estimated_bytes=1_001)
    with pytest.raises(QueryTooExpensiveError):
        make_tool(client).run_query("SELECT * FROM big")
    assert client.executed == []


def test_limit_is_configurable():
    client = FakeClient(estimated_bytes=5_000)
    make_tool(client, limit=10_000).run_query("SELECT 1")
    assert len(client.executed) == 1


def test_fork_has_its_own_byte_counter_and_same_limit():
    client = FakeClient(estimated_bytes=400)
    tool = make_tool(client)
    fork = tool.fork()
    fork.run_query("SELECT 1")
    assert (tool.bytes_processed, fork.bytes_processed) == (0, 400)
    assert fork.max_bytes_billed == tool.max_bytes_billed


def test_with_max_bytes_applies_new_limit():
    client = FakeClient(estimated_bytes=500)
    strict = make_tool(client, limit=10_000).with_max_bytes(100)
    with pytest.raises(QueryTooExpensiveError):
        strict.run_query("SELECT 1")
    assert client.executed == []


def test_refuses_tables_outside_scrm():
    client = FakeClient(estimated_bytes=10, tables=[SCRM_TABLE, GDELT_TABLE])
    with pytest.raises(DatasetNotAllowedError, match="gdelt-bq"):
        make_tool(client).run_query("SELECT * FROM `gdelt-bq.gdeltv2.events`")
    assert client.executed == []


def test_refuses_non_select_statements():
    client = FakeClient(estimated_bytes=10, statement_type="DELETE")
    with pytest.raises(StatementNotAllowedError):
        make_tool(client).run_query("DELETE FROM scrm.sites WHERE TRUE")
    assert client.executed == []


def test_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        make_tool(FakeClient(estimated_bytes=0), limit=0)
