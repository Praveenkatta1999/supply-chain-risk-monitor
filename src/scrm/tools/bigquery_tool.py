"""Guarded BigQuery access. Every query goes through here.

Two hard rules are enforced on every call, using the dry run's own metadata rather than
parsing SQL text:

1. Cost guard: a dry run runs first, and the query is refused if the estimated bytes
   exceed ``max_bytes_billed``. The real job also sets ``maximum_bytes_billed`` so
   BigQuery itself fails the job if the estimate was wrong.
2. Dataset allowlist: every table the query references must live in the allowed dataset
   (``scrm``). Direct reads from ``gdelt-bq`` or any other dataset are refused.
3. Read only: the dry run's statement type must be SELECT, so no DML or DDL runs.
"""

from dataclasses import dataclass
from typing import Any

from google.cloud import bigquery

from scrm.config import Settings
from scrm.telemetry import get_logger

log = get_logger(__name__)

QueryParam = bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter


class QueryRefusedError(Exception):
    """Base class for queries the guard will not execute."""


class QueryTooExpensiveError(QueryRefusedError):
    """The dry run estimated more bytes than the configured limit."""


class DatasetNotAllowedError(QueryRefusedError):
    """The query references a table outside the allowed dataset."""


class StatementNotAllowedError(QueryRefusedError):
    """The query is not a read-only SELECT."""


@dataclass(frozen=True)
class DryRunResult:
    total_bytes_processed: int
    referenced_tables: tuple[str, ...]
    statement_type: str | None


class BigQueryTool:
    """Run read-only queries against the allowed dataset, with a dry-run cost guard."""

    def __init__(
        self,
        client: bigquery.Client,
        *,
        max_bytes_billed: int,
        allowed_project: str,
        allowed_dataset: str,
        location: str = "US",
    ) -> None:
        if max_bytes_billed <= 0:
            raise ValueError("max_bytes_billed must be positive")
        self._client = client
        self.max_bytes_billed = max_bytes_billed
        self.allowed_project = allowed_project
        self.allowed_dataset = allowed_dataset
        self.location = location
        self.bytes_processed = 0  # running total of bytes processed by executed queries

    @classmethod
    def from_settings(cls, settings: Settings) -> "BigQueryTool":
        """Create a tool using Application Default Credentials."""
        if not settings.gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set")
        client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
        return cls(
            client,
            max_bytes_billed=settings.max_bytes_billed,
            allowed_project=settings.gcp_project,
            allowed_dataset=settings.bq_dataset,
            location=settings.bq_location,
        )

    def with_max_bytes(self, max_bytes_billed: int) -> "BigQueryTool":
        """Return a copy of this tool, sharing the client, with a different byte limit."""
        return BigQueryTool(
            self._client,
            max_bytes_billed=max_bytes_billed,
            allowed_project=self.allowed_project,
            allowed_dataset=self.allowed_dataset,
            location=self.location,
        )

    def dry_run(self, sql: str, params: list[QueryParam] | None = None) -> DryRunResult:
        """Estimate bytes and list referenced tables without running the query (free)."""
        config = bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False, query_parameters=params or []
        )
        job = self._client.query(sql, job_config=config, location=self.location)
        tables = tuple(
            f"{t.project}.{t.dataset_id}.{t.table_id}" for t in job.referenced_tables or []
        )
        return DryRunResult(int(job.total_bytes_processed or 0), tables, job.statement_type)

    def run_query(self, sql: str, params: list[QueryParam] | None = None) -> list[dict[str, Any]]:
        """Dry-run, check guards, then execute and return rows as dicts."""
        estimate = self.dry_run(sql, params)
        self._check_statement(estimate.statement_type)
        self._check_tables(estimate.referenced_tables)
        self._check_cost(estimate.total_bytes_processed)
        log.info(
            "bigquery.execute",
            extra={
                "estimated_bytes": estimate.total_bytes_processed,
                "tables": list(estimate.referenced_tables),
            },
        )
        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes_billed, query_parameters=params or []
        )
        job = self._client.query(sql, job_config=config, location=self.location)
        rows = [dict(row.items()) for row in job.result()]
        self.bytes_processed += int(job.total_bytes_processed or 0)
        return rows

    def _check_statement(self, statement_type: str | None) -> None:
        if statement_type != "SELECT":
            log.warning("bigquery.refused.statement", extra={"statement_type": statement_type})
            raise StatementNotAllowedError(f"Only SELECT is allowed, got {statement_type}")

    def _check_cost(self, estimated_bytes: int) -> None:
        if estimated_bytes > self.max_bytes_billed:
            log.warning(
                "bigquery.refused.cost",
                extra={"estimated_bytes": estimated_bytes, "limit": self.max_bytes_billed},
            )
            raise QueryTooExpensiveError(
                f"Dry run estimates {estimated_bytes:,} bytes, "
                f"over the limit of {self.max_bytes_billed:,} bytes."
            )

    def _check_tables(self, tables: tuple[str, ...]) -> None:
        prefix = f"{self.allowed_project}.{self.allowed_dataset}."
        disallowed = [t for t in tables if not t.startswith(prefix)]
        if disallowed:
            log.warning("bigquery.refused.dataset", extra={"tables": disallowed})
            raise DatasetNotAllowedError(
                f"Only {prefix}* may be queried; refused: {', '.join(disallowed)}"
            )
