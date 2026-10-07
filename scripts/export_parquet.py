"""Export the scrm tables from BigQuery to Parquet files in data/, and check row counts.

Usage:  uv run --env-file .env python scripts/export_parquet.py [--tables sites ...]

Every query goes through BigQueryTool (dry-run cost guard, scrm-only, SELECT-only).
Export and count both read the table as of one timestamp (BigQuery time travel), so rows
appended while the export runs cannot cause a false mismatch. Rows are streamed page by
page into the Parquet file, so a large table never has to fit in memory. Each file is
written to a .tmp path and renamed only after it is complete.

Exits non-zero if any file's row count differs from BigQuery's.
"""

import argparse
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

from scrm.config import get_settings
from scrm.telemetry import RunContext, configure_logging, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

TABLES = ("sites", "events_near_sites", "gkg_near_sites")
OUT_DIR = Path("data")
PAGE_SIZE = 50_000
log = get_logger("export_parquet")


def snapshot_sql(table: str, select: str) -> str:
    return f"SELECT {select} FROM `{table}` FOR SYSTEM_TIME AS OF @as_of"


def write_parquet(batches: Iterable[pa.RecordBatch], path: Path) -> int:
    """Stream record batches into ``path`` (atomically) and return the rows written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in batches:
            if writer is None:
                writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd")
            writer.write_batch(batch)
            rows += batch.num_rows
            log.info("export.progress", extra={"file": path.name, "rows": rows})
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"no rows returned for {path.name}")
    tmp.replace(path)
    return rows


def export_table(tool: BigQueryTool, name: str, as_of: datetime) -> tuple[int, int]:
    """Export one table; return (BigQuery row count, rows in the Parquet file)."""
    table = f"{tool.allowed_project}.{tool.allowed_dataset}.{name}"
    params = [bigquery.ScalarQueryParameter("as_of", "TIMESTAMP", as_of)]
    expected = tool.run_query(snapshot_sql(table, "COUNT(*) AS n"), params)[0]["n"]
    path = OUT_DIR / f"{name}.parquet"
    rows = tool.query_rows(snapshot_sql(table, "*"), params, page_size=PAGE_SIZE)
    write_parquet(rows.to_arrow_iterable(), path)
    in_file = pq.ParquetFile(path).metadata.num_rows  # re-read from disk, not our counter
    log.info(
        "export.table_done",
        extra={"table": name, "bigquery_rows": expected, "parquet_rows": in_file},
    )
    return expected, in_file


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=TABLES)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    tool = BigQueryTool.from_settings(settings)
    # A minute in the past, so the snapshot time is never ahead of BigQuery's clock.
    as_of = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=1)
    OUT_DIR.mkdir(exist_ok=True)

    with RunContext.new().bind():
        results = {name: export_table(tool, name, as_of) for name in args.tables}

    print(f"\nSnapshot as of {as_of.isoformat()}\n")
    print("| Table | BigQuery rows | Parquet rows | Match | File size |")
    print("|---|---|---|---|---|")
    for name, (expected, in_file) in results.items():
        size_mb = (OUT_DIR / f"{name}.parquet").stat().st_size / 1e6
        match = "yes" if expected == in_file else "NO"
        print(f"| {name} | {expected:,} | {in_file:,} | {match} | {size_mb:,.1f} MB |")
    print(f"\nBigQuery bytes processed: {tool.bytes_processed:,}")
    return 0 if all(e == f for e, f in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
