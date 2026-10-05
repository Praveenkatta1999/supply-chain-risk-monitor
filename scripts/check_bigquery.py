"""Smoke test for BigQueryTool against the real dataset.

1. Runs a small SELECT over scrm.sites through the guard.
2. Shows the guard refusing a query whose dry run exceeds the byte limit. To avoid
   needing a genuinely huge query, it uses a second tool with a 1 MiB limit; the dry run
   is free, and the refused query is never executed.

Run:  uv run --env-file .env python scripts/check_bigquery.py
"""

from scrm.config import get_settings
from scrm.telemetry import RunContext, configure_logging, get_logger
from scrm.tools.bigquery_tool import BigQueryTool, QueryTooExpensiveError

MIB = 1024**2
log = get_logger("check_bigquery")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    tool = BigQueryTool.from_settings(settings)
    ctx = RunContext.new()

    with ctx.bind():
        sql = (
            f"SELECT site_id, site_name, site_type, country, radius_km "
            f"FROM `{settings.gcp_project}.{settings.bq_dataset}.sites` "
            f"ORDER BY site_id"
        )
        rows = tool.run_query(sql)
        print(f"\n[1] scrm.sites returned {len(rows)} rows through the guard:")
        for row in rows:
            print(f"    {row['site_id']:<5} {row['site_type']:<11} {row['site_name']}")

        strict = tool.with_max_bytes(MIB)
        big_sql = (
            f"SELECT url, themes, organizations "
            f"FROM `{settings.gcp_project}.{settings.bq_dataset}.gkg_near_sites`"
        )
        estimated = strict.dry_run(big_sql).total_bytes_processed
        print(f"\n[2] Full scan of gkg_near_sites: dry run estimates {estimated:,} bytes")
        try:
            strict.run_query(big_sql)
        except QueryTooExpensiveError as exc:
            print(f"    Refused by guard (limit {MIB:,} bytes): {exc}")
        else:
            raise SystemExit("Guard did not refuse the over-limit query")


if __name__ == "__main__":
    main()
