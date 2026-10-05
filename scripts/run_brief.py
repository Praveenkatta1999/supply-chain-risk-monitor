"""Run query_agent -> verifier -> report_writer for one site and print the brief.

Usage:
    uv run --env-file .env python scripts/run_brief.py [--site P05] [--days 30] [--limit 20]

Logs (JSON, one line per event, all carrying the run ID) go to stderr; the Markdown brief
and a verification summary go to stdout.
"""

import argparse
import asyncio
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta

from scrm.agents import query_agent, report_writer, verifier
from scrm.config import get_settings
from scrm.schemas import (
    DateRange,
    QueryRequest,
    ReportRequest,
    Verdict,
    VerificationRequest,
    VerifiedEvent,
)
from scrm.telemetry import RunContext, configure_logging, get_logger
from scrm.tools.bigquery_tool import BigQueryTool

log = get_logger("run_brief")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default="P05", help="site_id from scrm.sites")
    parser.add_argument("--days", type=int, default=30, help="days back from today, inclusive")
    parser.add_argument("--limit", type=int, default=20, help="candidates to verify")
    return parser.parse_args(argv)


def print_verification_summary(verified: list[VerifiedEvent]) -> None:
    counts = Counter(v.verdict for v in verified)
    print(f"\n## Verification summary ({len(verified)} candidates)\n")
    for verdict in Verdict:
        print(f"- {verdict}: {counts[verdict]}")
    for verdict in (Verdict.NO, Verdict.UNVERIFIABLE):
        rows = [v for v in verified if v.verdict is verdict]
        if rows:
            print(f"\n### {verdict}\n")
            for v in rows:
                print(f"- {v.event.source_url}\n  {v.reason}")


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx = RunContext.new()
    tool = BigQueryTool.from_settings(settings)
    end = datetime.now(UTC).date()
    date_range = DateRange(start=end - timedelta(days=args.days - 1), end=end)

    with ctx.bind():
        log.info("run.start", extra={"site_id": args.site, "date_range": date_range.model_dump()})
        sites = query_agent.load_sites(tool, [args.site])
        query = await query_agent.run(
            QueryRequest(site_ids=[args.site], date_range=date_range, limit=args.limit),
            ctx,
            tool=tool,
        )
        verified = await verifier.run_many(
            [VerificationRequest(event=e, site=sites[0]) for e in query.events], ctx
        )
        brief = await report_writer.run(
            ReportRequest(
                run_id=ctx.run_id, date_range=date_range, sites=sites, verified_events=verified
            ),
            ctx,
        )
        log.info("run.done", extra={"items": len(brief.items)})

    print(brief.markdown)
    print_verification_summary(verified)
    print(f"\nBigQuery bytes processed: {query.bytes_processed:,}. Run ID: {ctx.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
