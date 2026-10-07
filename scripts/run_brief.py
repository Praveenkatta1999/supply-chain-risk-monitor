"""Run the full pipeline for one or more sites and print the brief.

Usage:
    uv run --env-file .env python scripts/run_brief.py [--sites P05 S01] [--days 30]
        [--end 2026-10-05]

Logs (JSON, one line per event, all carrying the run ID) go to stderr. The Markdown brief,
the false positive rate per site and the model call count go to stdout. The full result
(candidates, verdicts, scores, brief) is saved to data/runs/<run_id>.json for evals.
"""

import argparse
import asyncio
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from scrm.agents import orchestrator
from scrm.config import get_settings
from scrm.schemas import BriefRequest, DateRange, PipelineResult, SiteRun, Verdict
from scrm.telemetry import RunContext, configure_logging, get_logger

RUNS_DIR = Path("data/runs")
log = get_logger("run_brief")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", nargs="+", default=["P05"], help="site_ids from scrm.sites")
    parser.add_argument("--days", type=int, default=30, help="window length, inclusive")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (default: today, UTC)")
    return parser.parse_args(argv)


def false_positive_rate(run: SiteRun) -> tuple[int, int, int, str]:
    """(confirmed, rejected, unverifiable, rate), rate = rejected / (confirmed + rejected)."""
    counts = Counter(v.verdict for v in run.verified_events)
    yes, no = counts[Verdict.YES], counts[Verdict.NO]
    rate = f"{no / (yes + no):.0%}" if yes + no else "n/a"
    return yes, no, counts[Verdict.UNVERIFIABLE], rate


def print_summary(result: PipelineResult) -> None:
    print("\n## Verification by site\n")
    print("| Site | Stories | Confirmed | Rejected | Unverifiable | False positive rate |")
    print("|---|---|---|---|---|---|")
    for run in result.site_runs:
        yes, no, unverifiable, rate = false_positive_rate(run)
        stories = len(run.verified_events)
        print(f"| {run.site.site_id} | {stories} | {yes} | {no} | {unverifiable} | {rate} |")
    calls = result.model_calls
    detail = ", ".join(f"{name} {n}" for name, n in sorted(calls.items()))
    print(f"\nModel calls: {sum(calls.values())} ({detail})")
    print(f"BigQuery bytes processed: {sum(r.bytes_processed for r in result.site_runs):,}")


async def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_logging(get_settings().log_level)
    ctx = RunContext.new()
    end = args.end or datetime.now(UTC).date()
    request = BriefRequest(
        site_ids=args.sites,
        date_range=DateRange(start=end - timedelta(days=args.days - 1), end=end),
    )
    result = await orchestrator.run_pipeline(request, ctx)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    saved = RUNS_DIR / f"{ctx.run_id}.json"
    saved.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(result.brief.markdown)
    print_summary(result)
    print(f"Run ID: {ctx.run_id}. Full result saved to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
