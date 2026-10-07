"""Run the full pipeline for one or more sites and print the brief.

Usage:
    uv run --env-file .env python scripts/run_brief.py [--sites P05 S01] [--days 30]
        [--end 2026-10-05] [--compare RUN_ID]

Logs (JSON, one line per event, all carrying the run ID) go to stderr. The Markdown brief,
per-site verification numbers, investigations and model calls go to stdout; --compare
also prints the same numbers for an earlier saved run. The full result is saved to
data/runs/<run_id>.json for evals (scripts/add_eval_cases.py).
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
    parser.add_argument("--compare", metavar="RUN_ID", help="earlier saved run to compare")
    return parser.parse_args(argv)


def site_row(run: SiteRun) -> str:
    """One table row: candidates, triage, verifier outcomes and rejection rate."""
    counts = Counter(v.verdict for v in run.verified_events)
    yes, no = counts[Verdict.YES], counts[Verdict.NO]
    rate = f"{no / (yes + no):.0%}" if yes + no else "n/a"
    triaged = len(run.triage.selected) if run.triage else len(run.verified_events)
    return (
        f"| {run.site.site_id} | {len(run.candidates)} | {triaged} | {yes + no} | {yes} "
        f"| {no} | {counts[Verdict.UNVERIFIABLE]} | {rate} |"
    )


def print_verification(result: PipelineResult, label: str) -> None:
    print(f"\n## Verification by site: {label}\n")
    print(
        "| Site | Candidates | Sent to verifier | Judged | Confirmed | Rejected "
        "| Unverifiable | Rejection rate |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for run in result.site_runs:
        print(site_row(run))
    events = [v for r in result.site_runs for v in r.verified_events]
    counts = Counter(v.verdict for v in events)
    judged = counts[Verdict.YES] + counts[Verdict.NO]
    rate = f"{counts[Verdict.NO] / judged:.0%}" if judged else "n/a"
    calls = result.model_calls
    print(
        f"\nVerifier model calls: {calls.get('verifier', 0)}. Overall rejection rate: {rate} "
        f"({counts[Verdict.NO]} of {judged} judged). "
        f"Total model calls: {sum(calls.values())} "
        f"({', '.join(f'{name} {n}' for name, n in sorted(calls.items()))})."
    )


def print_investigations(result: PipelineResult) -> None:
    investigations = [i for r in result.site_runs for i in r.investigations]
    print(f"\n## Investigations: {len(investigations)}\n")
    for inv in investigations:
        print(
            f"- {inv.site_id} [{inv.trigger}] {inv.status}, {inv.confidence} confidence, "
            f"{inv.tool_calls} tool calls, {len(inv.corroborating_urls)} corroborating"
            f"{' (hit limit)' if inv.hit_limit else ''}\n  {inv.source_url}\n  {inv.summary}"
        )


def load_run(run_id: str) -> PipelineResult:
    return PipelineResult.model_validate_json(
        (RUNS_DIR / f"{run_id}.json").read_text(encoding="utf-8")
    )


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
    print_verification(result, f"this run ({ctx.run_id})")
    if args.compare:
        print_verification(load_run(args.compare), f"earlier run ({args.compare})")
    print_investigations(result)
    print(f"\nBigQuery bytes processed: {sum(r.bytes_processed for r in result.site_runs):,}")
    print(f"Run ID: {ctx.run_id}. Full result saved to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
