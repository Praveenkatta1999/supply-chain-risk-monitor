"""Ask the root agent a plain-language question and print its answer and the brief.

Usage:
    uv run --env-file .env python scripts/ask.py "What are the risks to our chip supply this week?"
        [--today 2026-10-07]

The agent chooses the sites and dates and runs the pipeline once. The terminal shows
one-line progress; full JSON logs (run ID on every line) go to data/logs/<run_id>.jsonl,
or to the terminal with --verbose. The pipeline result is saved to data/runs/<run_id>.json (the
same format as scripts/run_brief.py) and the answer, with model calls, to
data/runs/<run_id>.answer.json.
"""

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from scrm.agents import root_agent
from scrm.config import get_settings
from scrm.progress import setup_cli_logging
from scrm.telemetry import RunContext

RUNS_DIR = Path("data/runs")


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--today", type=date.fromisoformat, help="default: today, UTC")
    parser.add_argument(
        "--verbose", action="store_true", help="show the full JSON log instead of progress"
    )
    args = parser.parse_args(argv)
    # Claims can quote any language; don't depend on the console's code page.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ctx = RunContext.new()
    log_file = setup_cli_logging(ctx.run_id, verbose=args.verbose, level=get_settings().log_level)
    today = args.today or datetime.now(UTC).date()
    result = await root_agent.run(args.question, ctx, today=today)

    answer = result.answer
    print(f"# Question\n\n{args.question}\n")
    print(f"# Answer\n\n{answer.summary}\n")
    print(
        f"Sites: {', '.join(answer.site_ids)}; dates {answer.start_date} to {answer.end_date}. "
        f"{answer.rationale}\n"
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # The answer plus model calls for the whole question (the pipeline's own count stops
    # before the root agent's final answer).
    (RUNS_DIR / f"{ctx.run_id}.answer.json").write_text(
        result.model_copy(update={"pipeline": None}).model_dump_json(indent=2), encoding="utf-8"
    )
    if result.pipeline is None:
        print("The agent did not run the pipeline.")
    else:
        (RUNS_DIR / f"{ctx.run_id}.json").write_text(
            result.pipeline.model_dump_json(indent=2), encoding="utf-8"
        )
        print(result.pipeline.brief.markdown)
    calls = result.model_calls
    detail = ", ".join(f"{name} {n}" for name, n in sorted(calls.items()))
    print(f"\nModel calls: {sum(calls.values())} ({detail}). Run ID: {ctx.run_id}")
    print(f"Logs: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
