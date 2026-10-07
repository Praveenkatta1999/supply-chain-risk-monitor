"""Append a saved pipeline run's verifier verdicts to the eval dataset, unlabelled.

Usage:  uv run python -m scripts.add_eval_cases RUN_ID [--dataset evals/dataset.jsonl]

Reads data/runs/<RUN_ID>.json (written by scripts/run_brief.py) and appends one case per
verified story with ``human_label`` empty, ready for scripts/label_evals.py.

Existing cases are never rewritten: new cases are appended as new lines. A story whose
site and URL (or any of its supporting URLs) is already in the dataset is skipped, so the
same article is never labelled twice.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.run_evals import DEFAULT_DATASET, EvalCase, load_cases
from scrm.schemas import VerifiedEvent
from scrm.urls import article_key

RUNS_DIR = Path("data/runs")


def cases_from_run(run_id: str, runs_dir: Path = RUNS_DIR) -> list[EvalCase]:
    """One unlabelled case per verified story in the run, numbered by site and rank."""
    data = json.loads((runs_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    cases = []
    for site_run in data["site_runs"]:
        for rank, raw in enumerate(site_run["verified_events"], 1):
            verified = VerifiedEvent.model_validate(raw)
            cases.append(
                EvalCase(
                    case_id=f"{run_id}-{verified.event.site_id}-{rank:02d}",
                    run_id=run_id,
                    site_id=verified.event.site_id,
                    event=verified.event,
                    verifier_verdict=verified.verdict,
                    verifier_reason=verified.reason,
                )
            )
    return cases


def _story_keys(case: EvalCase) -> set[tuple[str, str]]:
    urls = [case.event.source_url, *case.event.supporting_urls]
    return {(case.site_id, article_key(url)) for url in urls}


def new_cases(existing: list[EvalCase], candidates: list[EvalCase]) -> list[EvalCase]:
    """Candidates whose case ID and story are not already in ``existing``."""
    seen_ids = {c.case_id for c in existing}
    seen_stories = set().union(*(_story_keys(c) for c in existing))
    fresh = []
    for case in candidates:
        if case.case_id in seen_ids or _story_keys(case) & seen_stories:
            continue
        fresh.append(case)
        seen_ids.add(case.case_id)
        seen_stories |= _story_keys(case)
    return fresh


def append_cases(cases: list[EvalCase], path: Path) -> None:
    """Append cases as new JSONL lines; existing lines are not touched."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    prefix = "" if not text or text.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + "".join(case.model_dump_json() + "\n" for case in cases))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    existing = load_cases(args.dataset)
    candidates = cases_from_run(args.run_id)
    fresh = new_cases(existing, candidates)
    append_cases(fresh, args.dataset)
    print(
        f"Appended {len(fresh)} unlabelled cases from run {args.run_id}; skipped "
        f"{len(candidates) - len(fresh)} already in the dataset. "
        f"Dataset now has {len(existing) + len(fresh)} cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
