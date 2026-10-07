"""Label eval cases one at a time: is this a real disruption that affects the site?

Usage:  uv run python -m scripts.label_evals [--all] [--dataset evals/dataset.jsonl]

Shows each unlabelled case (or every case with --all) and asks for a label:
    y = yes, a real current disruption at this site    n = no, false positive
    u = unsure                                          s = skip    q = quit
The verifier's verdict is shown only after you answer, so it cannot anchor your label.
Each label is saved immediately, so quitting part way loses nothing.
"""

import argparse
import sys
from pathlib import Path

from evals.run_evals import DEFAULT_DATASET, EvalCase, HumanLabel, load_cases, save_cases

KEYS: dict[str, HumanLabel] = {"y": "yes", "n": "no", "u": "unsure"}


def describe(case: EvalCase, position: int, total: int) -> str:
    event = case.event
    lines = [
        f"\n[{position}/{total}] {case.case_id}  site {case.site_id}  {event.event_date}",
        f"  URL:       {event.source_url}",
        f"  Place:     {event.place or '-'}  ({event.distance_km} km from site)",
        f"  Orgs:      {', '.join(event.organizations[:6]) or '-'}",
        f"  Themes:    {', '.join(event.themes[:8]) or '-'}",
    ]
    if event.supporting_urls:
        lines.append(f"  Also seen: {len(event.supporting_urls)} similar articles")
    return "\n".join(lines)


def ask(prompt: str = "  Label [y/n/u, s=skip, q=quit]: ") -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {*KEYS, "s", "q"}:
            return answer
        print("  Please type y, n, u, s or q.")


def label_cases(path: Path, relabel: bool) -> int:
    cases = load_cases(path)
    todo = [c for c in cases if relabel or c.human_label is None]
    print(f"{len(todo)} of {len(cases)} cases to label. Open each URL to read the article.")
    for position, case in enumerate(todo, 1):
        print(describe(case, position, len(todo)))
        answer = ask()
        if answer == "q":
            break
        if answer == "s":
            continue
        case.human_label = KEYS[answer]
        save_cases(cases, path)
        print(f"  Saved '{case.human_label}'. Verifier said: {case.verifier_verdict}")
        print(f"  Verifier reason: {case.verifier_reason}")
    labelled = sum(c.human_label is not None for c in cases)
    print(f"\n{labelled} of {len(cases)} cases labelled.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--all", action="store_true", help="also revisit labelled cases")
    args = parser.parse_args(argv)
    return label_cases(args.dataset, relabel=args.all)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
