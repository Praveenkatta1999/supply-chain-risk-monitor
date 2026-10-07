"""Offline evaluation of the verifier against human-labelled cases.

Usage:  uv run python -m evals.run_evals [path/to/dataset.jsonl]
        uv run --env-file .env python -m evals.run_evals --rerun-verifier

Each case is a real candidate from a pipeline run, stored with the verifier's verdict and
reason at the time, plus a ``human_label`` filled in with ``scripts/label_evals.py``.
By default the stored verdicts are scored. ``--rerun-verifier`` also runs the current
verifier (fetching each article again, calling Gemini) on the labelled cases and scores
that too, so prompt changes can be measured; articles may have changed or gone offline
since they were first verified.

The positive class is "yes" (a real, current disruption at the site). Only cases the
human labelled "yes" or "no" are scored; "unsure" and unlabelled cases are skipped.
An "unverifiable" verdict never reaches the brief, so for accuracy, precision and recall
it counts as a "no" prediction; coverage reports how often the verifier could judge at all.
Planned: severity mean absolute error once risk scores are labelled.
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from scrm.schemas import Event, Site, Verdict, VerificationRequest

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")

HumanLabel = Literal["yes", "no", "unsure"]


class EvalCase(BaseModel):
    case_id: str
    run_id: str
    site_id: str
    event: Event
    verifier_verdict: Verdict
    verifier_reason: str
    human_label: HumanLabel | None = None
    expected_severity: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""

    @property
    def scored(self) -> bool:
        return self.human_label in ("yes", "no")

    @property
    def predicted_yes(self) -> bool:
        return self.verifier_verdict is Verdict.YES


def _is_case(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def load_cases(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    """Parse the JSONL dataset, skipping blank lines and '#' comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if _is_case(line)]


def save_cases(cases: list[EvalCase], path: Path = DEFAULT_DATASET) -> None:
    """Rewrite the dataset, keeping its '#' header comments, via an atomic replace."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    header = [line for line in lines if line.lstrip().startswith("#")]
    body = [case.model_dump_json() for case in cases]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class Metrics:
    """Confusion counts over scored cases, positive class "yes"."""

    tp: int
    fp: int
    fn: int
    tn: int
    unverifiable: int  # scored cases the verifier could not judge (counted as "no")
    skipped: int  # unlabelled or "unsure"

    @property
    def scored(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float | None:
        return _ratio(self.tp + self.tn, self.scored)

    @property
    def precision(self) -> float | None:
        """Of the verifier's "yes" verdicts, the share the human also called "yes"."""
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        """Of the human's "yes" labels, the share the verifier also called "yes"."""
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def coverage(self) -> float | None:
        """Share of scored cases the verifier could judge (not "unverifiable")."""
        return _ratio(self.scored - self.unverifiable, self.scored)


def score(cases: list[EvalCase]) -> Metrics:
    scored = [c for c in cases if c.scored]
    human_yes = [c.human_label == "yes" for c in scored]
    predicted = [c.predicted_yes for c in scored]
    pairs = list(zip(predicted, human_yes, strict=True))
    return Metrics(
        tp=pairs.count((True, True)),
        fp=pairs.count((True, False)),
        fn=pairs.count((False, True)),
        tn=pairs.count((False, False)),
        unverifiable=sum(c.verifier_verdict is Verdict.UNVERIFIABLE for c in scored),
        skipped=len(cases) - len(scored),
    )


def disagreements(cases: list[EvalCase]) -> list[EvalCase]:
    """Scored cases where the verifier's yes/no call differs from the human label."""
    return [
        c
        for c in cases
        if c.scored
        and c.verifier_verdict is not Verdict.UNVERIFIABLE
        and c.verifier_verdict.value != c.human_label
    ]


def unverifiable(cases: list[EvalCase]) -> list[EvalCase]:
    """Scored cases the verifier could not judge: it neither agreed nor disagreed."""
    return [c for c in cases if c.scored and c.verifier_verdict is Verdict.UNVERIFIABLE]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def report(cases: list[EvalCase]) -> str:
    m = score(cases)
    lines = [
        f"{len(cases)} cases, {m.scored} scored (human yes/no), {m.skipped} skipped.",
        "",
        "| Metric | Value | Counts |",
        "|---|---|---|",
        f"| Accuracy | {_pct(m.accuracy)} | {m.tp + m.tn} of {m.scored} |",
        f"| Precision | {_pct(m.precision)} | {m.tp} of {m.tp + m.fp} verifier 'yes' |",
        f"| Recall | {_pct(m.recall)} | {m.tp} of {m.tp + m.fn} human 'yes' |",
        f"| Coverage | {_pct(m.coverage)} | {m.scored - m.unverifiable} of {m.scored} judged |",
        "",
        f"Confusion (positive = yes): TP {m.tp}, FP {m.fp}, FN {m.fn}, TN {m.tn}. "
        "'unverifiable' counts as a 'no' prediction.",
    ]
    for title, rows in (
        ("Disagreements", disagreements(cases)),
        ("Unverifiable (not judged)", unverifiable(cases)),
    ):
        lines += ["", f"## {title}: {len(rows)}", ""]
        for c in rows:
            lines += [
                f"- {c.case_id}  {c.event.source_url}",
                f"  human: {c.human_label}   verifier: {c.verifier_verdict}",
                f"  verifier reason: {c.verifier_reason}",
            ]
    return "\n".join(lines)


async def rerun_verifier(cases: list[EvalCase], sites: dict[str, Site]) -> list[EvalCase]:
    """Copies of the scored cases with the current verifier's verdict and reason."""
    from scrm.agents import verifier  # imported here: only this mode needs the agents
    from scrm.telemetry import RunContext

    scored = [c for c in cases if c.scored]
    requests = [VerificationRequest(event=c.event, site=sites[c.site_id]) for c in scored]
    results = await verifier.run_many(requests, RunContext.new())
    return [
        c.model_copy(update={"verifier_verdict": r.verdict, "verifier_reason": r.reason})
        for c, r in zip(scored, results, strict=True)
    ]


def _load_sites(site_ids: list[str]) -> dict[str, Site]:
    from scrm.agents import query_agent
    from scrm.config import get_settings
    from scrm.tools.bigquery_tool import BigQueryTool

    tool = BigQueryTool.from_settings(get_settings())
    return {s.site_id: s for s in query_agent.load_sites(tool, site_ids)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--rerun-verifier", action="store_true", help="also score the current verifier"
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cases = load_cases(args.dataset)
    print("# Stored verdicts (verifier at the time of each run)\n")
    print(report(cases))
    if args.rerun_verifier:
        scored = [c for c in cases if c.scored]
        sites = _load_sites(sorted({c.site_id for c in scored}))
        print("\n# Current verifier, re-run on the labelled cases\n")
        print(report(asyncio.run(rerun_verifier(scored, sites))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
