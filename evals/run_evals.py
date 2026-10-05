"""Offline evaluation of the verifier and risk scorer against labelled cases.

Usage:  uv run python -m evals.run_evals [path/to/dataset.jsonl]

Stub: loads and validates the dataset; scoring the agents is not implemented yet.
Planned metrics: verifier precision/recall on affects_site, and severity mean absolute error.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from scrm.schemas import Event

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")


class EvalCase(BaseModel):
    case_id: str
    site_id: str
    event: Event
    expected_affects_site: bool
    expected_severity: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""


def load_cases(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    """Parse the JSONL dataset, skipping blank lines and '#' comments."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        EvalCase.model_validate_json(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    cases = load_cases(Path(args[0]) if args else DEFAULT_DATASET)
    print(f"Loaded {len(cases)} eval cases. Agent scoring is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
