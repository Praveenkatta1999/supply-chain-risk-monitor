import json

from evals.run_evals import EvalCase, load_cases, save_cases
from scripts import add_eval_cases
from scrm.schemas import Verdict, VerifiedEvent


def write_run(tmp_path, event, urls):
    runs = tmp_path / "runs"
    runs.mkdir()
    verified = [
        VerifiedEvent(
            event=event.model_copy(update={"source_url": u, "event_id": u}),
            verdict=Verdict.NO,
            reason="Not here.",
            evidence_url=u,
        ).model_dump(mode="json")
        for u in urls
    ]
    (runs / "r2.json").write_text(
        json.dumps({"site_runs": [{"verified_events": verified}]}), encoding="utf-8"
    )
    return runs


def test_appends_unlabelled_cases_without_touching_existing_ones(tmp_path, event):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("# header\n", encoding="utf-8")
    labelled = EvalCase(
        case_id="old-1", run_id="r1", site_id=event.site_id, event=event,
        verifier_verdict=Verdict.YES, verifier_reason="r", human_label="yes",
    )  # fmt: skip
    save_cases([labelled], dataset)
    before = dataset.read_text(encoding="utf-8")

    runs = write_run(tmp_path, event, [str(event.source_url), "https://new.example/a"])
    fresh = add_eval_cases.new_cases(load_cases(dataset), add_eval_cases.cases_from_run("r2", runs))
    add_eval_cases.append_cases(fresh, dataset)

    assert dataset.read_text(encoding="utf-8").startswith(before)  # existing bytes intact
    cases = load_cases(dataset)
    assert [c.case_id for c in cases] == ["old-1", "r2-SUP-01-02"]  # same story skipped
    assert cases[0].human_label == "yes" and cases[1].human_label is None
