from evals.run_evals import (
    DEFAULT_DATASET,
    EvalCase,
    disagreements,
    load_cases,
    main,
    report,
    save_cases,
    score,
    unverifiable,
)
from scrm.schemas import Verdict


def case(event, case_id: str, verdict: Verdict, label=None) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        run_id="r1",
        site_id=event.site_id,
        event=event,
        verifier_verdict=verdict,
        verifier_reason="Because.",
        human_label=label,
    )


def test_committed_dataset_loads():
    cases = load_cases(DEFAULT_DATASET)
    assert len(cases) >= 20
    assert len({c.case_id for c in cases}) == len(cases)
    assert main([]) == 0


def test_save_keeps_header_comments_and_round_trips(tmp_path, event):
    path = tmp_path / "dataset.jsonl"
    path.write_text("# header line\n", encoding="utf-8")
    cases = [case(event, "c1", Verdict.YES, "no")]
    save_cases(cases, path)
    assert path.read_text(encoding="utf-8").startswith("# header line\n")
    assert load_cases(path) == cases


def mixed_cases(event) -> list[EvalCase]:
    return [
        case(event, "tp", Verdict.YES, "yes"),
        case(event, "fp", Verdict.YES, "no"),
        case(event, "fn", Verdict.NO, "yes"),
        case(event, "tn", Verdict.NO, "no"),
        case(event, "unv-no", Verdict.UNVERIFIABLE, "no"),
        case(event, "unv-yes", Verdict.UNVERIFIABLE, "yes"),
        case(event, "unsure", Verdict.YES, "unsure"),
        case(event, "unlabelled", Verdict.NO, None),
    ]


def test_metrics_treat_unverifiable_as_no_and_skip_unsure(event):
    m = score(mixed_cases(event))
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 2, 2)
    assert (m.unverifiable, m.skipped, m.scored) == (2, 2, 6)
    assert m.accuracy == 3 / 6
    assert m.precision == 1 / 2
    assert m.recall == 1 / 3
    assert m.coverage == 4 / 6


def test_disagreements_and_unverifiable_are_listed_separately(event):
    cases = mixed_cases(event)
    assert [c.case_id for c in disagreements(cases)] == ["fp", "fn"]
    assert [c.case_id for c in unverifiable(cases)] == ["unv-no", "unv-yes"]


def test_metrics_with_no_positives_are_not_a_crash(event):
    m = score([case(event, "tn", Verdict.NO, "no")])
    assert m.precision is None and m.recall is None
    assert "Precision | n/a" in report([case(event, "tn", Verdict.NO, "no")])
