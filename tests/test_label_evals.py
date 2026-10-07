from evals.run_evals import EvalCase, load_cases, save_cases
from scripts import label_evals
from scrm.schemas import Verdict


def write_dataset(path, event, n=3):
    path.write_text("# header\n", encoding="utf-8")
    cases = [
        EvalCase(
            case_id=f"c{i}",
            run_id="r1",
            site_id=event.site_id,
            event=event,
            verifier_verdict=Verdict.NO,
            verifier_reason="Not at the site.",
        )
        for i in range(n)
    ]
    save_cases(cases, path)


def test_labels_are_saved_one_at_a_time_and_quit_keeps_progress(tmp_path, event, monkeypatch):
    path = tmp_path / "dataset.jsonl"
    write_dataset(path, event)
    answers = iter(["y", "s", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    label_evals.label_cases(path, relabel=False)
    assert [c.human_label for c in load_cases(path)] == ["yes", None, None]


def test_verifier_verdict_is_hidden_until_after_labelling(tmp_path, event, monkeypatch, capsys):
    path = tmp_path / "dataset.jsonl"
    write_dataset(path, event, n=1)
    printed_before_answer = []

    def answer(prompt=""):
        printed_before_answer.append(capsys.readouterr().out)
        return "n"

    monkeypatch.setattr("builtins.input", answer)
    label_evals.label_cases(path, relabel=False)
    assert "Not at the site." not in printed_before_answer[0]
    assert "Verifier said: no" in capsys.readouterr().out
