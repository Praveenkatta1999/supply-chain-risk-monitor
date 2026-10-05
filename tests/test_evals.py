from evals.run_evals import DEFAULT_DATASET, load_cases, main


def test_empty_dataset_loads():
    assert load_cases(DEFAULT_DATASET) == []
    assert main([]) == 0
