from scrm.config import GIB, Settings


def test_defaults_when_env_empty(monkeypatch):
    for var in ("GOOGLE_CLOUD_PROJECT", "SCRM_MAX_BYTES_BILLED", "SCRM_BQ_DATASET"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings.from_env()
    assert settings.bq_dataset == "scrm"
    assert settings.max_bytes_billed == 10 * GIB


def test_reads_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "dopl-507205")
    monkeypatch.setenv("SCRM_MAX_BYTES_BILLED", "1000")
    settings = Settings.from_env()
    assert settings.gcp_project == "dopl-507205"
    assert settings.max_bytes_billed == 1000
