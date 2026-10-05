"""Application settings, read from environment variables.

Only plain environment variables are used; there is no key file support on purpose.
Authentication to Google Cloud happens through Application Default Credentials.
"""

import os
from functools import lru_cache

from pydantic import BaseModel, Field

GIB = 1024**3


class Settings(BaseModel):
    """Runtime configuration for the whole app."""

    gcp_project: str | None = Field(default=None, description="GCP project for BigQuery jobs.")
    bq_dataset: str = Field(default="scrm", description="The only dataset agents may read.")
    bq_location: str = Field(default="US", description="BigQuery location of the dataset.")
    max_bytes_billed: int = Field(
        default=10 * GIB,
        gt=0,
        description="Queries whose dry run exceeds this many bytes are refused.",
    )
    gemini_model: str = Field(default="gemini-3.5-flash", description="Model ID for all agents.")
    log_level: str = Field(default="INFO")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current process environment, falling back to defaults."""
        env = {
            "gcp_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
            "bq_dataset": os.getenv("SCRM_BQ_DATASET"),
            "bq_location": os.getenv("SCRM_BQ_LOCATION"),
            "max_bytes_billed": os.getenv("SCRM_MAX_BYTES_BILLED"),
            "gemini_model": os.getenv("SCRM_GEMINI_MODEL"),
            "log_level": os.getenv("SCRM_LOG_LEVEL"),
        }
        return cls.model_validate({k: v for k, v in env.items() if v})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings.from_env()
