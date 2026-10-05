import asyncio

import pytest

from scrm.agents import risk_scorer
from scrm.config import Settings
from scrm.telemetry import RunContext


def test_risk_scorer_builds_with_structured_output():
    agent = risk_scorer.build_agent(Settings())
    assert agent.name == "risk_scorer"
    assert agent.output_schema is risk_scorer.OUTPUT_SCHEMA


def test_risk_scorer_run_is_stubbed():
    with pytest.raises(NotImplementedError):
        asyncio.run(risk_scorer.run(None, RunContext.new()))  # type: ignore[arg-type]
