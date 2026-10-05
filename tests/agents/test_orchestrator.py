import asyncio

import pytest

from scrm.agents import orchestrator
from scrm.schemas import BriefRequest, RiskBrief
from scrm.telemetry import RunContext


def test_orchestrator_contract_and_stub(date_range):
    assert orchestrator.OUTPUT_SCHEMA is RiskBrief
    request = BriefRequest(site_ids=["SUP-01"], date_range=date_range)
    with pytest.raises(NotImplementedError):
        asyncio.run(orchestrator.run(request, RunContext.new()))
