import asyncio

import pytest

from scrm.agents import entity_resolver
from scrm.config import Settings
from scrm.telemetry import RunContext


def test_entity_resolver_builds_with_structured_output():
    agent = entity_resolver.build_agent(Settings())
    assert agent.name == "entity_resolver"
    assert agent.output_schema is entity_resolver.OUTPUT_SCHEMA


def test_entity_resolver_run_is_stubbed():
    with pytest.raises(NotImplementedError):
        asyncio.run(entity_resolver.run(None, RunContext.new()))  # type: ignore[arg-type]
