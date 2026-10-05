"""Orchestrator: takes a site list and date range, routes work, assembles the brief.

Pipeline (in order):
    query_agent -> entity_resolver -> verifier -> risk_scorer -> report_writer

The orchestrator is deterministic Python rather than an LLM: the order of steps is fixed,
so letting a model choose it would add cost and non-determinism without benefit. It
creates the RunContext so one run ID flows through every agent's logs.

Input:  BriefRequest
Output: RiskBrief
"""

from scrm.schemas import BriefRequest, RiskBrief
from scrm.telemetry import RunContext

NAME = "orchestrator"
INPUT_SCHEMA = BriefRequest
OUTPUT_SCHEMA = RiskBrief


async def run(request: BriefRequest, ctx: RunContext) -> RiskBrief:
    """Run the full pipeline for one request and return the brief."""
    raise NotImplementedError("orchestrator is not implemented yet")
