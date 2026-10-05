"""Risk scorer: rates severity and impact from 1 to 5 with a short reason.

Only scores events the verifier accepted. The score keeps the event's source URL.

Input:  ScoringRequest
Output: RiskScore
"""

from google.adk.agents import LlmAgent

from scrm.config import Settings
from scrm.schemas import RiskScore, ScoringRequest
from scrm.telemetry import RunContext

NAME = "risk_scorer"
INPUT_SCHEMA = ScoringRequest
OUTPUT_SCHEMA = RiskScore
DESCRIPTION = (__doc__ or "").splitlines()[0]

INSTRUCTION = """\
You score a verified event's severity (how bad the event is) and impact (how much it
disrupts this site) on 1-5 scales, with a reason under 500 characters.
TODO: full prompt. Respond only with JSON matching the output schema.
"""


def build_agent(settings: Settings) -> LlmAgent:
    """Return the ADK agent definition. Constructing it makes no network calls."""
    return LlmAgent(
        name=NAME,
        model=settings.gemini_model,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        output_key=NAME,
    )


async def run(request: ScoringRequest, ctx: RunContext) -> RiskScore:
    """Run the agent for one request within run ``ctx.run_id``."""
    raise NotImplementedError("risk_scorer is not implemented yet")
