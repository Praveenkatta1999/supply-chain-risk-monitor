"""Entity resolver: matches messy GDELT actor and organization names to scrm.sites companies.

Example: "FOXCONN TECHNOLOGY GRP" and "Hon Hai" should both resolve to the same company.
Low-confidence matches return company=None.

Input:  EntityResolutionRequest
Output: EntityResolutionResult
"""

from google.adk.agents import LlmAgent

from scrm.config import Settings
from scrm.schemas import EntityResolutionRequest, EntityResolutionResult
from scrm.telemetry import RunContext

NAME = "entity_resolver"
INPUT_SCHEMA = EntityResolutionRequest
OUTPUT_SCHEMA = EntityResolutionResult
DESCRIPTION = (__doc__ or "").splitlines()[0]

INSTRUCTION = """\
You match raw actor and organization names from news coverage to a fixed list of
companies. Return no match rather than guess.
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


async def run(request: EntityResolutionRequest, ctx: RunContext) -> EntityResolutionResult:
    """Run the agent for one request within run ``ctx.run_id``."""
    raise NotImplementedError("entity_resolver is not implemented yet")
