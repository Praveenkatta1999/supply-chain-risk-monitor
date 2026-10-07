"""Run an ADK LlmAgent once and return its structured output as a Pydantic model.

Each call uses a fresh in-memory session: every agent here is a function (JSON in, JSON
out), so no conversation state needs to survive between calls. Every LLM response (one
per model request, including each tool round-trip of an agent with tools) is counted on
the bound RunContext (``model_calls``) for cost reporting.
"""

from google.adk.agents import LlmAgent, RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from scrm.telemetry import current_context

APP_NAME = "scrm"
USER_ID = "pipeline"


class AgentOutputError(Exception):
    """The agent finished without writing a valid structured output."""


async def run_structured[M: BaseModel](
    agent: LlmAgent,
    payload: BaseModel,
    output_model: type[M],
    *,
    run_config: RunConfig | None = None,
) -> M:
    """Send ``payload`` as JSON to ``agent`` and validate the result as ``output_model``.

    Pass ``run_config`` to cap model calls (``max_llm_calls``); exceeding it raises ADK's
    ``LlmCallsLimitExceededError``.
    """
    if not agent.output_key:
        raise ValueError(f"agent {agent.name} needs an output_key")
    ctx = current_context()
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    try:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        message = types.Content(role="user", parts=[types.Part(text=payload.model_dump_json())])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=message, run_config=run_config
        ):
            if ctx and event.usage_metadata is not None:  # one per LLM response
                ctx.record_model_call(agent.name)
        session = await runner.session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session.id
        )
    finally:
        await runner.close()
    raw = session.state.get(agent.output_key) if session else None
    if raw is None:
        raise AgentOutputError(f"agent {agent.name} produced no output")
    return output_model.model_validate(raw)
