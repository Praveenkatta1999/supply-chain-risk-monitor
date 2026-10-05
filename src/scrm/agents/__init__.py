"""One module per agent. Each exposes INPUT_SCHEMA, OUTPUT_SCHEMA and an async ``run``.

LLM-backed agents also expose ``build_agent(settings)`` returning an ADK ``LlmAgent``
whose ``output_schema`` forces a Pydantic model instead of free text.
"""
