"""HTTP API: POST /brief runs the pipeline, GET /health reports liveness.

Run locally:  uv run uvicorn scrm.api:app --reload
"""

from fastapi import FastAPI, HTTPException, status

from scrm import __version__
from scrm.agents import orchestrator
from scrm.config import get_settings
from scrm.schemas import BriefRequest, RiskBrief
from scrm.telemetry import RunContext, configure_logging, get_logger

configure_logging(get_settings().log_level)
log = get_logger(__name__)

app = FastAPI(title="Supply Chain Risk Monitor", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/brief", response_model=RiskBrief)
async def create_brief(request: BriefRequest) -> RiskBrief:
    ctx = RunContext.new()
    with ctx.bind():
        log.info("brief.requested", extra={"site_ids": request.site_ids})
        try:
            return await orchestrator.run(request, ctx)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail={"run_id": ctx.run_id, "error": str(exc)},
            ) from exc
