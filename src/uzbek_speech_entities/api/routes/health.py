"""Read-only model readiness endpoint."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class HealthDevice(BaseModel):
    """Local accelerator and selected model devices."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mps_available: bool
    stt: str | None
    ner: str | None


class HealthModels(BaseModel):
    """Configured model identities and one-time load status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stt_loaded: bool
    stt_id: str | None
    ner_loaded: bool
    ner_path: str


class HealthResponse(BaseModel):
    """Stable health response returned for both ready and unavailable states."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "unavailable"]
    device: HealthDevice
    models: HealthModels


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def health(request: Request) -> HealthResponse | JSONResponse:
    """Report model/device readiness without loading models in the request."""
    analyzer = request.app.state.analyzer
    stt = analyzer.stt_service
    ner = analyzer.ner_predictor
    stt_loaded = bool(stt and stt.loaded)
    ner_loaded = bool(ner.loaded)
    body = HealthResponse(
        status="ok" if stt_loaded and ner_loaded else "unavailable",
        device=HealthDevice(
            mps_available=bool(request.app.state.mps_available),
            stt=stt.device if stt else None,
            ner=ner.device,
        ),
        models=HealthModels(
            stt_loaded=stt_loaded,
            stt_id=stt.model_id if stt else None,
            ner_loaded=ner_loaded,
            ner_path=str(ner.model_path),
        ),
    )
    if stt_loaded and ner_loaded:
        return body
    return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
