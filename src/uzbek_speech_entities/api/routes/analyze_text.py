"""JSON text-analysis endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from ...pipeline.analyzer import SpeechEntityAnalyzer
from ...pipeline.schemas import AnalysisResult
from ...stt.base import ModelLoadError
from ..dependencies import get_analyzer
from ..errors import ErrorResponse

router = APIRouter()


class TextAnalysisRequest(BaseModel):
    """Delimited JSON request body; semantic validation remains in the pipeline."""

    model_config = ConfigDict(extra="forbid")
    text: str


@router.post(
    "/analyze-text",
    response_model=AnalysisResult,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_text(
    payload: TextAnalysisRequest,
    analyzer: Annotated[SpeechEntityAnalyzer, Depends(get_analyzer)],
    request: Request,
) -> AnalysisResult:
    """Analyze text through the injected application-scoped analyzer."""
    if not analyzer.ner_predictor.loaded:
        raise ModelLoadError("NER model is unavailable.")
    async with request.app.state.inference_lock:
        return await run_in_threadpool(analyzer.analyze_text, payload.text)
