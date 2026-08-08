"""Bounded, temporary-file-backed multipart audio endpoint."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ...pipeline.analyzer import SpeechEntityAnalyzer
from ...pipeline.schemas import AnalysisResult
from ...stt.base import ModelLoadError
from ..dependencies import get_analyzer
from ..errors import ErrorResponse

router = APIRouter()
_CHUNK_SIZE = 64 * 1024


def _validated_suffix(upload: UploadFile, analyzer: SpeechEntityAnalyzer) -> str:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    allowed = {f".{item}" for item in analyzer.audio_config.allowed_extensions}
    if suffix not in allowed:
        raise HTTPException(status_code=422, detail="unsupported_file")
    content_type = (upload.content_type or "").lower()
    acceptable_type = content_type.startswith("audio/") or (
        content_type == "application/octet-stream"
    )
    if not content_type or not acceptable_type:
        raise HTTPException(status_code=422, detail="unsupported_file")
    return suffix


@router.post(
    "/analyze-audio",
    response_model=AnalysisResult,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_audio(
    file: Annotated[UploadFile, File(...)],
    analyzer: Annotated[SpeechEntityAnalyzer, Depends(get_analyzer)],
) -> AnalysisResult:
    """Stream one upload into a bounded temporary file and always remove it."""
    if not analyzer.ner_predictor.loaded:
        raise ModelLoadError("NER model is unavailable.")
    if analyzer.stt_service is None or not analyzer.stt_service.loaded:
        raise ModelLoadError("STT model is unavailable.")
    suffix = _validated_suffix(file, analyzer)
    temporary = NamedTemporaryFile(prefix="uzbek-upload-", suffix=suffix, delete=False)
    temporary_path = Path(temporary.name)
    written = 0
    try:
        while chunk := await file.read(_CHUNK_SIZE):
            written += len(chunk)
            if written > analyzer.audio_config.max_upload_bytes:
                raise HTTPException(status_code=413, detail="upload_too_large")
            temporary.write(chunk)
        temporary.close()
        if written == 0:
            raise HTTPException(status_code=400, detail="empty_audio")
        return analyzer.analyze_audio(temporary_path)
    finally:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        await file.close()
