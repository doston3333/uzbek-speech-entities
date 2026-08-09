"""Early, route-scoped ASGI request-size guard for multipart audio uploads."""

from __future__ import annotations

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import error_response

_AUDIO_PATH = "/api/analyze-audio"
# Multipart boundaries and part headers are outside the endpoint's exact file-byte limit.
MULTIPART_ENVELOPE_ALLOWANCE_BYTES = 16 * 1024


class _RequestBodyLimitExceeded(HTTPException, OSError):
    """Signal a body limit breach after Starlette closes partial multipart files."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="upload_too_large")


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name.lower() == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return length if length >= 0 else None
    return None


async def _send_upload_too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = error_response(413, "upload_too_large", "Audio file exceeds the upload limit.")
    await response(scope, receive, send)


class AudioRequestBodyLimitMiddleware:
    """Reject oversize audio requests before multipart parsing can spool their bodies."""

    def __init__(self, app: ASGIApp, *, max_file_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_file_bytes + MULTIPART_ENVELOPE_ALLOWANCE_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != _AUDIO_PATH:
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.max_request_bytes:
            await _send_upload_too_large(scope, receive, send)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes

            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_bytes:
                    raise _RequestBodyLimitExceeded()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyLimitExceeded:
            if response_started:
                raise
            await _send_upload_too_large(scope, receive, send)
