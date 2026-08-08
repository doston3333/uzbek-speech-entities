"""Projection of model spans from an analysis view back to display offsets."""

from __future__ import annotations

from .aligned_tokens import AnalysisNormalization


def project_analysis_span(
    view: AnalysisNormalization, start: int, end: int
) -> tuple[int, int] | None:
    """Project an overlapping analysis span exactly, rejecting hard boundaries."""
    if start < 0 or end <= start or end > len(view.analysis_text):
        return None
    covered = [
        token
        for token in view.tokens
        if token.analysis_start < end and start < token.analysis_end
    ]
    if not covered or any(
        token.source_start is None
        or token.source_end is None
        or token.hard_boundary_before
        or token.hard_boundary_after
        for token in covered
    ):
        return None
    source_start = min(token.source_start for token in covered if token.source_start is not None)
    source_end = max(token.source_end for token in covered if token.source_end is not None)
    if not 0 <= source_start < source_end <= len(view.display_text):
        return None
    return source_start, source_end
