"""Runtime and peak resident-memory measurement for local evaluation jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

import psutil  # type: ignore[import-untyped]

T = TypeVar("T")
_MEBIBYTE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement(Generic[T]):
    value: T
    elapsed_seconds: float
    starting_rss_mb: float
    peak_rss_mb: float
    peak_rss_delta_mb: float


def measure_runtime(
    operation: Callable[[], T], *, sample_interval_seconds: float = 0.02
) -> RuntimeMeasurement[T]:
    """Measure elapsed time and process RSS while ``operation`` runs."""
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    process = psutil.Process()
    starting_rss = process.memory_info().rss
    peak_rss = starting_rss
    stopped = threading.Event()

    def sample() -> None:
        nonlocal peak_rss
        while not stopped.wait(sample_interval_seconds):
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample, name="phase8-rss-sampler", daemon=True)
    started = monotonic()
    sampler.start()
    try:
        value = operation()
    finally:
        stopped.set()
        sampler.join(timeout=max(1.0, sample_interval_seconds * 4))
        try:
            peak_rss = max(peak_rss, process.memory_info().rss)
        except psutil.Error:
            pass
    elapsed = max(0.0, monotonic() - started)
    return RuntimeMeasurement(
        value=value,
        elapsed_seconds=elapsed,
        starting_rss_mb=starting_rss / _MEBIBYTE,
        peak_rss_mb=peak_rss / _MEBIBYTE,
        peak_rss_delta_mb=max(0.0, (peak_rss - starting_rss) / _MEBIBYTE),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the STT evaluator, whose output includes every required runtime metric."""
    from evaluation.evaluate_stt import main as evaluate_stt_main

    return evaluate_stt_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
