"""Download the pinned Uzbek NER TSV with checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from uzbek_speech_entities.ner.dataset import (
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    DATASET_URL,
    sha256_file,
)

EXPECTED_SHA256 = DATASET_SHA256
DEFAULT_OUTPUT = Path("data/raw") / DATASET_FILENAME


def download_dataset(
    *,
    url: str = DATASET_URL,
    output: Path = DEFAULT_OUTPUT,
    expected_sha256: str = EXPECTED_SHA256,
    retries: int = 3,
    timeout: float = 30.0,
) -> Path:
    """Download to a temporary file, verify it, then atomically publish it."""
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256 hex digest")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) == expected_sha256:
        return output

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
            ) as temporary_file:
                temporary_name = temporary_file.name
                digest = hashlib.sha256()
                with urlopen(url, timeout=timeout) as response:  # noqa: S310 - CLI-configured dataset URL.
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        temporary_file.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(
                    f"checksum mismatch: expected {expected_sha256}, got {digest.hexdigest()}"
                )
            Path(temporary_name).replace(output)
            return output
        except (HTTPError, URLError, OSError, ValueError) as error:
            last_error = error
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(attempt)

    raise RuntimeError(f"failed to download {url!r} after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATASET_URL, help="Pinned TSV URL to download.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Local TSV destination."
    )
    parser.add_argument("--sha256", default=EXPECTED_SHA256, help="Expected SHA-256 digest.")
    parser.add_argument("--retries", type=int, default=3, help="Maximum download attempts.")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request timeout in seconds."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = download_dataset(
        url=args.url,
        output=args.output,
        expected_sha256=args.sha256,
        retries=args.retries,
        timeout=args.timeout,
    )
    print(f"Verified {DATASET_ID}@{DATASET_REVISION}: {output} ({sha256_file(output)})")


if __name__ == "__main__":
    main()
