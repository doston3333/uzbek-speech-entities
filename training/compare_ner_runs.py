"""Compare clean and augmented NER runs using validation evidence, then promote inference files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from uzbek_speech_entities.config import resolve_project_path
from uzbek_speech_entities.ner.model_selection import (
    _read_json,
    build_comparison_report,
    finalize_selection,
    load_run,
    promote_selected_run,
    select_run,
    write_json_atomic,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths without importing model libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run", type=Path, default=Path("models/ner/clean"))
    parser.add_argument("--augmented-run", type=Path, default=Path("models/ner/augmented"))
    parser.add_argument("--report", type=Path, default=Path("reports/ner_model_comparison.json"))
    parser.add_argument("--final", type=Path, default=Path("models/ner/final"))
    parser.add_argument(
        "--finalization-evidence",
        type=Path,
        help="Required only for final-model promotion; omission writes a provisional report.",
    )
    parser.add_argument("--overwrite-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Create an atomic report and promote the validation-selected inference model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    clean = load_run("clean", resolve_project_path(args.clean_run))
    augmented = load_run("augmented", resolve_project_path(args.augmented_run))
    selection = select_run(clean, augmented)
    report_path = resolve_project_path(args.report)
    if args.finalization_evidence is None:
        write_json_atomic(report_path, build_comparison_report(selection))
        LOGGER.info(
            "Selected %s provisionally; wrote %s without promoting a final model",
            selection.selected.name,
            report_path,
        )
        return
    finalization_evidence = _read_json(resolve_project_path(args.finalization_evidence))
    finalized, _ = finalize_selection(selection, finalization_evidence)
    write_json_atomic(
        report_path,
        build_comparison_report(selection, finalization_evidence=finalization_evidence),
    )
    final_path = promote_selected_run(
        selection,
        resolve_project_path(args.final),
        report_path,
        finalization_evidence=finalization_evidence,
        overwrite_final=args.overwrite_final,
    )
    LOGGER.info(
        "Selected %s and promoted inference artifacts to %s", finalized.selected.name, final_path
    )


if __name__ == "__main__":
    main()
