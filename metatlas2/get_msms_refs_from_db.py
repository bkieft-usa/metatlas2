"""Export MSMS reference spectra from the main metatlas DuckDB.

Usage examples
--------------
Export all spectra to JSONL (default):

    metatlas2 get-msms-refs --output_path /tmp/msms_refs_export.json

Export only "metatlas" database, positive-mode spectra for two compounds:

    metatlas2 get-msms-refs \\
        --database_filter metatlas \\
        --polarity positive \\
        --inchikeys JPIJQSOTBSSVTP-STHAYSLISA-N,HDYANYHVCAPMJV-LXQIFKJMSA-N \\
        --output_path /tmp/subset.json

Export to tab-separated text instead of JSONL:

    metatlas2 get-msms-refs --tab-out --output_path /tmp/msms_refs_export.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg

logger = lcf.get_logger("workflow_objects")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_database_path(database_path: str | None) -> str:
    """Return *database_path* unchanged, or look it up from the environment."""
    if database_path is not None:
        return database_path
    paths = rtg.set_up_paths(config={})
    return paths["main_db_path"]


def _parse_inchikeys(raw: str) -> list[str]:
    """Split a comma-separated InChI-key string into a non-empty list."""
    keys = [k.strip() for k in raw.split(",")]
    return [k for k in keys if k]


# ---------------------------------------------------------------------------
# Core export functions
# ---------------------------------------------------------------------------

def export_msms_refs_jsonl(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Write *rows* to *output_path* in JSONL format.

    Each line is a JSON object with all columns from
    ``reference_fragmentation_data``.  The ``mz`` and ``intensities`` fields
    are written as JSON arrays.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d spectra to %s (JSONL)", len(rows), output_path)


def export_msms_refs_tsv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Write *rows* to *output_path* as a tab-separated text file.

    The ``mz`` and ``intensities`` columns are serialised as JSON arrays so
    the file can be round-tripped back into the database via the existing
    ``convert_msms_refs_tab_to_json.py`` converter.
    """
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
        logger.info("No rows to write; created empty file at %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            tsv_row = dict(row)
            # Serialise array columns as JSON strings for TSV compatibility
            tsv_row["mz"] = json.dumps(row["mz"])
            tsv_row["intensities"] = json.dumps(row["intensities"])
            writer.writerow(tsv_row)

    logger.info("Wrote %d spectra to %s (TSV)", len(rows), output_path)


def get_msms_refs(
    output_path: str,
    tab_out: bool = False,
    inchikeys: list[str] | None = None,
    database_filter: str | None = None,
    polarity: str | None = None,
    database_path: str | None = None,
) -> None:
    """Query ``reference_fragmentation_data`` and export to JSONL or TSV.

    Args:
        output_path:     Destination file path.  Defaults to
                         ``$HOME/msms_refs_export.json`` (or ``.tsv`` with
                         ``--tab-out``).
        tab_out:         If ``True``, write TSV instead of JSONL.
        inchikeys:       Optional list of InChI keys to restrict the export.
        database_filter: Optional exact-match filter on the ``database`` column.
        polarity:        Optional exact-match filter on the ``polarity`` column.
        database_path:   Path to the main DuckDB.  Resolved from
                         ``METATLAS_DATA_DIR`` when ``None``.
    """
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=None,
        log_to_stdout=True,
        reconfigure_existing=True,
    ):
        db = _resolve_database_path(database_path)

        logger.info(
            "Querying reference_fragmentation_data in %s"
            + (f" (database='{database_filter}')" if database_filter else "")
            + (f" (polarity='{polarity}')" if polarity else "")
            + (f" ({len(inchikeys)} inchi_keys)" if inchikeys is not None else "")
            + "...",
            db,
        )

        rows = dbi.query_msms_refs(
            db_path=db,
            inchi_keys=inchikeys,
            database_filter=database_filter,
            polarity=polarity,
        )

        if not rows:
            logger.warning(
                "No spectra found matching the supplied filters. Nothing written."
            )
            return

        out = Path(output_path).expanduser()
        if tab_out:
            export_msms_refs_tsv(rows, out)
        else:
            export_msms_refs_jsonl(rows, out)

        logger.info("Export complete: %d spectra written to %s", len(rows), out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="get_msms_refs_from_db",
        description=(
            "Export MSMS reference spectra from the reference_fragmentation_data\n"
            "table in the main metatlas DuckDB to a JSONL file (default) or a\n"
            "tab-separated text file (--tab-out).\n\n"
            "All filter arguments are optional; omitting them exports every row."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help=(
            "Path to the main metatlas DuckDB file. "
            "Defaults to the path derived from the METATLAS_DATA_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Destination file path. "
            "Defaults to $HOME/msms_refs_export.json (or .tsv with --tab-out)."
        ),
    )
    parser.add_argument(
        "--tab-out",
        action="store_true",
        default=False,
        help=(
            "Write output as a tab-separated text file instead of JSONL. "
            "The mz and intensities columns are serialised as JSON arrays."
        ),
    )
    parser.add_argument(
        "--inchikeys",
        type=str,
        default=None,
        help=(
            "Comma-separated list of InChI keys to export "
            "(e.g. JPIJQSOTBSSVTP-STHAYSLISA-N,HDYANYHVCAPMJV-LXQIFKJMSA-N). "
            "Omit to export all InChI keys."
        ),
    )
    parser.add_argument(
        "--database_filter",
        type=str,
        default=None,
        help=(
            "Exact-match filter on the 'database' column "
            "(e.g. 'metatlas', 'gnps'). Omit to include all databases."
        ),
    )
    parser.add_argument(
        "--polarity",
        type=str,
        default=None,
        help=(
            "Exact-match filter on the 'polarity' column "
            "('positive' or 'negative'). Omit to include both polarities."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    inchikeys = _parse_inchikeys(args.inchikeys) if args.inchikeys else None

    # Resolve default output path
    if args.output_path:
        output_path = args.output_path
    else:
        ext = ".tsv" if args.tab_out else ".json"
        output_path = str(Path.home() / f"msms_refs_export{ext}")

    get_msms_refs(
        output_path=output_path,
        tab_out=args.tab_out,
        inchikeys=inchikeys,
        database_filter=args.database_filter,
        polarity=args.polarity,
        database_path=args.database_path,
    )
