"""
get_atlases_from_db.py - standalone script called via metatlas2.sh get-atlases

Two sub-commands are available:

  fetch   - fetch one or more atlases by UID and write each to a CSV file
             (original behaviour, now explicit sub-command)

  query  - filter the atlases table by metadata attributes and print a
             summary table so the user can identify the UID(s) they need

Examples
--------
# Save a single atlas to $HOME/<uid>.csv
metatlas2.sh get-atlases fetch --atlas_uids <uid>

# Save two atlases to an explicit directory
metatlas2.sh get-atlases fetch --atlas_uids uid1,uid2 --output_path /path/to/dir

# List all C18 positive-mode atlases created by jsmith
metatlas2.sh get-atlases query --chromatography C18 --polarity positive --created_by jsmith

# List every atlas in the database (no filters)
metatlas2.sh get-atlases query
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg

logger = lcf.get_logger("workflow_objects")

# Columns printed by the query sub-command (in display order)
_QUERY_DISPLAY_COLUMNS = [
    "atlas_uid",
    "atlas_name",
    "chromatography",
    "polarity",
    "analysis_type",
    "analysis_name",
    "atlas_type",
    "created_by",
    "created_date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_values(raw_value: str) -> List[str]:
    """Parse a comma-separated CLI argument into non-empty values."""
    values = [value.strip() for value in raw_value.split(",")]
    return [value for value in values if value]


def _resolve_output_paths(atlas_uids: List[str], output_path: Optional[str]) -> List[Path]:
    """Resolve output CSV paths for each atlas UID."""
    if not output_path:
        return [Path.home() / f"{uid}.csv" for uid in atlas_uids]

    # Comma-separated explicit output file paths.
    if "," in output_path:
        output_paths = [Path(path).expanduser() for path in _parse_csv_values(output_path)]
        if len(output_paths) != len(atlas_uids):
            raise ValueError(
                "Number of output paths must match number of atlas UIDs when using "
                "comma-separated output paths."
            )
        return output_paths

    output = Path(output_path).expanduser()
    if len(atlas_uids) == 1 and output.suffix.lower() == ".csv":
        return [output]

    # Treat a single path as an output directory for one or more atlas UIDs.
    return [output / f"{uid}.csv" for uid in atlas_uids]


def _resolve_database_path(database_path: Optional[str]) -> str:
    """Return *database_path* unchanged, or look it up from the environment."""
    if database_path is not None:
        return database_path
    paths = rtg.set_up_paths(config={})
    return paths["main_db_path"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_atlases(
    atlas_uids: List[str],
    output_path: Optional[str] = None,
    database_path: Optional[str] = None,
) -> None:
    """Extract atlas DataFrames by UID and save each to CSV."""
    if not atlas_uids:
        raise ValueError("At least one atlas UID is required.")

    db = _resolve_database_path(database_path)
    output_paths = _resolve_output_paths(atlas_uids, output_path)

    logger.info(
        "Extracting %d atlas(es) from %s",
        len(atlas_uids),
        db,
    )

    for atlas_uid, csv_path in zip(atlas_uids, output_paths):
        atlas_df = dbi.get_atlas_compounds_table(db, atlas_uid)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        atlas_df.to_csv(csv_path, index=False)
        logger.info("Saved atlas %s to %s", atlas_uid, csv_path)


def query_atlases(
    database_path: Optional[str] = None,
    chromatography: Optional[str] = None,
    polarity: Optional[str] = None,
    analysis_type: Optional[str] = None,
    analysis_name: Optional[str] = None,
    created_by: Optional[str] = None,
) -> None:
    """Print atlas metadata rows matching the supplied filter criteria."""
    db = _resolve_database_path(database_path)

    logger.info("Querying atlas metadata in %s", db)

    df = dbi.query_atlases_metadata(
        database_path=db,
        chromatography=chromatography,
        polarity=polarity,
        analysis_type=analysis_type,
        analysis_name=analysis_name,
        created_by=created_by,
    )

    if df.empty:
        print("No atlases found matching the supplied filters.")
        return

    # Keep only columns that exist in the result (graceful for older DBs)
    display_cols = [c for c in _QUERY_DISPLAY_COLUMNS if c in df.columns]
    display_df = df[display_cols]

    # Pretty-print with pandas; widen the display so UIDs aren't truncated
    import pandas as pd
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 0,
        "display.max_colwidth", 80,
    ):
        print(display_df.to_string(index=False))

    print(f"\n{len(df)} atlas(es) found.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="get_atlases_from_db",
        description=(
            "Interact with atlas records in the metatlas2 database.\n\n"
            "Use the 'query' sub-command to discover atlas UIDs by filtering on\n"
            "metadata attributes, then use the 'fetch' sub-command to export the\n"
            "full compound table for one or more atlases to CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Path to the database file. Defaults to main_db_path from METATLAS_DATA_DIR.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.required = True

    # -- fetch sub-command ---------------------------------------------------
    fetch_parser = subparsers.add_parser(
        "fetch",
        parents=[common],
        help="Fetch atlas(es) by UID and fetch compound tables to CSV.",
        description="Fetch one or more atlases by UID and write each compound table to a CSV file.",
    )
    fetch_parser.add_argument(
        "--atlas_uids",
        required=True,
        type=str,
        help="Comma-separated atlas UID list (e.g. uid1,uid2,uid3).",
    )
    fetch_parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Optional output location. Defaults to $HOME/<uid>.csv for each UID. "
            "Accepts a directory path, a single .csv path (single UID only), or "
            "comma-separated output file paths matching the number of UIDs."
        ),
    )

    # -- query sub-command --------------------------------------------------
    query_parser = subparsers.add_parser(
        "query",
        parents=[common],
        help="List atlas metadata matching filter criteria.",
        description=(
            "Filter the atlases table by one or more metadata attributes and print\n"
            "a summary table.  All filters are optional; omitting all filters returns\n"
            "every atlas in the database.  Matching is case-insensitive and partial\n"
            "(substring match), so '--chromatography C18' matches 'HILIC-C18' etc."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    query_parser.add_argument(
        "--chromatography",
        type=str,
        default=None,
        help="Filter by chromatography method (partial, case-insensitive).",
    )
    query_parser.add_argument(
        "--polarity",
        type=str,
        default=None,
        help="Filter by polarity (partial, case-insensitive).",
    )
    query_parser.add_argument(
        "--analysis_type",
        type=str,
        default=None,
        help="Filter by analysis type (partial, case-insensitive).",
    )
    query_parser.add_argument(
        "--analysis_name",
        type=str,
        default=None,
        help="Filter by analysis name (partial, case-insensitive).",
    )
    query_parser.add_argument(
        "--created_by",
        type=str,
        default=None,
        help="Filter by creator username (partial, case-insensitive).",
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand == "fetch":
        get_atlases(
            atlas_uids=_parse_csv_values(args.atlas_uids),
            output_path=args.output_path,
            database_path=args.database_path,
        )
    elif args.subcommand == "query":
        query_atlases(
            database_path=args.database_path,
            chromatography=args.chromatography,
            polarity=args.polarity,
            analysis_type=args.analysis_type,
            analysis_name=args.analysis_name,
            created_by=args.created_by,
        )
    else:
        parser.print_help()
        sys.exit(1)
