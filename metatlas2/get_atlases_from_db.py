from __future__ import annotations

import pandas as pd
import logging

from pathlib import Path

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg

logger = lcf.get_logger("workflow_objects")

_QUERY_DISPLAY_COLUMNS = [
    "atlas_uid",
    "atlas_name",
    "chromatography",
    "polarity",
    "atlas_category",
    "created_by",
    "created_date",
]

def _parse_csv_values(raw_value: str) -> list[str]:
    """Parse a comma-separated CLI argument into non-empty values."""
    values = [value.strip() for value in raw_value.split(",")]
    return [value for value in values if value]

def _resolve_output_paths(atlas_uids: list[str], output_path: str | None) -> list[Path]:
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

def _resolve_database_path(database_path: str | None) -> str:
    """Return *database_path* unchanged, or look it up from the environment."""
    if database_path is not None:
        return database_path
    paths = rtg.set_up_paths(config={})
    return paths["main_db_path"]

def get_atlases(
    atlas_uids: list[str],
    output_path: str | None = None,
    database_path: str | None = None,
) -> None:
    """Extract atlas DataFrames by UID and save each to CSV."""
    with lcf.temporary_logging(log_level=logging.INFO, log_file=None, log_to_stdout=True, reconfigure_existing=True):
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
            logger.info(f"Saved atlas {atlas_uid} to {csv_path}")

def query_atlases(
    database_path: str | None = None,
    chromatography: str | None = None,
    polarity: str | None = None,
    atlas_category: str | None = None,
    atlas_name: str | None = None,
    created_by: str | None = None,
) -> None:
    """Print atlas metadata rows matching the supplied filter criteria."""
    with lcf.temporary_logging(log_level=logging.INFO, log_file=None, log_to_stdout=True, reconfigure_existing=True):
        db = _resolve_database_path(database_path)

        logger.info(f"Querying atlas metadata in {db}")

        df = dbi.query_atlases_metadata(
            database_path=db,
            chromatography=chromatography,
            polarity=polarity,
            atlas_category=atlas_category,
            atlas_name=atlas_name,
            created_by=created_by,
        )

        if df.empty:
            print("No atlases found matching the supplied filters.")
            return

        # Keep only columns that exist in the result (graceful for older DBs)
        display_cols = [c for c in _QUERY_DISPLAY_COLUMNS if c in df.columns]
        display_df = df[display_cols]

        # Pretty-print with pandas; widen the display so UIDs aren't truncated
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 0,
            "display.max_colwidth", 80,
        ):
            print(display_df.to_string(index=False))

        print(f"\n{len(df)} atlas(es) found.")

