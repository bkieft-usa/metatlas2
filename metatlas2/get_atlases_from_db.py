import argparse
from pathlib import Path
from typing import List, Optional

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg

logger = lcf.get_logger("workflow_objects")


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


def get_atlases(
    atlas_uids: List[str],
    output_path: Optional[str] = None,
    database_path: Optional[str] = None,
) -> None:
    """Extract atlas DataFrames by UID and save each to CSV."""
    if not atlas_uids:
        raise ValueError("At least one atlas UID is required.")

    if database_path is None:
        paths = rtg.set_up_paths(config={})
        database_path = paths["main_db_path"]

    output_paths = _resolve_output_paths(atlas_uids, output_path)

    logger.info(
        "Extracting %d atlas(es) from %s",
        len(atlas_uids),
        database_path,
    )

    for atlas_uid, csv_path in zip(atlas_uids, output_paths):
        atlas_df = dbi.get_atlas_compounds_table(database_path, atlas_uid)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        atlas_df.to_csv(csv_path, index=False)
        logger.info("Saved atlas %s to %s", atlas_uid, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract one or more atlas DataFrames from the database and save them to CSV files."
        )
    )
    parser.add_argument(
        "--atlas_uids",
        required=True,
        type=str,
        help="Comma-separated atlas UID list (e.g. uid1,uid2,uid3).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Optional output location. Defaults to $HOME/<uid>.csv for each UID. "
            "Accepts a directory path, a single .csv path (single UID only), or "
            "comma-separated output file paths matching the number of UIDs."
        ),
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=None,
        help="Optional path to the database file. Defaults to main_db_path from METATLAS_DATA_DIR.",
    )

    args = parser.parse_args()
    get_atlases(
        atlas_uids=_parse_csv_values(args.atlas_uids),
        output_path=args.output_path,
        database_path=args.database_path,
    )
