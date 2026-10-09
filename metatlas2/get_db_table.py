from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg

logger = lcf.get_logger("get_db_table")

# Tables that live in the main metatlas DuckDB.
# Used in --help text; the live list is always fetched from the DB at runtime.
_KNOWN_MAIN_DB_TABLES = [
    "atlas_compound_associations",
    "atlases",
    "compound_mzrt",
    "compounds",
    "lcmsruns",
    "manual_curation",
    "ms1_data",
    "ms2_data",
    "msms_refs",
    "project_config",
    "projects",
    "rt_alignment",
    "workflow_runs",
]


def _resolve_database_path(database_path: str | None) -> str:
    """Return *database_path* unchanged, or look it up from the environment."""
    if database_path is not None:
        return database_path
    paths = rtg.set_up_paths(config={})
    return paths["main_db_path"]


def _get_valid_tables(db_path: str) -> list[str]:
    """Return a sorted list of table names present in *db_path*."""
    with dbi.get_db_connection(db_path, read_only=True) as conn:
        rows = conn.execute("SHOW TABLES").fetchall()
    return sorted(row[0] for row in rows)


def export_table_to_csv(
    table_name: str,
    output_path: str | None = None,
    database_path: str | None = None,
) -> None:
    """Export *table_name* from the main metatlas DuckDB to a CSV file.

    The table name is validated against the live list of tables in the
    database before the query is executed, so an invalid name produces a
    clear error rather than a SQL exception.

    Args:
        table_name:    Name of the table to export (e.g. ``"msms_refs"``).
        output_path:   Destination CSV path.  Defaults to
                       ``$HOME/<table_name>_export.csv``.
        database_path: Path to the main DuckDB file.  Resolved from
                       ``METATLAS_DATA_DIR`` when ``None``.
    """
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=None,
        log_to_stdout=True,
        reconfigure_existing=True,
    ):
        db = _resolve_database_path(database_path)

        valid_tables = _get_valid_tables(db)
        if table_name not in valid_tables:
            raise ValueError(
                f"Table '{table_name}' not found in {db}.\n"
                f"Available tables: {', '.join(valid_tables)}"
            )

        logger.info("Querying table '%s' from %s ...", table_name, db)

        with dbi.get_db_connection(db, read_only=True) as conn:
            # table_name is allowlisted above — safe to interpolate
            df = conn.execute(f"SELECT * FROM {table_name}").df()

        out = Path(output_path).expanduser() if output_path else (
            Path.home() / f"{table_name}_export.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        logger.info(
            "Wrote %d row(s) to %s",
            len(df),
            out,
        )

