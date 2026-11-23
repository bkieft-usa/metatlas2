"""
metatlas2/standards_annotation/standards_annotation_db.py

Project DuckDB creation and all read/write helpers for the standards-annotation
workflow.  Provides resumability: extraction is skipped on re-run if the DB
already contains data.

Extraction layer (written once by ``standards-annotation``; never modified by review):
  sa_compounds          — one row per (compound_name, adduct): stable identity
  sa_extraction_targets — one row per (compound_uid, mz_rt_uid, filename)
  sa_ms1 / sa_ms2       — extracted MS1/MS2 data (wide format)
  sa_tic, sa_injbl_ms1  — TIC per file and injection-blank EICs
  sa_peaks              — detected peaks plus manual peaks (``created_in_annotation_run``)
  sa_peak_ms2           — best MS2 scan per peak and source file
  sa_ms1_sources        — selected MS1 file per compound, chromatography, polarity

Annotation runs (one per review iteration; ``draft`` until summarized, then ``frozen``):
  sa_annotation_runs    — run number, status, dates, output directory, GUI checkpoint
  sa_run_selections     — selected peaks and analyst notes per run
  sa_run_no_good_peaks  — explicit no-good-peaks choices per run
  sa_run_compound_notes — analyst notes per run, compound and chromatography
  sa_run_state          — per-run (compound_uid, chromatography) annotation status

RT alignment runs (each tied to one frozen annotation run):
  sa_alignment_runs, sa_alignment_models, sa_alignment_qc_points, sa_aligned_peaks

Deposit content (per alignment run; hand edits to deposit previews are written back here):
  sa_atlas_entries, sa_msms_entries

Deposits (one per preview; records exactly what went to the sheet and msms_refs):
  sa_deposits, sa_deposit_atlas_rows, sa_deposit_msms_rows, sa_deposit_changes
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

import metatlas2.shared_tools.logging_config as lcf
import metatlas2.shared_tools.workflow_objects as wfo
from metatlas2.shared_tools.atlas_sheet import ATLAS_COLUMNS
from metatlas2.shared_tools.utils import get_provenance

logger = lcf.get_logger("standards_annotation_db")


# ---------------------------------------------------------------------------
# Deposit file schemas
# ---------------------------------------------------------------------------

#: Columns of the MS/MS reference ``.tab`` importer format.
MSMS_COLUMNS = [
    "database", "id", "name", "spectrum", "decimal", "precursor_mz", "polarity",
    "adduct", "fragmentation_method", "collision_energy", "instrument", "instrument_type",
    "formula", "exact_mass", "inchi_key", "inchi", "smiles",
]

#: SQL types for each centralized-atlas column (keys must equal ATLAS_COLUMNS).
ATLAS_COLUMN_TYPES: dict[str, str] = {
    column: ("DOUBLE" if column in {"mz", "rt_peak", "rt_min", "rt_max", "mz_tolerance"} else "TEXT")
    for column in ATLAS_COLUMNS
}

#: SQL types for each MS/MS reference column (keys must equal MSMS_COLUMNS).
MSMS_COLUMN_TYPES: dict[str, str] = {
    column: {"decimal": "INTEGER", "precursor_mz": "DOUBLE", "exact_mass": "DOUBLE"}.get(column, "TEXT")
    for column in MSMS_COLUMNS
}

#: Linkage columns carried in deposit previews and stripped before upload/import.
ATLAS_LINK_COLUMNS = ["sa_entry_id", "sa_peak_uid"]
MSMS_LINK_COLUMNS = ["sa_entry_id", "sa_peak_uid", "sa_ce_label", "sa_measured_collision_energy"]


def _quoted_columns(types: dict[str, str]) -> str:
    return ",\n".join(f'    "{column}" {sql_type}' for column, sql_type in types.items())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_output_root(config: dict[str, Any]) -> Path:
    """Return the root output directory for this annotation run.

    Path: ``$METATLAS_DATA_DIR/projects/reference_standards_annotation/<username>/<standards_run_name>/``

    Args:
        config: Parsed config dict from
                :func:`metatlas2.shared_tools.load_tools.load_standards_config`.

    Returns:
        :class:`pathlib.Path` to the project output root.
    """
    run_name = config["project"]["standards_run_name"]
    username = get_provenance()["analyst"]
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    if data_dir:
        return (
            Path(data_dir)
            / "projects"
            / "reference_standards_annotation"
            / username
            / run_name
        )
    else:
        raise FileNotFoundError("METATLAS_DATA_DIR not set; cannot determine project output root")


def get_raw_data_root(config: dict[str, Any]) -> str:
    """Return the root directory for raw LCMS data (JGI convention).

    Returns ``$METATLAS_DATA_DIR/raw_data/<owner>`` when ``METATLAS_DATA_DIR``
    is set, otherwise raises.
    """
    owner = config["project"].get("owner", "jgi")
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    if data_dir:
        return str(Path(data_dir) / "raw_data" / owner)
    else:
        raise RuntimeError(f"METATLAS_DATA_DIR not set; cannot determine raw data root for owner '{owner}'")


def get_project_db_path(config: dict[str, Any]) -> str:
    """Return the absolute path to the project DuckDB file.

    Args:
        config: Parsed config dict.

    Returns:
        String path to ``<output_root>/<standards_run_name>.duckdb``.
    """
    run_name = config["project"]["standards_run_name"]
    root = get_project_output_root(config)
    return str(root / f"{run_name}.duckdb")


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------

def create_project_db(db_path: str) -> None:
    """Create the project DuckDB and all standards-annotation tables.

    Idempotent — safe to call on an existing database.

    Schema design:
      - ``sa_compounds``: one row per (compound_name, adduct) — stable identity.
        PK is ``compound_uid`` (uuid5 of compound_name|adduct).
      - ``sa_extraction_targets``: bridges compound_uid → mz_rt_uid → filename.
        The extraction engine uses mz_rt_uid as its join key; this table
        records which file each mz_rt_uid was extracted from.
      - ``sa_ms1`` / ``sa_ms2``: keyed by (mz_rt_uid, filename) as produced
        by the extraction engine.
      - ``sa_peaks``: carries both compound_uid (for grouping) and mz_rt_uid
        (to look up the source EIC/MS2 row).
      - ``sa_annotation_runs`` and ``sa_run_*``: one immutable snapshot of
        analyst decisions per review iteration (see module docstring).
      - ``sa_alignment_*``, ``sa_*_entries``, ``sa_deposit*``: alignment
        models and results, deposit content, and deposit records.

    Args:
        db_path: Absolute path to the DuckDB file.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as conn:

        # ------------------------------------------------------------------
        # sa_compounds — stable compound+adduct identity
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_compounds (
                compound_uid     TEXT PRIMARY KEY,
                compound_name    TEXT NOT NULL,
                inchi_key        TEXT,
                smiles           TEXT,
                inchi            TEXT,
                adduct           TEXT NOT NULL,
                polarity         TEXT,
                mz_theoretical   REAL
            )
        """)

        # ------------------------------------------------------------------
        # sa_extraction_targets — bridges compound_uid ↔ mz_rt_uid ↔ filename
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_extraction_targets (
                mz_rt_uid              TEXT PRIMARY KEY,
                compound_uid           TEXT NOT NULL REFERENCES sa_compounds(compound_uid),
                filename               TEXT NOT NULL,
                chromatography         TEXT NOT NULL,
                rt_min                 REAL,
                rt_max                 REAL,
                collision_energy_label TEXT DEFAULT '',
                annotation_notes       TEXT DEFAULT ''
            )
        """)

        # ------------------------------------------------------------------
        # sa_ms1 — wide-format MS1, one row per (mz_rt_uid, filename)
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_ms1 (
                mz_rt_uid   TEXT NOT NULL,
                filename    TEXT NOT NULL,
                inchi_key   TEXT,
                adduct      TEXT,
                spec_rts    REAL[],
                spec_ints   REAL[],
                spec_mzs    REAL[],
                in_feature  BOOLEAN[],
                PRIMARY KEY (mz_rt_uid, filename)
            )
        """)

        # ------------------------------------------------------------------
        # sa_ms2 — wide-format MS2, one row per (mz_rt_uid, filename, scan_rt)
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_ms2 (
                mz_rt_uid           TEXT NOT NULL,
                scan_rt             REAL NOT NULL,
                frag_mzs            REAL[],
                frag_ints           REAL[],
                in_feature          BOOLEAN,
                filename            TEXT NOT NULL,
                precursor_mz        REAL,
                precursor_intensity REAL,
                collision_energy    REAL,
                inchi_key           TEXT,
                adduct              TEXT,
                PRIMARY KEY (mz_rt_uid, filename, scan_rt)
            )
        """)

        # ------------------------------------------------------------------
        # sa_tic — total ion chromatogram per file
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_tic (
                filename      TEXT NOT NULL,
                rt            REAL NOT NULL,
                tic_intensity REAL,
                PRIMARY KEY (filename, rt)
            )
        """)

        # ------------------------------------------------------------------
        # sa_peaks — one row per detected peak
        # compound_uid: stable identity for grouping
        # mz_rt_uid:    points to the specific extraction target (file+compound+adduct)
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_peaks (
                peak_uid               TEXT PRIMARY KEY,
                compound_uid           TEXT NOT NULL REFERENCES sa_compounds(compound_uid),
                compound_name          TEXT NOT NULL,
                inchi_key              TEXT,
                adduct                 TEXT NOT NULL,
                polarity               TEXT NOT NULL,
                chromatography         TEXT NOT NULL,
                filename               TEXT NOT NULL,
                mz_rt_uid              TEXT NOT NULL REFERENCES sa_extraction_targets(mz_rt_uid),
                mz_theoretical         REAL,
                rt_peak                REAL NOT NULL,
                rt_min                 REAL NOT NULL,
                rt_max                 REAL NOT NULL,
                peak_intensity         REAL NOT NULL,
                mz_observed            REAL,
                peak_num               INTEGER NOT NULL DEFAULT 1,
                is_manual              BOOLEAN NOT NULL DEFAULT FALSE,
                collision_energy_label TEXT DEFAULT '',
                created_in_annotation_run INTEGER,
                UNIQUE (mz_rt_uid, rt_peak, filename)
            )
        """)

        # ------------------------------------------------------------------
        # sa_peak_ms2 — best MS2 scan per peak and source file
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_peak_ms2 (
                peak_uid              TEXT NOT NULL REFERENCES sa_peaks(peak_uid),
                scan_rt               REAL,
                precursor_mz          REAL,
                precursor_intensity   REAL,
                collision_energy      REAL,
                frag_mzs              TEXT,
                frag_ints             TEXT,
                filename              TEXT NOT NULL,
                PRIMARY KEY (peak_uid, filename)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_ms1_sources (
                compound_name TEXT NOT NULL,
                chromatography TEXT NOT NULL,
                polarity TEXT NOT NULL,
                filename TEXT NOT NULL,
                PRIMARY KEY (compound_name, chromatography, polarity)
            )
        """)

        # ------------------------------------------------------------------
        # sa_injbl_ms1 — InjBl (injection blank) EIC data
        # Mirrors sa_ms1 schema but keyed on injbl_filename.
        # No in_feature column — full RT window stored for visual context.
        # No peak detection is performed on InjBl data.
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_injbl_ms1 (
                mz_rt_uid      TEXT NOT NULL,
                injbl_filename TEXT NOT NULL,
                inchi_key      TEXT,
                adduct         TEXT,
                spec_rts       REAL[],
                spec_ints      REAL[],
                spec_mzs       REAL[],
                PRIMARY KEY (mz_rt_uid, injbl_filename)
            )
        """)

        _create_review_tables(conn)

    logger.info(f"Project DB ready at {db_path}")


def _create_review_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create annotation-run, alignment-run, and deposit tables.

    These tables have no FOREIGN KEY clauses: DuckDB cannot delete a
    referenced row in the same transaction as its referencing rows, which
    discarding a draft run requires. Referential integrity is enforced by
    the helpers in this module, which are the only writers.
    """
    # ------------------------------------------------------------------
    # Annotation runs: one per review iteration. At most one is 'draft'.
    # ------------------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_annotation_runs (
            annotation_run_id       INTEGER PRIMARY KEY,
            parent_run_id           INTEGER,
            status                  TEXT NOT NULL CHECK (status IN ('draft', 'frozen')),
            created_at              TEXT NOT NULL,
            created_by              TEXT,
            frozen_at               TEXT,
            frozen_by               TEXT,
            output_dir              TEXT,
            gdrive_url              TEXT,
            gui_checkpoint_compound TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_run_selections (
            annotation_run_id INTEGER NOT NULL,
            peak_uid          TEXT NOT NULL,
            analyst_notes     TEXT DEFAULT '',
            selected_by       TEXT,
            selected_at       TEXT,
            PRIMARY KEY (annotation_run_id, peak_uid)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_run_no_good_peaks (
            annotation_run_id INTEGER NOT NULL,
            compound_name     TEXT NOT NULL,
            chromatography    TEXT NOT NULL,
            polarity          TEXT NOT NULL,
            PRIMARY KEY (annotation_run_id, compound_name, chromatography, polarity)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_run_compound_notes (
            annotation_run_id INTEGER NOT NULL,
            compound_name     TEXT NOT NULL,
            chromatography    TEXT NOT NULL,
            analyst_notes     TEXT NOT NULL,
            PRIMARY KEY (annotation_run_id, compound_name, chromatography)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_run_state (
            annotation_run_id INTEGER NOT NULL,
            compound_uid      TEXT NOT NULL,
            chromatography    TEXT NOT NULL,
            annotation_status TEXT DEFAULT 'pending',
            last_updated      TEXT,
            PRIMARY KEY (annotation_run_id, compound_uid, chromatography)
        )
    """)

    # ------------------------------------------------------------------
    # RT alignment runs: one per alignment of a frozen annotation run.
    # ------------------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_alignment_runs (
            alignment_run_id  INTEGER PRIMARY KEY,
            annotation_run_id INTEGER NOT NULL,
            created_at        TEXT NOT NULL,
            created_by        TEXT,
            output_dir        TEXT NOT NULL,
            gdrive_url        TEXT,
            skip_alignment    BOOLEAN NOT NULL,
            tags              TEXT NOT NULL,
            config_snapshot   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_alignment_models (
            alignment_run_id  INTEGER NOT NULL,
            chromatography    TEXT NOT NULL,
            qc_atlas_uid      TEXT,
            qc_atlas_selector TEXT,
            model_type        TEXT,
            degree            INTEGER,
            coefficients      DOUBLE[],
            intercept         DOUBLE,
            "offset"          DOUBLE,
            r2                DOUBLE,
            rmse              DOUBLE,
            equation          TEXT,
            n_points          INTEGER,
            params            TEXT,
            qc_files          TEXT[],
            PRIMARY KEY (alignment_run_id, chromatography)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_alignment_qc_points (
            alignment_run_id INTEGER NOT NULL,
            chromatography   TEXT NOT NULL,
            point_num        INTEGER NOT NULL,
            mz_rt_uid        TEXT,
            compound_uid     TEXT,
            label            TEXT,
            inchi_key        TEXT,
            adduct           TEXT,
            atlas_rt_peak    DOUBLE,
            atlas_rt_min     DOUBLE,
            atlas_rt_max     DOUBLE,
            atlas_mz         DOUBLE,
            exp_rt_peak      DOUBLE,
            intensity_peak   DOUBLE,
            predicted_rt     DOUBLE,
            residual         DOUBLE,
            PRIMARY KEY (alignment_run_id, chromatography, point_num)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_aligned_peaks (
            alignment_run_id INTEGER NOT NULL,
            peak_uid         TEXT NOT NULL,
            row_order        INTEGER NOT NULL,
            peak_label       TEXT NOT NULL,
            peak_num         INTEGER NOT NULL,
            rt_peak          DOUBLE NOT NULL,
            rt_min           DOUBLE NOT NULL,
            rt_max           DOUBLE NOT NULL,
            rt_peak_aligned  DOUBLE NOT NULL,
            rt_min_aligned   DOUBLE NOT NULL,
            rt_max_aligned   DOUBLE NOT NULL,
            rt_delta         DOUBLE NOT NULL,
            PRIMARY KEY (alignment_run_id, peak_uid)
        )
    """)

    # ------------------------------------------------------------------
    # Deposit content per alignment run. Materialized from the database
    # on the first deposit preview; hand edits to previews are written back.
    # ------------------------------------------------------------------
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS sa_atlas_entries (
            alignment_run_id INTEGER NOT NULL,
            entry_id         INTEGER NOT NULL,
            sa_peak_uid      TEXT,
{_quoted_columns(ATLAS_COLUMN_TYPES)},
            origin           TEXT NOT NULL CHECK (origin IN ('generated', 'manual')),
            is_excluded      BOOLEAN NOT NULL DEFAULT FALSE,
            last_modified_by_deposit INTEGER,
            PRIMARY KEY (alignment_run_id, entry_id)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS sa_msms_entries (
            alignment_run_id INTEGER NOT NULL,
            entry_id         INTEGER NOT NULL,
            sa_peak_uid      TEXT,
            sa_ce_label      TEXT,
            sa_measured_collision_energy DOUBLE,
{_quoted_columns(MSMS_COLUMN_TYPES)},
            origin           TEXT NOT NULL CHECK (origin IN ('generated', 'manual')),
            is_excluded      BOOLEAN NOT NULL DEFAULT FALSE,
            last_modified_by_deposit INTEGER,
            PRIMARY KEY (alignment_run_id, entry_id)
        )
    """)

    # ------------------------------------------------------------------
    # Deposits: one per preview. 'generated' rows are what the preview
    # contained; 'final' rows are what was read back and deposited.
    # ------------------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_deposits (
            deposit_id          INTEGER PRIMARY KEY,
            alignment_run_id    INTEGER NOT NULL,
            status              TEXT NOT NULL CHECK (status IN
                                    ('previewed', 'superseded', 'in_progress', 'deposited')),
            output_dir          TEXT NOT NULL,
            chem_source         TEXT,
            created_at          TEXT NOT NULL,
            created_by          TEXT,
            gdrive_url          TEXT,
            atlas_preview_sha256 TEXT NOT NULL,
            msms_preview_sha256  TEXT NOT NULL,
            atlas_final_sha256   TEXT,
            msms_final_sha256    TEXT,
            deposit_started_at  TEXT,
            deposited_at        TEXT,
            deposited_by        TEXT,
            force_add           BOOLEAN,
            compounds_stage     TEXT NOT NULL DEFAULT 'pending'
                                    CHECK (compounds_stage IN ('pending', 'done', 'skipped')),
            compounds_stage_at  TEXT,
            atlas_stage         TEXT NOT NULL DEFAULT 'pending'
                                    CHECK (atlas_stage IN ('pending', 'done', 'skipped')),
            atlas_stage_at      TEXT,
            msms_stage          TEXT NOT NULL DEFAULT 'pending'
                                    CHECK (msms_stage IN ('pending', 'done', 'skipped')),
            msms_stage_at       TEXT
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS sa_deposit_atlas_rows (
            deposit_id    INTEGER NOT NULL,
            version       TEXT NOT NULL CHECK (version IN ('generated', 'final')),
            row_num       INTEGER NOT NULL,
            sa_entry_id   INTEGER,
            sa_peak_uid   TEXT,
{_quoted_columns(ATLAS_COLUMN_TYPES)},
            change_status TEXT CHECK (change_status IN ('unchanged', 'edited', 'added')),
            sheet_outcome TEXT CHECK (sheet_outcome IN ('appended', 'skipped_duplicate', 'skipped_by_flag')),
            PRIMARY KEY (deposit_id, version, row_num)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS sa_deposit_msms_rows (
            deposit_id    INTEGER NOT NULL,
            version       TEXT NOT NULL CHECK (version IN ('generated', 'final')),
            row_num       INTEGER NOT NULL,
            sa_entry_id   INTEGER,
            sa_peak_uid   TEXT,
            sa_ce_label   TEXT,
            sa_measured_collision_energy DOUBLE,
{_quoted_columns(MSMS_COLUMN_TYPES)},
            change_status TEXT CHECK (change_status IN ('unchanged', 'edited', 'added')),
            import_outcome TEXT CHECK (import_outcome IN
                ('inserted', 'skipped_existing', 'filtered', 'skipped_by_flag')),
            main_db_ref_uid TEXT,
            PRIMARY KEY (deposit_id, version, row_num)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_deposit_changes (
            deposit_id   INTEGER NOT NULL,
            file         TEXT NOT NULL CHECK (file IN ('atlas', 'msms')),
            sa_entry_id  INTEGER NOT NULL,
            change_type  TEXT NOT NULL CHECK (change_type IN ('edited', 'added', 'removed')),
            column_name  TEXT,
            old_value    TEXT,
            new_value    TEXT
        )
    """)


# ---------------------------------------------------------------------------
# Resumability checks
# ---------------------------------------------------------------------------

def is_extraction_complete(db_path: str) -> bool:
    """Return True if the extraction targets table already has data.

    Checks ``sa_extraction_targets`` (not ``sa_ms1``) so that a run that
    completed extraction but crashed before peak detection is correctly
    identified as having completed extraction.

    Args:
        db_path: Path to the project DuckDB.

    Returns:
        ``True`` if ``sa_extraction_targets`` is non-empty, ``False`` otherwise.
    """
    if not Path(db_path).exists():
        return False
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sa_extraction_targets").fetchone()[0]
        return int(count) > 0
    except Exception:
        return False


def is_peaks_complete(db_path: str) -> bool:
    """Return True if MS1 source selection and peak detection completed.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        return False
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sa_ms1_sources").fetchone()[0]
        return int(count) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Extraction persistence
# ---------------------------------------------------------------------------

def save_extraction_to_db(
    db_path: str,
    data: "wfo.StandardsAnnotationData",
) -> None:
    """Persist all extracted data to the project DuckDB.

    Populates in order:
      1. ``sa_compounds`` — one row per (compound_name, adduct)
      2. ``sa_extraction_targets`` — one row per (compound_uid, mz_rt_uid, filename)
      3. ``sa_ms1`` — wide-format MS1 data
      4. ``sa_ms2`` — wide-format MS2 data
      5. ``sa_tic`` — TIC per file

    Skips tables that already have data (idempotent / resumable).

    Args:
        db_path: Path to the project DuckDB.
        data:    :class:`metatlas2.wfo.StandardsAnnotationData` container.
    """
    if is_extraction_complete(db_path):
        logger.info(
            f"Extraction already exists for standards annotation run "
            f"{data.config['project']['standards_run_name']}. Not overwriting."
        )
        return

    expanded_table = data.expanded_table

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            # ------------------------------------------------------------------
            # sa_compounds — deduplicated by compound_uid
            # ------------------------------------------------------------------
            if expanded_table is not None and not expanded_table.empty:
                seen_compound_uids: set[str] = set()
                compound_rows = []
                for _, row in expanded_table.iterrows():
                    cuid = str(row["compound_uid"])
                    if cuid in seen_compound_uids:
                        continue
                    seen_compound_uids.add(cuid)
                    compound_rows.append((
                        cuid,
                        str(row.get("compound_name", "")),
                        str(row.get("inchi_key", "")),
                        str(row.get("smiles", "")),
                        str(row.get("inchi", "")),
                        str(row.get("adduct", "")),
                        str(row.get("polarity", "")),
                        float(row.get("mz_theoretical", float("nan"))),
                    ))
                if compound_rows:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO sa_compounds VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        compound_rows,
                    )
                    logger.info(f"Saved {len(compound_rows)} compound rows to sa_compounds")

            # ------------------------------------------------------------------
            # sa_extraction_targets — one row per (mz_rt_uid, filename)
            # ------------------------------------------------------------------
            if expanded_table is not None and not expanded_table.empty:
                target_rows = []
                for _, row in expanded_table.iterrows():
                    target_rows.append((
                        str(row["mz_rt_uid"]),
                        str(row["compound_uid"]),
                        str(row.get("filename", "")),
                        str(row.get("chromatography", "")),
                        float(row.get("rt_min", 0.0)),
                        float(row.get("rt_max", 20.0)),
                        str(row.get("collision_energy_label", "")),
                        str(row.get("annotation_notes", "")),
                    ))
                if target_rows:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO sa_extraction_targets VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        target_rows,
                    )
                    logger.info(f"Saved {len(target_rows)} extraction target rows to sa_extraction_targets")

            # ------------------------------------------------------------------
            # sa_ms1
            # ------------------------------------------------------------------
            if data.ms1_df is not None and not data.ms1_df.empty:
                ms1 = data.ms1_df.copy()
                for col in ("spec_rts", "spec_ints", "spec_mzs", "in_feature"):
                    if col in ms1.columns:
                        ms1[col] = ms1[col].apply(
                            lambda x: list(x) if x is not None else []
                        )
                conn.execute("INSERT OR IGNORE INTO sa_ms1 SELECT * FROM ms1")
                logger.info(f"Saved {len(ms1)} MS1 rows to sa_ms1")

            # ------------------------------------------------------------------
            # sa_ms2
            # ------------------------------------------------------------------
            if data.ms2_df is not None and not data.ms2_df.empty:
                ms2 = data.ms2_df.copy()
                for col in ("frag_mzs", "frag_ints"):
                    if col in ms2.columns:
                        ms2[col] = ms2[col].apply(
                            lambda x: list(x) if x is not None else []
                        )
                if "in_feature" in ms2.columns:
                    ms2["in_feature"] = ms2["in_feature"].astype(bool)
                # Normalise precursor_MZ → precursor_mz for DB storage
                if "precursor_MZ" in ms2.columns and "precursor_mz" not in ms2.columns:
                    ms2 = ms2.rename(columns={"precursor_MZ": "precursor_mz"})
                conn.execute("INSERT OR IGNORE INTO sa_ms2 SELECT * FROM ms2")
                logger.info(f"Saved {len(ms2)} MS2 rows to sa_ms2")

            # ------------------------------------------------------------------
            # sa_tic
            # ------------------------------------------------------------------
            if data.tic_by_file:
                tic_rows = []
                for filename, tic_df in data.tic_by_file.items():
                    for _, row in tic_df.iterrows():
                        tic_rows.append((
                            str(filename),
                            float(row["rt"]),
                            float(row["tic_intensity"]),
                        ))
                if tic_rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO sa_tic VALUES (?, ?, ?)",
                        tic_rows,
                    )
                    logger.info(f"Saved {len(tic_rows)} TIC rows to sa_tic")

            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to save extraction to DB: {exc}")
            raise

    # InjBl MS1 is saved separately (outside the main transaction) because it
    # is populated by a separate extraction pass that runs after the main one.
    # save_injbl_ms1_to_db() is called from run_standards_extraction() directly.


def load_extraction_from_db(
    db_path: str,
    config: dict[str, Any],
) -> "wfo.StandardsAnnotationData":
    """Reload all extracted data from the project DuckDB.

    Used on resume when extraction has already been completed.

    Args:
        db_path: Path to the project DuckDB.
        config:  Parsed config dict.

    Returns:
        :class:`metatlas2.wfo.StandardsAnnotationData`.
    """
    logger.info(f"Loading extraction data from {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as conn:
        has_sources = conn.execute("SELECT COUNT(*) FROM sa_ms1_sources").fetchone()[0] > 0
        if has_sources:
            ms1_df = conn.execute("""
                SELECT m.* FROM sa_ms1 m
                JOIN sa_extraction_targets t ON m.mz_rt_uid = t.mz_rt_uid
                JOIN sa_compounds c ON t.compound_uid = c.compound_uid
                JOIN sa_ms1_sources s ON s.compound_name = c.compound_name
                    AND s.chromatography = t.chromatography
                    AND s.polarity = c.polarity AND s.filename = t.filename
            """).df()
        else:
            ms1_df = conn.execute("SELECT * FROM sa_ms1").df()
        ms2_df = conn.execute("SELECT * FROM sa_ms2").df()
        tic_raw = conn.execute("SELECT * FROM sa_tic").df()
        # Rebuild expanded_table by joining sa_extraction_targets with sa_compounds
        expanded_table = conn.execute("""
            SELECT
                t.mz_rt_uid,
                t.compound_uid,
                t.filename,
                t.chromatography,
                t.rt_min,
                t.rt_max,
                t.collision_energy_label,
                t.annotation_notes,
                c.compound_name,
                c.inchi_key,
                c.smiles,
                c.inchi,
                c.adduct,
                c.polarity,
                c.mz_theoretical
            FROM sa_extraction_targets t
            JOIN sa_compounds c ON t.compound_uid = c.compound_uid
            ORDER BY c.compound_name, t.chromatography, c.adduct,
                     t.collision_energy_label, t.filename
        """).df()

    # Coerce list columns
    for col in ("spec_rts", "spec_ints", "spec_mzs", "in_feature"):
        if col in ms1_df.columns:
            ms1_df[col] = ms1_df[col].apply(
                lambda x: list(x) if x is not None else []
            )
    for col in ("frag_mzs", "frag_ints"):
        if col in ms2_df.columns:
            ms2_df[col] = ms2_df[col].apply(
                lambda x: list(x) if x is not None else []
            )

    # Rebuild tic_by_file dict
    tic_by_file: dict[str, pd.DataFrame] = {}
    if not tic_raw.empty:
        for filename, grp in tic_raw.groupby("filename"):
            tic_by_file[str(filename)] = (
                grp[["rt", "tic_intensity"]].sort_values("rt").reset_index(drop=True)
            )

    # Load InjBl MS1 if available
    injbl_ms1_df = load_injbl_ms1(db_path)

    logger.info(
        f"Loaded: {len(ms1_df)} MS1 rows, {len(ms2_df)} MS2 rows, "
        f"{len(tic_by_file)} TIC files, {len(expanded_table)} extraction target rows, "
        f"{len(injbl_ms1_df)} InjBl MS1 rows"
    )

    return wfo.StandardsAnnotationData(
        ms1_df=ms1_df,
        ms2_df=ms2_df,
        tic_by_file=tic_by_file,
        expanded_table=expanded_table,
        injbl_ms1_df=injbl_ms1_df,
        config=config,
        project_db_path=db_path,
    )


# ---------------------------------------------------------------------------
# InjBl MS1 persistence
# ---------------------------------------------------------------------------

def is_injbl_extraction_complete(db_path: str) -> bool:
    """Return True if ``sa_injbl_ms1`` already contains rows.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        return False
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM sa_injbl_ms1").fetchone()[0]
        return int(count) > 0
    except Exception:
        return False


def save_injbl_ms1_to_db(db_path: str, ms1_df: pd.DataFrame) -> None:
    """Bulk-insert InjBl MS1 EIC data into ``sa_injbl_ms1``.

    The DataFrame must have columns matching the ``sa_injbl_ms1`` schema:
    ``mz_rt_uid``, ``injbl_filename``, ``inchi_key``, ``adduct``,
    ``spec_rts``, ``spec_ints``, ``spec_mzs``.

    Existing rows with the same ``(mz_rt_uid, injbl_filename)`` PK are
    ignored (idempotent).

    Args:
        db_path: Path to the project DuckDB.
        ms1_df:  Wide-format InjBl MS1 DataFrame.
    """
    if ms1_df is None or ms1_df.empty:
        logger.info("save_injbl_ms1_to_db: no rows to save.")
        return

    df = ms1_df.copy()
    # Rename filename -> injbl_filename if the extraction engine used filename
    if "filename" in df.columns and "injbl_filename" not in df.columns:
        df = df.rename(columns={"filename": "injbl_filename"})

    # Ensure list columns are plain Python lists
    for col in ("spec_rts", "spec_ints", "spec_mzs"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if x is not None else [])

    # Keep only the columns that exist in the schema
    keep = ["mz_rt_uid", "injbl_filename", "inchi_key", "adduct",
            "spec_rts", "spec_ints", "spec_mzs"]
    df = df[[c for c in keep if c in df.columns]]

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("INSERT OR IGNORE INTO sa_injbl_ms1 SELECT * FROM df")
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to save InjBl MS1 data: {exc}")
            raise
    logger.info(f"Saved {len(df)} InjBl MS1 rows to sa_injbl_ms1")


def load_injbl_ms1(db_path: str) -> pd.DataFrame:
    """Load all rows from ``sa_injbl_ms1`` with list columns coerced.

    Returns an empty DataFrame if the table is empty, does not exist, or
    the DB file does not exist.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            df = conn.execute("SELECT * FROM sa_injbl_ms1").df()
        for col in ("spec_rts", "spec_ints", "spec_mzs"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: list(x) if x is not None else []
                )
        return df
    except Exception as exc:
        logger.warning(f"Could not load InjBl MS1 from {db_path}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Peak persistence
# ---------------------------------------------------------------------------

def _peak_values(p: dict) -> tuple:
    return (
        p["peak_uid"], p["compound_uid"], p["compound_name"], p.get("inchi_key", ""),
        p["adduct"], p["polarity"], p["chromatography"], p["filename"], p["mz_rt_uid"],
        float(p.get("mz_theoretical") or float("nan")), float(p["rt_peak"]),
        float(p["rt_min"]), float(p["rt_max"]), float(p["peak_intensity"]),
        float(p.get("mz_observed") or float("nan")), int(p.get("peak_num", 1)),
        bool(p.get("is_manual", False)), str(p.get("collision_energy_label", "")),
        _cell(p.get("created_in_annotation_run")),
    )


_PEAK_COLUMNS = (
    "peak_uid, compound_uid, compound_name, inchi_key, adduct, polarity, chromatography, "
    "filename, mz_rt_uid, mz_theoretical, rt_peak, rt_min, rt_max, peak_intensity, "
    "mz_observed, peak_num, is_manual, collision_energy_label, created_in_annotation_run"
)
_PEAK_INSERT = f"INSERT INTO sa_peaks ({_PEAK_COLUMNS}) VALUES ({', '.join('?' * 19)})"
_PEAK_MS2_INSERT = (
    "INSERT INTO sa_peak_ms2 (peak_uid, scan_rt, precursor_mz, precursor_intensity, "
    "collision_energy, frag_mzs, frag_ints, filename) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def _peak_ms2_values(r: dict) -> tuple:
    def number(key: str) -> float:
        value = r.get(key)
        return float(value) if value is not None else float("nan")

    return (
        r["peak_uid"], number("scan_rt"), number("precursor_mz"),
        number("precursor_intensity"), number("collision_energy"),
        json.dumps([float(v) for v in (r.get("frag_mzs") or [])]),
        json.dumps([float(v) for v in (r.get("frag_ints") or [])]),
        str(r.get("filename", "")),
    )


def save_detected_peaks_to_db(
    db_path: str, peaks: list[dict], peak_ms2_rows: list[dict],
    sources: list[tuple[str, str, str, str]] | None = None,
) -> None:
    """Insert a detection result and its MS2 associations in one transaction."""
    peak_uids = [peak["peak_uid"] for peak in peaks]
    ms2_uids = [row["peak_uid"] for row in peak_ms2_rows]
    ms2_keys = [(row["peak_uid"], row["filename"]) for row in peak_ms2_rows]
    if len(set(peak_uids)) != len(peak_uids) or len(set(ms2_keys)) != len(ms2_keys):
        raise ValueError("Detected peaks or MS2 associations contain duplicate keys.")
    if set(ms2_uids) - set(peak_uids):
        raise ValueError("MS2 associations reference peaks outside this detection result.")
    if not peaks and not sources:
        return
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            if sources:
                conn.executemany("INSERT INTO sa_ms1_sources VALUES (?, ?, ?, ?)", sources)
            if peaks:
                conn.executemany(_PEAK_INSERT, [_peak_values(peak) for peak in peaks])
            if peak_ms2_rows:
                conn.executemany(_PEAK_MS2_INSERT, [_peak_ms2_values(row) for row in peak_ms2_rows])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            logger.exception(
                "Could not save detected peaks and MS2 associations; neither was saved. "
                "Check for existing peaks or duplicate detection results."
            )
            raise
    logger.info(
        "Saved %d detected peaks and %d MS2 associations", len(peaks), len(peak_ms2_rows)
    )


def save_peaks_to_db(db_path: str, peaks: list[dict]) -> None:
    """Bulk-insert detected MS1 peaks into ``sa_peaks``.

    Each dict in *peaks* must have keys matching the ``sa_peaks`` schema.
    Existing rows with the same ``peak_uid`` are ignored.

    Args:
        db_path: Path to the project DuckDB.
        peaks:   List of peak dicts from the peak-detection pipeline.
    """
    if not peaks:
        return
    rows = [_peak_values(p) for p in peaks]
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.executemany(_PEAK_INSERT.replace("INSERT INTO", "INSERT OR IGNORE INTO"), rows)
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to save peaks: {exc}")
            raise
    logger.info(f"Saved {len(rows)} peaks to sa_peaks")


def _connection(db_path: str, connection: duckdb.DuckDBPyConnection | None = None, read_only: bool = False):
    """Return a context manager yielding *connection* or a new connection."""
    if connection is not None:
        return nullcontext(connection)
    return duckdb.connect(str(db_path), read_only=read_only)


def _visible_peak_clause(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return (f"({prefix}created_in_annotation_run IS NULL "
            f"OR {prefix}created_in_annotation_run <= ?)")


def load_peaks(db_path: str, annotation_run_id: int | None = None) -> pd.DataFrame:
    """Load rows from ``sa_peaks``.

    With *annotation_run_id*, manual peaks created in later annotation runs
    are excluded so a historical run is shown as it was reviewed.

    Returns an empty DataFrame if the DB does not exist.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    where, params = "", []
    if annotation_run_id is not None:
        where, params = f"WHERE {_visible_peak_clause()}", [int(annotation_run_id)]
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            return conn.execute(
                f"""
                SELECT * FROM sa_peaks {where}
                ORDER BY compound_name, chromatography, polarity, adduct, filename, peak_num
                """, params,
            ).df()
    except Exception as exc:
        raise RuntimeError(f"Could not load peaks from {db_path}: {exc}") from exc


def load_peak_ms2(db_path: str) -> pd.DataFrame:
    """Load all rows from ``sa_peak_ms2`` with frag columns deserialized.

    Returns an empty DataFrame if the table is empty or the DB does not exist.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            df = conn.execute("SELECT * FROM sa_peak_ms2").df()
        for col in ("frag_mzs", "frag_ints"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else (x or [])
                )
        return df
    except Exception as exc:
        raise RuntimeError(f"Could not load peak MS2 from {db_path}: {exc}") from exc




def save_manual_peak_to_db(db_path: str, peak: dict, ms2: list[dict]) -> None:
    """Atomically insert a manually added peak and its MS2 associations.

    If ``peak['created_in_annotation_run']`` is set, that run must be a draft.

    Args:
        db_path: Path to the project DuckDB.
        peak:    Peak dict with all ``sa_peaks`` fields.
        ms2:     MS2 association rows for the peak.
    """
    if any(row["peak_uid"] != peak["peak_uid"] for row in ms2):
        raise ValueError("Manual peak and MS2 association have different peak UIDs.")
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            run_id = _cell(peak.get("created_in_annotation_run"))
            if run_id is not None:
                _require_draft(conn, int(run_id))
            conn.execute(_PEAK_INSERT, _peak_values(peak))
            if ms2:
                conn.executemany(_PEAK_MS2_INSERT, [_peak_ms2_values(row) for row in ms2])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def delete_manual_peak_from_db(db_path: str, peak_uid: str) -> None:
    """Remove a manual peak (and its MS2 association) from the DB.

    Peaks selected in any frozen annotation run are part of the review
    history and cannot be deleted.

    Args:
        db_path:  Path to the project DuckDB.
        peak_uid: Primary key of the peak to remove.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            frozen = conn.execute(
                """SELECT DISTINCT s.annotation_run_id FROM sa_run_selections s
                   JOIN sa_annotation_runs r USING (annotation_run_id)
                   WHERE s.peak_uid = ? AND r.status = 'frozen'""", [peak_uid],
            ).fetchall()
            if frozen:
                raise ValueError(
                    f"Peak {peak_uid} is selected in frozen annotation run(s) "
                    f"{sorted(row[0] for row in frozen)} and cannot be deleted."
                )
            conn.execute("DELETE FROM sa_run_selections WHERE peak_uid = ?", [peak_uid])
            conn.execute("DELETE FROM sa_peak_ms2 WHERE peak_uid = ?", [peak_uid])
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to delete manual peak {peak_uid}: {exc}")
            raise
    _delete_unreferenced_peaks(db_path, [peak_uid])


# ---------------------------------------------------------------------------
# Annotation runs
# ---------------------------------------------------------------------------

def _scalar(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> Any:
    row = conn.execute(sql, params or []).fetchone()
    return row[0] if row else None


def _draft_run_id(conn: duckdb.DuckDBPyConnection) -> int | None:
    rows = conn.execute(
        "SELECT annotation_run_id FROM sa_annotation_runs WHERE status = 'draft' ORDER BY annotation_run_id"
    ).fetchall()
    if len(rows) > 1:
        raise RuntimeError(f"Project database has more than one draft annotation run: {[r[0] for r in rows]}")
    return int(rows[0][0]) if rows else None


def _latest_frozen_run_id(conn: duckdb.DuckDBPyConnection) -> int | None:
    value = _scalar(conn, "SELECT MAX(annotation_run_id) FROM sa_annotation_runs WHERE status = 'frozen'")
    return int(value) if value is not None else None


def _run_status(conn: duckdb.DuckDBPyConnection, annotation_run_id: int) -> str:
    status = _scalar(
        conn, "SELECT status FROM sa_annotation_runs WHERE annotation_run_id = ?", [int(annotation_run_id)],
    )
    if status is None:
        raise ValueError(f"Annotation run {annotation_run_id} does not exist.")
    return str(status)


def _require_draft(conn: duckdb.DuckDBPyConnection, annotation_run_id: int) -> None:
    if _run_status(conn, annotation_run_id) != "draft":
        raise ValueError(
            f"Annotation run {annotation_run_id} is frozen (already summarized) and cannot be changed. "
            "Open the GUI without an annotation run ID to start a new run."
        )


def _resolve_run(conn: duckdb.DuckDBPyConnection, annotation_run_id: int | None) -> int | None:
    if annotation_run_id is not None:
        _run_status(conn, annotation_run_id)
        return int(annotation_run_id)
    draft = _draft_run_id(conn)
    return draft if draft is not None else _latest_frozen_run_id(conn)


def resolve_annotation_run(db_path: str, annotation_run_id: int | None = None) -> int | None:
    """Return *annotation_run_id* if it exists, otherwise the current run.

    The current run is the draft if one exists, else the latest frozen run,
    else ``None``.
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return _resolve_run(conn, annotation_run_id)


def get_draft_annotation_run(db_path: str) -> int | None:
    """Return the draft annotation run ID, or ``None``."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return _draft_run_id(conn)


def get_latest_frozen_annotation_run(db_path: str) -> int | None:
    """Return the most recent frozen annotation run ID, or ``None``."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return _latest_frozen_run_id(conn)


def get_annotation_run(db_path: str, annotation_run_id: int) -> dict[str, Any]:
    """Return one ``sa_annotation_runs`` row as a dict; raise if missing."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(
            "SELECT * FROM sa_annotation_runs WHERE annotation_run_id = ?", [int(annotation_run_id)],
        ).df()
    if df.empty:
        raise ValueError(f"Annotation run {annotation_run_id} does not exist.")
    return df.iloc[0].to_dict()


def _create_annotation_run(conn: duckdb.DuckDBPyConnection, parent_run_id: int | None) -> int:
    """Insert a new draft run, copying the decisions of *parent_run_id*."""
    if _draft_run_id(conn) is not None:
        raise RuntimeError("A draft annotation run already exists.")
    prov = get_provenance()
    run_id = int(_scalar(conn, "SELECT COALESCE(MAX(annotation_run_id), 0) + 1 FROM sa_annotation_runs"))
    conn.execute(
        """INSERT INTO sa_annotation_runs
           (annotation_run_id, parent_run_id, status, created_at, created_by)
           VALUES (?, ?, 'draft', ?, ?)""",
        [run_id, parent_run_id, prov["timestamp"], prov["analyst"]],
    )
    if parent_run_id is None:
        conn.execute(
            """INSERT INTO sa_run_state
               SELECT DISTINCT ?, compound_uid, chromatography, 'pending', ?
               FROM sa_extraction_targets""",
            [run_id, prov["timestamp"]],
        )
        return run_id
    for table, columns in (
        ("sa_run_selections", "peak_uid, analyst_notes, selected_by, selected_at"),
        ("sa_run_no_good_peaks", "compound_name, chromatography, polarity"),
        ("sa_run_compound_notes", "compound_name, chromatography, analyst_notes"),
        ("sa_run_state", "compound_uid, chromatography, annotation_status, last_updated"),
    ):
        conn.execute(
            f"INSERT INTO {table} SELECT ?, {columns} FROM {table} WHERE annotation_run_id = ?",
            [run_id, int(parent_run_id)],
        )
    return run_id


def open_or_create_draft_annotation_run(db_path: str) -> int:
    """Return the draft run, creating one from the latest frozen run if needed.

    The first run starts with no decisions. Later runs start as a copy of
    the most recent frozen run.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            run_id = _draft_run_id(conn)
            created = run_id is None
            if created:
                parent = _latest_frozen_run_id(conn)
                run_id = _create_annotation_run(conn, parent)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    if created:
        source = f"a copy of annotation run {parent}" if parent is not None else "no selections"
        logger.info(f"Started draft annotation run {run_id} from {source}.")
    return run_id


def _delete_draft(conn: duckdb.DuckDBPyConnection, run_id: int) -> list[str]:
    """Delete a draft run's rows; return its manual peaks for later removal."""
    for table in ("sa_run_selections", "sa_run_no_good_peaks", "sa_run_compound_notes", "sa_run_state"):
        conn.execute(f"DELETE FROM {table} WHERE annotation_run_id = ?", [run_id])
    manual = [row[0] for row in conn.execute(
        "SELECT peak_uid FROM sa_peaks WHERE is_manual AND created_in_annotation_run = ?", [run_id],
    ).fetchall()]
    if manual:
        placeholders = ", ".join("?" for _ in manual)
        conn.execute(f"DELETE FROM sa_peak_ms2 WHERE peak_uid IN ({placeholders})", manual)
    conn.execute("DELETE FROM sa_annotation_runs WHERE annotation_run_id = ?", [run_id])
    return manual


def _delete_unreferenced_peaks(db_path: str, peak_uids: list[str]) -> None:
    """Delete manual peaks whose MS2 rows were removed in an earlier transaction.

    DuckDB cannot delete a foreign-key-referenced ``sa_peaks`` row in the
    transaction that deleted its ``sa_peak_ms2`` rows, so this runs separately.
    A failure here leaves only unselected, unreferenced manual peaks.
    """
    if not peak_uids:
        return
    placeholders = ", ".join("?" for _ in peak_uids)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            f"DELETE FROM sa_peaks WHERE is_manual AND peak_uid IN ({placeholders})", list(peak_uids),
        )


def discard_draft_annotation_run(db_path: str) -> int | None:
    """Delete the draft run, its decisions, and manual peaks added in it.

    Returns:
        The discarded run ID, or ``None`` if there was no draft.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            run_id = _draft_run_id(conn)
            manual = _delete_draft(conn, run_id) if run_id is not None else []
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    _delete_unreferenced_peaks(db_path, manual)
    if run_id is not None:
        logger.info(f"Discarded draft annotation run {run_id}.")
    return run_id


def start_new_annotation_run(
    db_path: str, from_annotation_run_id: int, discard_draft: bool = False,
) -> int:
    """Start a draft annotation run as a copy of any frozen run.

    Use this to revert to (and continue from) an earlier review iteration.

    Args:
        db_path:                Path to the project DuckDB.
        from_annotation_run_id: Frozen run to copy.
        discard_draft:          Discard an existing draft run first. Otherwise
                                an existing draft raises ``ValueError``.

    Returns:
        The new draft run ID.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            if _run_status(conn, from_annotation_run_id) != "frozen":
                raise ValueError(f"Annotation run {from_annotation_run_id} is a draft; copy a frozen run.")
            draft = _draft_run_id(conn)
            manual = []
            if draft is not None:
                if not discard_draft:
                    raise ValueError(
                        f"Draft annotation run {draft} already exists. Summarize it, or pass "
                        "discard_draft=True to discard it and its manual peaks."
                    )
                manual = _delete_draft(conn, draft)
            run_id = _create_annotation_run(conn, int(from_annotation_run_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    _delete_unreferenced_peaks(db_path, manual)
    logger.info(f"Started draft annotation run {run_id} from a copy of annotation run {from_annotation_run_id}.")
    return run_id


def freeze_annotation_run(db_path: str, annotation_run_id: int, output_dir: str | Path) -> None:
    """Mark a draft run frozen and record its summary output directory."""
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            _require_draft(conn, annotation_run_id)
            conn.execute(
                """UPDATE sa_annotation_runs
                   SET status = 'frozen', frozen_at = ?, frozen_by = ?, output_dir = ?
                   WHERE annotation_run_id = ?""",
                [prov["timestamp"], prov["analyst"], str(output_dir), int(annotation_run_id)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _set_field(db_path: str, table: str, key: str, key_value: int, field: str, value: Any) -> None:
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(f"UPDATE {table} SET {field} = ? WHERE {key} = ?", [value, int(key_value)])


def set_annotation_run_gdrive_url(db_path: str, annotation_run_id: int, url: str | None) -> None:
    _set_field(db_path, "sa_annotation_runs", "annotation_run_id", annotation_run_id, "gdrive_url", url)


def list_annotation_runs(db_path: str) -> pd.DataFrame:
    """Return one row per annotation run with decision and alignment counts."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute("""
            SELECT r.annotation_run_id, r.status, r.parent_run_id, r.created_at, r.created_by,
                   r.frozen_at,
                   (SELECT COUNT(*) FROM sa_run_selections s
                    WHERE s.annotation_run_id = r.annotation_run_id) AS n_selected_peaks,
                   (SELECT COUNT(*) FROM sa_run_no_good_peaks n
                    WHERE n.annotation_run_id = r.annotation_run_id) AS n_no_good_groups,
                   (SELECT COUNT(*) FROM sa_alignment_runs a
                    WHERE a.annotation_run_id = r.annotation_run_id) AS n_alignment_runs,
                   r.output_dir, r.gdrive_url
            FROM sa_annotation_runs r ORDER BY r.annotation_run_id
        """).df()


# ---------------------------------------------------------------------------
# Analyst decisions within an annotation run
# ---------------------------------------------------------------------------

def replace_compound_selections(
    db_path: str,
    compound_name: str,
    compound_uids: list[str],
    selections: list[tuple[str, str]],
    notes_by_chrom: dict[str, str],
    chromatographies: list[str],
    resume_compound: str | None = None,
    no_good_peaks: list[tuple[str, str]] | None = None,
    *,
    annotation_run_id: int | None = None,
) -> int:
    """Atomically save all adduct selections and state for a compound.

    Decisions are written to the draft annotation run *annotation_run_id*;
    with ``None`` the draft is used (and created from the latest frozen run
    if needed). Frozen runs cannot be changed.

    Returns:
        The annotation run ID that was written.
    """
    if not compound_uids:
        raise ValueError("Cannot save selections: compound has no database identities.")
    if len({uid for uid, _ in selections}) != len(selections):
        raise ValueError("Cannot save duplicate selected peak UIDs.")
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            run_id = annotation_run_id
            if run_id is None:
                run_id = _draft_run_id(conn)
                if run_id is None:
                    run_id = _create_annotation_run(conn, _latest_frozen_run_id(conn))
            run_id = int(run_id)
            _require_draft(conn, run_id)
            placeholders = ", ".join("?" for _ in compound_uids)
            valid = {
                row[0]: (row[1], row[2].upper()) for row in conn.execute(
                    f"""SELECT peak_uid, chromatography, polarity FROM sa_peaks
                        WHERE compound_uid IN ({placeholders}) AND compound_name = ?
                          AND {_visible_peak_clause()}""",
                    [*compound_uids, compound_name, run_id],
                ).fetchall()
            }
            missing = {uid for uid, _ in selections} - valid.keys()
            if missing:
                raise ValueError(f"Selected peaks not found for this compound: {sorted(missing)}")
            no_good = set(no_good_peaks or [])
            groups = {
                (row[0], row[1].upper()) for row in conn.execute(
                    """SELECT DISTINCT t.chromatography, c.polarity
                       FROM sa_extraction_targets t JOIN sa_compounds c USING (compound_uid)
                       WHERE c.compound_name = ?""", [compound_name],
                ).fetchall()
            }
            if no_good - groups:
                raise ValueError(f"Invalid no-good-peaks groups: {sorted(no_good - groups)}")
            conflicts = no_good & {valid[uid] for uid, _ in selections}
            if conflicts:
                raise ValueError(f"No-good-peaks choices conflict with selected peaks: {sorted(conflicts)}")
            conn.execute(
                f"""DELETE FROM sa_run_selections WHERE annotation_run_id = ? AND peak_uid IN
                    (SELECT peak_uid FROM sa_peaks WHERE compound_uid IN ({placeholders}))""",
                [run_id, *compound_uids],
            )
            if selections:
                conn.executemany(
                    "INSERT INTO sa_run_selections VALUES (?, ?, ?, ?, ?)",
                    [(run_id, uid, notes, prov["analyst"], prov["timestamp"]) for uid, notes in selections],
                )
            conn.execute(
                "DELETE FROM sa_run_no_good_peaks WHERE annotation_run_id = ? AND compound_name = ?",
                [run_id, compound_name],
            )
            if no_good:
                conn.executemany(
                    "INSERT INTO sa_run_no_good_peaks VALUES (?, ?, ?, ?)",
                    [(run_id, compound_name, chrom, pol) for chrom, pol in sorted(no_good)],
                )
            for chrom in chromatographies:
                conn.execute(
                    "INSERT OR REPLACE INTO sa_run_compound_notes VALUES (?, ?, ?, ?)",
                    [run_id, compound_name, chrom, notes_by_chrom.get(chrom, "")],
                )
            conn.execute(
                "UPDATE sa_annotation_runs SET gui_checkpoint_compound = ? WHERE annotation_run_id = ?",
                [resume_compound if resume_compound is not None else compound_name, run_id],
            )
            for uid in compound_uids:
                for chrom in chromatographies:
                    conn.execute(
                        """INSERT INTO sa_run_state VALUES (?, ?, ?, 'complete', ?)
                           ON CONFLICT (annotation_run_id, compound_uid, chromatography) DO UPDATE SET
                           annotation_status = excluded.annotation_status,
                           last_updated = excluded.last_updated""",
                        [run_id, uid, chrom, prov["timestamp"]],
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return run_id


def load_gui_checkpoint(db_path: str, annotation_run_id: int | None = None) -> str | None:
    """Return the compound at the run's last saved GUI checkpoint, if any."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        run_id = _resolve_run(conn, annotation_run_id)
        if run_id is None:
            return None
        return _scalar(
            conn, "SELECT gui_checkpoint_compound FROM sa_annotation_runs WHERE annotation_run_id = ?", [run_id],
        )


def load_compound_notes(
    db_path: str, compound_name: str, annotation_run_id: int | None = None,
) -> dict[str, str]:
    """Return saved analyst notes by chromatography for a compound."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        run_id = _resolve_run(conn, annotation_run_id)
        if run_id is None:
            return {}
        rows = conn.execute(
            """SELECT chromatography, analyst_notes FROM sa_run_compound_notes
               WHERE annotation_run_id = ? AND compound_name = ?""",
            [run_id, compound_name],
        ).fetchall()
    return dict(rows)


def load_no_good_peaks(
    db_path: str, compound_name: str, annotation_run_id: int | None = None,
) -> list[tuple[str, str]]:
    """Return saved no-good-peaks choices for a compound."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        run_id = _resolve_run(conn, annotation_run_id)
        if run_id is None:
            return []
        return conn.execute(
            """SELECT chromatography, polarity FROM sa_run_no_good_peaks
               WHERE annotation_run_id = ? AND compound_name = ?""",
            [run_id, compound_name],
        ).fetchall()


def validate_annotation_choices(db_path: str, annotation_run_id: int | None = None) -> None:
    """Require a peak or an explicit no-good choice for every target group."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        run_id = _resolve_run(conn, annotation_run_id)
        if run_id is None:
            raise ValueError("No annotation run exists; review peaks in the GUI first.")
        missing = conn.execute("""
            SELECT DISTINCT c.compound_name, t.chromatography, upper(c.polarity) AS polarity
            FROM sa_extraction_targets t JOIN sa_compounds c USING (compound_uid)
            WHERE NOT EXISTS (
                SELECT 1 FROM sa_run_selections s JOIN sa_peaks p USING (peak_uid)
                WHERE s.annotation_run_id = $run
                  AND p.compound_name = c.compound_name
                  AND p.chromatography = t.chromatography
                  AND upper(p.polarity) = upper(c.polarity)
            ) AND NOT EXISTS (
                SELECT 1 FROM sa_run_no_good_peaks n
                WHERE n.annotation_run_id = $run
                  AND n.compound_name = c.compound_name
                  AND n.chromatography = t.chromatography
                  AND n.polarity = upper(c.polarity)
            )
            ORDER BY c.compound_name, t.chromatography, polarity
        """, {"run": run_id}).fetchall()
    if missing:
        details = "\n".join(f"  {name} | {chrom} | {pol}" for name, chrom, pol in missing)
        raise ValueError(
            f"Annotation run {run_id}: select a peak or 'No good peaks' for each compound, "
            f"chromatography, and polarity in the GUI, then save:\n{details}"
        )


def load_analyst_selections(db_path: str, annotation_run_id: int | None = None) -> pd.DataFrame:
    """Load an annotation run's selections joined with peak metadata.

    Returns all ``sa_peaks`` columns plus ``analyst_notes``, ``selected_by``
    and ``selected_date``. Returns an empty DataFrame if there is no run.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            run_id = _resolve_run(conn, annotation_run_id)
            return conn.execute(
                """
                SELECT p.*, s.analyst_notes, s.selected_by, s.selected_at AS selected_date
                FROM sa_run_selections s
                JOIN sa_peaks p ON s.peak_uid = p.peak_uid
                WHERE s.annotation_run_id = ?
                ORDER BY p.compound_name, p.chromatography, p.polarity,
                         p.adduct, p.filename, p.peak_num
                """, [run_id if run_id is not None else -1],
            ).df()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not load analyst selections from {db_path}: {exc}") from exc


def load_selected_peaks_with_ms2(
    db_path: str, annotation_run_id: int | None = None, *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load an annotation run's selected peaks with a list of their best MS2 scans.

    Joins ``sa_run_selections``, ``sa_peaks``, ``sa_compounds``,
    ``sa_extraction_targets`` and ``sa_peak_ms2``.

    Args:
        db_path:           Path to the project DuckDB.
        annotation_run_id: Run to load; ``None`` loads the current run
                           (draft, else latest frozen).
    """
    if connection is None and not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    try:
        with _connection(db_path, connection, read_only=True) as conn:
            run_id = _resolve_run(conn, annotation_run_id)
            params = {"run": run_id if run_id is not None else -1}
            df = conn.execute(
                """
                SELECT
                    p.peak_uid, p.compound_uid, p.compound_name, p.inchi_key, p.adduct,
                    p.polarity, p.chromatography, p.collision_energy_label, p.filename,
                    p.mz_rt_uid, p.mz_theoretical, p.rt_peak, p.rt_min, p.rt_max,
                    p.peak_intensity, p.mz_observed, p.peak_num, p.is_manual,
                    s.analyst_notes, c.smiles, c.inchi, t.annotation_notes
                FROM sa_run_selections s
                JOIN sa_peaks p ON s.peak_uid = p.peak_uid
                JOIN sa_compounds c ON p.compound_uid = c.compound_uid
                JOIN sa_extraction_targets t ON p.mz_rt_uid = t.mz_rt_uid
                WHERE s.annotation_run_id = $run
                ORDER BY p.compound_name, p.chromatography, p.polarity,
                         p.collision_energy_label, p.adduct, p.filename, p.peak_num
                """, params,
            ).df()
            scans = conn.execute("""
                SELECT m.* FROM sa_peak_ms2 m
                JOIN sa_run_selections s ON s.peak_uid = m.peak_uid
                WHERE s.annotation_run_id = $run
                ORDER BY m.peak_uid, m.filename
            """, params).df()
        df["ms2_scans"] = df["peak_uid"].map(_group_scans(scans))
        df["ms2_scans"] = df["ms2_scans"].map(lambda value: value if isinstance(value, list) else [])
        return df
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not load selected peaks with MS2 from {db_path}: {exc}") from exc


def _group_scans(scans: pd.DataFrame) -> dict[str, list[dict]]:
    scans_by_peak: dict[str, list[dict]] = {}
    for row in scans.to_dict("records"):
        for col in ("frag_mzs", "frag_ints"):
            row[col] = json.loads(row[col]) if isinstance(row[col], str) else (row[col] or [])
        scans_by_peak.setdefault(row["peak_uid"], []).append(row)
    return scans_by_peak


def get_first_incomplete_compound(
    db_path: str, annotation_run_id: int | None = None,
) -> tuple[str | None, str | None]:
    """Return the (compound_uid, chromatography) of the run's first incomplete entry.

    Returns ``(None, None)`` if all are complete or there is no run.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        run_id = _resolve_run(conn, annotation_run_id)
        if run_id is None:
            return None, None
        row = conn.execute(
            """
            SELECT compound_uid, chromatography FROM sa_run_state
            WHERE annotation_run_id = ? AND annotation_status != 'complete'
            ORDER BY rowid LIMIT 1
            """, [run_id],
        ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def load_isomer_check_targets(db_path: str) -> pd.DataFrame:
    """Load theoretical m/z for all targets, including unselected adducts."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute("""
            SELECT DISTINCT c.compound_name, c.polarity, t.chromatography, c.mz_theoretical
            FROM sa_compounds c
            JOIN sa_extraction_targets t ON c.compound_uid = t.compound_uid
        """).df()


# ---------------------------------------------------------------------------
# RT alignment persistence
# ---------------------------------------------------------------------------

def load_extraction_target_filenames(db_path: str) -> list[str]:
    """Return distinct nonempty source filenames from the project database."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute("SELECT DISTINCT filename FROM sa_extraction_targets").fetchall()
    return sorted({str(row[0]).strip() for row in rows if row[0] and str(row[0]).strip()})


def load_sa_compounds(db_path: str) -> pd.DataFrame:
    """Load all rows from ``sa_compounds``.

    Used by the reverse RT alignment phase to discover which chromatographies
    and adducts are present in the project.

    Args:
        db_path: Path to the project DuckDB.

    Returns:
        DataFrame with all ``sa_compounds`` rows, or an empty DataFrame.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            return conn.execute("SELECT * FROM sa_compounds").df()
    except Exception as exc:
        logger.warning(f"Could not load sa_compounds from {db_path}: {exc}")
        return pd.DataFrame()




# ---------------------------------------------------------------------------
# RT alignment runs
# ---------------------------------------------------------------------------

def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def next_alignment_run_id(db_path: str) -> int:
    """Return the ID the next recorded alignment run will receive."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return int(_scalar(conn, "SELECT COALESCE(MAX(alignment_run_id), 0) + 1 FROM sa_alignment_runs"))


_QC_POINT_COLUMNS = [
    "mz_rt_uid", "compound_uid", "label", "inchi_key", "adduct", "atlas_rt_peak",
    "atlas_rt_min", "atlas_rt_max", "atlas_mz", "exp_rt_peak", "intensity_peak",
    "predicted_rt", "residual",
]
_ALIGNED_PEAK_COLUMNS = [
    "peak_uid", "peak_label", "peak_num", "rt_peak", "rt_min", "rt_max",
    "rt_peak_aligned", "rt_min_aligned", "rt_max_aligned", "rt_delta",
]


def record_alignment_run(
    db_path: str,
    *,
    alignment_run_id: int,
    annotation_run_id: int,
    output_dir: str | Path,
    skip_alignment: bool,
    tags: str,
    config_snapshot: str,
    models: list[dict[str, Any]],
    qc_points: dict[str, pd.DataFrame],
    aligned_peaks: pd.DataFrame,
) -> None:
    """Persist one alignment run, its models, QC points and aligned peaks atomically.

    Args:
        models:        One dict per chromatography with keys matching
                       ``sa_alignment_models`` columns (missing keys are NULL).
        qc_points:     Modeling DataFrame per chromatography.
        aligned_peaks: One row per selected peak, in output order.
    """
    prov = get_provenance()
    peaks = aligned_peaks.copy()
    if "rt_delta" not in peaks.columns:
        peaks["rt_delta"] = peaks["rt_peak_aligned"] - peaks["rt_peak"]
    peak_rows = [
        (alignment_run_id, row["peak_uid"], order, str(row["peak_label"]), int(row["peak_num"]),
         *(float(row[col]) for col in _ALIGNED_PEAK_COLUMNS[3:]))
        for order, row in enumerate(peaks.to_dict("records"), start=1)
    ]
    model_rows = []
    for model in models:
        model_rows.append((
            alignment_run_id, model["chromatography"], model.get("qc_atlas_uid"),
            json.dumps(model.get("qc_atlas_selector"), default=str),
            model.get("model_type"), _none_if_nan(model.get("degree")),
            [float(v) for v in np.ravel(model.get("coefficients", []))],
            _none_if_nan(model.get("intercept")), _none_if_nan(model.get("offset")),
            _none_if_nan(model.get("r2")), _none_if_nan(model.get("rmse")),
            model.get("equation"), _none_if_nan(model.get("n_points")),
            json.dumps(model.get("params") or {}, default=str),
            [str(f) for f in model.get("qc_files", [])],
        ))
    point_rows = []
    for chrom, frame in qc_points.items():
        frame = frame.reindex(columns=_QC_POINT_COLUMNS)
        for num, row in enumerate(frame.to_dict("records"), start=1):
            point_rows.append((alignment_run_id, chrom, num,
                               *(_none_if_nan(row[col]) for col in _QC_POINT_COLUMNS)))
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            if _run_status(conn, annotation_run_id) != "frozen":
                raise ValueError(f"Annotation run {annotation_run_id} is not frozen; summarize it before alignment.")
            conn.execute(
                "INSERT INTO sa_alignment_runs VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                [alignment_run_id, annotation_run_id, prov["timestamp"], prov["analyst"],
                 str(output_dir), bool(skip_alignment), tags, config_snapshot],
            )
            if model_rows:
                conn.executemany(
                    "INSERT INTO sa_alignment_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    model_rows,
                )
            if point_rows:
                conn.executemany(
                    f"INSERT INTO sa_alignment_qc_points VALUES ({', '.join('?' * 16)})", point_rows,
                )
            conn.executemany(
                f"INSERT INTO sa_aligned_peaks VALUES ({', '.join('?' * 12)})", peak_rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def set_alignment_run_gdrive_url(db_path: str, alignment_run_id: int, url: str | None) -> None:
    _set_field(db_path, "sa_alignment_runs", "alignment_run_id", alignment_run_id, "gdrive_url", url)


def get_alignment_run(db_path: str, alignment_run_id: int) -> dict[str, Any]:
    """Return one ``sa_alignment_runs`` row as a dict; raise if missing."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(
            "SELECT * FROM sa_alignment_runs WHERE alignment_run_id = ?", [int(alignment_run_id)],
        ).df()
    if df.empty:
        raise ValueError(f"Alignment run {alignment_run_id} does not exist.")
    return df.iloc[0].to_dict()


def list_alignment_runs(db_path: str) -> pd.DataFrame:
    """Return one row per alignment run with a per-chromatography model summary."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute("""
            SELECT a.alignment_run_id, a.annotation_run_id, a.created_at, a.created_by,
                   a.skip_alignment, a.tags,
                   (SELECT string_agg(
                        m.chromatography || ': ' || COALESCE(m.model_type, '?')
                        || ' R2=' || COALESCE(printf('%.4f', m.r2), 'NA')
                        || ' RMSE=' || COALESCE(printf('%.4f', m.rmse), 'NA')
                        || ' n=' || COALESCE(CAST(m.n_points AS TEXT), 'NA'), '; '
                        ORDER BY m.chromatography)
                    FROM sa_alignment_models m
                    WHERE m.alignment_run_id = a.alignment_run_id) AS models,
                   (SELECT COUNT(*) FROM sa_aligned_peaks p
                    WHERE p.alignment_run_id = a.alignment_run_id) AS n_aligned_peaks,
                   a.output_dir, a.gdrive_url
            FROM sa_alignment_runs a ORDER BY a.alignment_run_id
        """).df()


def load_alignment_models(db_path: str, alignment_run_id: int) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            "SELECT * FROM sa_alignment_models WHERE alignment_run_id = ? ORDER BY chromatography",
            [int(alignment_run_id)],
        ).df()


def load_alignment_qc_points(db_path: str, alignment_run_id: int) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            """SELECT * FROM sa_alignment_qc_points WHERE alignment_run_id = ?
               ORDER BY chromatography, point_num""",
            [int(alignment_run_id)],
        ).df()


def load_aligned_peaks(db_path: str, alignment_run_id: int) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            "SELECT * FROM sa_aligned_peaks WHERE alignment_run_id = ? ORDER BY row_order",
            [int(alignment_run_id)],
        ).df()


def load_deposit_source(db_path: str, alignment_run_id: int) -> pd.DataFrame:
    """Load everything needed to build deposit rows for an alignment run.

    One row per aligned peak (in alignment output order) with aligned RTs,
    peak and compound identity, structures, the annotation run's analyst
    notes, input annotation notes, the run's atlas tags, and an
    ``ms2_scans`` list from ``sa_peak_ms2``.
    """
    with duckdb.connect(str(db_path), read_only=True) as conn:
        params = {"aid": int(alignment_run_id)}
        df = conn.execute("""
            SELECT a.row_order, a.peak_uid, a.peak_label, a.peak_num,
                   a.rt_peak_aligned, a.rt_min_aligned, a.rt_max_aligned,
                   p.compound_name, p.inchi_key, p.adduct, p.polarity, p.chromatography,
                   p.filename, p.mz_theoretical, c.smiles, c.inchi,
                   COALESCE(s.analyst_notes, '') AS analyst_notes,
                   COALESCE(t.annotation_notes, '') AS annotation_notes,
                   r.tags
            FROM sa_aligned_peaks a
            JOIN sa_alignment_runs r USING (alignment_run_id)
            JOIN sa_peaks p ON p.peak_uid = a.peak_uid
            JOIN sa_compounds c ON c.compound_uid = p.compound_uid
            LEFT JOIN sa_run_selections s
                   ON s.annotation_run_id = r.annotation_run_id AND s.peak_uid = a.peak_uid
            LEFT JOIN sa_extraction_targets t ON t.mz_rt_uid = p.mz_rt_uid
            WHERE a.alignment_run_id = $aid
            ORDER BY a.row_order
        """, params).df()
        scans = conn.execute("""
            SELECT m.* FROM sa_peak_ms2 m
            JOIN sa_aligned_peaks a ON a.peak_uid = m.peak_uid
            WHERE a.alignment_run_id = $aid
            ORDER BY m.peak_uid, m.filename
        """, params).df()
    if df.empty:
        raise ValueError(f"Alignment run {alignment_run_id} has no aligned peaks.")
    grouped = _group_scans(scans)
    df["ms2_scans"] = df["peak_uid"].map(lambda uid: grouped.get(uid, []))
    return df


# ---------------------------------------------------------------------------
# Deposit content (per alignment run)
# ---------------------------------------------------------------------------

_ENTRY_TABLES = {
    "atlas": ("sa_atlas_entries", ATLAS_LINK_COLUMNS, ATLAS_COLUMNS),
    "msms": ("sa_msms_entries", MSMS_LINK_COLUMNS, MSMS_COLUMNS),
}
_ROW_TABLES = {"atlas": "sa_deposit_atlas_rows", "msms": "sa_deposit_msms_rows"}


def _cell(value: Any) -> Any:
    """Convert pandas/numpy missing values and scalars for DuckDB parameters."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if isinstance(value, np.generic) else value


def _quoted(columns: list[str]) -> str:
    return ", ".join(f'"{c}"' for c in columns)


def has_deposit_entries(db_path: str, alignment_run_id: int) -> bool:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return bool(_scalar(
            conn, "SELECT COUNT(*) FROM sa_atlas_entries WHERE alignment_run_id = ?", [int(alignment_run_id)],
        ))


def save_generated_entries(
    db_path: str, alignment_run_id: int, atlas: pd.DataFrame, msms: pd.DataFrame,
) -> None:
    """Store generated deposit content for an alignment run (once).

    *atlas* holds ``sa_peak_uid`` + ATLAS_COLUMNS; *msms* holds
    ``sa_peak_uid``, ``sa_ce_label``, ``sa_measured_collision_energy`` +
    MSMS_COLUMNS. Entry IDs are assigned in row order starting at 1.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            if _scalar(conn, "SELECT COUNT(*) FROM sa_atlas_entries WHERE alignment_run_id = ?",
                       [int(alignment_run_id)]):
                raise ValueError(f"Deposit entries already exist for alignment run {alignment_run_id}.")
            for kind, frame in (("atlas", atlas), ("msms", msms)):
                table, links, data_cols = _ENTRY_TABLES[kind]
                columns = [c for c in links if c != "sa_entry_id"] + data_cols
                rows = [
                    (int(alignment_run_id), num, *(_cell(row.get(c)) for c in columns), "generated")
                    for num, row in enumerate(frame.to_dict("records"), start=1)
                ]
                if rows:
                    conn.executemany(
                        f"""INSERT INTO {table} (alignment_run_id, entry_id, {_quoted(columns)}, origin)
                            VALUES ({', '.join('?' * (len(columns) + 3))})""",
                        rows,
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def load_deposit_entries(
    db_path: str, alignment_run_id: int, include_excluded: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (atlas, msms) entries for an alignment run in preview-file layout.

    Columns are the data columns followed by the ``sa_*`` linkage columns
    (``sa_entry_id`` is the entry ID). Excluded entries are omitted unless
    *include_excluded*.
    """
    frames = []
    with duckdb.connect(str(db_path), read_only=True) as conn:
        for kind in ("atlas", "msms"):
            table, links, data_cols = _ENTRY_TABLES[kind]
            other_links = [c for c in links if c != "sa_entry_id"]
            excluded = "" if include_excluded else "AND NOT is_excluded"
            frames.append(conn.execute(
                f"""SELECT {_quoted(data_cols)}, entry_id AS sa_entry_id, {_quoted(other_links)},
                           origin, is_excluded, last_modified_by_deposit
                    FROM {table} WHERE alignment_run_id = ? {excluded} ORDER BY entry_id""",
                [int(alignment_run_id)],
            ).df())
    return frames[0], frames[1]


# ---------------------------------------------------------------------------
# Deposits
# ---------------------------------------------------------------------------

def next_deposit_id(db_path: str) -> int:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return int(_scalar(conn, "SELECT COALESCE(MAX(deposit_id), 0) + 1 FROM sa_deposits"))


def get_deposit(db_path: str, deposit_id: int) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute("SELECT * FROM sa_deposits WHERE deposit_id = ?", [int(deposit_id)]).df()
    if df.empty:
        raise ValueError(f"Deposit {deposit_id} does not exist.")
    return df.iloc[0].to_dict()


def latest_deposit_for_alignment(db_path: str, alignment_run_id: int) -> dict[str, Any] | None:
    """Return the newest non-superseded deposit of an alignment run, or ``None``."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(
            """SELECT * FROM sa_deposits WHERE alignment_run_id = ? AND status != 'superseded'
               ORDER BY deposit_id DESC LIMIT 1""",
            [int(alignment_run_id)],
        ).df()
    return None if df.empty else df.iloc[0].to_dict()


def list_deposits(db_path: str) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute("""
            SELECT d.deposit_id, d.alignment_run_id, a.annotation_run_id, d.status, d.created_at,
                   d.deposited_at, d.compounds_stage, d.atlas_stage, d.msms_stage,
                   (SELECT COUNT(*) FROM sa_deposit_changes c WHERE c.deposit_id = d.deposit_id)
                       AS n_changes,
                   d.output_dir, d.gdrive_url
            FROM sa_deposits d JOIN sa_alignment_runs a USING (alignment_run_id)
            ORDER BY d.deposit_id
        """).df()


def _insert_rows(conn, kind: str, deposit_id: int, version: str, frame: pd.DataFrame) -> None:
    table = _ROW_TABLES[kind]
    _, links, data_cols = _ENTRY_TABLES[kind]
    extra = [c for c in ("change_status",) if c in frame.columns]
    columns = links + data_cols + extra
    rows = [
        (int(deposit_id), version, num, *(_cell(row.get(c)) for c in columns))
        for num, row in enumerate(frame.to_dict("records"), start=1)
    ]
    if rows:
        conn.executemany(
            f"""INSERT INTO {table} (deposit_id, version, row_num, {_quoted(columns)})
                VALUES ({', '.join('?' * (len(columns) + 3))})""",
            rows,
        )


def record_deposit_preview(
    db_path: str,
    *,
    deposit_id: int,
    alignment_run_id: int,
    output_dir: str | Path,
    chem_source: str,
    atlas: pd.DataFrame,
    msms: pd.DataFrame,
    atlas_sha256: str,
    msms_sha256: str,
) -> None:
    """Record a deposit preview and its generated rows; supersede older previews."""
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """UPDATE sa_deposits SET status = 'superseded'
                   WHERE alignment_run_id = ? AND status = 'previewed'""",
                [int(alignment_run_id)],
            )
            conn.execute(
                """INSERT INTO sa_deposits
                   (deposit_id, alignment_run_id, status, output_dir, chem_source, created_at,
                    created_by, atlas_preview_sha256, msms_preview_sha256)
                   VALUES (?, ?, 'previewed', ?, ?, ?, ?, ?, ?)""",
                [int(deposit_id), int(alignment_run_id), str(output_dir), chem_source,
                 prov["timestamp"], prov["analyst"], atlas_sha256, msms_sha256],
            )
            _insert_rows(conn, "atlas", deposit_id, "generated", atlas)
            _insert_rows(conn, "msms", deposit_id, "generated", msms)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def set_deposit_gdrive_url(db_path: str, deposit_id: int, url: str | None) -> None:
    _set_field(db_path, "sa_deposits", "deposit_id", deposit_id, "gdrive_url", url)


def load_deposit_rows(db_path: str, deposit_id: int, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (atlas, msms) snapshot rows of a deposit for *version*."""
    frames = []
    with duckdb.connect(str(db_path), read_only=True) as conn:
        for kind in ("atlas", "msms"):
            frames.append(conn.execute(
                f"SELECT * FROM {_ROW_TABLES[kind]} WHERE deposit_id = ? AND version = ? ORDER BY row_num",
                [int(deposit_id), version],
            ).df())
    return frames[0], frames[1]


def load_deposit_changes(db_path: str, deposit_id: int) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            "SELECT * FROM sa_deposit_changes WHERE deposit_id = ? ORDER BY file, sa_entry_id, column_name",
            [int(deposit_id)],
        ).df()


def _write_back_entries(
    conn, kind: str, alignment_run_id: int, deposit_id: int, final: pd.DataFrame,
    removed_ids: list[int],
) -> pd.DataFrame:
    """Apply final deposit rows to the alignment run's entries; return rows with IDs."""
    table, links, data_cols = _ENTRY_TABLES[kind]
    columns = [c for c in links if c != "sa_entry_id"] + data_cols
    final = final.copy()
    next_id = int(_scalar(
        conn, f"SELECT COALESCE(MAX(entry_id), 0) + 1 FROM {table} WHERE alignment_run_id = ?",
        [int(alignment_run_id)],
    ))
    for index, row in final.iterrows():
        status = row["change_status"]
        if status == "added":
            final.at[index, "sa_entry_id"] = next_id
            conn.execute(
                f"""INSERT INTO {table} (alignment_run_id, entry_id, {_quoted(columns)}, origin,
                                         last_modified_by_deposit)
                    VALUES ({', '.join('?' * (len(columns) + 4))})""",
                [int(alignment_run_id), next_id, *(_cell(row.get(c)) for c in columns),
                 "manual", int(deposit_id)],
            )
            conn.execute(
                "INSERT INTO sa_deposit_changes VALUES (?, ?, ?, 'added', NULL, NULL, NULL)",
                [int(deposit_id), kind, next_id],
            )
            next_id += 1
        elif status == "edited":
            assignments = ", ".join(f'"{c}" = ?' for c in columns)
            conn.execute(
                f"""UPDATE {table} SET {assignments}, is_excluded = FALSE, last_modified_by_deposit = ?
                    WHERE alignment_run_id = ? AND entry_id = ?""",
                [*(_cell(row.get(c)) for c in columns), int(deposit_id),
                 int(alignment_run_id), int(row["sa_entry_id"])],
            )
    for entry_id in removed_ids:
        conn.execute(
            f"""UPDATE {table} SET is_excluded = TRUE, last_modified_by_deposit = ?
                WHERE alignment_run_id = ? AND entry_id = ?""",
            [int(deposit_id), int(alignment_run_id), int(entry_id)],
        )
    return final


def commit_deposit_final(
    db_path: str,
    deposit_id: int,
    *,
    atlas: pd.DataFrame,
    msms: pd.DataFrame,
    changes: pd.DataFrame,
    removed: dict[str, list[int]],
    atlas_sha256: str,
    msms_sha256: str,
    force_add: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the final (read-back) deposit rows and apply edits to the entries.

    In one transaction: records edit/removal changes, writes edited, added
    (as ``manual``) and removed (``is_excluded``) entries back to the
    alignment run's deposit content, snapshots the final rows, and marks
    the deposit ``in_progress``.

    *atlas*/*msms* carry a ``change_status`` column (unchanged/edited/added)
    and ``sa_entry_id`` (NaN for added rows).

    Returns:
        Final (atlas, msms) rows with entry IDs assigned to added rows.
    """
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            deposit = conn.execute(
                "SELECT alignment_run_id, status FROM sa_deposits WHERE deposit_id = ?", [int(deposit_id)],
            ).fetchone()
            if deposit is None:
                raise ValueError(f"Deposit {deposit_id} does not exist.")
            alignment_run_id, status = deposit
            if status != "previewed":
                raise ValueError(f"Deposit {deposit_id} is {status}; only a previewed deposit can start.")
            if not changes.empty:
                conn.executemany(
                    "INSERT INTO sa_deposit_changes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(int(deposit_id), r["file"], int(r["sa_entry_id"]), r["change_type"],
                      _cell(r.get("column_name")), _cell(r.get("old_value")), _cell(r.get("new_value")))
                     for r in changes.to_dict("records")],
                )
            final = {}
            for kind, frame in (("atlas", atlas), ("msms", msms)):
                final[kind] = _write_back_entries(
                    conn, kind, alignment_run_id, deposit_id, frame, removed.get(kind, []),
                )
                _insert_rows(conn, kind, deposit_id, "final", final[kind])
            conn.execute(
                """UPDATE sa_deposits SET status = 'in_progress', atlas_final_sha256 = ?,
                   msms_final_sha256 = ?, deposit_started_at = ?, deposited_by = ?, force_add = ?
                   WHERE deposit_id = ?""",
                [atlas_sha256, msms_sha256, prov["timestamp"], prov["analyst"], bool(force_add),
                 int(deposit_id)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return final["atlas"], final["msms"]


_STAGES = {"compounds", "atlas", "msms"}


def mark_deposit_stage(
    db_path: str,
    deposit_id: int,
    stage: str,
    status: str,
    *,
    outcomes: list[str] | None = None,
    ref_uids: list[str | None] | None = None,
) -> None:
    """Record a deposit stage result and optional per-row outcomes.

    Args:
        stage:    ``compounds``, ``atlas`` or ``msms``.
        status:   ``done`` or ``skipped``.
        outcomes: One outcome per final row (atlas: sheet outcome; msms:
                  import outcome), in row order.
        ref_uids: For msms, the main-database ``ref_uid`` per final row.
    """
    if stage not in _STAGES or status not in {"done", "skipped"}:
        raise ValueError(f"Invalid deposit stage update: {stage}={status}")
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"UPDATE sa_deposits SET {stage}_stage = ?, {stage}_stage_at = ? WHERE deposit_id = ?",
                [status, prov["timestamp"], int(deposit_id)],
            )
            if outcomes is not None and stage in {"atlas", "msms"}:
                column = "sheet_outcome" if stage == "atlas" else "import_outcome"
                table = _ROW_TABLES[stage]
                ref_uids = ref_uids or [None] * len(outcomes)
                for num, (outcome, ref_uid) in enumerate(zip(outcomes, ref_uids), start=1):
                    extra = ", main_db_ref_uid = ?" if stage == "msms" else ""
                    params = [outcome] + ([ref_uid] if stage == "msms" else [])
                    conn.execute(
                        f"""UPDATE {table} SET {column} = ?{extra}
                            WHERE deposit_id = ? AND version = 'final' AND row_num = ?""",
                        [*params, int(deposit_id), num],
                    )
            remaining = _scalar(
                conn,
                """SELECT COUNT(*) FROM sa_deposits WHERE deposit_id = ? AND
                   'pending' IN (compounds_stage, atlas_stage, msms_stage)""",
                [int(deposit_id)],
            )
            if not remaining:
                conn.execute(
                    "UPDATE sa_deposits SET status = 'deposited', deposited_at = ? WHERE deposit_id = ?",
                    [prov["timestamp"], int(deposit_id)],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def reopen_skipped_stages(db_path: str, deposit_id: int, stages: list[str]) -> None:
    """Set previously skipped *stages* back to pending so a retry runs them."""
    if not set(stages) <= _STAGES:
        raise ValueError(f"Invalid deposit stages: {stages}")
    with duckdb.connect(str(db_path)) as conn:
        for stage in stages:
            conn.execute(
                f"""UPDATE sa_deposits SET {stage}_stage = 'pending', {stage}_stage_at = NULL,
                    status = 'in_progress' WHERE deposit_id = ? AND {stage}_stage = 'skipped'""",
                [int(deposit_id)],
            )
