"""
metatlas2/standards_annotation_db.py

Project DuckDB creation and all read/write helpers for the standards-annotation
workflow.  Provides resumability: extraction is skipped on re-run if the DB
already contains data.

Tables:
  sa_compounds          — one row per (compound_name, adduct): stable identity
  sa_extraction_targets — one row per (compound_uid, mz_rt_uid, filename):
                          bridges the extraction engine's per-file join key to
                          the stable compound identity
  sa_ms1                — extracted MS1 data (wide format), keyed by mz_rt_uid+filename
  sa_ms2                — extracted MS2 data (wide format), keyed by mz_rt_uid+filename+scan_rt
  sa_tic                — TIC per file
  sa_peaks              — detected MS1 peaks (one row per peak per file per adduct)
  sa_peak_ms2           — best MS2 scan per peak and source file
  sa_ms1_sources        — selected MS1 file per compound, chromatography, polarity
  sa_analyst_selections — analyst peak selections: (peak_uid, analyst_notes)
  sa_workflow_state     — per-(compound_uid, chromatography) annotation status
  sa_no_good_peaks      — explicit no-good-peaks choices by compound/chrom/polarity
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

import metatlas2.logging_config as lcf
import metatlas2.workflow_objects as wfo
from metatlas2.utils import get_provenance

logger = lcf.get_logger("standards_annotation_db")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_output_root(config: dict[str, Any]) -> Path:
    """Return the root output directory for this annotation run.

    Path: ``$METATLAS_DATA_DIR/projects/reference_standards_annotation/<username>/<standards_run_name>/``

    Args:
        config: Parsed config dict from
                :func:`metatlas2.load_tools.load_standards_config`.

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
      - ``sa_workflow_state``: keyed by (compound_uid, chromatography) so that
        C18 and HILIC states are independent for the same compound.

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
        # sa_analyst_selections — analyst peak selections
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_analyst_selections (
                peak_uid       TEXT PRIMARY KEY REFERENCES sa_peaks(peak_uid),
                analyst_notes  TEXT DEFAULT '',
                selected_by    TEXT,
                selected_date  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_compound_notes (
                compound_name TEXT NOT NULL,
                chromatography TEXT NOT NULL,
                analyst_notes TEXT NOT NULL,
                PRIMARY KEY (compound_name, chromatography)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_no_good_peaks (
                compound_name TEXT NOT NULL,
                chromatography TEXT NOT NULL,
                polarity TEXT NOT NULL,
                PRIMARY KEY (compound_name, chromatography, polarity)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_gui_checkpoint (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                compound_name TEXT NOT NULL
            )
        """)

        # ------------------------------------------------------------------
        # sa_workflow_state — per (compound_uid, chromatography) status
        # This allows C18 and HILIC states to be tracked independently.
        # ------------------------------------------------------------------
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sa_workflow_state (
                compound_uid      TEXT NOT NULL REFERENCES sa_compounds(compound_uid),
                chromatography    TEXT NOT NULL,
                annotation_status TEXT DEFAULT 'pending',
                last_updated      TEXT,
                PRIMARY KEY (compound_uid, chromatography)
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

    logger.info(f"Project DB ready at {db_path}")


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
      6. ``sa_workflow_state`` — initial 'pending' state per (compound_uid, chromatography)

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

    prov = get_provenance()
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

            # ------------------------------------------------------------------
            # sa_workflow_state — one row per (compound_uid, chromatography)
            # ------------------------------------------------------------------
            if expanded_table is not None and not expanded_table.empty:
                seen_state_keys: set[tuple] = set()
                state_rows = []
                for _, row in expanded_table.iterrows():
                    key = (str(row["compound_uid"]), str(row.get("chromatography", "")))
                    if key in seen_state_keys:
                        continue
                    seen_state_keys.add(key)
                    state_rows.append((
                        key[0],
                        key[1],
                        "pending",
                        prov["timestamp"],
                    ))
                if state_rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO sa_workflow_state VALUES (?, ?, ?, ?)",
                        state_rows,
                    )
                    logger.info(
                        f"Initialised workflow state for {len(state_rows)} "
                        f"(compound_uid, chromatography) pairs"
                    )

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
                conn.executemany(
                    "INSERT INTO sa_peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [_peak_values(peak) for peak in peaks],
                )
            if peak_ms2_rows:
                conn.executemany(
                    "INSERT INTO sa_peak_ms2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [_peak_ms2_values(row) for row in peak_ms2_rows],
                )
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
            conn.executemany(
                """
                INSERT OR IGNORE INTO sa_peaks VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                rows,
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to save peaks: {exc}")
            raise
    logger.info(f"Saved {len(rows)} peaks to sa_peaks")


def load_peaks(db_path: str) -> pd.DataFrame:
    """Load all rows from ``sa_peaks``.

    Returns an empty DataFrame if the table is empty or the DB does not exist.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            return conn.execute(
                """
                SELECT * FROM sa_peaks
                ORDER BY compound_name, chromatography, polarity, adduct, filename, peak_num
                """
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

    Args:
        db_path: Path to the project DuckDB.
        peak:    Peak dict with all ``sa_peaks`` fields.
    """
    if any(row["peak_uid"] != peak["peak_uid"] for row in ms2):
        raise ValueError("Manual peak and MS2 association have different peak UIDs.")
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                "INSERT INTO sa_peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _peak_values(peak),
            )
            if ms2:
                conn.executemany(
                    "INSERT INTO sa_peak_ms2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [_peak_ms2_values(row) for row in ms2],
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def delete_manual_peak_from_db(db_path: str, peak_uid: str) -> None:
    """Remove a manual peak (and its MS2 association) from the DB.

    Args:
        db_path:  Path to the project DuckDB.
        peak_uid: Primary key of the peak to remove.
    """
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM sa_peak_ms2 WHERE peak_uid = ?", [peak_uid])
            conn.execute(
                "DELETE FROM sa_peaks WHERE peak_uid = ? AND is_manual = TRUE",
                [peak_uid],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to delete manual peak {peak_uid}: {exc}")
            raise


# ---------------------------------------------------------------------------
# Analyst selection persistence (sa_analyst_selections)
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
) -> None:
    """Atomically save all adduct selections and workflow state for a compound."""
    if not compound_uids:
        raise ValueError("Cannot save selections: compound has no database identities.")
    if len({uid for uid, _ in selections}) != len(selections):
        raise ValueError("Cannot save duplicate selected peak UIDs.")
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            placeholders = ", ".join("?" for _ in compound_uids)
            valid = {
                row[0]: (row[1], row[2].upper()) for row in conn.execute(
                    f"SELECT peak_uid, chromatography, polarity FROM sa_peaks WHERE compound_uid IN ({placeholders}) AND compound_name = ?",
                    [*compound_uids, compound_name],
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
                f"DELETE FROM sa_analyst_selections WHERE peak_uid IN "
                f"(SELECT peak_uid FROM sa_peaks WHERE compound_uid IN ({placeholders}))",
                compound_uids,
            )
            if selections:
                conn.executemany(
                    "INSERT INTO sa_analyst_selections VALUES (?, ?, ?, ?)",
                    [(uid, notes, prov["analyst"], prov["timestamp"]) for uid, notes in selections],
                )
            conn.execute("DELETE FROM sa_no_good_peaks WHERE compound_name = ?", [compound_name])
            if no_good:
                conn.executemany(
                    "INSERT INTO sa_no_good_peaks VALUES (?, ?, ?)",
                    [(compound_name, chrom, pol) for chrom, pol in sorted(no_good)],
                )
            for chrom in chromatographies:
                conn.execute(
                    """INSERT OR REPLACE INTO sa_compound_notes VALUES (?, ?, ?)""",
                    [compound_name, chrom, notes_by_chrom.get(chrom, "")],
                )
            conn.execute(
                "INSERT OR REPLACE INTO sa_gui_checkpoint VALUES (1, ?)",
                [resume_compound if resume_compound is not None else compound_name],
            )
            for uid in compound_uids:
                for chrom in chromatographies:
                    conn.execute(
                        """INSERT INTO sa_workflow_state VALUES (?, ?, 'complete', ?)
                           ON CONFLICT (compound_uid, chromatography) DO UPDATE SET
                           annotation_status = excluded.annotation_status,
                           last_updated = excluded.last_updated""",
                        [uid, chrom, prov["timestamp"]],
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def load_gui_checkpoint(db_path: str) -> str | None:
    """Return the compound at the last saved GUI checkpoint, if one exists."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        row = conn.execute(
            "SELECT compound_name FROM sa_gui_checkpoint WHERE id = 1"
        ).fetchone()
    return row[0] if row else None


def load_compound_notes(db_path: str, compound_name: str) -> dict[str, str]:
    """Return saved analyst notes by chromatography for a compound."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            "SELECT chromatography, analyst_notes FROM sa_compound_notes WHERE compound_name = ?",
            [compound_name],
        ).fetchall()
    return dict(rows)


def load_no_good_peaks(db_path: str, compound_name: str) -> list[tuple[str, str]]:
    """Return saved no-good-peaks choices for a compound."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            "SELECT chromatography, polarity FROM sa_no_good_peaks WHERE compound_name = ?",
            [compound_name],
        ).fetchall()


def validate_annotation_choices(db_path: str) -> None:
    """Require a peak or an explicit no-good choice for every target group."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        missing = conn.execute("""
            SELECT DISTINCT c.compound_name, t.chromatography, upper(c.polarity) AS polarity
            FROM sa_extraction_targets t JOIN sa_compounds c USING (compound_uid)
            WHERE NOT EXISTS (
                SELECT 1 FROM sa_analyst_selections s JOIN sa_peaks p USING (peak_uid)
                WHERE p.compound_name = c.compound_name
                  AND p.chromatography = t.chromatography
                  AND upper(p.polarity) = upper(c.polarity)
            ) AND NOT EXISTS (
                SELECT 1 FROM sa_no_good_peaks n
                WHERE n.compound_name = c.compound_name
                  AND n.chromatography = t.chromatography
                  AND n.polarity = upper(c.polarity)
            )
            ORDER BY c.compound_name, t.chromatography, polarity
        """).fetchall()
    if missing:
        details = "\n".join(f"  {name} | {chrom} | {pol}" for name, chrom, pol in missing)
        raise ValueError(
            "Select a peak or 'No good peaks' for each compound, chromatography, "
            f"and polarity in the GUI, then save:\n{details}"
        )


def save_analyst_selection(
    db_path: str,
    peak_uid: str,
    analyst_notes: str = "",
) -> None:
    """Mark a peak as selected by the analyst (upsert into ``sa_analyst_selections``).

    Args:
        db_path:       Path to the project DuckDB.
        peak_uid:      Primary key of the peak in ``sa_peaks``.
        analyst_notes: Free-text notes from the analyst.
    """
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO sa_analyst_selections VALUES (?, ?, ?, ?)
                """,
                [peak_uid, analyst_notes, prov["analyst"], prov["timestamp"]],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to save analyst selection for {peak_uid}: {exc}")
            raise


def delete_analyst_selection(db_path: str, peak_uid: str) -> None:
    """Remove a single analyst selection by ``peak_uid``.

    Args:
        db_path:  Path to the project DuckDB.
        peak_uid: Primary key of the peak to deselect.
    """
    if not Path(db_path).exists():
        return
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                "DELETE FROM sa_analyst_selections WHERE peak_uid = ?",
                [peak_uid],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to delete analyst selection {peak_uid}: {exc}")


def delete_all_analyst_selections_for_compound(
    db_path: str,
    compound_uid: str,
) -> None:
    """Remove all analyst selections for a compound (full replace-on-save pattern).

    Joins ``sa_analyst_selections`` with ``sa_peaks`` on ``compound_uid`` to
    find all peak_uids belonging to this compound across all files and adducts.

    Args:
        db_path:      Path to the project DuckDB.
        compound_uid: Stable compound identity whose selections should be cleared.
    """
    if not Path(db_path).exists():
        return
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """
                DELETE FROM sa_analyst_selections
                WHERE peak_uid IN (
                    SELECT peak_uid FROM sa_peaks
                    WHERE compound_uid = ?
                )
                """,
                [compound_uid],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(
                f"Failed to delete analyst selections for compound_uid={compound_uid}: {exc}"
            )


def load_analyst_selections(db_path: str) -> pd.DataFrame:
    """Load all analyst selections joined with peak metadata.

    Returns a DataFrame with all columns from ``sa_peaks`` plus
    ``analyst_notes``, ``selected_by``, ``selected_date`` from
    ``sa_analyst_selections``.  Returns an empty DataFrame if none exist.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            df = conn.execute(
                """
                SELECT p.*, s.analyst_notes, s.selected_by, s.selected_date
                FROM sa_analyst_selections s
                JOIN sa_peaks p ON s.peak_uid = p.peak_uid
                ORDER BY p.compound_name, p.chromatography, p.polarity,
                         p.adduct, p.filename, p.peak_num
                """
            ).df()
        return df
    except Exception as exc:
        raise RuntimeError(f"Could not load analyst selections from {db_path}: {exc}") from exc


def load_selected_peaks_with_ms2(
    db_path: str, *, connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load selected peaks with a list of their best MS2 scans.

    Returns a DataFrame joining ``sa_analyst_selections``, ``sa_peaks``,
    ``sa_compounds``, and ``sa_peak_ms2`` for export to atlas CSV and
    msms-refs tab formats.

    Args:
        db_path: Path to the project DuckDB.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    try:
        with (nullcontext(connection) if connection is not None
              else duckdb.connect(str(db_path), read_only=True)) as conn:
            df = conn.execute(
                """
                SELECT
                    p.peak_uid,
                    p.compound_uid,
                    p.compound_name,
                    p.inchi_key,
                    p.adduct,
                    p.polarity,
                    p.chromatography,
                    p.collision_energy_label,
                    p.filename,
                    p.mz_rt_uid,
                    p.mz_theoretical,
                    p.rt_peak,
                    p.rt_min,
                    p.rt_max,
                    p.peak_intensity,
                    p.mz_observed,
                    p.peak_num,
                    p.is_manual,
                    s.analyst_notes,
                    c.smiles,
                    c.inchi,
                    t.annotation_notes
                FROM sa_analyst_selections s
                JOIN sa_peaks p ON s.peak_uid = p.peak_uid
                JOIN sa_compounds c ON p.compound_uid = c.compound_uid
                JOIN sa_extraction_targets t ON p.mz_rt_uid = t.mz_rt_uid
                ORDER BY p.compound_name, p.chromatography, p.polarity,
                         p.collision_energy_label, p.adduct, p.filename, p.peak_num
                """
            ).df()
            scans = conn.execute("""
                SELECT m.* FROM sa_peak_ms2 m
                JOIN sa_analyst_selections s ON s.peak_uid = m.peak_uid
                ORDER BY m.peak_uid, m.filename
            """).df()
        scans_by_peak: dict[str, list[dict]] = {}
        for row in scans.to_dict("records"):
            for col in ("frag_mzs", "frag_ints"):
                row[col] = json.loads(row[col])
            scans_by_peak.setdefault(row["peak_uid"], []).append(row)
        df["ms2_scans"] = df["peak_uid"].map(lambda uid: scans_by_peak.get(uid, []))
        return df
    except Exception as exc:
        raise RuntimeError(f"Could not load selected peaks with MS2 from {db_path}: {exc}") from exc


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
# Workflow state
# ---------------------------------------------------------------------------

def get_workflow_state(db_path: str) -> pd.DataFrame:
    """Load the per-(compound_uid, chromatography) annotation status.

    Args:
        db_path: Path to the project DuckDB.

    Returns:
        DataFrame with columns ``compound_uid``, ``chromatography``,
        ``annotation_status``, ``last_updated``.
    """
    if not Path(db_path).exists():
        return pd.DataFrame()
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            return conn.execute("SELECT * FROM sa_workflow_state").df()
    except Exception as exc:
        logger.warning(f"Could not load workflow state from {db_path}: {exc}")
        return pd.DataFrame()


def update_workflow_state(
    db_path: str,
    compound_uid: str,
    chromatography: str,
    status: str,
) -> None:
    """Update the annotation status for one (compound_uid, chromatography) pair.

    Args:
        db_path:        Path to the project DuckDB.
        compound_uid:   Stable compound identity key.
        chromatography: Chromatography label (e.g. ``'HILIC'``, ``'C18'``).
        status:         New status string: ``'pending'``, ``'in_progress'``,
                        or ``'complete'``.
    """
    prov = get_provenance()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                """
                INSERT INTO sa_workflow_state
                    (compound_uid, chromatography, annotation_status, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (compound_uid, chromatography) DO UPDATE
                  SET annotation_status = excluded.annotation_status,
                      last_updated      = excluded.last_updated
                """,
                [compound_uid, chromatography, status, prov["timestamp"]],
            )
            conn.execute("COMMIT")
        except Exception as exc:
            conn.execute("ROLLBACK")
            logger.error(
                f"Failed to update workflow state for "
                f"compound_uid={compound_uid}, chromatography={chromatography}: {exc}"
            )
            raise


def get_first_incomplete_compound(db_path: str) -> tuple[str | None, str | None]:
    """Return the (compound_uid, chromatography) of the first incomplete entry.

    Used by the GUI to resume from where the analyst left off.

    Args:
        db_path: Path to the project DuckDB.

    Returns:
        ``(compound_uid, chromatography)`` tuple, or ``(None, None)`` if all
        are complete.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            row = conn.execute(
                """
                SELECT compound_uid, chromatography FROM sa_workflow_state
                WHERE annotation_status != 'complete'
                ORDER BY rowid
                LIMIT 1
                """
            ).fetchone()
        if row:
            return row[0], row[1]
        return None, None
    except Exception as exc:
        raise RuntimeError(f"Could not load workflow state from {db_path}: {exc}") from exc
