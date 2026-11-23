"""Sync raw/mzML/h5 files on disk into the LIMS file tables (runs in the container).

Port of ``metatlas/io/update_lcmsfiles_in_lims.py``. Three steps, as before:

1. **File tables.** For each of ``raw_file`` (``*.raw``), ``mzml_file``
   (``*.mzML``) and ``hdf5_file`` (``*.h5``), insert every file under
   ``raw_data`` that is not in the table yet. Matching is case-insensitive on
   the extension, at any depth, and directories named ``*.raw`` count, as with
   ``find -iname``. Rows are never deleted, as before.
2. **lcmsrun names.** Insert a ``lcmsrun`` row for every file ``name`` (the
   path relative to ``raw_data`` with extensions stripped) that has none.
3. **lcmsrun links.** Point ``lcmsrun.raw_file`` / ``mzml_file`` /
   ``hdf5_file`` at the matching file-table rows (by ``name``).

Differences from the legacy script:

* **Disk scan:** one walk of ``raw_data`` collects all three file types. The
  legacy script ran three ``find`` passes.
* **Fewer queries:** each LIMS table is fetched once and re-fetched only after
  inserts, with no ``UNION`` query.
* **Deterministic names.** The legacy name builder joined the dot-separated
  pieces through a Python ``set``, so names with internal dots (e.g.
  ``n.__234.a``) came out in random order. Pieces now keep their original
  order. Names without internal dots are unchanged.
* **Stable links.** If several file rows share a ``name``, the legacy code
  linked whichever row LabKey returned first, which can flip between runs. Now
  an existing link that is still valid is kept; otherwise the newest (highest
  ``Key``) row wins.
* **Safety and logging:** a truncated LIMS read aborts the run instead of
  inserting duplicates; timings and counts are logged; ``--dry_run`` reports
  without writing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.shared_tools.lims import LimsBackend

logger = lcf.get_logger("ingest.lims_file_sync")

RAW_DATA_DIR = "/global/cfs/cdirs/metatlas/raw_data"

#: LIMS table -> (file_type value, case-insensitive extension matched on disk)
FILE_TABLES: dict[str, tuple[str, str]] = {
    "raw_file": ("raw", ".raw"),
    "mzml_file": ("mzml", ".mzml"),
    "hdf5_file": ("hdf5", ".h5"),
}
NAME_EXTENSIONS = {"raw", "tab", "gz", "pactolus", "mzML", "d", "h5"}
LCMSRUN_TABLE = "lcmsrun"


# ---------------------------------------------------------------------------
# Disk scan and names
# ---------------------------------------------------------------------------


def scan_raw_data(root: str = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Walk *root* once and collect files for every LIMS file table.

    Same matches as ``find ROOT -iname '*<ext>' -printf '%Ts %p'``: any depth,
    case-insensitive extension, files *and* directories (Waters ``.raw`` are
    directories), symlinks listed but not followed, unreadable directories
    skipped. ``timeepoch`` is the integer mtime (``lstat``).

    Returns:
        ``{table: DataFrame(filename, file_type, timeepoch)}``.
    """
    found: dict[str, list[tuple[str, int]]] = {t: [] for t in FILE_TABLES}
    suffixes = [(t, ext) for t, (_ft, ext) in FILE_TABLES.items()]
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            it = os.scandir(directory)
        except OSError as err:
            logger.debug(f"Skipping unreadable directory {directory}: {err}")
            continue
        with it:
            for entry in it:
                lower = entry.name.lower()
                for table, ext in suffixes:
                    if lower.endswith(ext):
                        try:
                            mtime = int(entry.stat(follow_symlinks=False).st_mtime)
                        except OSError:
                            continue
                        found[table].append((entry.path, mtime))
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                except OSError:
                    continue
    return {
        table: pd.DataFrame(found[table], columns=["filename", "timeepoch"]).assign(
            file_type=FILE_TABLES[table][0])[["filename", "file_type", "timeepoch"]]
        for table in FILE_TABLES
    }


def complex_name_splitter(filename: str, strippath: str = RAW_DATA_DIR) -> str:
    """LIMS ``name`` of a file: path relative to raw_data, with extensions removed.

    Same as the legacy function except that the remaining dot-separated pieces
    keep their original order (legacy used a ``set``, giving random order).
    """
    if "/project/projectdirs" in filename:
        filename = filename.replace("/project/projectdirs", "/cfs/cdirs")
    basename = os.path.basename(filename)
    pathname = filename.replace(basename, "")
    pathname = pathname.replace(strippath, "")
    pathname = pathname.strip("/")
    pieces: list[str] = []
    for piece in basename.split("."):
        if piece not in NAME_EXTENSIONS and piece not in pieces:
            pieces.append(piece)
    name = ".".join(pieces).replace("_spectral-hits", "")
    return os.path.join(pathname, name)


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    """Counts collected during a sync (logged at the end)."""

    on_disk: dict[str, int] = field(default_factory=dict)
    in_lims: dict[str, int] = field(default_factory=dict)
    inserted_files: dict[str, int] = field(default_factory=dict)
    lims_only: dict[str, int] = field(default_factory=dict)
    inserted_lcmsruns: int = 0
    lcmsrun_names_without_files: int = 0
    updated_links: dict[str, int] = field(default_factory=dict)
    ambiguous_names: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def _sample(values, n: int = 5) -> str:
    values = list(values)
    more = f" (+{len(values) - n} more)" if len(values) > n else ""
    return ", ".join(str(v) for v in values[:n]) + more


def sync_file_tables(lims: LimsBackend, disk: dict[str, pd.DataFrame], report: SyncReport,
                     strippath: str = RAW_DATA_DIR) -> set[str]:
    """Step 1: insert files on disk that are missing from each file table.

    Returns:
        Names of the tables that received inserts.
    """
    changed: set[str] = set()
    for table, df in disk.items():
        t0 = time.time()
        df_lims = lims.get_table(table, columns=["filename", "Key"])
        report.on_disk[table] = len(df)
        report.in_lims[table] = len(df_lims)
        diff = pd.merge(df, df_lims, on=["filename"], how="outer", indicator="Exist")
        new = diff.loc[diff["Exist"] == "left_only", ["file_type", "filename", "timeepoch"]].copy()
        report.lims_only[table] = int((diff["Exist"] == "right_only").sum())
        report.inserted_files[table] = len(new)
        if len(new):
            new["timeepoch"] = new["timeepoch"].astype(int)
            new["name"] = new["filename"].apply(lambda f: complex_name_splitter(f, strippath))
            new["basename"] = new["filename"].apply(os.path.basename)
            logger.info(f"    {table}: inserting {len(new)} new file(s): {_sample(new['basename'])}")
            lims.update_table(new, table, method="insert")
            changed.add(table)
        logger.info(
            f"    {table}: {len(df):,} on disk, {len(df_lims):,} in LIMS, {len(new):,} new, "
            f"{report.lims_only[table]:,} only in LIMS (not deleted) [{time.time() - t0:.1f}s]")
    return changed


def _choose_links(lcmsruns: pd.DataFrame, data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Rows of ``lcmsrun`` whose *column* must change, with the file-table Key to set."""
    merged = pd.merge(lcmsruns[["Key", "name", column]], data[["Key", "name"]], on="name", how="inner",
                      suffixes=("", "_file"))
    existing = pd.to_numeric(merged[column], errors="coerce").fillna(-1)
    merged["_existing"] = existing
    counts = merged.groupby("Key")["Key_file"].transform("size")
    ambiguous = int(merged.loc[counts > 1, "Key"].nunique())
    # Keep a run's current link if it is still one of the candidates.
    keep_keys = merged.loc[merged["Key_file"] == merged["_existing"], "Key"].unique()
    candidates = merged[~merged["Key"].isin(keep_keys)]
    chosen = candidates.sort_values("Key_file").drop_duplicates(subset=["Key"], keep="last")
    updates = chosen[["Key", "Key_file"]].rename(columns={"Key_file": column})
    updates[column] = updates[column].astype(int)
    return updates, ambiguous


def sync_lcmsruns(lims: LimsBackend, report: SyncReport) -> None:
    """Steps 2-3: add missing lcmsrun names, then fix the file links."""
    t0 = time.time()
    tables = {t: lims.get_table(t, columns=["Key", "name"]) for t in FILE_TABLES}
    names = set().union(*(set(df["name"]) for df in tables.values()))
    lcmsruns = lims.get_table(LCMSRUN_TABLE, columns=["Key", "name", *FILE_TABLES])
    existing = set(lcmsruns["name"])
    missing = sorted(names - existing, key=str)
    report.inserted_lcmsruns = len(missing)
    report.lcmsrun_names_without_files = len(existing - names)
    if missing:
        logger.info(f"    lcmsrun: inserting {len(missing)} new name(s): {_sample(missing)}")
        lims.update_table(pd.DataFrame({"name": missing}), LCMSRUN_TABLE, method="insert")
        lcmsruns = lims.get_table(LCMSRUN_TABLE, columns=["Key", "name", *FILE_TABLES])
    logger.info(f"    lcmsrun: {len(existing):,} rows, {len(missing):,} new names, "
                f"{report.lcmsrun_names_without_files:,} names with no file rows (not deleted)")
    report.seconds["lcmsrun names"] = round(time.time() - t0, 1)

    t0 = time.time()
    for column, data in tables.items():
        updates, ambiguous = _choose_links(lcmsruns, data, column)
        report.updated_links[column] = len(updates)
        report.ambiguous_names[column] = ambiguous
        if len(updates):
            lims.update_table(updates, LCMSRUN_TABLE, method="update")
        logger.info(f"    lcmsrun.{column}: {len(updates):,} link(s) updated"
                    + (f", {ambiguous:,} names match several file rows" if ambiguous else ""))
    report.seconds["lcmsrun links"] = round(time.time() - t0, 1)


def run_lims_file_sync(
    lims: LimsBackend,
    raw_data_dir: str = RAW_DATA_DIR,
    update_file_tables: bool = True,
    update_lcmsrun_names: bool = True,
    update_lcmsrun_links: bool = True,
) -> SyncReport:
    """Run the sync (any subset of the three steps). See module docstring."""
    report = SyncReport()
    t_all = time.time()
    if update_file_tables:
        t0 = time.time()
        disk = scan_raw_data(raw_data_dir)
        report.seconds["disk scan"] = round(time.time() - t0, 1)
        logger.info(f"Scanned {raw_data_dir} in {report.seconds['disk scan']:.0f}s: "
                    + ", ".join(f"{len(df):,} {FILE_TABLES[t][0]}" for t, df in disk.items()))
        logger.info("Step 1/3: updating LIMS file tables...")
        t0 = time.time()
        sync_file_tables(lims, disk, report, strippath=raw_data_dir)
        report.seconds["file tables"] = round(time.time() - t0, 1)
    if update_lcmsrun_names or update_lcmsrun_links:
        logger.info("Steps 2-3/3: updating lcmsrun names and file links...")
        if not update_lcmsrun_links:
            logger.info("    (link update skipped by request)")
        _run_lcmsrun_steps(lims, report, update_lcmsrun_names, update_lcmsrun_links)
    report.seconds["total"] = round(time.time() - t_all, 1)
    logger.info(
        "Summary: inserted "
        + ", ".join(f"{v:,} {k}" for k, v in report.inserted_files.items())
        + f", {report.inserted_lcmsruns:,} lcmsrun; updated "
        + ", ".join(f"{v:,} {k} links" for k, v in report.updated_links.items())
        + f"; timings {report.seconds}")
    return report


def _run_lcmsrun_steps(lims: LimsBackend, report: SyncReport, names: bool, links: bool) -> None:
    if names and links:
        sync_lcmsruns(lims, report)
        return

    class _NoInsert:
        """Wraps *lims* so lcmsrun inserts (names) or updates (links) are suppressed."""

        def __init__(self, inner: LimsBackend) -> None:
            self.inner = inner
            self.dry_run = inner.dry_run

        def get_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
            return self.inner.get_table(table, columns)

        def update_table(self, df: pd.DataFrame, table: str, method: str = "update") -> Any:
            if table == LCMSRUN_TABLE and ((method == "insert" and not names) or (method == "update" and not links)):
                return None
            return self.inner.update_table(df, table, method=method)

    sync_lcmsruns(_NoInsert(lims), report)
