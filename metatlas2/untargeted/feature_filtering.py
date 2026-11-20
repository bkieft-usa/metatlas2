"""Feature counts and background-filtered peak-height tables (Step 2 helpers).

Ported from the legacy ``create_filtered_peakheight_file`` and
``calculate_counts_for_lims_table``. The legacy row-by-row loop is
vectorised here, but the outputs (file names, columns, values and their
text formatting) are kept identical. Known legacy quirks that shape the
output are intentionally preserved and marked ``LEGACY QUIRK`` below.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import metatlas2.logging_config as lcf
from metatlas2.untargeted.status import tab_print

logger = lcf.get_logger("untargeted.feature_filtering")

# Large stand-in max used when a project has no control columns of a given type.
_NO_CONTROL_MAX = 1e20


def _sample_group(column: str) -> str | None:
    """Field 12 (sample group) of a peak-height column name, or None."""
    parts = column.split("_")
    return parts[12] if len(parts) > 12 else None


def control_columns(columns: list[str], background_designator: list[str]) -> list[str]:
    """Columns from extraction-control files (designator in field 12, ``mzml`` in name)."""
    out = []
    for col in columns:
        group = _sample_group(col)
        if group is None or "mzml" not in col.lower():
            continue
        if any(d.lower() in group.lower() for d in background_designator):
            out.append(col)
    return out


def txctrl_columns(columns: list[str]) -> list[str]:
    """Columns from treatment-control (``TxCtrl``) files."""
    return [
        col for col in columns
        if _sample_group(col) is not None and "txctrl" in _sample_group(col).lower() and "mzml" in col.lower()
    ]


def count_mgf_feature_ids(mgf_path: str | os.PathLike) -> int:
    """Number of lines containing ``FEATURE_ID`` in an MGF (legacy ``grep FEATURE_ID | wc -l``)."""
    if not os.path.isfile(mgf_path):
        return 0
    with open(mgf_path, errors="replace") as fh:
        return sum(1 for line in fh if "FEATURE_ID" in line)


def calculate_counts_for_lims_table(
    peakheight_filename: str | os.PathLike,
    mgf_filename: str | os.PathLike,
    background_designator: list[str],
) -> tuple[int, int, int]:
    """Count features for the LIMS ``num_*`` columns.

    Returns:
        ``(feature_count, exctrl_count, msms_count)`` where feature_count is the
        number of rows with any sample intensity > 0, exctrl_count the number
        with any control intensity > 0, and msms_count the number of
        ``FEATURE_ID`` lines in the MGF.
    """
    data_table = pd.read_csv(peakheight_filename, sep=",")
    columns = list(data_table.columns)
    exctrl_cols = control_columns(columns, background_designator)
    # LEGACY QUIRK: a column counts as a "sample" if ANY designator is absent
    # from its sample group (identical to "not a control" for one designator).
    sample_cols = [
        col for col in columns
        if _sample_group(col) is not None
        and any(d.lower() not in _sample_group(col).lower() for d in background_designator)
        and "mzml" in col.lower()
    ]
    exctrl_count = feature_count = 0
    if exctrl_cols:
        exctrl_count = int(data_table[exctrl_cols].gt(0).any(axis=1).sum())
        logger.info(tab_print(f"{exctrl_count} features in control samples", 4))
    if sample_cols:
        feature_count = int(data_table[sample_cols].gt(0).any(axis=1).sum())
        logger.info(tab_print(f"{feature_count} features in experimental samples", 4))
    if feature_count <= 1:
        logger.warning(tab_print("Warning! No features found in peak height file", 4))
    msms_count = count_mgf_feature_ids(mgf_filename)
    logger.info(tab_print(f"{msms_count} msms features in mgf file", 4))
    return feature_count, exctrl_count, msms_count


def _row_max(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Row-wise max as float (matches ``iterrows`` + ``Series.max`` formatting)."""
    if not cols:
        return pd.Series(_NO_CONTROL_MAX, index=df.index, dtype=float)
    return df[cols].apply(pd.to_numeric, errors="coerce").astype(float).max(axis=1)


def create_filtered_peakheight_file(
    peakheight_filename: str | os.PathLike,
    background_designator: list[str],
    output_dir: str | os.PathLike,
    project_name: str,
    polarity: str,
    background_ratio: int | float = 3,
    zero_value: float = 2 / 3,
    nonpolar_solvent_front: float = 0.5,
    polar_solvent_front: float = 0.8,
    nonpolar_solvent_end: float = 10,
    polar_solvent_end: float = 17.5,
) -> None:
    """Write background-filtered peak-height tables and a filtering summary.

    A feature passes a filter when its max sample height is >= the threshold
    derived from the control max (``ratio x``, ``+1e4``, ``+1e5``; for both
    ExCtrl and TxCtrl) **and** its RT is inside the chromatography window
    (``_HILIC`` -> polar, ``_C18`` -> nonpolar; otherwise empty files).

    JGI projects (``JGI`` in project field 1) only get the ``{ratio}x-exctrl``
    table, with zeros replaced by ``lowest_nonzero * zero_value``.
    """
    peakheight_filename = str(peakheight_filename)
    output_dir = str(output_dir)
    fields = project_name.split("_")
    jgi_project = len(fields) > 1 and "JGI" in fields[1]

    filtered_filenames = {
        f">{background_ratio}x_exctrl": f"{background_ratio}x-exctrl",
        ">1e4_exctrl": "1e4-exctrl",
        ">1e5_exctrl": "1e5-exctrl",
        f">{background_ratio}x_txctrl": f"{background_ratio}x-txctrl",
        ">1e4_txctrl": "1e4-txctrl",
        ">1e5_txctrl": "1e5-txctrl",
    }
    filtered_filenames = {
        key: os.path.join(output_dir, f"{project_name}_{polarity}_peak-height-filtered-{label}.csv")
        for key, label in filtered_filenames.items()
    }
    jgi_skipped = {">1e4_exctrl", ">1e5_exctrl", f">{background_ratio}x_txctrl", ">1e4_txctrl", ">1e5_txctrl"}

    data_table = pd.read_csv(peakheight_filename, sep=",")

    def write_empty(targets: dict[str, str]) -> None:
        # LEGACY QUIRK: empty tables drop the first ("row ID") column from the header.
        # LEGACY QUIRK: for JGI projects the legacy code tested for "x_exctrl" in the
        # file *path* (which uses "x-exctrl"), so no empty tables are ever written.
        if jgi_project:
            return
        empty = pd.DataFrame(columns=data_table.columns[1:])
        for filename in targets.values():
            empty.to_csv(filename, index=False)

    header = list(data_table.columns)
    exctrl_cols = control_columns(header, background_designator)
    txctrl_cols = txctrl_columns(header)
    # LEGACY QUIRK: "sample" columns are every non-control column, which
    # includes row ID / m/z / RT and any trailing "Unnamed" column.
    sample_cols = [c for c in header if c not in exctrl_cols and c not in txctrl_cols]
    if not exctrl_cols and not txctrl_cols:
        logger.warning(tab_print(
            f"Warning! No columns found in peak height table with designation {background_designator} "
            "or 'TxCtrl'. Returning empty dataframes.", 5))
        write_empty(filtered_filenames)
        return
    if not sample_cols:
        logger.warning(tab_print("Warning! No sample columns found in peak height table. Returning empty dataframes.", 5))
        write_empty(filtered_filenames)
        return

    if "_HILIC" in peakheight_filename:
        solvent_front, solvent_end = polar_solvent_front, polar_solvent_end
    elif "_C18" in peakheight_filename:
        solvent_front, solvent_end = nonpolar_solvent_front, nonpolar_solvent_end
    else:
        logger.warning(tab_print(
            "Warning! Could not infer solvent front and end for filtering, check peak height files and "
            "project names. Returning empty dataframes.", 5))
        write_empty(filtered_filenames)
        return
    column_limits = f"rt_in_{solvent_front}-{solvent_end}min_window"

    exctrl_max = _row_max(data_table, exctrl_cols)
    txctrl_max = _row_max(data_table, txctrl_cols)
    sample_max = _row_max(data_table, sample_cols)
    rt_peak = data_table["row retention time"].astype(float)
    in_window = (rt_peak >= solvent_front) & (rt_peak <= solvent_end)

    thresholds = {
        f">{background_ratio}x_exctrl": exctrl_max * background_ratio,
        ">1e4_exctrl": exctrl_max + 1e4,
        ">1e5_exctrl": exctrl_max + 1e5,
        f">{background_ratio}x_txctrl": txctrl_max * background_ratio,
        ">1e4_txctrl": txctrl_max + 1e4,
        ">1e5_txctrl": txctrl_max + 1e5,
    }

    summary = pd.DataFrame({
        "row ID": data_table["row ID"].astype(float).astype(int),
        "exctrl_max": exctrl_max,
        "txctrl_max": txctrl_max,
        "sample_max": sample_max,
        "rt_peak": rt_peak,
        column_limits: np.where(in_window, "1", "0"),
    })
    passing: dict[str, pd.Series] = {}
    for name, threshold in thresholds.items():
        passes = sample_max >= threshold
        summary[name] = np.where(passes, "1", "0")
        passing[name] = passes & in_window

    if not exctrl_cols:
        summary["exctrl_max"] = np.nan
    if not txctrl_cols:
        summary["txctrl_max"] = np.nan
    if jgi_project:
        drop = ["txctrl_max", *sorted(jgi_skipped)]
        summary = summary.drop(columns=[c for c in drop if c in summary.columns])
    summary_filename = os.path.join(output_dir, f"{project_name}_{polarity}_peak-height-filtering-summary.csv")
    summary.to_csv(summary_filename, index=False)

    for name, mask in passing.items():
        if jgi_project and name in jgi_skipped:
            continue
        indices = list(data_table.index[mask])
        if not indices:
            write_empty({name: filtered_filenames[name]})
            continue
        # LEGACY QUIRK: index 0 (the first feature) is always retained.
        filtered = data_table.loc[sorted(set([0] + indices))].copy()
        if name == f">{background_ratio}x_exctrl" and jgi_project:
            values = filtered[sample_cols + exctrl_cols]
            lowest_non_zero = values[values != 0].min().min()
            filtered = filtered.replace(0, lowest_non_zero * zero_value)
        removed = data_table.shape[0] - filtered.shape[0]
        logger.info(tab_print(
            f"{removed} features removed by the {name} background filter; {filtered.shape[0]} features "
            "retained. Saving filtered peak height table.", 5))
        filtered = filtered.drop(filtered.filter(regex="^Unnamed").columns, axis=1)
        filtered.to_csv(filtered_filenames[name], index=False)


def check_peak_height_table(peak_height_file: str | os.PathLike) -> int:
    """Number of data lines in a peak-height CSV (``wc -l`` minus the header line)."""
    n_newlines = 0
    with open(peak_height_file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n_newlines += chunk.count(b"\n")
    return n_newlines - 1
