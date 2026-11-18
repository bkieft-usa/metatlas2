"""Deterministic selected-peak labels and review directory names."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


GROUP_COLUMNS = ["chromatography", "polarity", "compound_uid", "adduct"]
TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{6}_\d{6}Z")


def fix_peak_labels(selections: pd.DataFrame) -> pd.DataFrame:
    """Derive review numbers without modifying the original detected peaks."""
    df = selections.copy()
    if df.empty:
        return df
    counts = df.groupby(GROUP_COLUMNS, dropna=False)["filename"].nunique(dropna=False)
    if (counts > 1).any():
        bad = counts[counts > 1].index
        rows = df.set_index(GROUP_COLUMNS).loc[bad].reset_index()
        details = rows[GROUP_COLUMNS + ["compound_name", "filename"]].drop_duplicates()
        raise ValueError(
            "Selections from multiple files for the same chromatography, polarity, "
            "compound and adduct are not allowed. Correct these selections in the GUI:\n"
            + details.to_string(index=False)
        )
    df["auto_peak_num"] = df["peak_num"]
    ordered = df.sort_values(GROUP_COLUMNS + ["filename", "rt_peak", "peak_uid"])
    groups = ordered.groupby(GROUP_COLUMNS + ["filename"], dropna=False)
    ordered["peak_num"] = groups.cumcount() + 1
    sizes = groups["peak_uid"].transform("size")
    ordered["peak_label"] = ordered["compound_name"]
    multi = sizes > 1
    ordered.loc[multi, "peak_label"] = (
        ordered.loc[multi, "compound_name"] + " (peak"
        + ordered.loc[multi, "peak_num"].astype(str) + ")"
    )
    return ordered


def validate_selected_peaks(selections: pd.DataFrame) -> None:
    """Require a nonempty set of uniquely identified selected peaks."""
    if selections.empty:
        raise ValueError("No selected peaks found; select at least one peak before continuing.")
    if selections["peak_uid"].duplicated().any():
        raise ValueError("Selected peak UIDs are not unique.")


def review_directory(output_root: Path, timestamp: str) -> Path:
    if not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError("Invalid timestamp; use the timestamp printed by the summary, e.g. 20260101T120000_000000Z.")
    return Path(output_root) / "outputs" / f"annotation_run_{timestamp}"

