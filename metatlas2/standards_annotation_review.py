"""Deterministic selected-peak labels, review directory names, and project history."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


GROUP_COLUMNS = ["chromatography", "polarity", "compound_uid", "adduct"]
ANNOTATION_RUN_DIR_PATTERN = re.compile(r"annotation_run_[1-9]\d*")


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


def _positive_id(value: int, kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int,)) and not str(value).isdigit():
        raise ValueError(f"Invalid {kind} ID: {value!r}; use a number from the project history.")
    value = int(value)
    if value < 1:
        raise ValueError(f"Invalid {kind} ID: {value!r}; use a number from the project history.")
    return value


def annotation_run_dir(output_root: Path, annotation_run_id: int) -> Path:
    """Return ``<output_root>/outputs/annotation_run_<N>``."""
    run_id = _positive_id(annotation_run_id, "annotation run")
    return Path(output_root) / "outputs" / f"annotation_run_{run_id}"


def alignment_run_dir(annotation_dir: Path, alignment_run_id: int) -> Path:
    """Return ``<annotation_dir>/rt_alignment_<M>``."""
    return Path(annotation_dir) / f"rt_alignment_{_positive_id(alignment_run_id, 'alignment run')}"


def deposit_dir(alignment_dir: Path, deposit_id: int) -> Path:
    """Return ``<alignment_dir>/deposit_<K>``."""
    return Path(alignment_dir) / f"deposit_{_positive_id(deposit_id, 'deposit')}"


def show_standards_project_history(db_path: str) -> dict[str, pd.DataFrame]:
    """Display annotation runs, alignment runs and deposits for choosing IDs.

    Returns the three tables (keys ``annotation_runs``, ``alignment_runs``,
    ``deposits``) so they can also be inspected programmatically.
    """
    from IPython.display import HTML, display

    from metatlas2 import standards_annotation_db as sadb

    tables = {
        "annotation_runs": sadb.list_annotation_runs(db_path),
        "alignment_runs": sadb.list_alignment_runs(db_path),
        "deposits": sadb.list_deposits(db_path),
    }
    titles = {
        "annotation_runs": "Annotation runs",
        "alignment_runs": "RT alignment runs",
        "deposits": "Deposits",
    }
    for key, frame in tables.items():
        display(HTML(f"<h4>{titles[key]}</h4>"))
        if frame.empty:
            display(HTML("<p><i>None yet.</i></p>"))
        else:
            shown = frame.copy()
            for column in ("created_at", "frozen_at", "deposited_at"):
                if column in shown:
                    shown[column] = shown[column].map(
                        lambda v: str(v)[:19].replace("T", " ") if isinstance(v, str) else v
                    )
            display(shown.drop(columns=["gdrive_url"], errors="ignore").set_index(shown.columns[0]))
    return tables
