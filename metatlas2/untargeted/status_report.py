"""Untargeted project status table (LIMS statuses + Google Drive upload state).

The tab-separated output format is relied on by
``scripts/untargeted/untargeted_project_summary_report.sh``: the first column
(``Modified``) must start with a digit and columns 3-7 are the four
MZmine/FBMN statuses followed by the Drive status.
"""

from __future__ import annotations

import subprocess

import pandas as pd

import metatlas2.logging_config as lcf
from metatlas2 import gdrive_upload as gdu
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.status import TaskStatus

logger = lcf.get_logger("untargeted.status_report")

REPORT_COLUMNS = [
    "Modified", "parent_dir",
    "mzmine_pos_status", "mzmine_neg_status", "fbmn_pos_status", "fbmn_neg_status",
    "gdrive_status", "upload_date",
]


def get_untargeted_status(ctx: RunContext, drive_listing: list[list[str]] | None = None) -> pd.DataFrame | None:
    """Build the status table for all (or ``ctx.direct_input``) projects.

    Args:
        ctx: Run context (LIMS + config).
        drive_listing: Optional pre-fetched ``rclone lsl`` rows
            (``[size, date, time, name]``); fetched from the configured remote if None.

    Returns:
        DataFrame with :data:`REPORT_COLUMNS` sorted by ``Modified``, or None if
        there are no matching projects or Drive could not be listed.
    """
    df = ctx.get_tasks()
    if df.empty:
        logger.warning("Could not find untargeted tasks to report! Check that the input is a comma-separated list.")
        return None
    if drive_listing is None:
        try:
            drive_listing = gdu.rclone_list(ctx.config["gdrive"]["remote"], long=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            logger.warning(f"Warning! Could not get list of projects from Google Drive ({err}). Skipping status report...")
            return None
    uploaded = {row[-1]: f"{row[1]} {row[2]}" for row in drive_listing if len(row) >= 4}
    out = df.copy()
    archives = out["parent_dir"].astype(str) + ".zip"
    out["gdrive_status"] = [
        TaskStatus.UPLOADED.value if a in uploaded else TaskStatus.NOT_UPLOADED.value for a in archives]
    out["upload_date"] = [uploaded.get(a, "NA") for a in archives]
    for col in REPORT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[REPORT_COLUMNS].sort_values(by="Modified", ascending=True)


def format_status_table(table: pd.DataFrame, print_recent: int | None = None) -> str:
    """Tab-separated text of the last *print_recent* rows (all rows if None)."""
    if print_recent is not None:
        table = table.tail(print_recent)
    return table.to_csv(sep="\t", index=False)
