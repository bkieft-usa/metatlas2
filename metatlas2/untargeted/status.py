"""Status vocabulary, per-polarity file layout, and LIMS dataframe helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

POLARITIES = ("positive", "negative")


class TaskStatus(str, Enum):
    """Status values stored in the LIMS ``untargeted_tasks`` list."""

    INITIATION = "01 initiation"
    RUNNING = "04 running"
    COMPLETE = "07 complete"
    UPLOADED = "08 uploaded"  # computed by the status report only, never stored
    ERROR = "09 error"
    NOT_RELEVANT = "12 not relevant"
    WAITING = "13 waiting"
    NOT_UPLOADED = "14 not uploaded"  # computed by the status report only, never stored

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def short_polarity(polarity: str) -> str:
    """``"positive"`` -> ``"pos"``, ``"negative"`` -> ``"neg"``."""
    return polarity[:3].lower()


def status_column(task: str, polarity: str) -> str:
    """LIMS column name, e.g. ``status_column("mzmine", "positive") == "mzmine_pos_status"``."""
    return f"{task}_{short_polarity(polarity)}_status"


# Suffixes appended to "{project}_{polarity}" for every per-polarity file.
FILE_SUFFIXES: dict[str, str] = {
    "metadata": "_metadata.tab",
    "batch_params": "_batch-params.xml",
    "filelist": "_filelist.txt",
    "mzmine_sbatch": "_mzmine-sbatch.sbatch",
    "mzmine_runner": "_mzmine.sh",
    "mzmine_out": "-mzmine.out",
    "mzmine_err": "-mzmine.err",
    "mzmine_job_id": "_mzmine-job-id.txt",
    "mzmine_status": "_mzmine-status.txt",
    "mzmine_queued": "_mzmine-queued.txt",
    "peak_height": "_peak-height.csv",
    "old_peak_height": "_MSMS_quant.csv",
    "quant": "_quant.csv",
    "mgf": ".mgf",
    "filtering_summary": "_peak-height-filtering-summary.csv",
    "fbmn_task": "_gnps2-fbmn-task.txt",
    "eb_task": "_gnps2-eb-task.txt",
    "gnps2_link": "_gnps2-page-link.txt",
    "fbmn_graphml": "_gnps2-fbmn-network.graphml",
    "fbmn_library_results": "_gnps2-fbmn-library-results.tsv",
}


@dataclass(frozen=True)
class PolarityTask:
    """File layout of one polarity of one untargeted project.

    All files live in ``{output_dir}/{project}_{polarity}/`` and are named
    ``{project}_{polarity}{suffix}`` (see :data:`FILE_SUFFIXES`).
    """

    output_dir: str
    project: str
    polarity: str

    @property
    def prefix(self) -> str:
        return f"{self.project}_{self.polarity}"

    @property
    def directory(self) -> Path:
        return Path(self.output_dir) / self.prefix

    @property
    def short(self) -> str:
        return short_polarity(self.polarity)

    def path(self, kind: str) -> Path:
        """Return the path of file *kind* (a key of :data:`FILE_SUFFIXES`)."""
        return self.directory / f"{self.prefix}{FILE_SUFFIXES[kind]}"

    def filtered_peak_height(self, label: str) -> Path:
        """Path of a background-filtered peak-height table, e.g. label ``"3x-exctrl"``."""
        return self.directory / f"{self.prefix}_peak-height-filtered-{label}.csv"

    def exists(self, kind: str) -> bool:
        return os.path.isfile(self.path(kind))


def check_for_polarities(output_dir: str, project: str) -> list[str] | None:
    """Return the polarities whose ``{project}_{polarity}`` directory exists, or ``None``."""
    found = [p for p in POLARITIES if os.path.exists(os.path.join(output_dir, f"{project}_{p}"))]
    return found or None


def subset_df_by_status(
    df: pd.DataFrame, tasktype: str, status: list[str], inverse: bool = False
) -> pd.DataFrame:
    """Rows where *either* polarity's ``{tasktype}_{pol}_status`` is in *status*.
    """
    pos = df[f"{tasktype}_pos_status"].isin(status)
    neg = df[f"{tasktype}_neg_status"].isin(status)
    if inverse:
        return df[~pos | ~neg]
    return df[pos | neg]


def filter_common_bad_project_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop LIMS rows whose ``parent_dir`` contains a space."""
    if df.empty or "parent_dir" not in df.columns:
        return df
    return df[~df["parent_dir"].astype(str).str.contains(" ")]


def tab_print(message: str = "", indent_level: int = 0) -> str:
    """Indent *message* with tabs."""
    return "\t" * indent_level + message


def start_message(script: str) -> str:
    """Log marker the error-report email script splits runs on. Do not change the wording."""
    return f"Successfully started {script} on {datetime.now()}:"


def end_message(script: str) -> str:
    """Log marker written when a run completes. Do not change the wording."""
    return f"Successfully completed {script} on {datetime.now()}:"


def read_task_id_file(path: str | os.PathLike) -> str | None:
    """Read the ID from a ``{name}={id}`` marker file (job-id / GNPS2 task files)."""
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    return text.split("=")[-1].strip() or None
