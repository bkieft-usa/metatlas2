"""Run-wide state shared by the pipeline steps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from metatlas2.untargeted.lims import LimsBackend, clean_update_df
from metatlas2.untargeted.status import filter_common_bad_project_names

if TYPE_CHECKING:  # pragma: no cover
    from metatlas2.untargeted.gnps2 import Gnps2Client


@dataclass
class RunContext:
    """Everything a pipeline step needs: config, LIMS backend, flags.

    Attributes:
        config: Loaded config (see :mod:`metatlas2.untargeted.config`).
        lims: LIMS backend (:class:`LimsClient` or :class:`InMemoryLims`).
        dry_run: Read everything, but do not write LIMS, submit jobs, mirror
            to GNPS2, modify MGF files, or upload to Google Drive.
        direct_input: Optional list of project names; bypasses status filters.
        sbatch_queue_file: When set, MZmine sbatch scripts are appended here for
            the host wrapper to submit (container mode). When unset, ``sbatch``
            is called directly if it is on PATH.
        clock: Returns "now" as epoch seconds (overridable in tests).
    """

    config: dict[str, Any]
    lims: LimsBackend
    dry_run: bool = False
    direct_input: list[str] | None = None
    overwrite_mzmine: bool = False
    overwrite_fbmn: bool = False
    overwrite_zip: bool = False
    overwrite_drive: bool = False
    force_reprocess: bool = False
    sbatch_queue_file: str | None = None
    gnps2: "Gnps2Client | None" = None
    clock: Callable[[], float] = field(default=time.time)

    # -- convenience accessors -------------------------------------------------
    @property
    def paths(self) -> dict[str, Any]:
        return self.config["paths"]

    @property
    def output_dir(self) -> str:
        return self.paths["output_dir"]

    @property
    def tasks_table(self) -> str:
        return self.config["lims"]["tasks_table"]

    @property
    def background_designator(self) -> list[str]:
        return self.config["project_selection"]["background_designator"]

    @property
    def group(self) -> str | None:
        return self.paths.get("group")

    def get_gnps2(self) -> "Gnps2Client":
        """Return the GNPS2 client, creating it from config on first use."""
        if self.gnps2 is None:
            from metatlas2.untargeted.gnps2 import Gnps2Client

            self.gnps2 = Gnps2Client.from_config(self.config["gnps2"], dry_run=self.dry_run)
        return self.gnps2

    # -- LIMS helpers ----------------------------------------------------------
    def get_tasks(self) -> pd.DataFrame:
        """Load ``untargeted_tasks``, drop bad names, and apply ``direct_input``."""
        df = self.lims.get_table(self.tasks_table)
        df = filter_common_bad_project_names(df)
        if self.direct_input is not None and not df.empty:
            df = df[df["parent_dir"].isin(self.direct_input)]
        return df

    def write_task_updates(self, df: pd.DataFrame, index_list: list, columns: list[str]) -> None:
        """Send the given rows/columns of a tasks dataframe back to LIMS (method=update)."""
        if not index_list:
            return
        cols = ["Key"] + [c for c in columns if c != "Key"]
        updates = df.loc[df.index.isin(sorted(set(index_list))), cols]
        self.lims.update_table(clean_update_df(updates), self.tasks_table, method="update")
