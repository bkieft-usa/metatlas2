"""MZmine inputs, Slurm job tracking, and pipeline Steps 2-3.

Slurm in the container
----------------------
``sbatch``/``squeue`` are not available inside the metatlas2 container, so:

* Step 3 does not submit jobs itself. It appends one line per job to a
  *queue file* (``--sbatch-queue-file``); ``scripts/metatlas2.sh untargeted run``
  submits those on the host after the container exits and writes the usual
  ``*_mzmine-job-id.txt`` file. (If no queue file is given and ``sbatch`` is on
  PATH — e.g. running on a login node with the repo venv — jobs are submitted
  directly.)
* Each MZmine sbatch script records its own lifecycle in
  ``*_mzmine-status.txt`` (``RUNNING <jobid> <epoch>`` then ``DONE <epoch>`` or
  ``FAILED <rc> <epoch>``), and Step 3 records every queueing in
  ``*_mzmine-queued.txt`` (one epoch per line). Step 2 uses these markers in
  place of ``squeue`` to decide whether a ``04 running`` job is still alive.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.feature_filtering import (
    calculate_counts_for_lims_table,
    create_filtered_peakheight_file,
)
from metatlas2.untargeted.status import (
    PolarityTask,
    TaskStatus,
    check_for_polarities,
    status_column,
    subset_df_by_status,
    tab_print,
)
from metatlas2.untargeted.utils import set_group

logger = lcf.get_logger("untargeted.mzmine")

# ---------------------------------------------------------------------------
# Batch parameters and sbatch files
# ---------------------------------------------------------------------------


def select_batch_template(instrument: str | None, mzmine_cfg: dict[str, Any]) -> tuple[str, int]:
    """Choose the MZmine batch template for an instrument (project field 6).

    Returns:
        ``(template_path, lims_parameter_row)``. Unknown or missing instruments
        fall back to the (lenient) default template.
    """
    batch_dir = mzmine_cfg["batch_file_dir"]
    if instrument:
        lowered = instrument.lower()
        for rule in mzmine_cfg["template_rules"]:
            if any(sub.lower() in lowered for sub in rule["match"]):
                return os.path.join(batch_dir, rule["template"]), int(rule["parameter_row"])
    return os.path.join(batch_dir, mzmine_cfg["default_template"]), int(mzmine_cfg["default_parameter_row"])


def render_batch_params(template_path: str | os.PathLike, output_prefix: str, placeholder: str) -> str:
    """Return the template XML with *placeholder* replaced by the output prefix path."""
    return Path(template_path).read_text().replace(placeholder, output_prefix)


def build_sbatch_script(task: PolarityTask, mzmine_cfg: dict[str, Any]) -> str:
    """Text of the MZmine sbatch script for one polarity.

    The header and MZmine command match the legacy script; the status-marker
    lines (``RUNNING``/``DONE``/``FAILED``) are new.
    """
    slurm = mzmine_cfg["slurm"]
    prefix = str(task.directory / task.prefix)
    header = ["#!/bin/bash", f"#SBATCH -N {slurm['nodes']}"]
    if slurm.get("exclusive", True):
        header.append("#SBATCH --exclusive")
    header += [
        f'#SBATCH --error="{prefix}-mzmine.err"',
        f'#SBATCH --output="{prefix}-mzmine.out"',
        f"#SBATCH -A {slurm['account']}",
        f"#SBATCH -C {slurm['constraint']}",
        f"#SBATCH --qos={slurm['qos']}",
        f"#SBATCH -t {slurm['time']}",
    ]
    launcher = f"{mzmine_cfg['executable']} {mzmine_cfg.get('extra_args', '')}".strip()
    status_file = task.path("mzmine_status")
    body = f"""
STATUS_FILE="{status_file}"
echo "RUNNING ${{SLURM_JOB_ID:-unknown}} $(date +%s)" >> "$STATUS_FILE"
_mzmine_finish() {{
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "DONE $(date +%s)" >> "$STATUS_FILE"
    else
        echo "FAILED $rc $(date +%s)" >> "$STATUS_FILE"
    fi
}}
trap _mzmine_finish EXIT
trap 'exit 143' TERM INT

{launcher} -batch {task.path('batch_params')} -input {task.path('filelist')}
rc=$?
echo MZMINE IS DONE
exit $rc
"""
    return "\n".join(header) + "\n" + body


def write_mzmine_sbatch_and_runner(task: PolarityTask, mzmine_cfg: dict[str, Any]) -> Path:
    """Write ``*_mzmine-sbatch.sbatch`` and the ``*_mzmine.sh`` runner (``sbatch <file>``)."""
    sbatch_path = task.path("mzmine_sbatch")
    sbatch_path.write_text(build_sbatch_script(task, mzmine_cfg))
    task.path("mzmine_runner").write_text(f"sbatch {sbatch_path}")
    return sbatch_path


_TIME_RE = re.compile(r"^#SBATCH\s+(?:-t|--time)[\s=]+(\S+)", re.MULTILINE)


def parse_slurm_time(value: str) -> float:
    """Convert a Slurm time string (``M``, ``H:M:S``, ``D-H:M:S``, ...) to seconds."""
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
    parts = [int(p) for p in value.split(":")]
    if len(parts) == 1:
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
        if days:
            h, m, s = parts[0], parts[1], 0
    else:
        h, m, s = parts[-3], parts[-2], parts[-1]
    return float(days * 86400 + h * 3600 + m * 60 + s)


def sbatch_time_limit_seconds(sbatch_path: str | os.PathLike, default: str = "3:00:00") -> float:
    """Time limit declared in an sbatch file (falls back to *default*)."""
    try:
        match = _TIME_RE.search(Path(sbatch_path).read_text())
    except OSError:
        match = None
    return parse_slurm_time(match.group(1) if match else default)


def set_sbatch_time(sbatch_path: str | os.PathLike, new_time: str) -> None:
    """Rewrite the ``#SBATCH -t`` line of an sbatch file (used for retries)."""
    path = Path(sbatch_path)
    text = _TIME_RE.sub(f"#SBATCH -t {new_time}", path.read_text())
    path.write_text(text)


# ---------------------------------------------------------------------------
# Job markers
# ---------------------------------------------------------------------------


def queued_times(task: PolarityTask) -> list[float]:
    """Epoch times this polarity was queued by the new pipeline (empty if never)."""
    try:
        lines = task.path("mzmine_queued").read_text().split()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(float(line))
        except ValueError:
            continue
    return out


def _last_status_line(task: PolarityTask) -> list[str] | None:
    try:
        lines = [l.split() for l in task.path("mzmine_status").read_text().splitlines() if l.strip()]
    except OSError:
        return None
    return lines[-1] if lines else None


def mzmine_job_failure_reason(task: PolarityTask, mzmine_cfg: dict[str, Any], now: float) -> str | None:
    """Decide whether a ``04 running`` MZmine job (outputs missing) is dead.

    Returns:
        A human-readable reason if the job should be marked ``09 error``,
        otherwise ``None`` (still pending/running).
    """
    grace = float(mzmine_cfg.get("status_grace_hours", 1)) * 3600
    max_pending = float(mzmine_cfg.get("max_pending_hours", 48)) * 3600
    last = _last_status_line(task)
    if last:
        state = last[0]
        if state == "FAILED":
            rc = last[1] if len(last) > 1 else "?"
            return f"MZmine job exited with code {rc}"
        if state == "DONE":
            return "MZmine job finished but expected output files are missing"
        if state == "RUNNING":
            try:
                started = float(last[2])
            except (IndexError, ValueError):
                started = task.path("mzmine_status").stat().st_mtime
            limit = sbatch_time_limit_seconds(task.path("mzmine_sbatch"), mzmine_cfg["slurm"]["time"])
            if now - started > limit + grace:
                return "MZmine job has been RUNNING longer than its time limit (likely killed)"
            return None
    job_id_file = task.path("mzmine_job_id")
    queued = queued_times(task)
    if queued:
        since = now - max(queued)
        if not os.path.isfile(job_id_file):
            if since > grace:
                return "MZmine job was queued but never submitted to Slurm"
            return None
        if since > max_pending:
            return f"MZmine job has not started after {mzmine_cfg.get('max_pending_hours', 48)} hours"
        return None
    # Job submitted by the legacy pipeline (no markers): judge by job-id file age.
    if os.path.isfile(job_id_file):
        if now - job_id_file.stat().st_mtime > max_pending:
            return "legacy MZmine job (no status markers) is older than max_pending_hours"
        return None
    return "no MZmine job ID or status markers found"


# ---------------------------------------------------------------------------
# Step 2: MZmine status
# ---------------------------------------------------------------------------

MZMINE_STATUS_UPDATE_COLUMNS = [
    "mzmine_neg_status", "mzmine_pos_status",
    "num_pos_features", "num_neg_features",
    "num_neg_features_in_exctrl", "num_pos_features_in_exctrl",
    "num_neg_msms", "num_pos_msms",
]


def mzmine_outputs_complete(task: PolarityTask) -> bool:
    """True when the MGF, metadata, MZmine stdout, and peak-height table all exist."""
    return (
        task.exists("mgf")
        and task.exists("metadata")
        and task.exists("mzmine_out")
        and (task.exists("peak_height") or task.exists("old_peak_height"))
    )


def update_mzmine_status(ctx: RunContext) -> None:
    """Step 2: mark finished MZmine jobs complete (with counts + filtering) or errored."""
    logger.info("Step 2/7: Checking and updating status of MZmine jobs in LIMS...")
    df = ctx.get_tasks()
    if ctx.direct_input is None and not df.empty:
        df = subset_df_by_status(df, "mzmine", [TaskStatus.INITIATION, TaskStatus.RUNNING, TaskStatus.ERROR])
    if df.empty:
        logger.info(tab_print("No MZmine jobs to update!", 1))
        return
    filt = ctx.config["filtering"]
    mzmine_cfg = ctx.config["mzmine"]
    now = ctx.clock()
    index_list: list = []
    df = df.copy()
    for i, row in df.iterrows():
        project = row["parent_dir"]
        polarity_list = check_for_polarities(row["output_dir"], project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 2))
            continue
        for polarity in polarity_list:
            col = status_column("mzmine", polarity)
            if ctx.direct_input is None and row[col] in (TaskStatus.NOT_RELEVANT, TaskStatus.COMPLETE):
                continue
            task = PolarityTask(row["output_dir"], project, polarity)
            if mzmine_outputs_complete(task):
                logger.info(tab_print(f"Working on {project} in {polarity} mode", 2))
                logger.info(tab_print(f"All MZmine output files found in {polarity} directory, continuing...", 3))
                logger.info(tab_print("Calculating feature and background counts and updating LIMS table", 3))
                features, exctrl, msms = calculate_counts_for_lims_table(
                    task.path("peak_height"), task.path("mgf"), ctx.background_designator)
                df[col] = df[col].astype(object)
                df.loc[i, col] = TaskStatus.COMPLETE.value
                df.loc[i, f"num_{task.short}_features"] = int(features)
                df.loc[i, f"num_{task.short}_features_in_exctrl"] = int(exctrl)
                df.loc[i, f"num_{task.short}_msms"] = int(msms)
                index_list.append(i)
                if ctx.dry_run:
                    logger.info(tab_print("[dry-run] Not writing filtered peak height files.", 4))
                elif features > 0:
                    logger.info(tab_print("Filtering peak height file to remove features below background...", 4))
                    create_filtered_peakheight_file(
                        peakheight_filename=task.path("peak_height"),
                        background_designator=ctx.background_designator,
                        output_dir=task.directory,
                        project_name=project,
                        polarity=polarity,
                        background_ratio=filt["background_ratio"],
                        zero_value=filt["zero_value"],
                        nonpolar_solvent_front=filt["nonpolar_solvent_front"],
                        polar_solvent_front=filt["polar_solvent_front"],
                        nonpolar_solvent_end=filt["nonpolar_solvent_end"],
                        polar_solvent_end=filt["polar_solvent_end"],
                    )
                else:
                    logger.warning(tab_print("Warning! No features were found, not writing a filtered peak height file.", 4))
                if not ctx.dry_run:
                    set_group(task.directory, ctx.group)
            elif row[col] == TaskStatus.RUNNING:
                reason = mzmine_job_failure_reason(task, mzmine_cfg, now)
                if reason:
                    logger.critical(tab_print(
                        f"Warning! MZmine status for {polarity} mode of {project} is ['04 running'] but "
                        f"{reason}. Changing to ['09 error'].", 1))
                    df[col] = df[col].astype(object)
                    df.loc[i, col] = TaskStatus.ERROR.value
                    index_list.append(i)
    if index_list:
        logger.info(tab_print(f"Updating statuses in LIMS table for {len(set(index_list))} projects...", 1))
        ctx.write_task_updates(df, index_list, MZMINE_STATUS_UPDATE_COLUMNS)
        logger.info(tab_print("MZmine status update complete.", 1))
    else:
        logger.info(tab_print("No MZmine jobs to update!", 1))


# ---------------------------------------------------------------------------
# Step 3: queue / submit MZmine jobs
# ---------------------------------------------------------------------------


def _record_queued(task: PolarityTask, now: float) -> None:
    with open(task.path("mzmine_queued"), "a") as fh:
        fh.write(f"{now:.0f}\n")


def _clear_job_markers(task: PolarityTask) -> None:
    for kind in ("mzmine_status", "mzmine_job_id"):
        try:
            task.path(kind).unlink()
        except FileNotFoundError:
            pass


def _submit_now(task: PolarityTask) -> str | None:
    """Submit an sbatch file directly (host mode) and write the job-id file."""
    result = subprocess.run(
        ["sbatch", str(task.path("mzmine_sbatch"))], capture_output=True, text=True, check=False)
    out = result.stdout.strip()
    logger.info(tab_print(out or result.stderr.strip(), 3))
    if result.returncode != 0 or not out:
        return None
    job_id = out.split()[-1]
    task.path("mzmine_job_id").write_text(f"{task.prefix}={job_id}\n")
    return job_id


def queue_mzmine_job(ctx: RunContext, task: PolarityTask) -> bool:
    """Hand one MZmine job to Slurm (queue file or direct ``sbatch``).

    Returns:
        True if the job was queued/submitted (so status should become ``04 running``).
    """
    if ctx.dry_run:
        logger.info(tab_print(f"[dry-run] Would submit {task.path('mzmine_sbatch')}", 3))
        return False
    now = ctx.clock()
    if ctx.sbatch_queue_file:
        _clear_job_markers(task)
        queue = Path(ctx.sbatch_queue_file)
        queue.parent.mkdir(parents=True, exist_ok=True)
        with open(queue, "a") as fh:
            fh.write(f"{task.path('mzmine_sbatch')}\t{task.path('mzmine_job_id')}\t{task.prefix}\n")
        _record_queued(task, now)
        logger.info(tab_print(f"Queued {task.prefix} MZmine job for host submission.", 3))
        return True
    if shutil.which("sbatch"):
        _clear_job_markers(task)
        job_id = _submit_now(task)
        if job_id is None:
            logger.error(tab_print(f"sbatch submission failed for {task.prefix}", 3))
            return False
        _record_queued(task, now)
        logger.info(tab_print("Wrote job ID to file and setting status to ['04 running].", 3))
        return True
    logger.error(tab_print(
        "Cannot submit MZmine job: no --sbatch-queue-file given and 'sbatch' is not on PATH.", 3))
    return False


def submit_mzmine_jobs(ctx: RunContext, new_projects: pd.DataFrame | None = None) -> None:
    """Step 3: submit MZmine jobs for initiated (and retryable errored) polarities.

    Automatic mode considers projects created by Step 1 in this run (``01
    initiation``) and (bug fix vs. legacy) projects in ``09 error`` that this
    pipeline queued before and that have retries left. Anything else needs
    ``--direct_input``.
    """
    logger.info("Step 3/7: Submitting new MZmine jobs...")
    mzmine_cfg = ctx.config["mzmine"]
    df = ctx.get_tasks()
    if df.empty:
        logger.info(tab_print("No new MZmine jobs to submit!", 1))
        return
    if ctx.direct_input is None:
        df = subset_df_by_status(df, "mzmine", [TaskStatus.INITIATION, TaskStatus.ERROR])
    if df.empty:
        logger.info(tab_print("No new MZmine jobs to submit!", 1))
        return
    new_names = set(new_projects["parent_dir"]) if new_projects is not None and not new_projects.empty else set()
    max_attempts = 1 + int(mzmine_cfg.get("max_retries", 1))
    index_list: list = []
    df = df.copy()
    for col in ("mzmine_pos_status", "mzmine_neg_status", "fbmn_pos_status", "fbmn_neg_status"):
        df[col] = df[col].astype(object)
    logger.info(tab_print(f"Total of {df.shape[0]} candidate project(s) for MZmine submission", 1))
    for i, row in df.iterrows():
        project = row["parent_dir"]
        polarity_list = check_for_polarities(row["output_dir"], project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 2))
            continue
        for polarity in polarity_list:
            task = PolarityTask(row["output_dir"], project, polarity)
            col = status_column("mzmine", polarity)
            status = row[col]
            if status == TaskStatus.COMPLETE and not ctx.overwrite_mzmine:
                continue
            if task.exists("peak_height") and task.exists("mgf") and not ctx.overwrite_mzmine:
                logger.info(tab_print(
                    f"Note: Overwrite is False and MZmine files for {project} in {polarity} mode already exist. "
                    "Skipping submission...", 2))
                continue
            if status == TaskStatus.NOT_RELEVANT:
                continue
            if status == TaskStatus.RUNNING and not (ctx.direct_input and ctx.overwrite_mzmine):
                # Never resubmit a job that may still be running unless explicitly forced.
                continue
            if status == TaskStatus.ERROR:
                if row.get(f"num_{task.short}_files", 0) == 0:
                    logger.warning(tab_print(
                        f"Warning! No raw data files for {polarity} mode. Setting status to '12 not relevant'", 2))
                    df.loc[i, col] = TaskStatus.NOT_RELEVANT.value
                    df.loc[i, status_column("fbmn", polarity)] = TaskStatus.NOT_RELEVANT.value
                    index_list.append(i)
                    continue
                if ctx.direct_input is None:
                    attempts = len(queued_times(task))
                    if not mzmine_cfg.get("auto_resubmit_errors", True):
                        continue
                    if attempts == 0:
                        # Errored before this pipeline managed it (legacy); needs --direct_input.
                        continue
                    if attempts >= max_attempts:
                        logger.debug(tab_print(f"{task.prefix}: retry limit reached ({attempts}).", 2))
                        continue
                logger.info(tab_print(
                    f"Note: MZmine task for {project} in {polarity} mode has error status but has raw files. "
                    "Attempting to resubmit with longer runtime.", 2))
                if task.exists("mzmine_sbatch") and not ctx.dry_run:
                    set_sbatch_time(task.path("mzmine_sbatch"), mzmine_cfg["slurm"]["retry_time"])
            elif status == TaskStatus.INITIATION and ctx.direct_input is None and project not in new_names:
                # Legacy behaviour: only projects created by Step 1 in this run are submitted
                # automatically. Stale '01 initiation' rows (LIMS has some from years ago) need
                # --direct_input.
                logger.debug(tab_print(f"Project {project} is '01 initiation' from an earlier run; skipping.", 2))
                continue
            if not task.exists("mzmine_sbatch"):
                logger.warning(tab_print(f"Warning! No sbatch file for {task.prefix}. Skipping.", 2))
                continue
            logger.info(tab_print(f"Submitting {polarity} mode mzmine job for project {project}", 2))
            if queue_mzmine_job(ctx, task):
                df.loc[i, col] = TaskStatus.RUNNING.value
                index_list.append(i)
    if index_list:
        ctx.write_task_updates(
            df, index_list, ["mzmine_neg_status", "mzmine_pos_status", "fbmn_neg_status", "fbmn_pos_status"])
        logger.info(tab_print("MZmine submission(s) complete.", 1))


__all__ = [
    "build_sbatch_script",
    "mzmine_job_failure_reason",
    "mzmine_outputs_complete",
    "parse_slurm_time",
    "queue_mzmine_job",
    "render_batch_params",
    "select_batch_template",
    "submit_mzmine_jobs",
    "update_mzmine_status",
    "write_mzmine_sbatch_and_runner",
]
