"""Steps 4-6: GNPS2 FBMN status, submission (plus Everything Bagel), and downloads."""

from __future__ import annotations

import os

import metatlas2.logging_config as lcf
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.gnps2 import (
    build_eb_file_list,
    count_mgf_lines,
    everything_bagel_parameters,
    fbmn_parameters,
    remove_contaminant_from_mgf,
)
from metatlas2.untargeted.project_setup import find_raw_data_directory, raw_data_subdir_for_project
from metatlas2.untargeted.status import (
    PolarityTask,
    TaskStatus,
    check_for_polarities,
    read_task_id_file,
    status_column,
    subset_df_by_status,
    tab_print,
)
from metatlas2.untargeted.utils import set_group

logger = lcf.get_logger("untargeted.fbmn")

FBMN_STATUS_COLUMNS = ["fbmn_neg_status", "fbmn_pos_status"]

# GNPS2 status -> LIMS status
_GNPS2_STATUS_MAP = {
    "DONE": TaskStatus.COMPLETE,
    "FAILED": TaskStatus.ERROR,
    "RUNNING": TaskStatus.RUNNING,
    "QUEUED": TaskStatus.RUNNING,
}


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------


def update_fbmn_status(ctx: RunContext) -> None:
    """Step 4: poll GNPS2 for every running/waiting/errored FBMN task and update LIMS."""
    logger.info("Step 4/7: Checking and updating status of FBMN jobs in LIMS...")
    df = ctx.get_tasks()
    if ctx.direct_input is None and not df.empty:
        df = subset_df_by_status(df, "fbmn", [TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.ERROR])
    if df.empty:
        logger.info(tab_print("No FBMN jobs to update!", 1))
        return
    gnps2 = ctx.get_gnps2()
    df = df.copy()
    for col in FBMN_STATUS_COLUMNS:
        df[col] = df[col].astype(object)
    index_list: list = []
    for i, row in df.iterrows():
        project = row["parent_dir"]
        polarity_list = check_for_polarities(row["output_dir"], project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 1))
            continue
        for polarity in polarity_list:
            col = status_column("fbmn", polarity)
            if row[col] in (TaskStatus.NOT_RELEVANT, TaskStatus.COMPLETE):
                continue
            task = PolarityTask(row["output_dir"], project, polarity)
            task_id = read_task_id_file(task.path("fbmn_task"))
            if not task_id:
                continue
            status, _version = gnps2.check_status(task_id)
            index_list.append(i)
            if status == "N/A":
                logger.critical(tab_print(f"Warning! Could not get status from GNPS2 for {project} in {polarity} mode", 1))
                continue
            new_status = _GNPS2_STATUS_MAP.get(status, TaskStatus.WAITING)
            if df.loc[i, col] != new_status.value:
                if new_status in (TaskStatus.COMPLETE, TaskStatus.ERROR, TaskStatus.RUNNING):
                    word = {TaskStatus.COMPLETE: "complete", TaskStatus.ERROR: "error", TaskStatus.RUNNING: "running"}
                    logger.info(tab_print(
                        f"Updating FBMN job status for {project} in {polarity} mode to {word[new_status]}", 1))
                df.loc[i, col] = new_status.value
    if index_list:
        ctx.write_task_updates(df, index_list, FBMN_STATUS_COLUMNS)
        logger.info(tab_print("FBMN status update complete.", 1))
    else:
        logger.info(tab_print("No FBMN jobs to update!", 1))


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------


def _write_task_file(path: os.PathLike, prefix: str, task_id: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"{prefix}={task_id}\n")


def submit_fbmn_jobs(ctx: RunContext) -> None:
    """Step 5: mirror inputs to GNPS2 and submit FBMN (+ Everything Bagel) jobs."""
    logger.info("Step 5/7: Submitting new FBMN jobs to GNPS2...")
    gcfg = ctx.config["gnps2"]
    paths = ctx.paths
    df = ctx.get_tasks()
    if ctx.direct_input is None and not df.empty:
        df = subset_df_by_status(df, "fbmn", [TaskStatus.WAITING, TaskStatus.ERROR])
        if not df.empty:
            df = subset_df_by_status(df, "mzmine", [TaskStatus.COMPLETE])
        if not df.empty:
            df = subset_df_by_status(df, "mzmine", [TaskStatus.ERROR], inverse=True)
    if df.empty:
        logger.info(tab_print("No new FBMN jobs to submit!", 1))
        return
    gnps2 = ctx.get_gnps2()
    df = df.copy()
    for col in FBMN_STATUS_COLUMNS:
        df[col] = df[col].astype(object)
    index_list: list = []
    for i, row in df.iterrows():
        project = row["parent_dir"]
        polarity_list = check_for_polarities(row["output_dir"], project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 2))
            continue
        gnps2_subdir = paths.get("raw_data_subdir") or raw_data_subdir_for_project(project)
        raw_data = f"{gnps2.user_upload_prefix}/raw_data/{gnps2_subdir}/{project}"
        for polarity in polarity_list:
            task = PolarityTask(row["output_dir"], project, polarity)
            col = status_column("fbmn", polarity)
            mz_col = status_column("mzmine", polarity)
            if row[col] == TaskStatus.COMPLETE and row[mz_col] == TaskStatus.COMPLETE and not ctx.overwrite_fbmn:
                continue
            logger.info(tab_print(f"Working on {polarity} mode for project {project}:", 1))
            if row[col] == TaskStatus.ERROR:
                if row.get(f"num_{task.short}_files", 0) == 0:
                    logger.info(tab_print(
                        f"Notice! No raw data files for {polarity} mode. Setting status to '12 not relevant'", 2))
                    df.loc[i, col] = TaskStatus.NOT_RELEVANT.value
                    index_list.append(i)  # bug fix: legacy never wrote this change to LIMS
                    continue
                if (row.get(f"num_{task.short}_msms", 0) or 0) < 2:
                    logger.info(tab_print(
                        f"Notice! Insufficient MSMS detected for {polarity} mode. Setting status to '12 not relevant'", 2))
                    df.loc[i, col] = TaskStatus.NOT_RELEVANT.value
                    index_list.append(i)  # bug fix: legacy never wrote this change to LIMS
                    continue
                logger.warning(tab_print(
                    f"Warning! Project {polarity} mode has an error status but appears to have raw data and Mzmine "
                    "results. Attempting to resubmit...", 2))
                if task.exists("fbmn_task") and not ctx.dry_run:
                    task.path("fbmn_task").unlink()
            if row[col] == TaskStatus.NOT_RELEVANT:
                logger.info(tab_print(f"Bailed out because FBMN status is '12 not relevant' for {polarity} mode", 2))
                continue
            if row[mz_col] != TaskStatus.COMPLETE:
                logger.info(tab_print(f"Bailed out because MZmine status is not '07 complete' for {polarity} mode", 2))
                continue
            if task.exists("fbmn_task") and not ctx.overwrite_fbmn and row[col] != TaskStatus.ERROR:
                logger.info(tab_print(
                    f"Bailed out because FBMN task file already exists for {polarity} mode and overwrite is False", 2))
                continue

            if gcfg.get("mirror_mzmine_results", True):
                logger.info(tab_print("Ensuring MZmine results are at GNPS2 before submitting FBMN job...", 2))
                if not gnps2.mirror_mzmine_results(project, polarity, row["output_dir"]):
                    logger.warning(tab_print(
                        f"Skipping FBMN submission for {polarity} mode because of MZmine results upload failure", 3))
                    continue
            else:
                logger.info(tab_print("Skipping MZmine results mirroring to GNPS2...", 2))

            local_raw = find_raw_data_directory(
                project, paths["raw_data_dir"], paths.get("raw_data_subdir"), paths.get("fallback_raw_data_subdirs"))
            if gcfg.get("mirror_raw_data", True):
                logger.info(tab_print("Ensuring raw mzML data are at GNPS2 before submitting FBMN job...", 2))
                if local_raw is None or not gnps2.mirror_raw_data(project, local_raw, mode="pipeline"):
                    logger.info(tab_print("Notice! Proceeding with FBMN submission even though raw data mirror failed", 3))
            else:
                logger.info(tab_print(f"Skipping raw data mirroring to GNPS2 for {polarity} mode...", 2))

            remote_task_dir = f"{gnps2.user_upload_prefix}/untargeted_tasks/{task.prefix}"
            spectra_file = f"{remote_task_dir}/{task.prefix}.mgf"
            quant_file = f"{remote_task_dir}/{task.prefix}_quant.csv"
            metadata_file = f"{remote_task_dir}/{task.prefix}_metadata.tab"

            if count_mgf_lines(task.path("mgf")) == 0:
                logger.info(tab_print(
                    f"Note! MGF file in {polarity} mode has no MSMS data. Updating FBMN status to '12 not relevant'", 2))
                df.loc[i, col] = TaskStatus.NOT_RELEVANT.value
                index_list.append(i)
                continue

            if ctx.dry_run:
                logger.info(tab_print("[dry-run] Would remove internal calibrant from MGF and submit FBMN.", 2))
                continue
            logger.info(tab_print("Removing internal calibrant from MGF file before FBMN submission...", 2))
            remove_contaminant_from_mgf(task.path("mgf"), gcfg["calibrant_mz"], gcfg.get("calibrant_ppm", 5))

            params = fbmn_parameters(gcfg["fbmn_params"], task.prefix, quant_file, spectra_file, metadata_file, raw_data)
            response = gnps2.submit(params) or {}
            logger.info(tab_print(f"Submitted FBMN job for {polarity} mode and set LIMS status to ['04 running'].", 2))
            df.loc[i, col] = TaskStatus.RUNNING.value
            index_list.append(i)
            task_id = response.get("task")
            if task_id:
                _write_task_file(task.path("fbmn_task"), task.prefix, task_id)
                logger.info(tab_print(
                    f"GNPS2 task file for {polarity} mode written to {task.path('fbmn_task').name}", 3))
            else:
                logger.warning(tab_print(
                    "Warning! GNPS2 FBMN task ID not found. File gnps2-fbmn-task.txt not written.", 3))

            if gcfg.get("submit_everything_bagel", True):
                _submit_everything_bagel(ctx, task, local_raw, raw_data, metadata_file)

    if index_list:
        ctx.write_task_updates(df, index_list, FBMN_STATUS_COLUMNS)
        logger.info(tab_print("FBMN submission(s) complete. Use GNPS2 *gnps2-page-link.txt to monitor progress.", 1))


def _submit_everything_bagel(
    ctx: RunContext, task: PolarityTask, local_raw: str | None, raw_data: str, metadata_file: str
) -> None:
    """Fire-and-forget Everything Bagel submission for one polarity."""
    gnps2 = ctx.get_gnps2()
    try:
        if local_raw is None:
            raise ValueError(f"Raw data directory not found for {task.project}")
        eb_files = build_eb_file_list(local_raw, raw_data, task.polarity)
    except (OSError, ValueError) as err:
        logger.warning(tab_print(f"Skipping Everything Bagel for {task.polarity}: {err}", 2))
        return
    params = everything_bagel_parameters(
        ctx.config["gnps2"]["everything_bagel_params"], f"{task.prefix}_everything_bagel", eb_files, metadata_file)
    response = gnps2.submit(params) or {}
    task_id = response.get("task")
    if task_id:
        logger.info(tab_print(
            f"Submitted Everything Bagel job for {task.polarity} mode to GNPS2 with job ID: {task_id}", 2))
        _write_task_file(task.path("eb_task"), task.prefix, task_id)
    else:
        logger.warning(tab_print(
            f"Warning! Everything Bagel job for {task.polarity} mode submitted but no task ID was returned. "
            f"Full response: {response}", 2))


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------


def download_fbmn_results(ctx: RunContext) -> None:
    """Step 6: download the network graphml and library hits for completed FBMN tasks."""
    logger.info("Step 6/7: Checking for completed FBMN jobs and downloading results...")
    df = ctx.get_tasks()
    if not df.empty:
        if ctx.direct_input is None:
            df = subset_df_by_status(df, "fbmn", [TaskStatus.COMPLETE, TaskStatus.ERROR])
        if not df.empty:
            df = subset_df_by_status(df, "mzmine", [TaskStatus.COMPLETE])
    if df.empty:
        logger.info(tab_print("No completed FBMN data to download!", 1))
        return
    gnps2 = ctx.get_gnps2()
    count = 0
    for _, row in df.iterrows():
        project = row["parent_dir"]
        polarity_list = check_for_polarities(ctx.output_dir, project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 1))
            continue
        for polarity in polarity_list:
            task = PolarityTask(ctx.output_dir, project, polarity)
            task_id = read_task_id_file(task.path("fbmn_task"))
            if not task_id:
                continue
            outputs = [task.path("fbmn_graphml"), task.path("fbmn_library_results"), task.path("gnps2_link")]
            if not ctx.overwrite_fbmn and all(os.path.exists(p) for p in outputs):
                continue
            logger.info(tab_print(f"Working on {polarity} mode for project {project}:", 1))
            fbmn_status = row[status_column("fbmn", polarity)]
            if fbmn_status == TaskStatus.NOT_RELEVANT:
                logger.info(tab_print(
                    f"Bailed out because FBMN status for {polarity} mode is '12 not relevant'. Skipping download.", 2))
                continue
            if fbmn_status != TaskStatus.COMPLETE:
                logger.info(tab_print(
                    f"Bailed out because FBMN status for {polarity} mode is not '07 complete'. Skipping download.", 2))
                continue
            if row[status_column("mzmine", polarity)] != TaskStatus.COMPLETE:
                logger.info(tab_print(
                    f"Bailed out because MZmine status for {polarity} mode is not '07 complete'. Skipping download.", 2))
                continue
            logger.info(tab_print(f"Downloading {polarity} mode FBMN results with task ID {task_id}", 2))
            if ctx.dry_run:
                logger.info(tab_print("[dry-run] Not downloading.", 3))
                continue
            if ctx.overwrite_fbmn or not os.path.exists(task.path("gnps2_link")):
                task.path("gnps2_link").write_text(f"{gnps2.status_url(task_id)}\n")
                logger.info(tab_print("Wrote GNPS2 link", 3))
            if ctx.overwrite_fbmn or not os.path.exists(task.path("fbmn_graphml")):
                if gnps2.download(gnps2.graphml_url(task_id), task.path("fbmn_graphml")):
                    logger.info(tab_print("Downloaded graphml file", 3))
                    count += 1
                else:
                    logger.critical(tab_print("Error: Failed to download graphml file", 3))
            if ctx.overwrite_fbmn or not os.path.exists(task.path("fbmn_library_results")):
                if gnps2.download(gnps2.library_results_url(task_id), task.path("fbmn_library_results")):
                    logger.info(tab_print("Downloaded result table file", 3))
                    count += 1
                else:
                    logger.critical(tab_print("Error: Failed to download results table", 3))
            set_group(task.directory, ctx.group)
    if count > 0:
        logger.info(tab_print("All new FBMN results downloaded.", 1))
    else:
        logger.info(tab_print("No new FBMN results to download!", 1))
