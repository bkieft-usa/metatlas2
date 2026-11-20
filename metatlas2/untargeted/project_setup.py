"""Step 1: find new projects in LIMS and write their MZmine inputs.

A project (raw-data directory, LIMS ``parent_dir``) becomes a new untargeted
task when all of these hold:

* it has POS or NEG mzML files in LIMS ``lcmsrun_plus``;
* it is not already in ``untargeted_tasks`` (unless ``--force_reprocess`` with
  ``--direct_input``);
* its newest file is older than ``min_file_age_hours`` (3 h);
* at least one file name contains ``MS2``/``MSMS`` and not ``_MS1_``;
* its raw-data directory has no ``nountargeted.txt`` opt-out file.

For each polarity with files, ``{output_dir}/{project}_{polarity}/`` gets the
metadata table, MZmine batch XML, mzML file list, and sbatch/runner scripts,
and one row per project is inserted into ``untargeted_tasks``.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

import metatlas2.logging_config as lcf
from metatlas2.file_and_project_format import (
    get_name_field,
    parent_dir_num_fields,
    validate_file_name_minimal,
)
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.mzmine import (
    render_batch_params,
    select_batch_template,
    write_mzmine_sbatch_and_runner,
)
from metatlas2.untargeted.status import POLARITIES, PolarityTask, TaskStatus, short_polarity, tab_print
from metatlas2.untargeted.utils import set_group

logger = lcf.get_logger("untargeted.project_setup")

EXCLUDED_SAMPLE_GROUPS = ("QC", "InjBl", "ISTD")


def raw_data_subdir_for_project(project: str) -> str:
    """Raw-data subdirectory inferred from project field 1 (``EB`` -> ``egsb``, default ``jgi``)."""
    department = get_name_field(project, 1)
    if department is None:
        return "jgi"
    subdir = department.lower()
    return "egsb" if subdir == "eb" else subdir


def find_raw_data_directory(
    project: str,
    raw_data_dir: str,
    raw_data_subdir: str | None = None,
    fallback_subdirs: list[str] | None = None,
) -> str | None:
    """Locate ``{raw_data_dir}/{subdir}/{project}``.

    With *raw_data_subdir* only that location is checked; otherwise the
    subdir inferred from the project name is tried first, then *fallback_subdirs*.
    """
    if raw_data_subdir is not None:
        path = os.path.join(raw_data_dir, raw_data_subdir, project)
        return path if os.path.exists(path) else None
    candidates = [raw_data_subdir_for_project(project), *(fallback_subdirs or [])]
    for subdir in candidates:
        path = os.path.join(raw_data_dir, subdir, project)
        if os.path.exists(path):
            return path
    return None


def metadata_file_filter(
    data: pd.DataFrame, polarity: str, skip_blank_filter: bool = False, nonstandard_filename: bool = False
) -> list[str]:
    """Basenames of mzML files to feed MZmine for one polarity (``"POS"``/``"NEG"``).

    Standard names: polarity must appear in field 9 and QC/InjBl/ISTD must not
    appear in field 12. Non-standard names fall back to substring matching.
    Names containing ``Blank`` are dropped unless *skip_blank_filter*.
    """

    def keep(name: str) -> bool:
        if not skip_blank_filter and "Blank" in name:
            return False
        if nonstandard_filename:
            return f"_{polarity}_" in name and not any(g in name for g in EXCLUDED_SAMPLE_GROUPS)
        fields = name.split("_")
        if len(fields) <= 12:
            return False
        return polarity in fields[9] and not any(g in fields[12] for g in EXCLUDED_SAMPLE_GROUPS)

    return [name for name in data["basename"] if keep(name)]


def build_metadata_df(df_project: pd.DataFrame, polarity: str, background_designator: list[str]) -> pd.DataFrame:
    """GNPS-style metadata table for one polarity.

    Columns: ``filename``, ``ATTRIBUTE_sampletype`` (lower-cased sample group),
    plus up to three constant columns holding the sample groups that contain a
    background designator.

    LEGACY QUIRK: those constant columns are named ``CONTROL``, ``CASE`` and
    ``ATTRIBUTE_media`` regardless of content, and the table includes every
    file of the polarity (QC/InjBl/ISTD too), unlike the MZmine file list.
    """
    pol_short = polarity[:3].upper()
    meta = df_project[["basename", "sample_group"]].copy()
    meta["basename"] = meta["basename"].fillna("")
    meta["sample_group"] = meta["sample_group"].fillna("").astype(str).str.lower()
    groups = [g for g in meta["sample_group"].unique() if any(c.lower() in g for c in background_designator)]
    meta = meta.rename(columns={"basename": "filename", "sample_group": "ATTRIBUTE_sampletype"})
    meta["filename"] = meta["filename"].apply(os.path.basename)
    for col, group in zip(["CONTROL", "CASE", "ATTRIBUTE_media"], groups, strict=False):
        meta.loc[:, col] = group
    return meta[meta["filename"].str.split("_").str[9].str.contains(pol_short, na=False)]


def _validate_names(project: str, file_list: list[str]) -> bool:
    """Minimal filename + parent-directory validation (``--validate_names``)."""
    if any(validate_file_name_minimal(f) for f in file_list):
        logger.warning(tab_print(
            f"Warning! Project {project} has one or more mzML files that do not have valid format. "
            "Not adding project to untargeted tasks...", 2))
        return False
    if any(parent_dir_num_fields(f) for f in file_list):
        logger.warning(tab_print(
            f"Warning! Parent dir {project} does not have valid format. Not adding project to untargeted tasks...", 2))
        return False
    return True


def write_metadata_per_new_project(
    df: pd.DataFrame,
    background_designator: list[str],
    validate_names: bool,
    raw_data_dir: str,
    raw_data_subdir: str | None = None,
    skip_blank_filter: bool = False,
    fallback_subdirs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect per-polarity file lists and metadata for each new project.

    Returns:
        A list of dicts: ``{"parent_dir", "polarities", "positive": {...}, "negative": {...}}``
        where each polarity dict has ``file_count``, ``file_list`` (full mzML paths)
        and ``metadata_df``. Projects without files or raw-data directory are omitted.
    """
    projects = []
    for project in df["parent_dir"].unique():
        df_project = df[df["parent_dir"] == project]
        mzml_dir = find_raw_data_directory(project, raw_data_dir, raw_data_subdir, fallback_subdirs)
        if mzml_dir is None:
            where = f"{raw_data_dir}/{raw_data_subdir}" if raw_data_subdir else "several possible alternatives"
            logger.error(tab_print(
                f"Raw data directory for {project} could not be found after trying {where}. Not creating metadata.", 2))
            continue
        info: dict[str, Any] = {"parent_dir": project, "positive": {}, "negative": {}}
        counts: dict[str, int] = {}
        skip_project = False
        for polarity in POLARITIES:
            pol_short = polarity[:3].upper()
            try:
                subset = metadata_file_filter(df_project, pol_short, skip_blank_filter, nonstandard_filename=False)
            except IndexError:  # pragma: no cover - kept for parity with legacy fallback
                subset = metadata_file_filter(df_project, pol_short, skip_blank_filter, nonstandard_filename=True)
            file_list = [os.path.join(mzml_dir, f) for f in subset]
            counts[polarity] = len(file_list)
            if file_list:
                if validate_names and not _validate_names(project, file_list):
                    skip_project = True
                    break
                info[polarity]["file_count"] = len(file_list)
                info[polarity]["file_list"] = file_list
        if skip_project:
            continue
        polarities = [p for p in POLARITIES if counts[p] > 0]
        if not polarities:
            continue
        if len(polarities) == 2:
            logger.info(tab_print(
                f"Project {project} has both polarities with {counts['positive']} positive and "
                f"{counts['negative']} negative files", 2))
        else:
            logger.info(tab_print(
                f"Project {project} has {polarities[0]} mode only with {counts[polarities[0]]} files", 2))
        info["polarities"] = polarities
        for polarity in polarities:
            info[polarity]["metadata_df"] = build_metadata_df(df_project, polarity, background_designator)
        projects.append(info)
    return projects


def _prepare_lcmsruns(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy cleanup of the ``lcmsrun_plus`` table."""
    df = df[pd.notna(df["mzml_file"])].copy()
    df["basename"] = df["mzml_file"].apply(os.path.basename)
    df = df.sort_values("timeepoch", ascending=False)
    df = df.drop_duplicates(["basename", "parent_dir"])
    df = df.drop(columns=["mzml_file_container"], errors="ignore")
    return df.replace("", np.nan)


def find_new_projects(ctx: RunContext, lcmsruns: pd.DataFrame, tasks: pd.DataFrame) -> list[str]:
    """Names of projects eligible to become new untargeted tasks (see module docstring)."""
    sel = ctx.config["project_selection"]
    cutoff = ctx.clock() - float(sel["min_file_age_hours"]) * 3600
    newest = lcmsruns.groupby("parent_dir")["timeepoch"].max()
    old_enough = set(newest[newest < cutoff].index)
    too_new = sorted(newest[newest >= cutoff].index)
    has_ms2 = lcmsruns.groupby("parent_dir")["basename"].apply(
        lambda names: any(("MS2" in n or "MSMS" in n) and "_MS1_" not in n for n in names))
    ms2_dirs = set(has_ms2[has_ms2].index)
    if too_new:
        logger.info(tab_print(
            f"Note: There are data directories with files less than {sel['min_file_age_hours']} hours old! "
            "Skipping these for now:", 2))
        for folder in too_new:
            logger.info(tab_print(folder, 3))
    all_folders = [a.strip() for a in lcmsruns.loc[lcmsruns["polarity"].isin(["POS", "NEG"]), "parent_dir"].unique()]
    in_tasks = [a.strip() for a in tasks["parent_dir"]] if not tasks.empty else []
    logger.info(tab_print("Finding new projects to initate...", 1))
    if ctx.force_reprocess and ctx.direct_input is not None:
        candidates = list(all_folders)
        logger.info(tab_print(f"Force reprocessing {len(candidates)} projects regardless of LIMS status", 2))
    else:
        candidates = list(np.setdiff1d(all_folders, in_tasks))
    return sorted(set(candidates) & old_enough & ms2_dirs)


def _write_polarity_inputs(
    ctx: RunContext, project: str, polarity: str, info: dict[str, Any], template: str
) -> str | None:
    """Create the polarity directory and MZmine inputs. Returns the metadata path or None."""
    task = PolarityTask(ctx.output_dir, project, polarity)
    basepath = task.directory
    if ctx.dry_run:
        logger.info(tab_print(
            f"[dry-run] Would write metadata, batch params, file list ({info[polarity]['file_count']} files) "
            f"and sbatch files to {basepath} using {os.path.basename(template)}", 2))
        return str(task.path("metadata"))
    logger.info(tab_print("Writing MZmine submission input files...", 2))
    if os.path.exists(basepath):
        if ctx.force_reprocess:
            logger.info(tab_print(f"Force reprocess enabled: Removing existing directory {basepath}...", 3))
            shutil.rmtree(basepath)
        else:
            logger.warning(tab_print(
                f"Warning! Directory {basepath} already exists. Not writing new metadata or MZmine submission files...", 3))
            return None
    basepath.mkdir(parents=False)
    set_group(basepath, ctx.group)

    logger.info(tab_print(f"{polarity} metadata file (*_metadata.tab)", 3))
    info[polarity]["metadata_df"].to_csv(task.path("metadata"), sep="\t", index=False)

    logger.info(tab_print(f"{polarity} MZmine parameter file (*_batch-params.xml)", 3))
    mzmine_cfg = ctx.config["mzmine"]
    task.path("batch_params").write_text(
        render_batch_params(template, str(basepath / task.prefix), mzmine_cfg["output_placeholder"]))

    logger.info(tab_print(f"{polarity} mzML path list file (*_filelist.txt)", 3))
    with open(task.path("filelist"), "w") as fh:
        for mzml in info[polarity]["file_list"]:
            fh.write(f"{mzml}\n")

    logger.info(tab_print(f"{polarity} sbatch submission script and runner files (*.sh, *.sbatch)", 3))
    write_mzmine_sbatch_and_runner(task, mzmine_cfg)
    return str(task.path("metadata"))


def update_new_untargeted_tasks(ctx: RunContext) -> pd.DataFrame | None:
    """Step 1: create task directories/inputs for new projects and insert them in LIMS.

    Returns:
        DataFrame of the rows inserted into ``untargeted_tasks`` (or None).
    """
    logger.info("Step 1/7: Syncing LIMS and NERSC to identify new projects with raw data that are not yet "
                "in the untargeted task list...")
    lims_cfg = ctx.config["lims"]
    sel = ctx.config["project_selection"]
    paths = ctx.paths

    df = ctx.lims.get_table(lims_cfg["files_table"])
    if ctx.direct_input is not None:
        df = df[df["parent_dir"].isin(ctx.direct_input)]
        if df.empty:
            logger.info(tab_print(f"Projects in direct input ({ctx.direct_input}) are not in LIMS lcmsruns table", 1))
            return None
    if df.empty:
        logger.info(tab_print("No new projects to add to untargeted tasks!", 2))
        return None
    df = _prepare_lcmsruns(df)
    tasks = ctx.lims.get_table(ctx.tasks_table)

    logger.info(tab_print("Comparing LIMS untargeted tasks with raw files on disk...", 1))
    new_folders = find_new_projects(ctx, df, tasks)
    if not new_folders:
        logger.info(tab_print("No new projects to add to untargeted tasks!", 2))
        return None

    logger.info(tab_print("Checking for polarities in new projects and validating mzml file names...", 1))
    new_projects = write_metadata_per_new_project(
        df=df[df["parent_dir"].isin(new_folders)],
        background_designator=ctx.background_designator,
        validate_names=bool(sel["validate_names"]),
        raw_data_dir=paths["raw_data_dir"],
        raw_data_subdir=paths.get("raw_data_subdir"),
        skip_blank_filter=bool(sel["skip_blank_filter"]),
        fallback_subdirs=paths.get("fallback_raw_data_subdirs"),
    )
    new_projects = [
        d for d in new_projects
        if not glob.glob(os.path.join(paths["raw_data_dir"], "*", d["parent_dir"], "nountargeted.txt"))
    ]
    if not new_projects:
        logger.info(tab_print("No new projects to add to untargeted tasks!", 2))
        return None
    logger.info(tab_print("New projects to add to untargeted tasks:", 2))
    logger.info(tab_print("\n" + "\n".join("\t" * 7 + d["parent_dir"] for d in new_projects), 0))

    mzmine_cfg = ctx.config["mzmine"]
    rows = []
    for info in new_projects:
        project = info["parent_dir"]
        logger.info(tab_print(f"Working on {project}...", 1))
        row: dict[str, Any] = {
            "parent_dir": project,
            "conforming_filenames": True,
            "file_conversion_complete": True,
            "output_dir": ctx.output_dir,
        }
        if mzmine_cfg.get("custom_batch_params"):
            template, param_row = mzmine_cfg["custom_batch_params"], 5
            logger.info(tab_print(f"Using custom MZmine parameter file: {template}", 2))
        else:
            instrument = get_name_field(project, 6)
            logger.info(tab_print(f"Inferred machine name: {instrument}", 2))
            if instrument is None:
                logger.warning(tab_print(
                    "Warning! Could not validate machine name. Using lenient (IQX) MZmine parameters...", 2))
            template, param_row = select_batch_template(instrument, mzmine_cfg)
            logger.info(tab_print(f"Using MZmine parameters: {os.path.basename(template)}", 2))
        row["mzmine_parameter_sheet"] = template
        row["mzmine_parameter_row"] = param_row

        for polarity in POLARITIES:
            short = short_polarity(polarity)
            row[f"mzmine_{short}_status"] = TaskStatus.ERROR.value
            row[f"fbmn_{short}_status"] = TaskStatus.ERROR.value
            if polarity not in info["polarities"]:
                row[f"{short}_metadata_file"] = ""
                row[f"num_{short}_files"] = 0
                row[f"mzmine_{short}_status"] = TaskStatus.NOT_RELEVANT.value
                row[f"fbmn_{short}_status"] = TaskStatus.NOT_RELEVANT.value
                continue
            metadata_path = _write_polarity_inputs(ctx, project, polarity, info, template)
            if metadata_path is None:
                # LEGACY: directory existed; the row is still inserted with '09 error'.
                continue
            row[f"{short}_metadata_file"] = metadata_path
            row[f"num_{short}_files"] = info[polarity]["file_count"]
            row[f"mzmine_{short}_status"] = TaskStatus.INITIATION.value
            row[f"fbmn_{short}_status"] = TaskStatus.WAITING.value
        rows.append(row)

    inserted = pd.DataFrame(rows)
    if not inserted.empty:
        logger.info(tab_print(f"Updating LIMS table with {inserted.shape[0]} new projects...", 1))
        ctx.lims.update_table(inserted, ctx.tasks_table, method="insert")
        logger.info(tab_print("LIMS untargeted tasks table update complete.", 2))
    logger.info(tab_print(
        f"Exported table for {inserted.shape[0]} new projects to pass to MZmine submission function.", 1))
    return inserted
