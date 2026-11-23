"""Step 7: zip completed projects and upload them to Google Drive (rclone)."""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.shared_tools import gdrive_upload as gdu
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.feature_filtering import check_peak_height_table
from metatlas2.untargeted.status import (
    FILE_SUFFIXES,
    PolarityTask,
    TaskStatus,
    check_for_polarities,
    subset_df_by_status,
    tab_print,
)
from metatlas2.untargeted.utils import set_group

logger = lcf.get_logger("untargeted.packaging")

#: Pipeline bookkeeping files (new in metatlas2) kept out of delivered zips.
ZIP_EXCLUDED_SUFFIXES = (FILE_SUFFIXES["mzmine_status"], FILE_SUFFIXES["mzmine_queued"])


def abridged_project_name(project: str, strip_suffixes: list[str]) -> str | None:
    """``{date}_{dept}_{submitter}_{pid}_{chromatography}`` for conforming names, else None."""
    base = project
    for suffix in strip_suffixes:
        if project.endswith(suffix):
            base = project[: -len(suffix)]
            break
    fields = base.split("_")
    if len(fields) < 9:
        logger.info(tab_print(
            f"Note! Project name {project} has fewer than expected fields. Skipping renaming files before zip...", 1))
        return None
    date, department, submitter, pid, chromatography = fields[0], fields[1], fields[2], fields[3], fields[7]
    if (not any(s.lower() in chromatography.lower() for s in ("C18", "LIPID", "HILIC"))
            or not date.isdigit() or len(date) != 8):
        logger.warning(tab_print(
            f"Warning! Project name {project} does not follow the standard naming convention. "
            "Skipping renaming files before zip...", 1))
        logger.warning(tab_print(
            f"Here is what could be extracted: Date: {date}, Department: {department}, Submitter: {submitter}, "
            f"PID: {pid}, Chromatography: {chromatography}", 2))
        return None
    return f"{date}_{department}_{submitter}_{pid}_{chromatography}"


def zip_untargeted_results(
    target_dirs: list[str | os.PathLike],
    output_zip_archive: str | os.PathLike,
    project: str,
    extra_files: list[str | os.PathLike] | None = None,
    abridged_name: str | None = None,
) -> Path:
    """Zip all files under *target_dirs* (plus *extra_files*) flat into one archive.
    """
    output_zip_archive = Path(output_zip_archive)
    entries: list[tuple[Path, str]] = []
    for target in target_dirs:
        for root, _dirs, files in os.walk(target):
            for name in sorted(files):
                if name.endswith(ZIP_EXCLUDED_SUFFIXES):
                    continue
                entries.append((Path(root) / name, name))
    for extra in extra_files or []:
        entries.append((Path(extra), Path(extra).name))

    tmp = output_zip_archive.with_name(output_zip_archive.name + ".tmp")
    seen: set[str] = set()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in entries:
            if abridged_name and project in arcname:
                arcname = arcname.replace(project, abridged_name)
            if arcname in seen:
                logger.warning(tab_print(f"Duplicate file name {arcname} in zip; keeping the first copy.", 2))
                continue
            seen.add(arcname)
            zf.write(path, arcname)
    tmp.replace(output_zip_archive)
    logger.info(tab_print(f"New untargeted results in {[str(t) for t in target_dirs]} zipped", 1))
    return output_zip_archive


def _split_remote(remote: str) -> tuple[str, Path]:
    drive, _, path = remote.partition(":")
    return drive, Path(path or "/")


def _remote_has_file(remote: str, filename: str) -> bool:
    return any(row[-1] == filename for row in gdu.rclone_list(remote))


def add_gnps2_documentation(download_dir: str, doc_name: str | None, remote: str) -> bool:
    """Fetch the GNPS2 user guide from the Drive *remote* into *download_dir* (``--update``)."""
    if not doc_name:
        logger.warning(tab_print("Warning! GNPS2 user guide documentation file name not provided. Not downloaded!", 2))
        return False
    try:
        gdu.rclone_run(["copy", "--update", f"{remote.rstrip('/')}/{doc_name}", str(download_dir)])
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        logger.warning(tab_print(f"Warning! GNPS2 user guide documentation not downloaded! ({err})", 2))
        return False
    if os.path.isfile(Path(download_dir) / doc_name):
        logger.info(tab_print(f"GNPS2 user guide documentation ({doc_name}) downloaded.", 2))
        return True
    logger.warning(tab_print("Warning! GNPS2 user guide documentation not downloaded!", 2))
    return False


def upload_to_google_drive(file_path: str | os.PathLike, remote: str, overwrite: bool = False) -> bool:
    """Upload a zip to the Drive *remote* and confirm it is listed there.

    With ``overwrite=False`` an existing remote file is left alone (returns False).
    With ``overwrite=True`` rclone ``--update`` replaces an older remote copy.
    """
    file_path = Path(file_path)
    name = file_path.name
    logger.info(tab_print("Uploading zip to Google Drive...", 2))
    drive, dest = _split_remote(remote)
    try:
        if not overwrite and _remote_has_file(remote, name):
            logger.info(tab_print(
                f"Note: Overwrite is False and Google Drive folder for {name} already exists. Skipping upload...", 2))
            return False
        gdu._rclone_copy(file_path, drive, dest, apply_excludes=False, required=True,
                         update=overwrite, ignore_existing=not overwrite)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        logger.critical(tab_print(
            f"Warning! Google Drive upload failed with overwrite={overwrite}: {err}", 3))
        return False
    try:
        if _remote_has_file(remote, name):
            logger.info(tab_print("Google Drive upload confirmed!", 3))
            return True
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        logger.warning(tab_print(f"Warning! Google Drive upload check failed: {err}", 3))
        return False
    logger.warning(tab_print("Warning! Google Drive upload check failed. Upload may not have been successful.", 3))
    return False


def zip_and_upload_untargeted_results(ctx: RunContext) -> None:
    """Step 7: zip projects whose MZmine and FBMN finished, then upload to Drive."""
    logger.info("Step 7/7: Zipping up and (optionally) uploading output folders to gdrive...")
    gcfg: dict[str, Any] = ctx.config["gdrive"]
    download_dir = ctx.paths["download_dir"]
    df = ctx.get_tasks()
    if ctx.direct_input is None and not df.empty:
        df = subset_df_by_status(df, "fbmn", [TaskStatus.COMPLETE])
        if not df.empty:
            df = subset_df_by_status(df, "mzmine", [TaskStatus.COMPLETE])
        if not df.empty:
            df = subset_df_by_status(df, "mzmine", [TaskStatus.ERROR], inverse=True)
        if not df.empty:
            df = subset_df_by_status(df, "fbmn", [TaskStatus.ERROR], inverse=True)
    if df.empty:
        logger.info(tab_print("No completed untargeted results to zip and upload!", 1))
        return
    logger.info(tab_print(f"{df.shape[0]} project(s) total with complete mzmine and fbmn status.", 1))
    zip_count = upload_count = 0
    for _, row in df.iterrows():
        project = row["parent_dir"]
        archive = Path(download_dir) / f"{project}.zip"
        if os.path.exists(archive) and not ctx.overwrite_zip:
            continue
        polarity_list = check_for_polarities(ctx.output_dir, project)
        if polarity_list is None:
            logger.warning(tab_print(
                f"Warning! Project {project} does not have a negative or a positive polarity directory. Skipping...", 1))
            continue
        tasks = [PolarityTask(ctx.output_dir, project, p) for p in polarity_list]
        if not all(t.exists("peak_height") and t.exists("fbmn_library_results") for t in tasks):
            continue
        n_features = sum(check_peak_height_table(t.path("peak_height")) for t in tasks)
        if n_features <= int(gcfg.get("min_features", 0)):
            logger.warning(tab_print(
                f"Warning! Project {project} has less than {gcfg.get('min_features', 0)} features in the "
                f"{polarity_list} peak height table(s). Skipping zip and upload...", 1))
            continue
        if ctx.dry_run:
            logger.info(tab_print(f"[dry-run] Would zip {project} ({n_features} features) to {archive}"
                                  + (" and upload to Google Drive" if gcfg.get("upload", True) else ""), 1))
            continue
        for t in tasks:
            set_group(t.directory, ctx.group)
            try:
                t.path("mzmine_job_id").unlink()
            except FileNotFoundError:
                pass
        extra = []
        if gcfg.get("add_documentation", True):
            logger.info(tab_print("Downloading latest GNPS2 user guide documentation to add to zip...", 1))
            if add_gnps2_documentation(download_dir, gcfg.get("doc_name"), gcfg["remote"]):
                extra.append(Path(download_dir) / gcfg["doc_name"])
            else:
                logger.warning(tab_print(
                    "Warning! Add documentation flag is True but GNPS2 user guide documentation not available. "
                    "Not adding to zip...", 2))
        abridged = None
        if gcfg.get("abridged_filenames", True):
            abridged = abridged_project_name(project, gcfg.get("abridged_strip_suffixes", []))
        zip_untargeted_results([t.directory for t in tasks], archive, project, extra, abridged)
        set_group(archive, ctx.group)
        zip_count += 1
        if gcfg.get("upload", True) and os.path.exists(archive):
            if upload_to_google_drive(archive, gcfg["remote"], overwrite=ctx.overwrite_drive):
                upload_count += 1
    if zip_count == 0:
        logger.info(tab_print("No newly completed untargeted projects to be zipped.", 2))
    logger.info(tab_print(f"{upload_count} new complete untargeted projects uploaded.", 1))
