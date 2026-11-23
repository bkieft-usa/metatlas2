from __future__ import annotations

import configparser
import json
import logging
import os
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen

from IPython.display import HTML, display
from tqdm.auto import tqdm

from metatlas2.shared_tools.utils import should_disable_tqdm

logger = logging.getLogger(__name__)

_NERSC_RCLONE_PATH = "/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone"

def _get_rclone_path() -> str:
    """Return the path to the rclone binary.

    Checks the NERSC shared environment path first (for backward compatibility
    on Perlmutter), then falls back to the system PATH via shutil.which.
    If neither is found, returns the NERSC path so that downstream
    FileNotFoundError handling in _rclone_copy() and _rclone_config_file()
    is preserved unchanged.
    """
    if os.path.isfile(_NERSC_RCLONE_PATH):
        return _NERSC_RCLONE_PATH
    system_rclone = shutil.which("rclone")
    if system_rclone:
        return system_rclone
    return _NERSC_RCLONE_PATH

RCLONE_UPLOAD_EXCLUDES = [
    "*.yaml",
    "*.ipynb",
    "curated_atlases.csv",
    "auto_ided_atlases.csv",
    ".*",
    ".*/**",
    "**/.*",
    "**/.*/**",
]

def _rclone_config_file() -> str | None:
    """Return the path to the rclone config file, or None if not found."""
    try:
        result = subprocess.check_output([_get_rclone_path(), "config", "file"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    lines = [l for l in result.splitlines() if l.strip()]
    return lines[-1] if lines else None

def _get_drive_name_for_id(folder_id: str) -> str | None:
    """
    Look up the rclone remote name corresponding to a Google Drive folder ID.
    Returns None if the config file is missing or the ID is not found.
    """
    ini_file = _rclone_config_file()
    if ini_file is None:
        return None
    config = configparser.ConfigParser()
    config.read(ini_file)
    for name in config.sections():
        props = config[name]
        if props.get("type") == "drive" and props.get("root_folder_id") == folder_id:
            return name
    return None

def _rclone_copy(
    source: Path, drive: str, dest_path: Path, overwrite: bool = False,
    *, apply_excludes: bool = True, required: bool = False,
    update: bool = False, ignore_existing: bool = False,
) -> None:
    """
    Copy *source* (a directory or a single file) to *drive*:*dest_path*.

    Args:
        overwrite: Add ``--ignore-times`` so every file is re-transferred.
        apply_excludes: Apply :data:`RCLONE_UPLOAD_EXCLUDES`.
        required: Raise (instead of warning) if the rclone binary is missing.
        update: Add ``--update`` (skip files that are newer on the remote).
        ignore_existing: Add ``--ignore-existing`` (never replace remote files).
    """
    dest = f"{drive}:{dest_path}"
    cmd = [
        _get_rclone_path(), "copy", str(source), dest,
        "--progress",
        "--transfers", "32",
        "--checkers", "64",
        "--drive-chunk-size", "8M",
        "--fast-list",
        "--drive-pacer-burst", "200",
        "--drive-pacer-min-sleep", "10ms",
        "--drive-disable-http2=false",
    ]
    if apply_excludes:
        for pattern in RCLONE_UPLOAD_EXCLUDES:
            cmd.extend(["--exclude", pattern])
    if overwrite:
        cmd.append("--ignore-times")
    if update:
        cmd.append("--update")
    if ignore_existing:
        cmd.append("--ignore-existing")

    try:
        logger.info(f"Starting rclone upload: {source} -> {dest}")
        with tqdm(total=100, desc="Uploading to Google Drive", unit="%", disable=should_disable_tqdm()) as pbar:
            with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as proc:
                for line in proc.stdout or []:
                    line = line.strip()
                    if line.startswith("Transferred:") and line.endswith("%"):
                        try:
                            percent = float(line.split(",")[1].split("%")[0])
                            pbar.n = percent
                            pbar.refresh()
                        except (IndexError, ValueError):
                            pass
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
                pbar.n = 100
                pbar.refresh()
    except subprocess.CalledProcessError as err:
        logger.exception("rclone copy failed: %s", err)
        raise
    except FileNotFoundError:
        if required:
            logger.exception("rclone binary not found at %s", _get_rclone_path())
            raise
        logger.warning("rclone binary not found at %s — skipping upload.", _get_rclone_path())

def rclone_run(args: list[str]) -> str:
    """Run ``rclone <args...>`` and return its stdout.

    Raises:
        FileNotFoundError: If the rclone binary cannot be found.
        subprocess.CalledProcessError: If rclone exits non-zero.
    """
    cmd = [_get_rclone_path(), *args]
    logger.debug("Running rclone command: %s", " ".join(cmd))
    return subprocess.check_output(cmd, text=True)


def rclone_list(remote_path: str, *, long: bool = False, max_depth: int = 1) -> list[list[str]]:
    """List files at an rclone remote path (``rclone ls``/``lsl``).

    Args:
        remote_path: e.g. ``"my_remote:/some/folder"``.
        long: Use ``lsl`` (size, date, time, name) instead of ``ls`` (size, name).
        max_depth: Value for ``--max-depth``.

    Returns:
        One list of whitespace-split fields per listed file. File names
        containing spaces are re-joined into the final field.
    """
    out = rclone_run(["lsl" if long else "ls", remote_path, "--max-depth", str(max_depth)])
    n_leading = 3 if long else 1
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) <= n_leading:
            continue
        rows.append(parts[:n_leading] + [" ".join(parts[n_leading:])])
    return rows


def _has_drive_access(drive: str) -> tuple[bool, str | None]:
    """Return whether the configured remote is accessible and an optional error message."""
    cmd = [_get_rclone_path(), "lsjson", "--dirs-only", f"{drive}:"]
    try:
        subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return True, None
    except FileNotFoundError:
        return False, f"rclone binary not found at {_get_rclone_path()}."
    except subprocess.CalledProcessError as err:
        message = err.output.strip() if isinstance(err.output, str) else str(err)
        if not message:
            message = str(err)
        return False, message

def _get_drive_id_for_path(drive: str, dest_path: Path) -> str | None:
    """
    Return the Google Drive folder ID for *drive*:*dest_path*.
    Returns None if the folder cannot be found.
    """
    parts = dest_path.parts
    if not parts:
        return None
    parent = f"{drive}:{'/'.join(parts[:-1])}" if len(parts) > 1 else f"{drive}:"
    cmd = [_get_rclone_path(), "lsjson", "--dirs-only", parent]
    try:
        result = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as err:
        logger.exception("rclone lsjson failed: %s", err)
        return None
    for entry in json.loads(result):
        if entry.get("Name") == parts[-1]:
            return entry.get("ID")
    return None

def _drive_path_to_url(drive: str, dest_path: Path) -> str | None:
    """Return a browser URL for *drive*:*dest_path*, or None on failure."""
    folder_id = _get_drive_id_for_path(drive, dest_path)
    if folder_id is None:
        return None
    return f"https://drive.google.com/drive/folders/{folder_id}"


def upload_standards_outputs(config: dict, annotation_dir: Path, sources: list[Path]) -> str:
    """Upload standards outputs under their annotation run and return its Drive URL."""
    project = config["project"]
    run_name = project["standards_run_name"]
    if not isinstance(run_name, str) or not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError(f"Invalid standards_run_name for Google Drive: {run_name!r}")
    drive_config = config.get("gdrive")
    if not isinstance(drive_config, dict):
        raise ValueError("GDRIVE must contain the JGI_Metabolomics_Projects root_folder_id.")
    root_id = drive_config.get("root_folder_id")
    if not isinstance(root_id, str) or not root_id.strip():
        raise ValueError("GDRIVE.root_folder_id must identify the JGI_Metabolomics_Projects folder.")
    annotation_dir = Path(annotation_dir).resolve()
    from metatlas2.standards_annotation.standards_annotation_review import ANNOTATION_RUN_DIR_PATTERN
    if not ANNOTATION_RUN_DIR_PATTERN.fullmatch(annotation_dir.name):
        raise ValueError(f"Invalid annotation run directory: {annotation_dir}")
    if not annotation_dir.is_dir() or not sources:
        raise ValueError(f"Annotation directory or upload sources missing: {annotation_dir}")
    resolved_sources = []
    for source in sources:
        source = Path(source).resolve()
        if not source.exists() or not source.is_relative_to(annotation_dir):
            raise ValueError(f"Upload source is missing or outside {annotation_dir}: {source}")
        resolved_sources.append(source)
    drive = _get_drive_name_for_id(root_id.strip())
    if drive is None:
        raise RuntimeError(f"No rclone Google Drive remote configured for folder ID {root_id!r}.")
    has_access, error = _has_drive_access(drive)
    if not has_access:
        raise RuntimeError(f"Cannot access rclone remote {drive!r}: {error}")
    dest = Path("Reference_Standards_Annotation") / run_name / annotation_dir.name
    for source in resolved_sources:
        relative = source.relative_to(annotation_dir)
        source_dest = dest / (relative if source.is_dir() else relative.parent)
        _rclone_copy(source, drive, source_dest, overwrite=True, apply_excludes=False, required=True)
    url = _drive_path_to_url(drive, dest)
    if not url:
        raise RuntimeError(f"Uploaded outputs but could not find Drive folder {drive}:{dest}.")
    logger.info("Standards annotation Drive folder: %s", url)
    return url

def copy_outputs_to_google_drive(
    obj: "RTAlign" | "AnalysisSummary",
    stage: "ANALYSIS_SUMMARY",
    overwrite: bool = False
) -> None:
    """
    Recursively copy the analysis output directory to Google Drive using rclone.

    The destination hierarchy under ``JGI_Metabolomics_Projects/Analysis_Uploads/``:

        JGI_Metabolomics_Projects/
        └── Analysis_Uploads/
            └── <project_name>/
                └── <username>_<OWNER>-<CHROMATOGRAPHY>_<rta_number>_<tga_number>/
                    └── Targeted/
                        └── <OWNER>-<CHROMATOGRAPHY>_<project_name>/
                            ├── <analysis_name>-<polarity>_<YYYY-MM-DD-HH-MM>/  (ANALYSIS_SUMMARY)
                            └── RT-Alignment_<YYYY-MM-DD-HH-MM>/                (RT_ALIGNMENT)
    """
    if stage == "ANALYSIS_SUMMARY":
        if obj.override_parameters.get("upload_to_gdrive", True) is False:
            logger.info("upload_to_gdrive parameter is False — skipping upload.")
            return

    fail_suffix = "skipping upload to Google Drive"

    config_file = _rclone_config_file()
    if config_file is None:
        logger.warning(f"rclone config file not found — {fail_suffix}.")
        return

    drive = _get_drive_name_for_id(obj.config.gdrive_subfolder)
    if drive is None:
        logger.warning(
            "rclone config does not contain Google Drive folder ID '%s' — %s.",
            obj.config.gdrive_subfolder,
            fail_suffix,
        )
        return

    has_access, access_err = _has_drive_access(drive)
    if not has_access:
        msg = f"No access to Google Drive remote '{drive}' via rclone"
        if access_err:
            msg = f"{msg}: {access_err}"
        logger.warning(f"{msg} — {fail_suffix}.")
        return

    env_user = os.environ.get("USER", "unknown")
    run_folder = (
        f"{env_user}_{obj.owner.upper()}-{obj.chromatography.upper()}"
        f"_{obj.rt_alignment_number}_{obj.analysis_number}"
    )

    owner_chrom = f"{obj.owner.upper()}-{obj.chromatography.upper()}"
    targeted_subdir = Path("Analysis_Uploads") / obj.project_name / run_folder / "Targeted" / f"{owner_chrom}_{obj.project_name}"

    path_string = f"{drive}:{targeted_subdir}"
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if stage == "RT_ALIGNMENT":
        rt_results_dir = Path(obj.paths.get("rt_alignment_results_dir"))
        if rt_results_dir.is_dir():
            rt_dest = targeted_subdir / f"RT-Alignment_{date_str}"
            logger.info(f"Uploading RT alignment results: {rt_results_dir} -> {drive}:{rt_dest}")
            _rclone_copy(rt_results_dir, drive, rt_dest, overwrite=overwrite)
        else:
            logger.warning(f"rt_alignment_results_dir '{rt_results_dir}' does not exist — skipping RT alignment upload.")
    elif stage == "ANALYSIS_SUMMARY":
        tga_results_dir = Path(obj.paths.get("analysis_results_output_dir"))
        if tga_results_dir.is_dir():
            analysis_name = obj.ta.atlas_name if obj.ta else Path(tga_results_dir).name
            polarity = obj.polarity if obj.polarity else "unknown"
            dated_label = f"{analysis_name}-{polarity}_{date_str}"
            tga_dest = targeted_subdir / dated_label
            logger.info(f"Uploading targeted analysis results: {tga_results_dir} -> {drive}:{tga_dest}")
            _rclone_copy(tga_results_dir, drive, tga_dest, overwrite=overwrite)
            url = _drive_path_to_url(drive, tga_dest)
            if url:
                logger.info(f"Upload complete: {path_string} ({url})")
                #display(HTML(f'Upload complete: <a href="{url}">{path_string}</a>'))
                owner = (getattr(obj, "owner", "") or "").lower()
                if owner == "jgi":
                    chromatography = (getattr(obj, "chromatography", "") or "").upper()
                    polarity_display = (polarity or "").upper()
                    lines = [
                        f"Chromatography: {chromatography}",
                        f"Analysis: {analysis_name}",
                        f"Polarity: {polarity_display}",
                        "Comment: ",
                        "Action: ",
                        f"Link: {url}",
                    ]
                    text = "\n".join(lines)
                    logger.info(f"JIRA template for analysis release:\n")
                    display(HTML(
                        f"<textarea rows='{len(lines)}' cols='120' "
                        f"style='font-family:monospace;resize:none;border:1px solid #ccc;"
                        f"padding:6px;background:#f9f9f9;color:black;overflow:hidden' readonly>"
                        f"{text}</textarea>"
                    ))
        else:
            logger.warning(f"analysis_results_output_dir '{tga_results_dir}' does not exist — skipping targeted analysis upload.")