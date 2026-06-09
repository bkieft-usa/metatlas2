"""Transfer files to Google Drive using rclone."""

import configparser
import json
import logging
import subprocess

from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen
from typing import List, Optional, Tuple

from IPython.display import HTML, display
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

RCLONE_PATH = "/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone"

RCLONE_UPLOAD_EXCLUDES = [
    "*.yaml",
    "*.ipynb",
    "atl-*csv",
    "manually_curated_compound_data.csv",
    "curated_atlases.csv",
    "auto_ided_atlases.csv",
    ".*",
    ".*/**",
    "**/.*",
    "**/.*/**",
]

# ------------------------------------------------------------------ #
#  Low-level rclone helpers                                           #
# ------------------------------------------------------------------ #

def _rclone_config_file() -> Optional[str]:
    """Return the path to the rclone config file, or None if not found."""
    try:
        result = subprocess.check_output([RCLONE_PATH, "config", "file"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    lines = [l for l in result.splitlines() if l.strip()]
    return lines[-1] if lines else None


def _get_drive_name_for_id(folder_id: str) -> Optional[str]:
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


def _rclone_copy(source: Path, drive: str, dest_path: Path, overwrite: bool = False) -> None:
    """
    Copy *source* directory to *drive*:*dest_path*
    """
    dest = f"{drive}:{dest_path}"
    cmd = [
        RCLONE_PATH, "copy", str(source), dest,
        "--progress",
        "--transfers", "4",
        "--checkers", "8",
        "--drive-chunk-size", "16M",
    ]
    for pattern in RCLONE_UPLOAD_EXCLUDES:
        cmd.extend(["--exclude", pattern])
    if overwrite:
        cmd.append("--ignore-times")
    
    try:
        logger.info("Starting rclone upload: %s -> %s", source, dest)
        with tqdm(total=100, desc="Uploading to Google Drive", unit="%") as pbar:
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
        logger.warning("rclone binary not found at %s — skipping upload.", RCLONE_PATH)


def _has_drive_access(drive: str) -> Tuple[bool, Optional[str]]:
    """Return whether the configured remote is accessible and an optional error message."""
    cmd = [RCLONE_PATH, "lsjson", "--dirs-only", f"{drive}:"]
    try:
        subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return True, None
    except FileNotFoundError:
        return False, f"rclone binary not found at {RCLONE_PATH}."
    except subprocess.CalledProcessError as err:
        message = err.output.strip() if isinstance(err.output, str) else str(err)
        if not message:
            message = str(err)
        return False, message


def _get_drive_id_for_path(drive: str, dest_path: Path) -> Optional[str]:
    """
    Return the Google Drive folder ID for *drive*:*dest_path*.
    Returns None if the folder cannot be found.
    """
    parts = dest_path.parts
    if not parts:
        return None
    parent = f"{drive}:{'/'.join(parts[:-1])}" if len(parts) > 1 else f"{drive}:"
    cmd = [RCLONE_PATH, "lsjson", "--dirs-only", parent]
    try:
        result = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as err:
        logger.exception("rclone lsjson failed: %s", err)
        return None
    for entry in json.loads(result):
        if entry.get("Name") == parts[-1]:
            return entry.get("ID")
    return None


def _drive_path_to_url(drive: str, dest_path: Path) -> Optional[str]:
    """Return a browser URL for *drive*:*dest_path*, or None on failure."""
    folder_id = _get_drive_id_for_path(drive, dest_path)
    if folder_id is None:
        return None
    return f"https://drive.google.com/drive/folders/{folder_id}"


# ------------------------------------------------------------------ #
#  Public upload function                                             #
# ------------------------------------------------------------------ #

def copy_outputs_to_google_drive(summary_obj: "AnalysisSummary", overwrite: bool = False) -> None:
    """
    Recursively copy the analysis output directory to Google Drive using rclone.

    The destination folder name is formed from the final 3 path components of
    analysis_output_dir joined by underscores, suffixed with a timestamp:
        PROJECT_NAME_RTA0_TGA0_2025-01-15-10-30-00

    Parameters
    ----------
    summary_obj:
        The AnalysisSummary object whose analysis_output_dir will be uploaded.
    overwrite:
        If True, overwrite existing files on Google Drive.
    """
    if summary_obj.override_parameters.get("upload_to_gdrive", True) is False:
        logger.info("upload_to_gdrive parameter is False — skipping upload.")
        return

    output_dir = Path(summary_obj.paths.get("analysis_output_dir"))
    gdrive_subfolder = summary_obj.config.gdrive_subfolder

    fail_suffix = "skipping upload to Google Drive"

    config_file = _rclone_config_file()
    if config_file is None:
        logger.warning("rclone config file not found — %s.", fail_suffix)
        return

    drive = _get_drive_name_for_id(gdrive_subfolder)
    if drive is None:
        logger.warning(
            "rclone config does not contain Google Drive folder ID '%s' — %s.",
            gdrive_subfolder,
            fail_suffix,
        )
        return

    has_access, access_err = _has_drive_access(drive)
    if not has_access:
        msg = f"No access to Google Drive remote '{drive}' via rclone"
        if access_err:
            msg = f"{msg}: {access_err}"
        logger.warning("%s — %s.", msg, fail_suffix)
        display(HTML(f"Upload skipped: {msg}"))
        return

    if not output_dir.is_dir():
        logger.warning("analysis_output_dir '%s' does not exist — %s.", output_dir, fail_suffix)
        return

    # Build destination name from the final 3 path components + timestamp
    final_parts = output_dir.parts[-3:]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = "_".join(final_parts) + f"_{date_str}"
    dest_path = Path("Analysis_uploads") / folder_name

    path_string = f"{drive}:{dest_path}"
    display(HTML(f"Uploading targeted analysis to Google Drive at {path_string}"))

    _rclone_copy(output_dir, drive, dest_path, overwrite=overwrite)

    url = _drive_path_to_url(drive, dest_path)
    if url:
        display(HTML(f'Upload complete: <a href="{url}">{path_string}</a>'))
    logger.info("Upload complete.")