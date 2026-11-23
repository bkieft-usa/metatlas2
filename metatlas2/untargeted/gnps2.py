"""GNPS2 access: workflow submission, status, downloads, and SFTP mirroring.

Also holds the FBMN / Everything Bagel parameter builders and MGF helpers.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.untargeted.status import tab_print

logger = lcf.get_logger("untargeted.gnps2")

_TASK_ID_RE = re.compile(r"[\"'/]status\?task=([a-f0-9]{32})")


def _quiet_paramiko() -> None:
    plog = logging.getLogger("paramiko")
    plog.setLevel(logging.CRITICAL)
    plog.addHandler(logging.NullHandler())
    plog.propagate = False


def read_gnps2_password(credentials_file: str | os.PathLike) -> str | None:
    """Read the GNPS2 password from a ``NAME="password"`` credentials file.

    Prefers a ``MYVARIABLE=`` line (legacy SFTP format) and otherwise uses the
    first line containing ``=`` (legacy HTTP format). Quotes are stripped.
    """
    try:
        lines = Path(credentials_file).read_text().splitlines()
    except OSError:
        logger.error(f"GNPS2 credentials file not found or unreadable: {credentials_file}")
        return None
    candidates = [l for l in lines if l.startswith("MYVARIABLE=")] or [l for l in lines if "=" in l]
    if not candidates:
        return None
    password = candidates[0].split("=", 1)[1].replace('"', "").replace("'", "").strip()
    return password or None


def extract_task_id(response_text: str) -> str | None:
    """Find a 32-hex GNPS2 task ID in an HTML response (``status?task=<id>``)."""
    match = _TASK_ID_RE.search(response_text)
    return match.group(1) if match else None


def parse_status_page(html: bytes | str) -> tuple[str, str]:
    """Return ``(status, version)`` from a GNPS2 task status page, or ``("N/A", "N/A")``."""
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "html.parser")
        version = soup.find("td", string="Version").find_next_sibling("td").text
        status = soup.find("td", string="Status").find_next_sibling("td").text
    except Exception:
        return "N/A", "N/A"
    if status and version:
        return status, version
    return "N/A", "N/A"


# ---------------------------------------------------------------------------
# Parameter builders
# ---------------------------------------------------------------------------


def fbmn_parameters(
    base_params: dict[str, str],
    description: str,
    quant_file: str,
    spectra_file: str,
    metadata_file: str,
    raw_data: str,
) -> dict[str, str]:
    """FBMN workflow parameters: config defaults plus the per-task file locations."""
    params = {"description": description, **base_params}
    params.update({
        "inputfeatures": quant_file,
        "inputspectra": spectra_file,
        "metadata_filename": metadata_file,
        "input_raw_spectra": raw_data,
    })
    return params


def everything_bagel_parameters(
    base_params: dict[str, str], description: str, raw_data: str, metadata_file: str
) -> dict[str, str]:
    """Everything Bagel parameters: config defaults plus input spectra and metadata."""
    params = {"description": description, **base_params}
    params.update({"input_spectra": raw_data, "metadata_file": metadata_file})
    return params


def build_eb_file_list(local_dir: str | os.PathLike, raw_data: str, polarity: str) -> str:
    """``;``-joined GNPS2 paths of all mzML files of one polarity in *local_dir*.

    Raises:
        ValueError: On unknown polarity, unparseable/duplicate/``;``-containing
            names, or when no files match.
    """
    if polarity not in ("positive", "negative"):
        raise ValueError(f"Unsupported polarity: {polarity}")
    selected: set[str] = set()
    for path in Path(local_dir).rglob("*.mzML"):
        if not path.is_file():
            continue
        fields = path.name.split("_")
        if len(fields) <= 9 or fields[9] not in ("POS", "NEG", "FPS", "EI"):
            raise ValueError(f"Cannot determine polarity of {path.name}")
        if fields[9] == polarity[:3].upper():
            if ";" in path.name or path.name in selected:
                raise ValueError(f"Invalid or duplicate filename: {path.name}")
            selected.add(path.name)
    if not selected:
        raise ValueError(f"No {polarity} mzML files found in {local_dir}")
    return ";".join(f"{raw_data}/{name}" for name in sorted(selected))


# ---------------------------------------------------------------------------
# MGF helpers
# ---------------------------------------------------------------------------


def count_mgf_lines(mgf_file: str | os.PathLike) -> int:
    """Number of ``END IONS`` lines in an MGF (0 if the file is missing)."""
    if not os.path.isfile(mgf_file):
        return 0
    with open(mgf_file, errors="replace") as fh:
        return sum(1 for line in fh if "END" in line and "IONS" in line)


def _is_number_token(token: str) -> bool:
    return token.replace(".", "", 1).isdigit()


def remove_contaminant_from_mgf(
    mgf_file: str | os.PathLike, calibrant_mz: dict[str, float], ppm: float = 5
) -> int:
    """Remove lines mentioning the internal-calibrant m/z from an MGF, in place.

    Polarity is taken from ``negative``/``positive`` in the file path.

    LEGACY QUIRK: *any* line with a whitespace-separated number within *ppm*
    of the calibrant is removed, which can include header lines (kept as-is).

    Returns:
        Number of removed lines (-1 if polarity could not be determined).
    """
    name = str(mgf_file)
    if "negative" in name:
        target = float(calibrant_mz["negative"])
    elif "positive" in name:
        target = float(calibrant_mz["positive"])
    else:
        logger.critical(tab_print(f"Error: Could not determine polarity of file {name}", 1))
        return -1
    diff = target * ppm / 1e6
    with open(mgf_file) as fh:
        lines = fh.readlines()
    kept = [
        line for line in lines
        if not any(abs(float(tok) - target) < diff for tok in line.split() if _is_number_token(tok))
    ]
    removed = len(lines) - len(kept)
    if removed:
        logger.info(tab_print(f"Deleting {removed} lines from {os.path.basename(name)}", 1))
        with open(mgf_file, "w") as fh:
            fh.writelines(kept)
    return removed


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Gnps2Client:
    """GNPS2 HTTP + SFTP client.

    Args:
        username: GNPS2 account (also the ``USERUPLOAD/<username>`` prefix).
        credentials_file: File holding the GNPS2 password.
        base_url: GNPS2 web root.
        sftp_host / sftp_port: GNPS2 SFTP endpoint.
        dry_run: Log instead of submitting or uploading.
    """

    def __init__(
        self,
        username: str,
        credentials_file: str,
        base_url: str = "https://gnps2.org",
        sftp_host: str = "sftp.gnps2.org",
        sftp_port: int = 443,
        sftp_retry_wait_seconds: float = 30,
        request_timeout_seconds: float = 300,
        dry_run: bool = False,
    ) -> None:
        self.username = username
        self.credentials_file = credentials_file
        self.base_url = base_url.rstrip("/")
        self.sftp_host = sftp_host
        self.sftp_port = int(sftp_port)
        self.sftp_retry_wait_seconds = float(sftp_retry_wait_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.dry_run = dry_run

    @classmethod
    def from_config(cls, cfg: dict[str, Any], dry_run: bool = False) -> "Gnps2Client":
        return cls(
            username=cfg["username"],
            credentials_file=cfg["credentials_file"],
            base_url=cfg.get("base_url", "https://gnps2.org"),
            sftp_host=cfg.get("sftp_host", "sftp.gnps2.org"),
            sftp_port=cfg.get("sftp_port", 443),
            sftp_retry_wait_seconds=cfg.get("sftp_retry_wait_seconds", 30),
            request_timeout_seconds=cfg.get("request_timeout_seconds", 300),
            dry_run=dry_run,
        )

    # -- paths / urls ----------------------------------------------------------
    @property
    def user_upload_prefix(self) -> str:
        return f"USERUPLOAD/{self.username}"

    def status_url(self, task_id: str) -> str:
        return f"{self.base_url}/status?task={task_id}"

    def graphml_url(self, task_id: str) -> str:
        return f"{self.base_url}/result?task={task_id}&viewname=graphml&resultdisplay_type=task"

    def library_results_url(self, task_id: str) -> str:
        return f"{self.base_url}/resultfile?task={task_id}&file=nf_output/library/merged_results_with_gnps.tsv"

    # -- HTTP ------------------------------------------------------------------
    def submit(self, params: dict[str, str]) -> dict[str, Any]:
        """Log in and launch a GNPS2 workflow.

        Returns:
            ``{"task": <id>}`` (or GNPS2's JSON response); ``{}`` on failure or dry run.
        """
        if self.dry_run:
            logger.info(tab_print(f"[dry-run] Would submit GNPS2 workflow {params.get('workflowname')}: "
                                  f"{params.get('description')}", 2))
            return {}
        import requests

        password = read_gnps2_password(self.credentials_file)
        if not password:
            logger.error(tab_print("GNPS2 password is required to submit jobs.", 2))
            return {}
        session = requests.Session()
        session.post(f"{self.base_url}/login", data={"username": self.username, "password": password},
                     timeout=self.request_timeout_seconds)
        response = session.post(f"{self.base_url}/launchworkflow", data=params,
                                timeout=self.request_timeout_seconds)
        try:
            return response.json()
        except Exception:
            task_id = extract_task_id(response.text)
            if task_id:
                logger.info(tab_print(f"Extracted task ID from GNPS2 HTML response: {task_id}", 2))
                return {"task": task_id}
            logger.warning(tab_print(
                f"Warning! GNPS2 submission response (status {response.status_code}) could not be parsed as "
                "JSON and no task ID found in HTML.", 2))
            return {}

    def check_status(self, task_id: str) -> tuple[str, str]:
        """Return ``(status, version)`` of a task from its public status page."""
        import requests

        try:
            response = requests.get(self.status_url(task_id), timeout=self.request_timeout_seconds)
        except Exception as err:
            logger.warning(tab_print(f"GNPS2 status request failed for {task_id}: {err}", 2))
            return "N/A", "N/A"
        return parse_status_page(response.content)

    def download(self, url: str, target_path: str | os.PathLike) -> bool:
        """Stream *url* to *target_path*. Returns True on HTTP 200."""
        if self.dry_run:
            logger.info(tab_print(f"[dry-run] Would download {url} -> {target_path}", 3))
            return False
        import requests

        try:
            response = requests.get(url, stream=True, timeout=self.request_timeout_seconds)
        except Exception as err:
            logger.warning(tab_print(f"Download failed for {url}: {err}", 3))
            return False
        if response.status_code != 200:
            return False
        with open(target_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        return True

    # -- SFTP ------------------------------------------------------------------
    def connect_sftp(self):
        """Open ``(transport, sftp)``; retries once after a wait. Returns ``(None, None)`` on failure."""
        import paramiko

        _quiet_paramiko()
        password = read_gnps2_password(self.credentials_file)
        if not password:
            logger.error("Password is required to connect to GNPS2.")
            return None, None
        for attempt in (1, 2):
            try:
                transport = paramiko.Transport((self.sftp_host, self.sftp_port))
                transport.connect(username=self.username, password=password)
                return transport, paramiko.SFTPClient.from_transport(transport)
            except (paramiko.SSHException, OSError) as err:
                if attempt == 1:
                    logger.info(tab_print(f"Failed to connect to GNPS2: {err}", 4))
                    logger.info(tab_print(
                        f"Waiting {self.sftp_retry_wait_seconds:.0f} seconds and attempting to connect again...", 5))
                    time.sleep(self.sftp_retry_wait_seconds)
                else:
                    logger.error(tab_print(f"Failed to connect to GNPS2 again: {err}", 4))
        return None, None

    def mirror_mzmine_results(self, project: str, polarity: str, output_dir: str) -> bool:
        """Upload ``.mgf``/``.csv``/``.tab`` files of a polarity to ``/untargeted_tasks/{project}_{polarity}``."""
        if self.dry_run:
            logger.info(tab_print(f"[dry-run] Would mirror MZmine results for {project}_{polarity} to GNPS2", 3))
            return True
        logger.info(tab_print(f"Mirroring MZmine results for {project} to GNPS2...", 3))
        transport, sftp = self.connect_sftp()
        if sftp is None:
            return False
        remote_dir = f"/untargeted_tasks/{project}_{polarity}"
        try:
            try:
                sftp.mkdir(remote_dir)
            except Exception as err:
                if "mkdir file exists" not in str(err).lower():
                    logger.warning(tab_print(f"Notice! Did not create {remote_dir} at GNPS2 for reason: {err}", 3))
            local_dir = Path(output_dir) / f"{project}_{polarity}"
            for file_path in local_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in (".mgf", ".csv", ".tab"):
                    sftp.put(str(file_path), f"{remote_dir}/{file_path.name}")
                    logger.info(tab_print(f"Uploaded {file_path.name} to GNPS2...", 4))
            logger.info(tab_print(f"Completed MZmine results mirror to GNPS2 for {project}...", 3))
            return True
        except Exception as err:
            logger.error(tab_print(f"Failed to mirror MZmine results for {project} to GNPS2: {err}", 3))
            return False
        finally:
            sftp.close()
            transport.close()

    def mirror_raw_data(self, project: str, local_dir: str, mode: str = "pipeline") -> bool:
        """Upload a project's mzML files to ``/raw_data/{subdir}/{project}`` on GNPS2.

        Modes:
            ``pipeline``: upload everything to a new directory; if it already
            exists, behave like ``soft``.
            ``hard``: upload every file, overwriting.
            ``soft``: upload only files that are new or changed (size/mtime).
        """
        if mode not in ("pipeline", "hard", "soft"):
            raise ValueError(f"Unknown mirror mode: {mode}")
        local_path = Path(local_dir)
        remote_dir = f"/raw_data/{local_path.parent.name}/{project}"
        if self.dry_run:
            logger.info(tab_print(f"[dry-run] Would {mode}-mirror raw data {local_path} -> GNPS2:{remote_dir}", 2))
            return True
        label = {"pipeline": "Pipeline mirroring", "hard": "Hard mirroring (force overwrite all)",
                 "soft": "Soft mirroring (sync only new/modified)"}[mode]
        logger.info(tab_print(f"{label} raw data for {project} to GNPS2...", 2))
        transport, sftp = self.connect_sftp()
        if sftp is None:
            return False
        try:
            effective = mode
            if mode == "pipeline":
                try:
                    sftp.listdir(remote_dir)
                    logger.info(tab_print(f"Target directory {remote_dir} exists. Switching to soft sync mode.", 3))
                    effective = "soft"
                except FileNotFoundError:
                    pass
            try:
                sftp.mkdir(remote_dir)
            except Exception as err:
                if effective != "hard" and "mkdir file exists" not in str(err).lower():
                    logger.warning(tab_print(f"Could not create directory {remote_dir}: {err}", 3))
                    if effective == "pipeline":
                        return False
            uploaded = skipped = 0
            for file_path in local_path.rglob("*.mzML"):
                if not file_path.is_file():
                    continue
                remote_path = f"{remote_dir}/{file_path.name}"
                if effective == "soft":
                    try:
                        rstat = sftp.stat(remote_path)
                        lstat = file_path.stat()
                        if rstat.st_size == lstat.st_size and rstat.st_mtime >= lstat.st_mtime:
                            skipped += 1
                            continue
                    except FileNotFoundError:
                        pass
                sftp.put(str(file_path), remote_path)
                uploaded += 1
                verb = {"hard": "Force uploaded", "soft": "Synced"}.get(effective, "Uploaded")
                logger.info(tab_print(f"{verb} {file_path.name}", 3))
            if effective == "soft" and skipped:
                kind = "Pipeline sync" if mode == "pipeline" else "Soft sync"
                logger.info(tab_print(f"{kind} complete: {uploaded} uploaded, {skipped} skipped", 2))
            elif effective == "hard":
                logger.info(tab_print(f"Hard mirror complete: {uploaded} files uploaded (forced)", 2))
            else:
                logger.info(tab_print(f"Pipeline mirror complete: {uploaded} files uploaded", 2))
            logger.info(tab_print(f"Completed raw data mirror to GNPS2 for {project}", 2))
            return True
        except Exception as err:
            logger.error(f"Failed to mirror raw data for {project} to GNPS2: {err}")
            return False
        finally:
            sftp.close()
            transport.close()
