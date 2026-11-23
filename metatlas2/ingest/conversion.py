"""Host-side .raw -> .mzML -> .h5 converter (standard library only).

Port of ``metatlas-repo/utils/convert_files.sh`` + ``raw_to_h5.sh`` +
``metatlas/io/file_converter.py``. It runs on the host because
ThermoRawFileParser (TRFP) runs in its own shifter image, and shifter can't
be called from inside the metatlas2 container. The mzML -> h5 step runs in
the metatlas2 container (one shifter call per file); see
:mod:`metatlas2.ingest.h5_writer`.

What is unchanged (so the conversion email report and downstream tools keep working):

* Files are picked up from ``{raw_data_dir}/{subdir}/{project}/*.raw``
  (exactly two levels deep, regular files only) when there is no ``.h5`` and
  no ``.failed`` next to them.
* Outputs land next to the .raw: ``X.mzML`` (TRFP 1.4.2, ``-f=1``) and ``X.h5``.
* While a file is being converted a ``X.progress`` file holds its log (group
  ``metatlas``, mode 640). On success it is deleted. On failure it is renamed
  ``X.failed`` (this stops retries) and contains ``ERROR`` lines.
* When the h5 step fails, the mzML is moved to the matching path under
  ``conversion_failures/`` (``raw_data`` -> ``conversion_failures``).

What changed:

* The TRFP image is pulled once per run instead of once per file.
* File-name validation runs in-process; the old code started a container per file for this.
* A missing or empty mzML now marks the file ``.failed``. The old code exited
  without doing that, so the file was retried every 5 minutes forever.
* The h5 is written under a temporary name, so a killed run never leaves a
  half-written ``.h5`` that counts as converted.
* One run log with per-file stage timings and a summary. TRFP's verbose
  output goes only to the per-file ``.progress``/``.failed`` files.
* A lock file prevents two runs for the same subdirectory overlapping.
* No record is written to the legacy metatlas MySQL database.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from metatlas2.shared_tools.file_and_project_format import validate_file_name_minimal

logger = logging.getLogger("metatlas2.ingest.conversion")

#: ThermoRawFileParser 1.4.2 biocontainer (same digest as the legacy converter).
TRFP_IMAGE = (
    "quay.io/biocontainers/thermorawfileparser"
    "@sha256:3b930ef774b3d4e0d559f38903da2390f9b24b96a016a1761805b88ae78c2b40"
)

H5Runner = Callable[[Path, Path], dict]


@dataclass
class ConversionSettings:
    """Settings for one converter run.

    Attributes:
        raw_data_dir: Root holding ``{subdir}/{project}/*.raw``.
        trfp_image: Shifter image reference for ThermoRawFileParser.
        h5_command: Command prefix that runs ``metatlas2.ingest.h5_writer`` (the
            mzML path is appended). ``None`` runs the writer in this process
            (needs numpy/pymzml/tables, e.g. inside the container or a venv).
        max_workers: Files converted concurrently.
        group: Unix group for ``.progress``/``.failed`` files (None to skip).
        validate_names: Fail files whose names don't pass the minimal checks.
        pull_image: Run ``shifterimg pull`` for the TRFP image once per run.
        step_timeout_seconds: Optional per-step timeout (None = no limit, as before).
        dry_run: Only report what would be converted.
    """

    raw_data_dir: str = "/global/cfs/cdirs/metatlas/raw_data"
    trfp_image: str = TRFP_IMAGE
    h5_command: list[str] | None = None
    max_workers: int = 2
    group: str | None = "metatlas"
    validate_names: bool = True
    pull_image: bool = True
    step_timeout_seconds: float | None = None
    dry_run: bool = False
    trfp_command: list[str] | None = None  # override for tests; default uses shifter


@dataclass
class FileResult:
    raw_file: str
    ok: bool
    stage: str = ""
    error: str = ""
    seconds: dict[str, float] = field(default_factory=dict)
    h5_summary: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    projects: int = 0
    raw_files: int = 0
    converted: int = 0
    failed_before: int = 0
    to_convert: list[str] = field(default_factory=list)
    per_project: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def find_unconverted_files(base_dir: str | os.PathLike) -> ScanResult:
    """Find ``{base_dir}/{project}/X.raw`` files with no ``X.h5`` and no ``X.failed``.

    Matches the legacy ``find -mindepth 2 -maxdepth 2 -type f`` selection:
    symlinked project directories are not followed and only regular files count.
    """
    result = ScanResult()
    base = Path(base_dir)
    try:
        with os.scandir(base) as it:
            projects = sorted((e for e in it if e.is_dir(follow_symlinks=False)), key=lambda e: e.name)
    except OSError as err:
        logger.error(f"Cannot list {base}: {err}")
        return result
    for project in projects:
        result.projects += 1
        raws: list[str] = []
        done: set[str] = set()
        failed: set[str] = set()
        try:
            with os.scandir(project.path) as it:
                for entry in it:
                    name = entry.name
                    if not name.endswith((".raw", ".h5", ".failed")):
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    stem, ext = os.path.splitext(name)
                    if ext == ".raw":
                        raws.append(stem)
                    elif ext == ".h5":
                        done.add(stem)
                    else:
                        failed.add(stem)
        except OSError as err:
            logger.warning(f"Cannot list project directory {project.path}: {err}")
            continue
        result.raw_files += len(raws)
        for stem in raws:
            if stem in failed:
                result.failed_before += 1
            elif stem in done:
                result.converted += 1
            else:
                result.to_convert.append(os.path.join(project.path, stem + ".raw"))
                result.per_project[project.name] = result.per_project.get(project.name, 0) + 1
    result.to_convert.sort()
    return result


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------


class _ProgressLog:
    """Writes timestamped lines to the per-file ``.progress`` file (legacy ``ts`` format)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, level: str, message: str) -> None:
        stamp = datetime.now()
        with open(self.path, "a") as fh:
            for line in str(message).splitlines() or [""]:
                fh.write(f"{stamp}, {level}, {line}\n")


class _Cancelled(Exception):
    pass


class Converter:
    """Runs the raw -> mzML -> h5 pipeline for many files with a worker pool."""

    def __init__(self, settings: ConversionSettings, h5_runner: H5Runner | None = None) -> None:
        self.settings = settings
        self._h5_runner = h5_runner
        self._stop = threading.Event()
        self._procs: dict[str, subprocess.Popen] = {}
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def paths(raw_file: str) -> dict[str, Path]:
        stem = raw_file[: -len(".raw")] if raw_file.endswith(".raw") else raw_file
        return {k: Path(stem + ext) for k, ext in (
            ("raw", ".raw"), ("mzml", ".mzML"), ("h5", ".h5"), ("h5_partial", ".h5.partial"),
            ("progress", ".progress"), ("failed", ".failed"))}

    def _set_group(self, path: Path, mode: int) -> None:
        try:
            if self.settings.group:
                shutil.chown(path, group=self.settings.group)
            os.chmod(path, mode)
        except (OSError, LookupError) as err:
            logger.debug(f"Could not set group/mode on {path}: {err}")

    def _run(self, raw_file: str, cmd: list[str], progress: _ProgressLog, tag: str) -> tuple[int, list[str]]:
        """Run *cmd*, streaming its output into the progress file. Returns (rc, lines)."""
        if self._stop.is_set():
            raise _Cancelled()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True)
        with self._lock:
            self._procs[raw_file] = proc
        lines: list[str] = []
        timer = None
        if self.settings.step_timeout_seconds:
            timer = threading.Timer(self.settings.step_timeout_seconds, self._kill, args=(proc,))
            timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                lines.append(line)
                progress.write("INFO" if "ERROR" not in line.upper() else "ERROR", f"{tag}: {line}")
            rc = proc.wait()
        finally:
            if timer:
                timer.cancel()
            with self._lock:
                self._procs.pop(raw_file, None)
        if self._stop.is_set():
            raise _Cancelled()
        return rc, lines

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass

    def _trfp_cmd(self, raw: Path) -> list[str]:
        if self.settings.trfp_command is not None:
            return [*self.settings.trfp_command, f"-i={raw}", f"-o={raw.parent}", "-f=1"]
        return ["shifter", "--clearenv", "--module=none", f"--image={self.settings.trfp_image}",
                "ThermoRawFileParser.sh", f"-i={raw}", f"-o={raw.parent}", "-f=1"]

    def _h5_step(self, raw_file: str, mzml: Path, h5: Path, progress: _ProgressLog) -> dict:
        if self._h5_runner is not None:
            return self._h5_runner(mzml, h5)
        if self.settings.h5_command is None:
            from metatlas2.ingest.h5_writer import mzml_to_h5

            return {"ok": True, **mzml_to_h5(mzml, h5)}
        rc, lines = self._run(raw_file, [*self.settings.h5_command, str(mzml), "--h5", str(h5)], progress, "h5")
        summary: dict = {}
        for line in reversed(lines):
            if line.startswith("{"):
                try:
                    summary = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if rc != 0 or not summary.get("ok"):
            raise RuntimeError(summary.get("error") or f"h5 writer exited with code {rc}")
        return summary

    # -- one file --------------------------------------------------------------
    def convert_file(self, raw_file: str) -> FileResult:
        """Convert one .raw file. Never raises (except on cancellation)."""
        raw_file = os.path.realpath(raw_file)
        p = self.paths(raw_file)
        result = FileResult(raw_file=raw_file, ok=False)
        with self._lock:
            self._in_flight.add(raw_file)
        try:
            return self._convert_file(raw_file, p, result)
        finally:
            with self._lock:
                self._in_flight.discard(raw_file)

    def _convert_file(self, raw_file: str, p: dict[str, Path], result: FileResult) -> FileResult:
        p["progress"].unlink(missing_ok=True)
        p["progress"].touch()
        self._set_group(p["progress"], 0o640)
        progress = _ProgressLog(p["progress"])
        if p["h5"].exists() or p["failed"].exists():
            what = ".h5" if p["h5"].exists() else ".failed"
            progress.write("INFO", f"{what} file found for {raw_file} - skipping conversion.")
            p["progress"].unlink(missing_ok=True)
            result.ok, result.stage = True, "skipped"
            return result

        def fail(stage: str, message: str, move_mzml: bool = False) -> FileResult:
            progress.write("ERROR", message)
            if move_mzml and p["mzml"].exists() and "exists but it can not be written" not in message:
                dest = Path(str(p["mzml"]).replace("raw_data", "conversion_failures"))
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p["mzml"]), str(dest))
                    progress.write("ERROR", f"Moved mzML file to {dest}")
                except OSError as err:
                    progress.write("ERROR", f"Could not move mzML to {dest}: {err}")
            for key in ("mzml", "h5", "h5_partial"):
                p[key].unlink(missing_ok=True)
            os.replace(p["progress"], p["failed"])
            result.stage, result.error = stage, message
            return result

        # 1. file name validation
        if self.settings.validate_names:
            errors = validate_file_name_minimal(raw_file)
            if errors:
                for err in errors:
                    progress.write("ERROR", err)
                return fail("validate", f"Invalid file name: {raw_file}")

        # 2. raw -> mzML
        t0 = time.time()
        progress.write("INFO", f"Converting {raw_file} to mzML with ThermoRawFileParser")
        try:
            rc, lines = self._run(raw_file, self._trfp_cmd(p["raw"]), progress, "ThermoRawFileParser")
        except OSError as err:
            return fail("raw->mzML", f"Could not start ThermoRawFileParser: {err}")
        result.seconds["raw->mzML"] = round(time.time() - t0, 1)
        if rc != 0:
            last = next((l for l in reversed(lines) if l.strip()), "")
            return fail("raw->mzML", f"ThermoRawFileParser exited with code {rc}: {last}")
        if not p["mzml"].exists() or p["mzml"].stat().st_size == 0:
            return fail("raw->mzML", f"raw to mzml conversion failed for {raw_file}, but return code was still 0.")

        # 3. mzML -> h5
        t0 = time.time()
        progress.write("INFO", f"Converting mzML file {p['mzml']}")
        progress.write("INFO", f"Generating h5 file: {p['h5']}")
        try:
            summary = self._h5_step(raw_file, p["mzml"], p["h5"], progress)
        except _Cancelled:
            raise
        except Exception as err:
            result.seconds["mzML->h5"] = round(time.time() - t0, 1)
            return fail("mzML->h5", f"During file conversion: {err}", move_mzml=True)
        result.seconds["mzML->h5"] = round(time.time() - t0, 1)
        if not p["h5"].exists():
            return fail("mzML->h5", f"h5 file was not created: {p['h5']}", move_mzml=True)
        result.h5_summary = summary
        progress.write("INFO", "raw to h5 conversion sucessfully completed.")
        p["progress"].unlink(missing_ok=True)
        result.ok, result.stage = True, "converted"
        return result

    # -- cancellation ----------------------------------------------------------
    def cancel(self) -> list[str]:
        """Stop everything and clean up in-flight files (SIGTERM handling). Returns the files cleaned."""
        self._stop.set()
        with self._lock:
            procs = list(self._procs.values())
            in_flight = sorted(self._in_flight)
        for proc in procs:
            self._kill(proc)
        for raw_file in in_flight:
            p = self.paths(raw_file)
            for key in ("mzml", "h5_partial", "progress"):
                p[key].unlink(missing_ok=True)
            if not p["h5"].exists():
                p["mzml"].unlink(missing_ok=True)
        return in_flight

    # -- many files ------------------------------------------------------------
    def convert_many(self, raw_files: list[str]) -> list[FileResult]:
        """Convert *raw_files* with ``max_workers`` threads, logging progress."""
        total = len(raw_files)
        results: list[FileResult] = []
        with ThreadPoolExecutor(max_workers=max(1, self.settings.max_workers)) as pool:
            futures = {pool.submit(self.convert_file, f): f for f in raw_files}
            try:
                for n, fut in enumerate(as_completed(futures), start=1):
                    raw = futures[fut]
                    try:
                        res = fut.result()
                    except _Cancelled:
                        continue
                    except Exception as err:  # unexpected bug: report, keep going
                        logger.exception(f"[{n}/{total}] ERROR unexpected failure for {raw}")
                        res = FileResult(raw_file=raw, ok=False, stage="internal", error=str(err))
                    results.append(res)
                    _log_file_result(res, n, total)
            except KeyboardInterrupt:
                for fut in futures:
                    fut.cancel()
                raise
        return results


def _fmt_rows(summary: dict) -> str:
    rows = summary.get("rows") or {}
    return ", ".join(f"{k}={v:,}" for k, v in rows.items() if v)


def _log_file_result(res: FileResult, n: int, total: int) -> None:
    name = os.path.basename(res.raw_file)
    timing = ", ".join(f"{k} {v:.1f}s" for k, v in res.seconds.items())
    if res.ok and res.stage == "converted":
        logger.info(f"[{n}/{total}] OK {name} ({timing}; {_fmt_rows(res.h5_summary) or 'no rows'})")
    elif res.ok:
        logger.info(f"[{n}/{total}] SKIPPED {name} (already has .h5 or .failed)")
    else:
        logger.error(f"[{n}/{total}] FAILED at {res.stage}: {name}: {res.error}"
                     + (f" ({timing})" if timing else ""))


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


def pull_trfp_image(image: str) -> bool:
    """``shifterimg pull`` the TRFP image once (logs and continues on failure)."""
    t0 = time.time()
    try:
        res = subprocess.run(["shifterimg", "pull", image], capture_output=True, text=True, check=False)
    except OSError as err:
        logger.warning(f"Could not run shifterimg pull: {err}")
        return False
    if res.returncode != 0:
        logger.warning(f"shifterimg pull failed ({res.returncode}): {(res.stderr or res.stdout).strip()[-300:]}")
        return False
    logger.info(f"ThermoRawFileParser image ready ({time.time() - t0:.0f}s)")
    return True


class RunLock:
    """Non-blocking exclusive lock (``flock``) so two runs can't overlap."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            self._fh = None
            return False
        return True

    def release(self) -> None:
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def run_conversion(subdir: str, settings: ConversionSettings, lock_path: str | os.PathLike | None = None,
                   h5_runner: H5Runner | None = None) -> int:
    """Convert every pending .raw under ``{raw_data_dir}/{subdir}``.

    Returns:
        Process exit code: 0 if nothing failed, 1 if any file failed, 2 if
        another run holds the lock.
    """
    t_start = time.time()
    base_dir = os.path.join(settings.raw_data_dir, subdir)
    logger.info("=" * 78)
    logger.info(f"Successfully started file conversion for {subdir} on {datetime.now()}:")
    lock = RunLock(lock_path) if lock_path else None
    if lock and not lock.acquire():
        logger.info(f"Another conversion run for {subdir} holds {lock_path}; exiting.")
        return 2
    converter = Converter(settings, h5_runner=h5_runner)
    previous = signal.getsignal(signal.SIGTERM)

    def on_term(signum, frame):  # noqa: ARG001
        cleaned = converter.cancel()
        logger.warning(f"Received signal {signum}; cancelled. Cleaned up in-flight files: {cleaned}")
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    try:
        t0 = time.time()
        scan = find_unconverted_files(base_dir)
        logger.info(
            f"Scanned {scan.projects} project directories in {base_dir} ({time.time() - t0:.1f}s): "
            f"{scan.raw_files} .raw files, {scan.converted} already converted, "
            f"{scan.failed_before} previously failed, {len(scan.to_convert)} to convert.")
        for project, count in sorted(scan.per_project.items()):
            logger.info(f"    {count:5d} to convert in {project}")
        if not scan.to_convert:
            return 0
        if settings.dry_run:
            for f in scan.to_convert:
                logger.info(f"[dry-run] would convert {f}")
            return 0
        if settings.pull_image and settings.trfp_command is None:
            pull_trfp_image(settings.trfp_image)
        logger.info(f"Converting {len(scan.to_convert)} files with {settings.max_workers} workers")
        try:
            results = converter.convert_many(scan.to_convert)
        except KeyboardInterrupt:
            logger.warning("Conversion run interrupted.")
            return 1
        converted = [r for r in results if r.ok and r.stage == "converted"]
        failed = [r for r in results if not r.ok]
        elapsed = time.time() - t_start
        logger.info(
            f"Summary for {subdir}: {len(converted)} converted, {len(failed)} failed, "
            f"{len(results) - len(converted) - len(failed)} skipped in {elapsed / 60:.1f} min")
        for r in failed:
            logger.error(f"    FAILED ({r.stage}) {r.raw_file}: {r.error}")
        return 1 if failed else 0
    finally:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, previous)
        if lock:
            lock.release()
        logger.info(f"Successfully completed file conversion for {subdir} on {datetime.now()}:")
