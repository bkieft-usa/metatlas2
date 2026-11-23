"""Tests for the host-side converter (metatlas2.ingest.conversion) with a fake ThermoRawFileParser."""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metatlas2.ingest.conversion import ConversionSettings, Converter, find_unconverted_files, run_conversion

REPO = Path(__file__).resolve().parents[1]
PROJECT = "20260101_JGI_AB_123456_Test_final_EXP120B_C18_USDAY00001"


def raw_name(i: int, pol: str = "POS") -> str:
    return f"{PROJECT}_{pol}_MS2_{i}_Sample_1_Rg80to1200-CE102040norm_Run{i}.raw"


@pytest.fixture
def logtext():
    """Capture the converter's log (metatlas2 loggers don't propagate, so caplog can't)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    log = logging.getLogger("metatlas2.ingest.conversion")
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield buf
    log.removeHandler(handler)
    log.setLevel(old_level)


@pytest.fixture
def tree(tmp_path):
    raw_data = tmp_path / "raw_data"
    proj = raw_data / "jgi" / PROJECT
    proj.mkdir(parents=True)
    return raw_data, proj


def fake_trfp(tmp_path: Path, mode: str = "ok") -> list[str]:
    """A ThermoRawFileParser stand-in: writes <stem>.mzML next to the raw file."""
    script = tmp_path / f"trfp_{mode}.py"
    script.write_text(f'''
import sys, os
args = dict(a.split("=", 1) for a in sys.argv[1:])
raw = args["-i"]; out = args["-o"]
print("2026-01-01 00:00:00 INFO Started parsing " + raw)
mode = {mode!r}
if mode == "fail":
    print("2026-01-01 00:00:01 ERROR raw file is corrupt"); sys.exit(1)
target = os.path.join(out, os.path.basename(raw)[:-4] + ".mzML")
open(target, "w").write("" if mode == "empty" else "<mzML/>")
print("2026-01-01 00:00:02 INFO Finished parsing")
''')
    return [sys.executable, str(script)]


def ok_h5(mzml: Path, h5: Path) -> dict:
    h5.write_text("h5")
    return {"ok": True, "rows": {"ms1_pos": 10, "ms2_pos": 3}}


def bad_h5(mzml: Path, h5: Path) -> dict:
    raise RuntimeError("Not a valid mzML file")


def settings(raw_data: Path, tmp_path: Path, mode: str = "ok", **kw) -> ConversionSettings:
    return ConversionSettings(raw_data_dir=str(raw_data), trfp_command=fake_trfp(tmp_path, mode),
                              group=None, max_workers=2, **kw)


def test_scan_matches_legacy_selection(tree):
    raw_data, proj = tree
    for i in range(1, 6):
        (proj / raw_name(i)).write_text("raw")
    (proj / raw_name(2)[:-4]).with_suffix(".h5").write_text("h5")       # converted
    (proj / raw_name(3)[:-4]).with_suffix(".failed").write_text("x")    # failed before
    (proj / raw_name(4)[:-4]).with_suffix(".progress").write_text("x")  # stale progress: retried
    (proj / "orphan.h5").write_text("h5")                                # no raw: ignored
    (proj / "sub").mkdir()
    (proj / "sub" / raw_name(9)).write_text("raw")                       # depth 3: ignored
    (raw_data / "jgi" / "loose.raw").write_text("raw")                   # depth 1: ignored
    other = raw_data / "elsewhere"
    other.mkdir()
    (other / raw_name(7)).write_text("raw")
    os.symlink(other, raw_data / "jgi" / "linked_project")               # symlinked dir: not followed
    scan = find_unconverted_files(raw_data / "jgi")
    assert [os.path.basename(f) for f in scan.to_convert] == [raw_name(1), raw_name(4), raw_name(5)]
    assert (scan.raw_files, scan.converted, scan.failed_before) == (5, 1, 1)
    assert scan.per_project == {PROJECT: 3}


def test_successful_conversion(tree, tmp_path):
    raw_data, proj = tree
    raw = proj / raw_name(1)
    raw.write_text("raw")
    res = Converter(settings(raw_data, tmp_path), h5_runner=ok_h5).convert_file(str(raw))
    assert res.ok and res.stage == "converted" and "raw->mzML" in res.seconds
    assert (proj / raw_name(1).replace(".raw", ".mzML")).exists()
    assert (proj / raw_name(1).replace(".raw", ".h5")).exists()
    assert not (proj / raw_name(1).replace(".raw", ".progress")).exists()


def _failed_text(proj: Path, i: int) -> str:
    return (proj / raw_name(i).replace(".raw", ".failed")).read_text()


def test_invalid_name_fails_with_error_lines(tree, tmp_path):
    raw_data, proj = tree
    raw = proj / "bad_name.raw"
    raw.write_text("raw")
    res = Converter(settings(raw_data, tmp_path), h5_runner=ok_h5).convert_file(str(raw))
    assert not res.ok and res.stage == "validate"
    text = (proj / "bad_name.failed").read_text()
    assert ", ERROR, Invalid file name" in text and "minimum of 16" in text
    assert not (proj / "bad_name.progress").exists()


def test_trfp_failure(tree, tmp_path):
    raw_data, proj = tree
    (proj / raw_name(1)).write_text("raw")
    res = Converter(settings(raw_data, tmp_path, "fail"), h5_runner=ok_h5).convert_file(str(proj / raw_name(1)))
    assert not res.ok and res.stage == "raw->mzML" and "exited with code 1" in res.error
    text = _failed_text(proj, 1)
    assert "ThermoRawFileParser: 2026-01-01 00:00:01 ERROR raw file is corrupt" in text
    assert not (proj / raw_name(1).replace(".raw", ".mzML")).exists()


def test_empty_mzml_is_marked_failed(tree, tmp_path):
    """Legacy bug: this case exited without a .failed file, so it was retried forever."""
    raw_data, proj = tree
    (proj / raw_name(1)).write_text("raw")
    res = Converter(settings(raw_data, tmp_path, "empty"), h5_runner=ok_h5).convert_file(str(proj / raw_name(1)))
    assert not res.ok and "return code was still 0" in _failed_text(proj, 1)


def test_h5_failure_moves_mzml_to_conversion_failures(tree, tmp_path):
    raw_data, proj = tree
    (proj / raw_name(1)).write_text("raw")
    res = Converter(settings(raw_data, tmp_path), h5_runner=bad_h5).convert_file(str(proj / raw_name(1)))
    assert not res.ok and res.stage == "mzML->h5"
    moved = tmp_path / "conversion_failures" / "jgi" / PROJECT / raw_name(1).replace(".raw", ".mzML")
    assert moved.exists()
    assert "ERROR, During file conversion: Not a valid mzML file" in _failed_text(proj, 1)
    assert not (proj / raw_name(1).replace(".raw", ".h5")).exists()


def test_existing_h5_is_skipped_without_leftovers(tree, tmp_path):
    raw_data, proj = tree
    (proj / raw_name(1)).write_text("raw")
    (proj / raw_name(1).replace(".raw", ".h5")).write_text("h5")
    res = Converter(settings(raw_data, tmp_path), h5_runner=ok_h5).convert_file(str(proj / raw_name(1)))
    assert res.ok and res.stage == "skipped"
    assert not (proj / raw_name(1).replace(".raw", ".progress")).exists()


def test_cancel_cleans_in_flight_files(tree, tmp_path):
    raw_data, proj = tree
    raw = proj / raw_name(1)
    raw.write_text("raw")
    conv = Converter(settings(raw_data, tmp_path))
    for ext in (".mzML", ".h5.partial", ".progress"):
        (proj / raw_name(1).replace(".raw", ext)).write_text("x")
    conv._in_flight.add(os.path.realpath(raw))
    assert conv.cancel() == [os.path.realpath(raw)]
    assert sorted(p.name for p in proj.iterdir()) == [raw_name(1)]


def test_run_conversion_end_to_end_and_lock(tree, tmp_path, logtext):
    raw_data, proj = tree
    for i in (1, 2):
        (proj / raw_name(i)).write_text("raw")
    (proj / "bad.raw").write_text("raw")
    lock = tmp_path / "logs" / ".jgi.convert.lock"
    rc = run_conversion("jgi", settings(raw_data, tmp_path), lock_path=lock, h5_runner=ok_h5)
    assert rc == 1  # bad.raw failed
    text = logtext.getvalue()
    assert "2 converted, 1 failed" in text
    assert "Successfully started file conversion" in text and "Successfully completed" in text
    # nothing left to do on the second run
    logtext.seek(0)
    logtext.truncate(0)
    assert run_conversion("jgi", settings(raw_data, tmp_path), lock_path=lock, h5_runner=ok_h5) == 0
    assert "0 to convert" in logtext.getvalue()
    # a held lock stops a concurrent run
    from metatlas2.ingest.conversion import RunLock

    held = RunLock(lock)
    assert held.acquire()
    try:
        assert run_conversion("jgi", settings(raw_data, tmp_path), lock_path=lock, h5_runner=ok_h5) == 2
    finally:
        held.release()


def test_dry_run_changes_nothing(tree, tmp_path, logtext):
    raw_data, proj = tree
    (proj / raw_name(1)).write_text("raw")
    assert run_conversion("jgi", settings(raw_data, tmp_path, dry_run=True), h5_runner=ok_h5) == 0
    assert "[dry-run] would convert" in logtext.getvalue()
    assert sorted(p.name for p in proj.iterdir()) == [raw_name(1)]


def test_conversion_report_reads_our_files(tree, tmp_path):
    """The ported email report still parses .h5/.failed/.progress files written by the converter."""
    raw_data, proj = tree
    for i in (1, 2, 3):
        (proj / raw_name(i)).write_text("raw")
    (proj / "bad.raw").write_text("raw")
    conv = Converter(settings(raw_data, tmp_path), h5_runner=ok_h5)
    for f in (raw_name(1), raw_name(2), "bad.raw"):
        conv.convert_file(str(proj / f))
    env = {**os.environ, "RAW_DATA_DIR": str(raw_data)}
    res = subprocess.run(["bash", str(REPO / "scripts" / "ingest" / "conversion_report.sh"), "jgi", "7"],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert "converted=2 failed=1 pending=1 in_progress=0" in res.stderr
    assert "Extracted ERROR lines from .failed files" in res.stdout
    assert f"Failed files:\n    {PROJECT}/bad.raw" in res.stdout
    assert "ERROR, Filename contains 1 fields but a minimum of 16 are required." in res.stdout


def test_host_modules_import_with_stdlib_only():
    """The host-side CLI must not import numpy/pandas/pymzml/tables at module level."""
    code = ("import sys; import metatlas2.ingest.run_file_conversion; "
            "bad = {'numpy','pandas','pymzml','tables','yaml'} & set(sys.modules); "
            "print(sorted(bad)); sys.exit(1 if bad else 0)")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert res.returncode == 0, res.stdout + res.stderr
