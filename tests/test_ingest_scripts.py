"""Tests for the shell scripts in scripts/ingest (run with fake external commands)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SYNC = REPO / "scripts" / "ingest" / "file_transfer_sync.sh"

# Exact rsync arguments of the legacy ~/bin/sync.sh on metabo-dtn (before src/dest).
LEGACY_RSYNC_ARGS = [
    "--archive", "--no-perms", "--no-group", "--chmod=ugo=rwX", "--prune-empty-dirs",
    "--exclude=$RECYCLE.BIN/", "--exclude=filechecks/", "--exclude=Robo_copy_test/",
    "--include=*/[Oo]ther/***", "--exclude=*/*/", "--include=*/", "--include=*.raw",
    "--include=NO_UNTARGETED.TXT", "--exclude=*",
]

pytestmark = pytest.mark.skipif(not (shutil.which("flock") and shutil.which("stdbuf")),
                                reason="needs flock and stdbuf")


def _fake_rsync(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_rsync"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_OUT"\necho "sent 10 bytes"\necho "oops" >&2\n')
    fake.chmod(0o755)
    return fake


def test_file_transfer_sync_matches_legacy_rsync_call(tmp_path):
    args_out = tmp_path / "args.txt"
    env = {**os.environ, "HOME": str(tmp_path), "RSYNC": str(_fake_rsync(tmp_path)), "ARGS_OUT": str(args_out),
           "NETAPP_MOUNT": "/mnt/netapp"}
    res = subprocess.run(["bash", str(SYNC), "jgi"], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    args = args_out.read_text().splitlines()
    assert args[:len(LEGACY_RSYNC_ARGS)] == LEGACY_RSYNC_ARGS
    assert args[len(LEGACY_RSYNC_ARGS):] == [
        f"--password-file={tmp_path}/msdata_pw",
        f"--log-file={tmp_path}/logs/jgi.log",
        "/mnt/netapp/jgi/",
        "rsync://msdata@dtn04.nersc.gov:39653/jgi/",
    ]
    lines = (tmp_path / "logs" / "jgi.error.log").read_text().splitlines()
    assert len(lines) == 2
    assert all(re.match(r"^[A-Z][a-z]{2} [ 0-9]\d \d{2}:\d{2}:\d{2} ", line) for line in lines)
    assert lines[0].endswith("sent 10 bytes") and lines[1].endswith("oops")


def test_file_transfer_sync_rejects_bad_volume(tmp_path):
    res = subprocess.run(["bash", str(SYNC), "other"], capture_output=True, text=True,
                         env={**os.environ, "HOME": str(tmp_path)})
    assert res.returncode == 128 and "Usage" in res.stderr
