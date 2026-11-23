"""End-to-end tests of the untargeted pipeline with in-memory LIMS and fake GNPS2/rclone/Slurm."""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from metatlas2.shared_tools import gdrive_upload as gdu
from metatlas2.untargeted.run_untargeted_pipeline import main as cli_main
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.lims import InMemoryLims
from metatlas2.untargeted.pipeline import parse_skip_steps, run_untargeted_pipeline
from metatlas2.untargeted.status import PolarityTask
from metatlas2.untargeted.status_report import REPORT_COLUMNS, format_status_table, get_untargeted_status
from untargeted_fixtures import PROJECT_JGI, lcmsruns_df, make_config, make_raw_project, simulate_mzmine_outputs

REPO = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000.0


class FakeGnps2:
    """Records submissions; status per task is controlled by the test."""

    user_upload_prefix = "USERUPLOAD/u"

    def __init__(self):
        self.submitted: list[dict] = []
        self.status: dict[str, str] = {}
        self.mirrored: list[tuple] = []

    def submit(self, params):
        task_id = f"{len(self.submitted):032x}"
        self.submitted.append(params)
        self.status[task_id] = "RUNNING"
        return {"task": task_id}

    def check_status(self, task_id):
        return self.status.get(task_id, "N/A"), "v1"

    def mirror_mzmine_results(self, project, polarity, output_dir):
        self.mirrored.append(("results", project, polarity))
        return True

    def mirror_raw_data(self, project, local_dir, mode="pipeline"):
        self.mirrored.append(("raw", project, mode))
        return True

    def status_url(self, t):
        return f"https://gnps2.org/status?task={t}"

    def graphml_url(self, t):
        return f"graphml:{t}"

    def library_results_url(self, t):
        return f"library:{t}"

    def download(self, url, target):
        Path(target).write_text(url)
        return True


@pytest.fixture
def drive(monkeypatch):
    """Fake Google Drive remote: rclone list/copy/run operate on a dict."""
    files: dict[str, str] = {}

    def fake_copy(source, drive_name, dest, overwrite=False, **kw):
        files[Path(source).name] = "2026-11-20 10:00:00.000"

    def fake_list(remote, long=False, max_depth=1):
        if long:
            return [["1", *files[n].split(" "), n] for n in files]
        return [["1", n] for n in files]

    def fake_run(args):
        if args[0] == "copy":  # documentation download
            Path(args[-1], Path(args[-2]).name).write_text("guide")
        return ""

    monkeypatch.setattr(gdu, "_rclone_copy", fake_copy)
    monkeypatch.setattr(gdu, "rclone_list", fake_list)
    monkeypatch.setattr(gdu, "rclone_run", fake_run)
    return files


def _status(ctx):
    return ctx.lims.get_table("untargeted_tasks").iloc[0]


def test_full_lifecycle(tmp_path, drive):
    cfg = make_config(tmp_path)
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    lims = InMemoryLims({"lcmsrun_plus": lcmsruns_df(PROJECT_JGI, names, raw, "jgi", NOW - 5 * 3600),
                         "untargeted_tasks": pd.DataFrame(columns=["parent_dir"])})
    fake = FakeGnps2()
    queue = tmp_path / "queue.tsv"
    ctx = RunContext(config=cfg, lims=lims, gnps2=fake, sbatch_queue_file=str(queue), clock=lambda: NOW)

    # Run 1: project created and MZmine queued for both polarities.
    assert run_untargeted_pipeline(ctx)
    row = _status(ctx)
    assert (row["mzmine_pos_status"], row["mzmine_neg_status"]) == ("04 running", "04 running")
    assert (row["fbmn_pos_status"], row["fbmn_neg_status"]) == ("13 waiting", "13 waiting")
    assert len(queue.read_text().splitlines()) == 2

    # Host wrapper submits; MZmine runs and finishes.
    tasks = [PolarityTask(cfg["paths"]["output_dir"], PROJECT_JGI, p) for p in ("positive", "negative")]
    for i, t in enumerate(tasks):
        t.path("mzmine_job_id").write_text(f"{t.prefix}={100 + i}\n")
        t.path("mzmine_status").write_text(f"RUNNING {100 + i} {NOW:.0f}\nDONE {NOW:.0f}\n")
        simulate_mzmine_outputs(t)
    queue.unlink()

    # Run 2: MZmine complete -> FBMN + Everything Bagel submitted.
    assert run_untargeted_pipeline(ctx)
    row = _status(ctx)
    assert row["mzmine_pos_status"] == "07 complete" and row["num_pos_msms"] == 3
    assert row["fbmn_pos_status"] == "04 running"
    workflows = sorted(p["workflowname"] for p in fake.submitted)
    assert workflows == ["everything_bagel_workflow"] * 2 + ["feature_based_molecular_networking_workflow"] * 2
    fbmn = next(p for p in fake.submitted if p["description"] == f"{PROJECT_JGI}_positive")
    assert fbmn["inputspectra"] == f"USERUPLOAD/u/untargeted_tasks/{PROJECT_JGI}_positive/{PROJECT_JGI}_positive.mgf"
    assert fbmn["input_raw_spectra"] == f"USERUPLOAD/u/raw_data/jgi/{PROJECT_JGI}"
    assert all(t.exists("fbmn_task") and t.exists("eb_task") for t in tasks)
    assert not queue.exists()  # nothing resubmitted

    # GNPS2 finishes.
    for k in fake.status:
        fake.status[k] = "DONE"

    # Run 3: FBMN complete -> downloads -> zip -> upload.
    assert run_untargeted_pipeline(ctx)
    row = _status(ctx)
    assert row["fbmn_pos_status"] == "07 complete" and row["fbmn_neg_status"] == "07 complete"
    archive = Path(cfg["paths"]["download_dir"]) / f"{PROJECT_JGI}.zip"
    assert archive.exists() and f"{PROJECT_JGI}.zip" in drive
    names_in_zip = zipfile.ZipFile(archive).namelist()
    assert "Untargeted_metabolomics_GNPS2_Guide.docx" in names_in_zip
    assert any(n.endswith("_positive_gnps2-fbmn-library-results.tsv") for n in names_in_zip)
    assert not any("mzmine-status" in n or "mzmine-job-id" in n for n in names_in_zip)
    assert all(n.startswith("20260101_JGI_AB_123456_C18") or n.endswith(".docx") for n in names_in_zip)

    # Status report reflects the upload.
    table = get_untargeted_status(ctx)
    assert list(table.columns) == REPORT_COLUMNS
    assert table.iloc[0]["gdrive_status"] == "08 uploaded"
    line = format_status_table(table).splitlines()[1]
    assert re.match(r"^\d", line)
    assert "07 complete\t07 complete\t07 complete\t07 complete\t08 uploaded" in line


def test_step5_insufficient_msms_is_written_to_lims(tmp_path):
    """Legacy bug: '12 not relevant' set on errored FBMN tasks was never saved."""
    cfg = make_config(tmp_path)
    out = cfg["paths"]["output_dir"]
    PolarityTask(out, PROJECT_JGI, "positive").directory.mkdir()
    row = {"Key": 1, "parent_dir": PROJECT_JGI, "output_dir": out,
           "mzmine_pos_status": "07 complete", "mzmine_neg_status": "12 not relevant",
           "fbmn_pos_status": "09 error", "fbmn_neg_status": "12 not relevant",
           "num_pos_files": 5, "num_neg_files": 0, "num_pos_msms": 1, "num_neg_msms": 0}
    ctx = RunContext(config=cfg, lims=InMemoryLims({"untargeted_tasks": pd.DataFrame([row])}),
                     gnps2=FakeGnps2(), clock=lambda: NOW)
    run_untargeted_pipeline(ctx, skip_steps="Sync,MZmine_Status,MZmine_Submit,FBMN_Status,FBMN_Download,Zip_and_Upload")
    assert _status(ctx)["fbmn_pos_status"] == "12 not relevant"


def test_failing_step_does_not_stop_later_steps(tmp_path, monkeypatch):
    import metatlas2.untargeted.pipeline as pl

    cfg = make_config(tmp_path)
    ctx = RunContext(config=cfg, lims=InMemoryLims({"untargeted_tasks": pd.DataFrame(columns=["parent_dir"])}))
    called = []
    monkeypatch.setattr(pl, "update_new_untargeted_tasks", lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pl, "zip_and_upload_untargeted_results", lambda c: called.append("zip"))
    assert run_untargeted_pipeline(ctx) is False
    assert called == ["zip"]


def test_parse_skip_steps():
    assert parse_skip_steps("Sync,FBMN_Status") == {"Sync", "FBMN_Status"}
    with pytest.raises(ValueError):
        parse_skip_steps(["Nope"])


def test_cli_dry_run_makes_no_changes(tmp_path, monkeypatch, drive):
    cfg = make_config(tmp_path)
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    lims = InMemoryLims({"lcmsrun_plus": lcmsruns_df(PROJECT_JGI, names, raw, "jgi", 0),
                         "untargeted_tasks": pd.DataFrame(columns=["parent_dir"])})
    import metatlas2.untargeted.lims as lims_mod

    def fake_from_config(cls, c, dry_run=False):
        lims.dry_run = dry_run  # mirror LimsClient.from_config honouring dry_run
        return lims

    monkeypatch.setattr(lims_mod.LimsClient, "from_config", classmethod(fake_from_config))
    rc = cli_main(["run", "--dry_run", "--raw_data_dir", str(raw), "--output_dir", cfg["paths"]["output_dir"],
                   "--download_dir", cfg["paths"]["download_dir"], "--log_file", str(tmp_path / "log.txt")])
    assert rc == 0
    lims.dry_run = False
    assert lims.get_table("untargeted_tasks").empty
    assert not list(Path(cfg["paths"]["output_dir"]).iterdir())
    log = (tmp_path / "log.txt").read_text()
    assert "Successfully started run_untargeted_pipeline.py" in log
    assert "Successfully completed run_untargeted_pipeline.py" in log
    assert "[dry-run] Would write metadata" in log
    assert lims.writes and lims.writes[0][1] == "insert"  # attempted, but not applied


def test_wrapper_drains_sbatch_queue(tmp_path):
    """Run the _drain_sbatch_queue bash function from scripts/metatlas2.sh with a fake sbatch."""
    script = (REPO / "scripts" / "metatlas2.sh").read_text()
    func = re.search(r"^_drain_sbatch_queue\(\) \{.*?^\}", script, re.S | re.M).group(0)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sbatch").write_text("#!/bin/bash\necho '4242;perlmutter'\n")
    (bindir / "sbatch").chmod(0o755)
    queue = tmp_path / "q.tsv"
    job_file = tmp_path / "P_positive_mzmine-job-id.txt"
    queue.write_text(f"{tmp_path}/P_positive_mzmine-sbatch.sbatch\t{job_file}\tP_positive\n")
    cmd = f'{func}\n_drain_sbatch_queue "{queue}"'
    res = subprocess.run(["bash", "-c", cmd], env={"PATH": f"{bindir}:/usr/bin:/bin"},
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert job_file.read_text() == "P_positive=4242\n"
    assert not queue.exists()
