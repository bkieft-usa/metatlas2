"""Tests for metatlas2.untargeted.mzmine (templates, sbatch, job markers, Steps 2-3)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from metatlas2.untargeted.config import load_untargeted_config
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.lims import InMemoryLims
from metatlas2.untargeted.mzmine import (
    build_sbatch_script,
    mzmine_job_failure_reason,
    parse_slurm_time,
    select_batch_template,
    submit_mzmine_jobs,
    update_mzmine_status,
    write_mzmine_sbatch_and_runner,
)
from metatlas2.untargeted.status import PolarityTask
from untargeted_fixtures import PROJECT_JGI, make_config, simulate_mzmine_outputs

NOW = 2_000_000_000.0
NEW = pd.DataFrame({"parent_dir": [PROJECT_JGI]})  # Step 1 output


def _task_row(output_dir, key=1, pos="01 initiation", neg="12 not relevant", **extra):
    row = {
        "Key": key, "parent_dir": PROJECT_JGI, "output_dir": output_dir,
        "mzmine_pos_status": pos, "mzmine_neg_status": neg,
        "fbmn_pos_status": "13 waiting", "fbmn_neg_status": "12 not relevant",
        "num_pos_files": 6, "num_neg_files": 0,
        "num_pos_features": 0, "num_neg_features": 0, "num_pos_features_in_exctrl": 0,
        "num_neg_features_in_exctrl": 0, "num_pos_msms": 0, "num_neg_msms": 0,
    }
    row.update(extra)
    return row


@pytest.fixture
def setup(tmp_path):
    cfg = make_config(tmp_path)
    task = PolarityTask(cfg["paths"]["output_dir"], PROJECT_JGI, "positive")
    task.directory.mkdir()
    write_mzmine_sbatch_and_runner(task, cfg["mzmine"])
    task.path("metadata").write_text("filename\tATTRIBUTE_sampletype\n")
    return cfg, task


def _ctx(cfg, rows, **kw):
    lims = InMemoryLims({"untargeted_tasks": pd.DataFrame(rows)})
    return RunContext(config=cfg, lims=lims, clock=lambda: NOW, **kw)


def test_select_batch_template():
    cfg = load_untargeted_config()["mzmine"]
    assert select_batch_template("IQX", cfg)[0].endswith("/IQX-mzmine-3.7.2-batchparams.xml")
    assert select_batch_template("EXP120B", cfg) == (
        f"{cfg['batch_file_dir']}/mzmine-3.7.2-batchparams.xml", 2)
    assert select_batch_template("QE144", cfg)[1] == 2
    assert select_batch_template("Exploris", cfg)[1] == 2
    assert select_batch_template("WEIRD", cfg)[1] == 5
    assert select_batch_template(None, cfg)[1] == 5


@pytest.mark.parametrize("value,seconds", [("3:00:00", 10800), ("6:00:00", 21600), ("30", 1800),
                                           ("1-00:00:00", 86400), ("10:30", 630)])
def test_parse_slurm_time(value, seconds):
    assert parse_slurm_time(value) == seconds


def test_sbatch_script_header_matches_legacy(setup):
    cfg, task = setup
    text = build_sbatch_script(task, cfg["mzmine"])
    prefix = task.directory / task.prefix
    assert text.startswith("#!/bin/bash\n#SBATCH -N 1\n#SBATCH --exclusive\n"
                           f'#SBATCH --error="{prefix}-mzmine.err"\n#SBATCH --output="{prefix}-mzmine.out"\n'
                           "#SBATCH -A m2650\n#SBATCH -C cpu\n#SBATCH --qos=regular\n#SBATCH -t 3:00:00\n")
    assert "/opt/MZmine/bin/MZmine -threads auto -t /tmp -batch" in text
    assert "exit $rc" in text


@pytest.mark.parametrize("status_text,queued,job_id,expect_dead", [
    ("RUNNING 123 {now_minus_1h}\nFAILED 1 {now}\n", True, True, True),
    ("RUNNING 123 {now_minus_1h}\nDONE {now}\n", True, True, True),
    ("RUNNING 123 {now_minus_1h}\n", True, True, False),
    ("RUNNING 123 {now_minus_5h}\n", True, True, True),  # 3h limit + 1h grace exceeded
    (None, True, False, True),   # queued >1h ago but never submitted
    (None, True, True, False),   # submitted, pending
    (None, False, True, False),  # legacy job, fresh job-id file
    (None, False, False, True),  # nothing at all
])
def test_failure_reason(setup, status_text, queued, job_id, expect_dead):
    cfg, task = setup
    fmt = {"now": int(NOW), "now_minus_1h": int(NOW - 3600), "now_minus_5h": int(NOW - 5 * 3600)}
    if status_text:
        task.path("mzmine_status").write_text(status_text.format(**fmt))
    if queued:
        task.path("mzmine_queued").write_text(f"{NOW - 2 * 3600:.0f}\n")
    if job_id:
        task.path("mzmine_job_id").write_text(f"{task.prefix}=123\n")
        os.utime(task.path("mzmine_job_id"), (NOW - 60, NOW - 60))
    reason = mzmine_job_failure_reason(task, cfg["mzmine"], NOW)
    assert (reason is not None) == expect_dead, reason


def test_step3_queues_job_and_sets_running(setup, tmp_path):
    cfg, task = setup
    queue = tmp_path / "queue.tsv"
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"])], sbatch_queue_file=str(queue))
    submit_mzmine_jobs(ctx, NEW)
    line = queue.read_text().strip().split("\t")
    assert line == [str(task.path("mzmine_sbatch")), str(task.path("mzmine_job_id")), task.prefix]
    assert task.path("mzmine_queued").exists()
    assert ctx.lims.get_table("untargeted_tasks").loc[0, "mzmine_pos_status"] == "04 running"


def test_step3_ignores_stale_initiation_rows(setup, tmp_path):
    """Projects left in '01 initiation' by earlier runs are not auto-submitted (legacy behaviour)."""
    cfg, task = setup
    queue = tmp_path / "queue.tsv"
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"])], sbatch_queue_file=str(queue))
    submit_mzmine_jobs(ctx, pd.DataFrame({"parent_dir": ["some_other_project"]}))
    assert not queue.exists()
    ctx.direct_input = [PROJECT_JGI]
    submit_mzmine_jobs(ctx)
    assert queue.exists()


def test_step3_dry_run_queues_nothing(setup, tmp_path):
    cfg, task = setup
    queue = tmp_path / "queue.tsv"
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"])], sbatch_queue_file=str(queue), dry_run=True)
    submit_mzmine_jobs(ctx, NEW)
    assert not queue.exists() and not task.path("mzmine_queued").exists()
    assert ctx.lims.get_table("untargeted_tasks").loc[0, "mzmine_pos_status"] == "01 initiation"


def test_step3_resubmits_own_errored_job_with_longer_time_until_cap(setup, tmp_path):
    cfg, task = setup
    queue = tmp_path / "queue.tsv"
    task.path("mzmine_queued").write_text(f"{NOW - 86400:.0f}\n")
    task.path("mzmine_status").write_text("FAILED 1 0\n")
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="09 error")], sbatch_queue_file=str(queue))
    submit_mzmine_jobs(ctx)
    assert "#SBATCH -t 6:00:00" in task.path("mzmine_sbatch").read_text()
    assert not task.path("mzmine_status").exists()  # stale markers cleared
    assert ctx.lims.get_table("untargeted_tasks").loc[0, "mzmine_pos_status"] == "04 running"
    # second failure: retry cap (1) reached -> not resubmitted again
    queue.unlink()
    ctx.lims.update_table(pd.DataFrame([{"Key": 1, "mzmine_pos_status": "09 error"}]), "untargeted_tasks")
    submit_mzmine_jobs(ctx)
    assert not queue.exists()


def test_step3_does_not_resubmit_legacy_errored_projects(setup, tmp_path):
    cfg, task = setup
    queue = tmp_path / "queue.tsv"
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="09 error")], sbatch_queue_file=str(queue))
    submit_mzmine_jobs(ctx)
    assert not queue.exists()
    # ...but --direct_input does
    ctx.direct_input = [PROJECT_JGI]
    submit_mzmine_jobs(ctx)
    assert queue.exists()


def test_step3_error_without_files_becomes_not_relevant(setup, tmp_path):
    cfg, task = setup
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="09 error", num_pos_files=0)],
               sbatch_queue_file=str(tmp_path / "q.tsv"))
    submit_mzmine_jobs(ctx)
    row = ctx.lims.get_table("untargeted_tasks").iloc[0]
    assert row["mzmine_pos_status"] == "12 not relevant" and row["fbmn_pos_status"] == "12 not relevant"


def test_step2_completes_with_counts_and_filtering(setup):
    cfg, task = setup
    simulate_mzmine_outputs(task)
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="04 running")])
    update_mzmine_status(ctx)
    row = ctx.lims.get_table("untargeted_tasks").iloc[0]
    assert row["mzmine_pos_status"] == "07 complete"
    assert row["num_pos_features"] == 5 and row["num_pos_features_in_exctrl"] == 4 and row["num_pos_msms"] == 3
    assert task.filtered_peak_height("3x-exctrl").exists()


def test_step2_marks_dead_job_as_error(setup):
    cfg, task = setup
    task.path("mzmine_queued").write_text(f"{NOW - 7200:.0f}\n")
    task.path("mzmine_status").write_text(f"RUNNING 1 {NOW - 3600:.0f}\nFAILED 137 {NOW:.0f}\n")
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="04 running")])
    update_mzmine_status(ctx)
    assert ctx.lims.get_table("untargeted_tasks").loc[0, "mzmine_pos_status"] == "09 error"


def test_step2_leaves_live_job_running(setup):
    cfg, task = setup
    task.path("mzmine_queued").write_text(f"{NOW - 7200:.0f}\n")
    task.path("mzmine_job_id").write_text(f"{task.prefix}=5\n")
    task.path("mzmine_status").write_text(f"RUNNING 5 {NOW - 3600:.0f}\n")
    ctx = _ctx(cfg, [_task_row(cfg["paths"]["output_dir"], pos="04 running")])
    update_mzmine_status(ctx)
    assert ctx.lims.get_table("untargeted_tasks").loc[0, "mzmine_pos_status"] == "04 running"
    assert ctx.lims.writes == []
