"""Tests for metatlas2.untargeted.project_setup (Step 1)."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.lims import InMemoryLims
from metatlas2.untargeted.project_setup import (
    build_metadata_df,
    find_raw_data_directory,
    metadata_file_filter,
    raw_data_subdir_for_project,
    update_new_untargeted_tasks,
)
from metatlas2.untargeted.status import PolarityTask
from untargeted_fixtures import (
    PLACEHOLDER,
    PROJECT_EB,
    PROJECT_JGI,
    lcmsruns_df,
    make_config,
    make_raw_project,
    mzml_name,
)

NOW = 2_000_000_000.0
OLD = NOW - 5 * 3600


def _ctx(tmp_path, lcms: pd.DataFrame, tasks: pd.DataFrame | None = None, **kw) -> RunContext:
    cfg = make_config(tmp_path)
    lims = InMemoryLims({"lcmsrun_plus": lcms,
                         "untargeted_tasks": tasks if tasks is not None else pd.DataFrame(columns=["parent_dir"])})
    return RunContext(config=cfg, lims=lims, clock=lambda: NOW, **kw)


def test_raw_data_subdir_for_project():
    assert raw_data_subdir_for_project(PROJECT_JGI) == "jgi"
    assert raw_data_subdir_for_project(PROJECT_EB) == "egsb"
    assert raw_data_subdir_for_project("single") == "jgi"


def test_find_raw_data_directory_fallback(tmp_path):
    (tmp_path / "bpb" / PROJECT_JGI).mkdir(parents=True)
    assert find_raw_data_directory(PROJECT_JGI, str(tmp_path), None, ["egsb", "bpb"]).endswith(f"bpb/{PROJECT_JGI}")
    assert find_raw_data_directory(PROJECT_JGI, str(tmp_path), "jgi") is None
    assert find_raw_data_directory("nope", str(tmp_path), None, ["bpb"]) is None


def test_metadata_file_filter_excludes_qc_injbl_istd_blank():
    names = [mzml_name(PROJECT_JGI, "POS", g, i) for i, g in enumerate(
        ["ExCtrl", "Sample", "QC", "InjBl", "ISTD", "Blank"], start=1)]
    names.append(mzml_name(PROJECT_JGI, "NEG", "Sample", 9))
    df = pd.DataFrame({"basename": names})
    kept = metadata_file_filter(df, "POS")
    assert [n.split("_")[12] for n in kept] == ["ExCtrl", "Sample"]
    kept_blank = metadata_file_filter(df, "POS", skip_blank_filter=True)
    assert "Blank" in [n.split("_")[12] for n in kept_blank]


def test_build_metadata_df_quirky_columns():
    names = [mzml_name(PROJECT_JGI, "POS", g, i) for i, g in enumerate(["ExCtrl", "Sample", "QC"], start=1)]
    names.append(mzml_name(PROJECT_JGI, "NEG", "Sample", 9))
    df = pd.DataFrame({"basename": names, "sample_group": ["ExCtrl", "Sample", "QC", "Sample"]})
    meta = build_metadata_df(df, "positive", ["ExCtrl"])
    assert list(meta.columns) == ["filename", "ATTRIBUTE_sampletype", "CONTROL"]
    assert len(meta) == 3  # QC included in metadata (legacy), NEG file excluded
    assert set(meta["CONTROL"]) == {"exctrl"}


def test_step1_creates_inputs_and_inserts_lims_row(tmp_path):
    ctx = _ctx(tmp_path, pd.DataFrame())
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    ctx.lims.tables["lcmsrun_plus"] = lcmsruns_df(PROJECT_JGI, names, raw, "jgi", OLD)
    inserted = update_new_untargeted_tasks(ctx)
    assert list(inserted["parent_dir"]) == [PROJECT_JGI]
    row = ctx.lims.get_table("untargeted_tasks").iloc[0]
    assert row["mzmine_pos_status"] == "01 initiation" and row["fbmn_pos_status"] == "13 waiting"
    assert row["num_pos_files"] == 6  # 2 ExCtrl + 1 TxCtrl + 3 Sample
    assert row["mzmine_parameter_sheet"].endswith("/mzmine-3.7.2-batchparams.xml")  # EXP120B
    assert row["mzmine_parameter_row"] == 2

    task = PolarityTask(ctx.output_dir, PROJECT_JGI, "positive")
    xml = task.path("batch_params").read_text()
    assert PLACEHOLDER not in xml and f"{task.directory / task.prefix}_peak-height.csv" in xml
    filelist = task.path("filelist").read_text().split()
    assert len(filelist) == 6 and all(f.startswith(str(raw / "jgi" / PROJECT_JGI)) for f in filelist)
    meta = pd.read_csv(task.path("metadata"), sep="\t")
    assert len(meta) == 9  # every POS file, including QC/InjBl/ISTD
    sbatch = task.path("mzmine_sbatch").read_text()
    assert "#SBATCH -t 3:00:00" in sbatch and "#SBATCH -A m2650" in sbatch
    assert f"-batch {task.path('batch_params')} -input {task.path('filelist')}" in sbatch
    assert str(task.path("mzmine_status")) in sbatch
    assert task.path("mzmine_runner").read_text() == f"sbatch {task.path('mzmine_sbatch')}"


@pytest.mark.parametrize("case", ["too_new", "no_ms2", "opt_out", "already_tracked"])
def test_step1_skips_ineligible_projects(tmp_path, case):
    ctx = _ctx(tmp_path, pd.DataFrame())
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    epoch = OLD
    if case == "too_new":
        epoch = NOW - 600
    if case == "no_ms2":
        names = [n.replace("_MS2_", "_MS1_") for n in names]
    if case == "opt_out":
        (raw / "jgi" / PROJECT_JGI / "nountargeted.txt").write_text("")
    if case == "already_tracked":
        ctx.lims.tables["untargeted_tasks"] = pd.DataFrame({"parent_dir": [PROJECT_JGI]})
    ctx.lims.tables["lcmsrun_plus"] = lcmsruns_df(PROJECT_JGI, names, raw, "jgi", epoch)
    assert update_new_untargeted_tasks(ctx) is None
    assert not list((tmp_path / "untargeted_tasks").iterdir())


def test_step1_single_polarity_and_iqx_template(tmp_path):
    ctx = _ctx(tmp_path, pd.DataFrame())
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "egsb", PROJECT_EB, polarities=("NEG",))
    ctx.lims.tables["lcmsrun_plus"] = lcmsruns_df(PROJECT_EB, names, raw, "egsb", OLD)
    update_new_untargeted_tasks(ctx)
    row = ctx.lims.get_table("untargeted_tasks").iloc[0]
    assert row["mzmine_pos_status"] == "12 not relevant" and row["num_pos_files"] == 0
    assert row["mzmine_neg_status"] == "01 initiation"
    assert "IQX-" in row["mzmine_parameter_sheet"] and row["mzmine_parameter_row"] == 5
    assert not PolarityTask(ctx.output_dir, PROJECT_EB, "positive").directory.exists()


def test_step1_dry_run_writes_nothing(tmp_path):
    ctx = _ctx(tmp_path, pd.DataFrame(), dry_run=True)
    ctx.lims.dry_run = True
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    ctx.lims.tables["lcmsrun_plus"] = lcmsruns_df(PROJECT_JGI, names, raw, "jgi", OLD)
    inserted = update_new_untargeted_tasks(ctx)
    assert len(inserted) == 1
    assert not list((tmp_path / "untargeted_tasks").iterdir())
    assert ctx.lims.get_table("untargeted_tasks").empty


def test_step1_existing_directory_is_not_overwritten(tmp_path):
    ctx = _ctx(tmp_path, pd.DataFrame())
    raw = tmp_path / "raw_data"
    names = make_raw_project(raw, "jgi", PROJECT_JGI)
    ctx.lims.tables["lcmsrun_plus"] = lcmsruns_df(PROJECT_JGI, names, raw, "jgi", OLD)
    PolarityTask(ctx.output_dir, PROJECT_JGI, "positive").directory.mkdir()
    update_new_untargeted_tasks(ctx)
    row = ctx.lims.get_table("untargeted_tasks").iloc[0]
    assert row["mzmine_pos_status"] == "09 error"  # legacy behaviour
    assert row["mzmine_neg_status"] == "01 initiation"


def test_clock_default_is_wall_time(tmp_path):
    ctx = RunContext(config=make_config(tmp_path), lims=InMemoryLims())
    assert abs(ctx.clock() - time.time()) < 5
