"""Tests for metatlas2.untargeted.feature_filtering (counts and background filtering)."""

from __future__ import annotations

import pandas as pd

from metatlas2.untargeted.feature_filtering import (
    calculate_counts_for_lims_table,
    check_peak_height_table,
    count_mgf_feature_ids,
    create_filtered_peakheight_file,
)
from untargeted_fixtures import PROJECT_EB, PROJECT_JGI, peak_height_table, write_mgf


def _write(tmp_path, project, pol="positive", **kw):
    d = tmp_path / f"{project}_{pol}"
    d.mkdir()
    ph = d / f"{project}_{pol}_peak-height.csv"
    peak_height_table(project, pol[:3].upper(), **kw).to_csv(ph, index=False)
    return d, ph


def test_counts(tmp_path):
    d, ph = _write(tmp_path, PROJECT_JGI)
    mgf = d / "x.mgf"
    write_mgf(mgf, 4)
    features, exctrl, msms = calculate_counts_for_lims_table(ph, mgf, ["ExCtrl"])
    # sample columns = TxCtrl + Sample columns; rows with any >0: rows 0,1,3,4,5
    assert features == 5
    assert exctrl == 4  # ExCtrl >0 in rows 0,1,3,4
    assert msms == 4
    assert count_mgf_feature_ids(tmp_path / "missing.mgf") == 0


def test_eb_hilic_writes_all_six_tables(tmp_path):
    d, ph = _write(tmp_path, PROJECT_EB)
    create_filtered_peakheight_file(ph, ["ExCtrl"], d, PROJECT_EB, "positive")
    prefix = f"{PROJECT_EB}_positive_peak-height-filtered-"
    for label in ("3x-exctrl", "1e4-exctrl", "1e5-exctrl", "3x-txctrl", "1e4-txctrl", "1e5-txctrl"):
        assert (d / f"{prefix}{label}.csv").exists(), label
    summary = pd.read_csv(d / f"{PROJECT_EB}_positive_peak-height-filtering-summary.csv")
    assert "rt_in_0.8-17.5min_window" in summary.columns
    # RT 0.2 is outside the HILIC window
    assert summary.loc[0, "rt_in_0.8-17.5min_window"] == 0
    three_x = pd.read_csv(d / f"{prefix}3x-exctrl.csv")
    # row 3 (ID 4): sample max 1e6 >= 3*5000 and RT 3.0 in window -> kept; row 0 always kept (legacy quirk)
    assert 4 in set(three_x["row ID"]) and 1 in set(three_x["row ID"])
    assert not any(c.startswith("Unnamed") for c in three_x.columns)


def test_jgi_c18_only_3x_exctrl_with_zero_replacement(tmp_path):
    d, ph = _write(tmp_path, PROJECT_JGI)
    create_filtered_peakheight_file(ph, ["ExCtrl"], d, PROJECT_JGI, "positive", zero_value=0.5)
    files = sorted(p.name for p in d.iterdir())
    assert f"{PROJECT_JGI}_positive_peak-height-filtered-3x-exctrl.csv" in files
    assert not any("1e4" in f or "txctrl" in f for f in files)
    summary = pd.read_csv(d / f"{PROJECT_JGI}_positive_peak-height-filtering-summary.csv")
    assert list(summary.columns) == ["row ID", "exctrl_max", "sample_max", "rt_peak",
                                     "rt_in_0.5-10min_window", ">3x_exctrl"]
    table = pd.read_csv(d / f"{PROJECT_JGI}_positive_peak-height-filtered-3x-exctrl.csv")
    sample_cols = [c for c in table.columns if "_Sample_" in c or "_ExCtrl_" in c]
    assert (table[sample_cols] != 0).all().all()


def test_unknown_chromatography_writes_empty_tables(tmp_path):
    project = PROJECT_EB.replace("HILICZ", "LIPID")
    d, ph = _write(tmp_path, project)
    create_filtered_peakheight_file(ph, ["ExCtrl"], d, project, "positive")
    empty = pd.read_csv(d / f"{project}_positive_peak-height-filtered-3x-exctrl.csv")
    assert empty.empty
    assert "row ID" not in empty.columns  # legacy quirk: first column dropped from header
    assert not (d / f"{project}_positive_peak-height-filtering-summary.csv").exists()


def test_check_peak_height_table(tmp_path):
    p = tmp_path / "ph.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    assert check_peak_height_table(p) == 2
