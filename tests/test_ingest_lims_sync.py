"""Tests for metatlas2.ingest.lims_file_sync against the in-memory LIMS."""

from __future__ import annotations

import os

import pandas as pd

from metatlas2.ingest.lims_file_sync import complex_name_splitter, run_lims_file_sync, scan_raw_data
from metatlas2.shared_tools.lims import InMemoryLims

ROOT = "/global/cfs/cdirs/metatlas/raw_data"


def test_complex_name_splitter():
    assert complex_name_splitter(f"{ROOT}/jgi/P/run_1.raw") == "jgi/P/run_1"
    assert complex_name_splitter(f"{ROOT}/jgi/P/run_1.mzML") == "jgi/P/run_1"
    assert complex_name_splitter(f"{ROOT}/jgi/P/run_1.h5") == "jgi/P/run_1"
    # internal dots keep their order (legacy used a set -> random order)
    assert complex_name_splitter(f"{ROOT}/egsb/P/n.__234.a.mzML") == "egsb/P/n.__234.a"
    assert complex_name_splitter(f"{ROOT}/egsb/P/n.__234.a.raw") == "egsb/P/n.__234.a"
    assert complex_name_splitter("/global/project/projectdirs/metatlas/raw_data/x/P/f.raw") == "x/P/f"
    assert complex_name_splitter(f"{ROOT}/x/P/f_spectral-hits.tab.gz") == "x/P/f"


def _make_tree(root):
    p = root / "jgi" / "P"
    p.mkdir(parents=True)
    for name in ("a.raw", "a.mzML", "a.h5", "b.RAW", "b.mzml", "notes.txt"):
        (p / name).write_text("x")
    (p / "waters.raw").mkdir()                  # Waters raw data is a directory
    (p / "waters.raw" / "inner.h5").write_text("x")
    other = root.parent / "outside"
    other.mkdir()
    (other / "c.raw").write_text("x")
    os.symlink(other, root / "linked")          # symlinked dir: listed, not followed
    os.symlink(p / "a.raw", root / "jgi" / "alias.raw")  # symlink file: listed (find -iname)
    return p


def test_scan_matches_find_iname(tmp_path):
    root = tmp_path / "raw_data"
    _make_tree(root)
    disk = scan_raw_data(str(root))
    rel = {t: sorted(os.path.relpath(f, root) for f in df["filename"]) for t, df in disk.items()}
    assert rel["raw_file"] == ["jgi/P/a.raw", "jgi/P/b.RAW", "jgi/P/waters.raw", "jgi/alias.raw"]
    assert rel["mzml_file"] == ["jgi/P/a.mzML", "jgi/P/b.mzml"]
    assert rel["hdf5_file"] == ["jgi/P/a.h5", "jgi/P/waters.raw/inner.h5"]
    assert set(disk["raw_file"]["file_type"]) == {"raw"}
    assert disk["raw_file"]["timeepoch"].dtype.kind == "i"


def _empty_lims():
    return InMemoryLims({
        "raw_file": pd.DataFrame(columns=["filename", "Key", "name", "basename", "file_type", "timeepoch"]),
        "mzml_file": pd.DataFrame(columns=["filename", "Key", "name", "basename", "file_type", "timeepoch"]),
        "hdf5_file": pd.DataFrame(columns=["filename", "Key", "name", "basename", "file_type", "timeepoch"]),
        "lcmsrun": pd.DataFrame(columns=["Key", "name", "raw_file", "mzml_file", "hdf5_file"]),
    })


def test_full_sync_and_idempotence(tmp_path):
    root = tmp_path / "raw_data"
    _make_tree(root)
    lims = _empty_lims()
    report = run_lims_file_sync(lims, raw_data_dir=str(root))
    assert report.inserted_files == {"raw_file": 4, "mzml_file": 2, "hdf5_file": 2}
    raw = lims.get_table("raw_file")
    row = raw[raw["basename"] == "a.raw"].iloc[0]
    assert row["name"] == "jgi/P/a" and row["file_type"] == "raw"
    runs = lims.get_table("lcmsrun").set_index("name")
    # Extension stripping is case-sensitive (legacy): b.RAW / b.mzml keep their suffix in the name.
    assert set(runs.index) == {"jgi/P/a", "jgi/P/b.RAW", "jgi/P/b.mzml", "jgi/P/waters", "jgi/alias",
                               "jgi/P/waters.raw/inner"}
    mzml = lims.get_table("mzml_file").set_index("name")
    assert runs.loc["jgi/P/a", "mzml_file"] == mzml.loc["jgi/P/a", "Key"]
    assert runs.loc["jgi/P/a", "hdf5_file"] == lims.get_table("hdf5_file").set_index("name").loc["jgi/P/a", "Key"]
    # second run: nothing to do
    n_writes = len(lims.writes)
    again = run_lims_file_sync(lims, raw_data_dir=str(root))
    assert len(lims.writes) == n_writes
    assert sum(again.inserted_files.values()) == 0 and again.inserted_lcmsruns == 0
    assert sum(again.updated_links.values()) == 0


def test_new_file_gets_linked(tmp_path):
    root = tmp_path / "raw_data"
    p = _make_tree(root)
    lims = _empty_lims()
    run_lims_file_sync(lims, raw_data_dir=str(root))
    (p / "c.raw").write_text("x")
    run_lims_file_sync(lims, raw_data_dir=str(root))
    (p / "c.h5").write_text("x")
    report = run_lims_file_sync(lims, raw_data_dir=str(root))
    assert report.inserted_files["hdf5_file"] == 1 and report.updated_links["hdf5_file"] == 1
    runs = lims.get_table("lcmsrun").set_index("name")
    h5 = lims.get_table("hdf5_file").set_index("name")
    assert runs.loc["jgi/P/c", "hdf5_file"] == h5.loc["jgi/P/c", "Key"]
    assert runs.loc["jgi/P/c", "raw_file"] == lims.get_table("raw_file").set_index("name").loc["jgi/P/c", "Key"]


def test_ambiguous_names_prefer_existing_link_then_newest():
    lims = _empty_lims()
    lims.tables["raw_file"] = pd.DataFrame({"Key": [10, 20], "name": ["x/P/r", "x/P/r"],
                                            "filename": ["/a/x/P/r.raw", "/b/x/P/r.raw"]})
    lims.tables["lcmsrun"] = pd.DataFrame({"Key": [1, 2], "name": ["x/P/r", "x/P/q"],
                                           "raw_file": [10, None], "mzml_file": [None, None],
                                           "hdf5_file": [None, None]})
    report = run_lims_file_sync(lims, update_file_tables=False)
    assert report.updated_links["raw_file"] == 0 and report.ambiguous_names["raw_file"] == 1  # 10 still valid
    lims.tables["lcmsrun"].loc[0, "raw_file"] = None
    report = run_lims_file_sync(lims, update_file_tables=False)
    assert report.updated_links["raw_file"] == 1
    assert lims.get_table("lcmsrun").set_index("name").loc["x/P/r", "raw_file"] == 20  # newest


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "raw_data"
    _make_tree(root)
    lims = _empty_lims()
    lims.dry_run = True
    report = run_lims_file_sync(lims, raw_data_dir=str(root))
    assert report.inserted_files["raw_file"] == 4
    assert lims.get_table("raw_file").empty and lims.get_table("lcmsrun").empty


def test_skip_steps(tmp_path):
    root = tmp_path / "raw_data"
    _make_tree(root)
    lims = _empty_lims()
    run_lims_file_sync(lims, raw_data_dir=str(root), update_lcmsrun_names=False, update_lcmsrun_links=False)
    assert len(lims.get_table("raw_file")) == 4 and lims.get_table("lcmsrun").empty
    run_lims_file_sync(lims, raw_data_dir=str(root), update_file_tables=False, update_lcmsrun_links=False)
    runs = lims.get_table("lcmsrun")
    assert len(runs) == 6 and runs["raw_file"].isna().all()
