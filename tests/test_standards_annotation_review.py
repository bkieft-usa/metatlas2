from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import duckdb
import pandas as pd
import pytest

from metatlas2 import standards_annotation_db as db
from metatlas2 import standards_annotation_gui as gui
from metatlas2 import standards_annotation_review as review
from metatlas2 import standards_annotation_rt_align as align
from metatlas2 import standards_annotation_summary as summary
from metatlas2 import standards_annotation_tools as tools
from metatlas2 import deposit_ref_standards_atlas_and_msms_refs as deposit
from metatlas2.deposit_ref_standards_atlas_and_msms_refs import run_atlas_and_msms_deposit
from metatlas2.standards_annotation_rt_align import run_reverse_rt_align
from metatlas2.run_standards_annotation import run_standards_annotation
from metatlas2.workflow_objects import NewMsmsRefsConfig


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path_factory, monkeypatch):
    """Point METATLAS_DATA_DIR away from real data; tests may override it."""
    monkeypatch.setenv("METATLAS_DATA_DIR", str(tmp_path_factory.mktemp("metatlas_data_dir")))


@pytest.fixture(autouse=True)
def mock_standards_drive_upload(monkeypatch):
    upload = Mock(return_value="https://drive.google.com/drive/folders/review")
    monkeypatch.setattr("metatlas2.gdrive_upload.upload_standards_outputs", upload)
    return upload


def mock_alignment_model(monkeypatch):
    monkeypatch.setattr(align, "get_qc_runs_for_chrom", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(align, "load_qc_atlas", lambda *_: SimpleNamespace(atlas_uid="qc-uid", to_dataframe=lambda: pd.DataFrame([{"x": 1}])))
    monkeypatch.setattr(align, "extract_qc_ms1", lambda *_: pd.DataFrame([{"x": 1}]))
    monkeypatch.setattr(align, "build_reverse_rt_model", lambda *_: {
        "model_type": "polynomial", "r2": 1.0, "rmse": 0.0,
        "modeling_df": pd.DataFrame([{"x": 1}]),
    })
    monkeypatch.setattr(align, "apply_reverse_rt_model", lambda df, _: df.assign(
        rt_peak_aligned=df["rt_peak"], rt_min_aligned=df["rt_min"],
        rt_max_aligned=df["rt_max"], rt_delta=0.0,
    ))
    def plot(_df, _model, output_dir, run_name):
        path = Path(output_dir) / f"RTAlignment-Summary__{run_name}.pdf"
        path.write_bytes(b"test plot")
        return str(path)
    monkeypatch.setattr(align, "make_rt_alignment_summary_plot", plot)


@pytest.fixture
def project(tmp_path):
    path = str(tmp_path / "project.duckdb")
    db.create_project_db(path)
    with duckdb.connect(path) as conn:
        conn.execute(
            "INSERT INTO sa_compounds VALUES ('c', 'A', 'key', '', '', '[M+H]+', 'POS', 100)"
        )
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('t', 'c', 'file.h5', 'C18', 0, 10, 'CE', 'annotation')"
        )
    peaks = [
        dict(peak_uid=uid, compound_uid="c", compound_name="A", inchi_key="key",
             adduct="[M+H]+", polarity="POS", chromatography="C18",
             filename="file.h5", mz_rt_uid="t", mz_theoretical=100, rt_peak=rt,
             rt_min=rt - .1, rt_max=rt + .1, peak_intensity=100,
             mz_observed=100, peak_num=num, is_manual=False, collision_energy_label="CE")
        for uid, rt, num in [("p1", 1, 1), ("p3", 3, 3), ("p2", 2, 2)]
    ]
    db.save_peaks_to_db(path, peaks)
    db.replace_compound_selections(path, "A", ["c"], [("p1", "note"), ("p3", "note")], {"C18": "note"}, ["C18"])
    return path


def test_peak_labels_order_groups_and_deterministic_ties():
    rows = [
        dict(peak_uid=uid, compound_uid=compound, compound_name=compound,
             chromatography=chrom, polarity=pol, adduct=adduct, filename=filename,
             rt_peak=rt, peak_num=num)
        for uid, compound, chrom, pol, adduct, filename, rt, num in [
            ("b", "A", "C18", "POS", "H", "f", 2., 3),
            ("a", "A", "C18", "POS", "H", "f", 1., 5),
            ("c", "A", "C18", "POS", "Na", "f", 2., 7),
            ("d", "A", "C18", "NEG", "H", "f", 3., 9),
            ("f", "B", "C18", "POS", "H", "f", 1., 8),
            ("e", "B", "C18", "POS", "H", "f", 1., 4),
        ]
    ]
    source = pd.DataFrame(rows)
    result = review.fix_peak_labels(source).set_index("peak_uid")
    assert (result.loc["a", "peak_num"], result.loc["b", "peak_num"]) == (1, 2)
    assert (result.loc["a", "auto_peak_num"], result.loc["b", "auto_peak_num"]) == (5, 3)
    assert result.loc["b", "peak_label"] == "A (peak2)"
    assert result.loc["c", "peak_label"] == "A"
    assert result.loc["c", "peak_num"] == result.loc["d", "peak_num"] == 1
    assert (result.loc["e", "peak_num"], result.loc["f", "peak_num"]) == (1, 2)
    assert source.loc[0, "peak_num"] == 3
    assert review.fix_peak_labels(source.sample(frac=1)).set_index("peak_uid")["peak_num"].to_dict() == result["peak_num"].to_dict()
    source.loc[0, "filename"] = "other"
    with pytest.raises(ValueError, match="multiple files"):
        review.fix_peak_labels(source)


def test_selection_save_is_atomic_and_leaves_alignment_export_untouched(project, tmp_path):
    csv = tmp_path / "current.csv"
    csv.write_text("current")
    with pytest.raises(ValueError, match="not found"):
        db.replace_compound_selections(project, "A", ["c"], [("missing", "")], {}, ["C18"])
    assert csv.read_text() == "current"
    assert set(db.load_analyst_selections(project)["peak_uid"]) == {"p1", "p3"}
    db.replace_compound_selections(project, "A", ["c"], [("p1", "note"), ("p3", "note")], {"C18": "note"}, ["C18"])
    assert csv.read_text() == "current"
    db.replace_compound_selections(project, "A", ["c"], [("p3", "new")], {"C18": "new"}, ["C18"], resume_compound="next")
    assert csv.read_text() == "current"
    assert db.load_analyst_selections(project)["peak_uid"].tolist() == ["p3"]
    assert db.load_gui_checkpoint(project) == "next"
    assert db.load_compound_notes(project, "A") == {"C18": "new"}
    db.replace_compound_selections(project, "A", ["c"], [], {"C18": "empty note"}, ["C18"])
    assert db.load_compound_notes(project, "A") == {"C18": "empty note"}
    assert csv.read_text() == "current"


def test_selection_save_rolls_back_after_database_write_failure(project):
    with pytest.raises(duckdb.ConstraintException):
        db.replace_compound_selections(
            project, "A", ["c"], [("p3", "new")], {"C18": None}, ["C18"],
            resume_compound="next",
        )
    assert set(db.load_analyst_selections(project)["peak_uid"]) == {"p1", "p3"}
    assert db.load_compound_notes(project, "A") == {"C18": "note"}
    assert db.load_gui_checkpoint(project) == "A"


def test_annotation_notes_round_trip_per_chromatography(tmp_path):
    project = str(tmp_path / "notes.duckdb")
    db.create_project_db(project)
    targets = pd.DataFrame([
        dict(mz_rt_uid=f"target-{chrom}", compound_uid="compound-adduct", compound_name="A",
             adduct="[M+H]+", polarity="POS", chromatography=chrom,
             filename=f"{chrom}.h5", mz_theoretical=100, rt_min=0, rt_max=10,
             annotation_notes=f"{chrom} input note")
        for chrom in ("C18", "HILIC")
    ])
    data = SimpleNamespace(expanded_table=targets, ms1_df=pd.DataFrame(),
                           ms2_df=pd.DataFrame(), tic_by_file={}, config={})
    db.save_extraction_to_db(project, data)
    reloaded = db.load_extraction_from_db(project, {})
    assert dict(zip(reloaded.expanded_table.chromatography,
                    reloaded.expanded_table.annotation_notes)) == {
                        "C18": "C18 input note", "HILIC": "HILIC input note"}

    peaks = [
        dict(peak_uid=f"peak-{row.chromatography}", compound_uid=row.compound_uid,
             compound_name="A", adduct=row.adduct, polarity=row.polarity,
             chromatography=row.chromatography, filename=row.filename, mz_rt_uid=row.mz_rt_uid,
             mz_theoretical=100, rt_peak=3, rt_min=2.9, rt_max=3.1, peak_intensity=10)
        for row in targets.itertuples()
    ]
    db.save_peaks_to_db(project, peaks)
    db.replace_compound_selections(project, "A", ["compound-adduct"],
                                   [(peak["peak_uid"], "") for peak in peaks],
                                   {}, ["C18", "HILIC"])
    selections = db.load_selected_peaks_with_ms2(project)
    assert dict(zip(selections.chromatography, selections.annotation_notes)) == {
        "C18": "C18 input note", "HILIC": "HILIC input note"}


@pytest.mark.parametrize("upload_to_gdrive", [False, True])
def test_no_good_choices_persist_and_summary_requires_every_group(project, tmp_path, monkeypatch, mock_standards_drive_upload, upload_to_gdrive):
    with duckdb.connect(project) as conn:
        conn.execute(
            "INSERT INTO sa_compounds VALUES ('neg', 'A', 'key', '', '', '[M-H]-', 'NEG', 100)"
        )
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('neg-target', 'neg', 'file.h5', 'C18', 0, 10, 'CE', '')"
        )
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('hilic-target', 'c', 'file.h5', 'HILIC', 0, 10, 'CE', '')"
        )
    data = SimpleNamespace(config={"project": {"standards_run_name": "run", "owner": "jgi"}}, project_db_path=project)
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    with pytest.raises(ValueError, match=r"A \| C18 \| NEG"):
        summary.run_standards_annotation_summary(data, upload_to_gdrive=upload_to_gdrive)
    assert not (tmp_path / "outputs").exists()

    db.replace_compound_selections(
        project, "A", ["c", "neg"], [("p1", "note")], {"C18": "note"},
        ["C18", "HILIC"], no_good_peaks=[("C18", "NEG"), ("HILIC", "POS")],
    )
    assert set(db.load_no_good_peaks(project, "A")) == {("C18", "NEG"), ("HILIC", "POS")}
    db.validate_annotation_choices(project)
    summary.run_standards_annotation_summary(data, upload_to_gdrive=upload_to_gdrive)
    csv = next(tmp_path.glob("outputs/annotation_run_*/StandardsAnnotation_Summary_run.csv"))
    assert pd.read_csv(csv)["Compound Label"].tolist() == ["A"]
    if upload_to_gdrive:
        mock_standards_drive_upload.assert_called_once_with(data.config, csv.parent, [csv.parent])
    else:
        mock_standards_drive_upload.assert_not_called()
    with pytest.raises(ValueError, match="conflict"):
        db.replace_compound_selections(
            project, "A", ["c", "neg"], [("p1", "")], {}, ["C18"],
            no_good_peaks=[("C18", "POS")],
        )
    assert set(db.load_no_good_peaks(project, "A")) == {("C18", "NEG"), ("HILIC", "POS")}
    assert db.load_analyst_selections(project)["peak_uid"].tolist() == ["p1"]
    db.replace_compound_selections(
        project, "A", ["c", "neg"], [], {}, ["C18", "HILIC"],
        no_good_peaks=[("C18", "POS"), ("C18", "NEG"), ("HILIC", "POS")],
    )
    db.validate_annotation_choices(project)
    with pytest.raises(ValueError, match="No selected peaks"):
        summary.run_standards_annotation_summary(data)


def test_no_good_choice_clears_and_is_cleared_by_peaks():
    resolve = gui._resolve_polarity_choices
    assert resolve({"p1", "p2"}, {"p1", "p2", "none"}, "none") == {"none"}
    assert resolve({"none"}, {"none", "p1"}, "none") == {"p1"}
    assert resolve({"p1", "p2"}, {"p2"}, "none") == {"p2"}
    assert resolve({"none"}, set(), "none") == set()


def test_gui_no_good_option_without_detected_peaks_and_resume(project, monkeypatch):
    with duckdb.connect(project) as conn:
        conn.execute(
            "INSERT INTO sa_compounds VALUES ('neg', 'A', 'key', '', '', '[M-H]-', 'NEG', 100)"
        )
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('neg-target', 'neg', 'file.h5', 'C18', 0, 10, 'CE', '')"
        )
    db.replace_compound_selections(
        project, "A", ["c", "neg"], [("p1", "")], {}, ["C18"],
        no_good_peaks=[("C18", "NEG")],
    )
    expanded = pd.DataFrame([
        {"compound_name": "A", "compound_uid": uid, "mz_rt_uid": target,
         "adduct": adduct, "chromatography": "C18", "polarity": pol,
         "filename": "file.h5", "mz_theoretical": 100, "collision_energy_label": "CE"}
        for uid, target, adduct, pol in [
            ("c", "t", "[M+H]+", "POS"), ("neg", "neg-target", "[M-H]-", "NEG")
        ]
    ])
    data = SimpleNamespace(
        project_db_path=project, expanded_table=expanded,
        ms1_df=pd.DataFrame(columns=["mz_rt_uid"]), ms2_df=pd.DataFrame(), tic_by_file={},
    )
    apps = []
    real_dash = gui.dash.Dash
    def create_dash(*args, **kwargs):
        app = real_dash(*args, **kwargs)
        apps.append(app)
        return app
    monkeypatch.setattr(gui.dash, "Dash", create_dash)
    monkeypatch.setattr("werkzeug.serving.make_server", lambda *_: SimpleNamespace(
        shutdown=lambda: None, serve_forever=lambda: None,
    ))
    monkeypatch.setattr("IPython.display.display", lambda *_: None)
    monkeypatch.setattr(gui.time, "sleep", lambda _: None)
    gui.build_standards_annotation_gui(data, {"project": {"standards_run_name": "run", "owner": "jgi"}}, port=0)
    restored = apps[0].layout.children[0].data
    assert restored["checked_peaks"]["__no_good__:C18:NEG"]
    assert restored["checked_peaks"]["p1"]
    callbacks = {}
    for callback in apps[0].callback_map.values():
        if "callback" in callback:
            func = callback["callback"].__wrapped__
            callbacks.setdefault(func.__name__, func)
    session = {"compound_idx": 0, "checked_peaks": {"p1": True, "__no_good__:C18:NEG": True},
               "analyst_notes_by_chrom": {}}
    options = callbacks["update_checklists"](session)
    assert any(o["value"] == "__no_good__:C18:NEG" for o in options[4])
    assert options[5] == ["__no_good__:C18:NEG"]
    session = callbacks["update_pos_checked"](["p1", "__no_good__:C18:POS"], session, 0)
    assert session["checked_peaks"]["__no_good__:C18:POS"]
    assert not session["checked_peaks"]["p1"]
    session = callbacks["update_pos_checked"](["p1", "__no_good__:C18:POS"], session, 0)
    assert session["checked_peaks"]["p1"]
    assert not session["checked_peaks"]["__no_good__:C18:POS"]
    callbacks["save_and_exit"](1, session)
    assert db.load_no_good_peaks(project, "A") == [("C18", "NEG")]
    assert db.load_analyst_selections(project)["peak_uid"].tolist() == ["p1"]


def test_setup_refuses_existing_project_without_modifying_it(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    existing = project_dir / "sentinel"
    existing.write_text("unchanged")
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: {"project": {"standards_run_name": "run", "owner": "jgi"}})
    monkeypatch.setattr(db, "get_project_output_root", lambda _: project_dir)
    monkeypatch.setattr(db, "get_raw_data_root", lambda _: str(tmp_path))
    with pytest.raises(FileExistsError, match="different standards_run_name"):
        run_standards_annotation("config.yaml")
    assert existing.read_text() == "unchanged"
    assert sorted(p.name for p in project_dir.iterdir()) == ["sentinel"]


def test_setup_creates_new_project_and_runs_extraction(tmp_path, monkeypatch):
    root = tmp_path / "new-project"
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text("project: {}\n")
    input_file = tmp_path / "my_standards.csv"
    input_file.write_text("compound_name\nA\n")
    config = {"project": {"standards_run_name": "run", "owner": "jgi",
                          "standards_input_file": str(input_file)},
              "extraction": {"peak_finding": {"x": 1}}}
    calls = []
    data = SimpleNamespace()
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(db, "get_project_output_root", lambda _: root)
    monkeypatch.setattr(db, "get_project_db_path", lambda _: str(root / "run.duckdb"))
    monkeypatch.setattr(db, "get_raw_data_root", lambda _: str(tmp_path))
    monkeypatch.setattr(db, "create_project_db", lambda path: calls.append(("db", path)))
    monkeypatch.setattr("metatlas2.standards_annotation_tools.detect_and_save_peaks",
                        lambda obj, peak_params: calls.append(("peaks", obj, peak_params)))
    def extraction(*_):
        from metatlas2.standards_annotation_tools import detect_and_save_peaks
        calls.append(("extract",))
        detect_and_save_peaks(data, peak_params=config["extraction"]["peak_finding"])
        return data
    monkeypatch.setattr("metatlas2.standards_annotation_tools.run_standards_extraction", extraction)
    monkeypatch.setattr("metatlas2.notebook_generator.generate_standards_annotation_notebook",
                        lambda **_: calls.append(("notebook",)) or str(root / "notebook.ipynb"))
    run_standards_annotation(str(config_file))
    assert [call[0] for call in calls] == ["db", "extract", "peaks", "notebook"]
    assert calls[2][1:] == (data, {"x": 1})
    assert root.is_dir()
    assert (root / "my_config.yaml").read_text() == config_file.read_text()
    assert (root / "my_standards.csv").read_text() == input_file.read_text()


def test_detection_saves_matching_peak_and_ms2_once(project):
    with duckdb.connect(project) as conn:
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('new-target', 'c', 'file.h5', 'C18', 0, 10, 'CE', '')"
        )
    meta = {"mz_rt_uid": "new-target", "compound_uid": "c", "compound_name": "A",
            "adduct": "[M+H]+", "polarity": "POS", "chromatography": "C18",
            "mz_theoretical": 100, "collision_energy_label": "CE", "filename": "file.h5"}
    data = SimpleNamespace(
        project_db_path=project, expanded_table=pd.DataFrame([meta]),
        ms1_df=pd.DataFrame([{"mz_rt_uid": "new-target", "filename": "file.h5",
                              "spec_rts": [0., 1., 2., 3., 4., 5., 6.],
                              "spec_ints": [0., 0., 3., 10., 3., 0., 0.],
                              "spec_mzs": [100.] * 7}]),
        ms2_df=pd.DataFrame([{"mz_rt_uid": "new-target", "filename": "file.h5",
                              "scan_rt": 3., "frag_mzs": [50.], "frag_ints": [100.]}]),
    )
    params = {"min_width": 0, "prominence_snr": 0}
    tools.detect_and_save_peaks(data, params)
    with duckdb.connect(project, read_only=True) as conn:
        saved = conn.execute(
            """SELECT p.peak_uid, m.peak_uid FROM sa_peaks p
               JOIN sa_peak_ms2 m USING (peak_uid) WHERE p.mz_rt_uid = 'new-target'"""
        ).fetchall()
    assert len(saved) == 1
    assert saved[0][0] == saved[0][1]
    with pytest.raises(duckdb.ConstraintException):
        tools.detect_and_save_peaks(data, params)
    with duckdb.connect(project, read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM sa_peaks WHERE mz_rt_uid = 'new-target'"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM sa_peak_ms2").fetchone()[0] == 1


def test_detection_ms2_failure_rolls_back_peaks(project):
    peak = db.load_peaks(project).iloc[0].to_dict()
    peak.update(peak_uid="new", rt_peak=9, rt_min=8.9, rt_max=9.1)
    bad_ms2 = {"peak_uid": "new", "filename": "file.h5", "frag_mzs": ["invalid"], "frag_ints": [1]}
    with pytest.raises(ValueError):
        db.save_detected_peaks_to_db(project, [peak], [bad_ms2])
    assert "new" not in set(db.load_peaks(project)["peak_uid"])
    assert db.load_peak_ms2(project).empty


def test_detection_selects_one_file_across_adducts_and_saves_both_ms2(project, tmp_path, monkeypatch):
    template = ("20260720_JGI_PLM-RS_512633_SorghSwg_final_EXP120B_HILICZ_"
                "USHXG03401_POS_MS2_RefStd-1_FA-CoA_1_Rg70to1050-{}norm-RxnMix-S1_Run{}.h5")
    filenames = [template.format("CE102040", 135), template.format("CE205060", 136)]
    meta, ms1, ms2 = [], [], []
    with duckdb.connect(project) as conn:
        for adduct, compound_uid in [("[M+H]+", "b-h"), ("[M+Na]+", "b-na")]:
            conn.execute("INSERT INTO sa_compounds VALUES (?, 'B', 'key', '', '', ?, 'POS', 100)",
                         [compound_uid, adduct])
            for i, filename in enumerate(filenames):
                uid = f"{compound_uid}-{i}"
                conn.execute("INSERT INTO sa_extraction_targets VALUES (?, ?, ?, 'C18', 0, 10, ?, '')",
                             [uid, compound_uid, filename, ("CE102040", "CE205060")[i]])
                meta.append(dict(mz_rt_uid=uid, compound_uid=compound_uid, compound_name="B",
                                 filename=filename, adduct=adduct, polarity="POS", chromatography="C18",
                                 mz_theoretical=100, collision_energy_label=("CE102040", "CE205060")[i]))
                height = {("b-h", 0): 20, ("b-h", 1): 10, ("b-na", 0): 5, ("b-na", 1): 30}[compound_uid, i]
                ms1.append(dict(mz_rt_uid=uid, filename=filename, spec_rts=list(range(7)),
                                spec_ints=[0, 0, 2, height, 2, 0, 0], spec_mzs=[100] * 7))
                for rt in (2.8 + i * .1, 4.5):
                    ms2.append(dict(mz_rt_uid=uid, filename=filename, scan_rt=rt,
                                    precursor_mz=100 + i, precursor_intensity=200 + i,
                                    collision_energy=10 + i, frag_mzs=[50 + i], frag_ints=[100 + i]))
    data = SimpleNamespace(project_db_path=project, expanded_table=pd.DataFrame(meta),
                           ms1_df=pd.DataFrame(ms1), ms2_df=pd.DataFrame(ms2))
    tools.detect_and_save_peaks(data, {"min_width": 0, "prominence_snr": 0})
    peaks = db.load_peaks(project)
    peaks = peaks[peaks.compound_name == "B"]
    assert len(peaks) == 2
    assert set(peaks.filename) == {filenames[1]}
    assert set(data.ms1_df.filename) == {filenames[1]}
    scans = db.load_peak_ms2(project)
    assert len(scans) == 4
    assert set(scans.filename) == set(filenames)
    assert sorted(scans.scan_rt.round(1).unique()) == [2.8, 2.9]
    db.replace_compound_selections(project, "B", ["b-h", "b-na"],
                                   [(uid, "") for uid in peaks.peak_uid], {}, ["C18"])
    selections = review.fix_peak_labels(db.load_selected_peaks_with_ms2(project))
    selections = selections[selections.compound_name == "B"]
    assert len(selections) == 2
    assert all(len(scans) == 2 for scans in selections.ms2_scans)
    summary.make_standards_annotation_table(selections, tmp_path, "test")
    output = pd.read_csv(tmp_path / "StandardsAnnotation_Summary_test.csv")
    assert len(output) == 2
    assert json.loads(output.loc[0, "CE102040 spectrum"]) == [[50.0], [100.0]]
    assert json.loads(output.loc[0, "CE205060 spectrum"]) == [[51.0], [101.0]]
    assert output.loc[0, "CE205060 MS2 Precursor Intensity"] == 201
    ms1_plot = data.ms1_df.copy()
    ms1_plot["adduct"] = ms1_plot.mz_rt_uid.map({row["mz_rt_uid"]: row["adduct"] for row in meta})
    titles = []
    monkeypatch.setattr(summary, "_plot_raw_ms2", lambda ax, **kwargs: titles.append(kwargs["title"]))
    figure_path = tmp_path / "summary.pdf"
    summary._sa_figure_worker({
        "compound_name": "B", "fig_path": str(figure_path), "chrom": "C18", "pol": "POS",
        "adduct_list": ["[M+H]+", "[M+Na]+"], "selections_dicts": selections.to_dict("records"),
        "ms1_dicts": ms1_plot.to_dict("records"), "tic_dicts": {},
    })
    assert figure_path.is_file()
    assert len(titles) == 4
    assert sum("CE102040" in title for title in titles) == 2
    assert sum("CE205060" in title for title in titles) == 2

    hilic_uid = "b-h-hilic"
    hilic_filename = filenames[0]
    with duckdb.connect(project) as conn:
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES (?, 'b-h', ?, 'HILIC', 0, 10, 'CE102040', 'HILIC note')",
            [hilic_uid, hilic_filename],
        )
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('b-h-hilic-ce2', 'b-h', ?, 'HILIC', 0, 10, 'CE205060', 'HILIC note')",
            [filenames[1]],
        )
        conn.execute(
            "UPDATE sa_extraction_targets SET annotation_notes = 'C18 note' "
            "WHERE compound_uid IN ('b-h', 'b-na') AND chromatography = 'C18'"
        )
    hilic_peak = dict(peaks.iloc[0])
    hilic_peak.update(peak_uid="hilic-peak", compound_uid="b-h", adduct="[M+H]+",
                      chromatography="HILIC", filename=hilic_filename, mz_rt_uid=hilic_uid,
                      rt_peak=3.0, is_manual=True)
    db.save_manual_peak_to_db(project, hilic_peak, [{
        "peak_uid": "hilic-peak", "filename": hilic_filename, "scan_rt": 3.0,
        "frag_mzs": [5.0, 500.0], "frag_ints": [10.0, 20.0],
    }])
    hilic_peak_2 = dict(hilic_peak, peak_uid="hilic-peak-2", rt_peak=4.0, peak_num=2)
    db.save_manual_peak_to_db(project, hilic_peak_2, [{
        "peak_uid": "hilic-peak-2", "filename": filenames[1], "scan_rt": 4.0,
        "frag_mzs": [100.0], "frag_ints": [30.0],
    }])
    db.save_manual_peak_to_db(
        project, dict(hilic_peak, peak_uid="hilic-peak-3", rt_peak=5.0, peak_num=3), []
    )
    restored_targets = db.load_extraction_from_db(project, {}).expanded_table
    data.expanded_table = restored_targets[restored_targets.compound_name == "B"].reset_index(drop=True)
    data.ms1_df = pd.concat([data.ms1_df, pd.DataFrame([{
        "mz_rt_uid": hilic_uid, "filename": hilic_filename,
        "spec_rts": list(range(7)), "spec_ints": [0, 0, 2, 10, 2, 0, 0],
        "spec_mzs": [100] * 7,
    }])], ignore_index=True)

    import dash
    import werkzeug.serving

    app_holder = []
    dash_class = dash.Dash
    def create_app(*args, **kwargs):
        app = dash_class(*args, **kwargs)
        app_holder.append(app)
        return app
    monkeypatch.setattr(dash, "Dash", create_app)
    monkeypatch.setattr(werkzeug.serving, "make_server", lambda *_: SimpleNamespace(
        serve_forever=lambda: None, shutdown=lambda: None))
    monkeypatch.setattr(gui.time, "sleep", lambda _: None)
    data.tic_by_file = {}
    data.injbl_ms1_df = pd.DataFrame()
    gui.build_standards_annotation_gui(
        data, config={"project": {"standards_run_name": "test"}}, port=0,
    )
    callbacks = {value["callback"].__wrapped__.__name__: value["callback"].__wrapped__
                 for value in app_holder[0].callback_map.values() if "callback" in value}
    plotted = callbacks["update_plots"]({"compound_idx": 0})
    eic = plotted[12]
    lines = [trace for trace in eic.data if trace.mode == "lines"]
    triangles = [trace for trace in eic.data if trace.mode == "markers"
                 and trace.name and trace.name.endswith(" MS2")]
    assert len(lines) == 2
    assert len(triangles) == 4
    assert sorted({round(float(trace.x[0]), 1) for trace in triangles}) == [2.8, 2.9]
    assert sorted({float(trace.y[0]) for trace in triangles}) == [10, 30]
    assert plotted[6] == "C18 note"
    assert plotted[26] == "HILIC note"
    c18_rows = plotted[20]
    hilic_rows = plotted[40]
    assert [col.children.children for col in c18_rows[0].children] == ["CE102040", "CE205060"]
    assert len(c18_rows) == 3
    assert len(hilic_rows) == 4
    c18_figures = [col.children.figure for row in c18_rows[1:] for col in row.children]
    hilic_figure = hilic_rows[1].children[0].children.figure
    assert "no CE205060 scan" in hilic_rows[1].children[1].children.children
    assert "no CE102040 scan" in hilic_rows[2].children[0].children.children
    hilic_second_ce = hilic_rows[2].children[1].children.figure
    assert "CE205060" in hilic_second_ce.layout.title.text
    assert all("no CE" in col.children.children for col in hilic_rows[3].children)
    assert all(list(fig.layout.xaxis.range) == list(hilic_figure.layout.xaxis.range)
               and fig.layout.xaxis.fixedrange for fig in [*c18_figures, hilic_second_ce])
    assert hilic_figure.layout.xaxis.range[0] < 5
    assert hilic_figure.layout.xaxis.range[1] > 500
    colors_by_adduct = {line.name: line.line.color for line in lines}
    for row in c18_rows[1:]:
        for col in row.children:
            fig = col.children.figure
            adduct = fig.layout.title.text.split(" peak")[0]
            assert fig.data[0].marker.color == colors_by_adduct[adduct]
    checklist = callbacks["update_checklists"]({"compound_idx": 0})[0]
    assert not any(str(option["label"]).startswith("-- CE") for option in checklist)
    add_manual = next(value["callback"].__wrapped__ for key, value in app_holder[0].callback_map.items()
                      if "sa-manual-rt-0.value" in key and "callback" in value
                      and value["callback"].__wrapped__.__name__ == "add_manual_peak")
    add_manual(1, 2, "[M+H]+", {"compound_idx": 0}, 0)
    manual = db.load_peaks(project)
    manual = manual[(manual.compound_name == "B") & (manual.chromatography == "C18") & manual.is_manual]
    assert len(manual) == 1
    assert manual.iloc[0].filename == filenames[1]
    assert len(db.load_peak_ms2(project).loc[lambda df: df.peak_uid == manual.iloc[0].peak_uid]) == 2


def test_more_than_two_files_are_rejected(project):
    meta = pd.DataFrame([
        dict(mz_rt_uid=f"t{i}", compound_name="A", chromatography="C18",
             polarity="POS", filename=f"file{i}.h5")
        for i in range(3)
    ])
    data = SimpleNamespace(project_db_path=project, expanded_table=meta,
                           ms1_df=pd.DataFrame([dict(mz_rt_uid="t0", filename="file0.h5")]),
                           ms2_df=pd.DataFrame())
    with pytest.raises(ValueError, match=r"file0.h5.*file1.h5.*file2.h5"):
        tools.detect_and_save_peaks(data)


def test_no_detected_peaks_uses_raw_max_and_persists_source(project):
    rows = []
    with duckdb.connect(project) as conn:
        for i, intensity in enumerate((5, 10)):
            uid, filename = f"flat-{i}", f"flat-{i}.h5"
            conn.execute("INSERT INTO sa_extraction_targets VALUES (?, 'c', ?, 'C18', 0, 10, 'CE', '')",
                         [uid, filename])
            conn.execute("INSERT INTO sa_ms1 VALUES (?, ?, 'key', '[M+H]+', [0,1,2,3,4], ?, [100,100,100,100,100], [true,true,true,true,true])",
                         [uid, filename, [intensity] * 5])
            rows.append(dict(mz_rt_uid=uid, filename=filename, compound_uid="c", compound_name="A",
                             adduct="[M+H]+", polarity="POS", chromatography="C18",
                             collision_energy_label="CE", spec_rts=list(range(5)),
                             spec_ints=[intensity] * 5, spec_mzs=[100] * 5))
    data = SimpleNamespace(project_db_path=project, expanded_table=pd.DataFrame(rows),
                           ms1_df=pd.DataFrame(rows), ms2_df=pd.DataFrame())
    tools.detect_and_save_peaks(data)
    assert data.ms1_df.filename.tolist() == ["flat-1.h5"]
    assert db.is_peaks_complete(project)
    loaded = db.load_extraction_from_db(project, {})
    assert loaded.ms1_df.filename.tolist() == ["flat-1.h5"]


def test_closest_ms2_scan_is_selected_even_without_fragments():
    scans = pd.DataFrame([
        dict(scan_rt=2.9, frag_mzs=[], frag_ints=[]),
        dict(scan_rt=2.8, frag_mzs=[50], frag_ints=[100]),
    ])
    assert tools.pick_best_ms2_scan(scans, 3.0).scan_rt == 2.9


def test_missing_ms2_file_is_reported(monkeypatch):
    warnings = []
    monkeypatch.setattr(tools.logger, "warning", lambda message, *args: warnings.append(message % args))
    targets = pd.DataFrame([
        dict(mz_rt_uid=f"t{i}", compound_name="A", chromatography="C18",
             polarity="POS", adduct="H", filename=f"file{i}")
        for i in range(2)
    ])
    scans = pd.DataFrame([dict(mz_rt_uid="t0", filename="file0", scan_rt=3.0,
                               frag_mzs=[50], frag_ints=[100])])
    peak = dict(peak_uid="p", compound_name="A", chromatography="C18",
                polarity="POS", adduct="H", rt_peak=3.0)
    result = tools.associate_peak_ms2(peak, targets, scans)
    assert len(result) == 1
    assert result[0]["filename"] == "file0"
    assert any("file1" in warning for warning in warnings)


def test_alignment_does_not_create_missing_project(tmp_path, monkeypatch):
    root = tmp_path / "missing"
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: {"project": {"standards_run_name": "run", "owner": "jgi"}})
    monkeypatch.setattr(db, "get_project_output_root", lambda _: root)
    monkeypatch.setattr(db, "get_project_db_path", lambda _: str(root / "run.duckdb"))
    with pytest.raises(FileNotFoundError, match="project database not found"):
        run_reverse_rt_align("config.yaml", 1)
    with pytest.raises(ValueError, match="ANNOTATION_RUN_ID"):
        run_reverse_rt_align("config.yaml", None)
    assert not root.exists()


def test_standards_config_loads_and_validates_deposit_tags(tmp_path):
    from metatlas2.load_tools import load_standards_config

    input_file = tmp_path / "input.csv"
    input_file.touch()
    config = tmp_path / "config.yaml"
    prefix = f"PROJECT:\n  standards_run_name: run\n  standards_input_file: {input_file}\n"
    config.write_text(prefix + "TAGS:\n  atlas: [' EMA ', specific, specific]\n")
    assert load_standards_config(str(config))["tags"]["atlas"] == ["EMA", "specific"]
    config.write_text(prefix + "TAGS:\n  atlas: ['one;two']\n")
    with pytest.raises(ValueError, match="semicolons"):
        load_standards_config(str(config))


def test_standards_config_validates_chem_source(tmp_path):
    from metatlas2.load_tools import load_standards_config

    input_file = tmp_path / "input.csv"
    input_file.touch()
    config = tmp_path / "config.yaml"
    prefix = f"PROJECT:\n  standards_run_name: run\n  standards_input_file: {input_file}\n"
    config.write_text(prefix + "  chem_source: 'Vendor 123'\n")
    assert load_standards_config(str(config))["project"]["chem_source"] == "Vendor 123"
    config.write_text(prefix + "  chem_source: 123\n")
    with pytest.raises(ValueError, match="PROJECT.chem_source must be a string"):
        load_standards_config(str(config))


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("upload_to_gdrive", [False, True])
def test_reverse_alignment_uploads_output(project, tmp_path, monkeypatch, mock_standards_drive_upload, skip, upload_to_gdrive):
    config = {"project": {"standards_run_name": "run"}}
    annotation = tmp_path / "outputs" / "annotation_run_1"
    alignment = annotation / "rt_alignment_4"
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(db, "get_project_db_path", lambda _: project)
    run = Mock(return_value=(4, alignment))
    monkeypatch.setattr(align, "run_reverse_rt_alignment", run)
    display = Mock()
    monkeypatch.setattr(align, "display", display)
    run_reverse_rt_align("config.yaml", 1, skip_rt_alignment=skip, upload_to_gdrive=upload_to_gdrive)
    assert run.call_args.args[3] == 1
    assert run.call_args.kwargs == {"skip_rt_alignment": skip}
    assert "Alignment run 4" in display.call_args.args[0].data
    assert "annotation run 1" in display.call_args.args[0].data
    if upload_to_gdrive:
        mock_standards_drive_upload.assert_called_once_with(config, annotation, [alignment])
    else:
        mock_standards_drive_upload.assert_not_called()


def test_database_helpers_load_state(project):
    assert db.load_extraction_target_filenames(project) == ["file.h5"]
    assert db.load_gui_checkpoint(project) == "A"
    assert db.load_compound_notes(project, "absent") == {}
    db.validate_annotation_choices(project)


def test_annotation_runs_and_multiple_alignments_use_chosen_run(project, tmp_path, monkeypatch):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}},
              "tags": {"atlas": ["EMA", "test"]}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    data = SimpleNamespace(config=config, project_db_path=project)
    summary.run_standards_annotation_summary(data, max_workers=1)
    with pytest.raises(ValueError, match="No draft annotation run.*run 1 is already summarized"):
        summary.run_standards_annotation_summary(data, max_workers=1)
    annotation_dir = tmp_path / "outputs" / "annotation_run_1"
    assert [p.name for p in (tmp_path / "outputs").iterdir()] == ["annotation_run_1"]
    assert db.get_annotation_run(project, 1)["status"] == "frozen"
    assert db.get_annotation_run(project, 1)["output_dir"] == str(annotation_dir)
    table = pd.read_csv(annotation_dir / "StandardsAnnotation_Summary_run.csv")
    assert table.columns[:2].tolist() == ["Compound Label", "Compound Name"]
    assert table.columns[-1] == "GUI Peak Num"
    assert table["GUI Peak Num"].tolist() == [1, 3]
    assert table["Compound Label"].tolist() == ["A (peak1)", "A (peak2)"]
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    mock_alignment_model(monkeypatch)
    out = tmp_path / "outputs"
    alignment_id, first_run = align.run_reverse_rt_alignment("config", project, out, 1)
    assert (alignment_id, first_run) == (1, annotation_dir / "rt_alignment_1")
    csv = first_run / "StandardsAnnotation_Summary_Aligned_run.csv"
    first = pd.read_csv(csv)
    assert first["tags"].tolist() == ["EMA;test", "EMA;test"]
    assert first["Peak #"].tolist() == [1, 2]
    assert first["Compound Label"].tolist() == ["A (peak1)", "A (peak2)"]
    original_csv = csv.read_bytes()
    original_plot = (first_run / "model_figures" / "RTAlignment-Summary__run__C18.pdf").read_bytes()

    # GUI changes after freezing go to a new draft run; run 1 is unchanged.
    db.replace_compound_selections(project, "A", ["c"], [("p3", "note")], {"C18": "note"}, ["C18"])
    assert db.get_draft_annotation_run(project) == 2
    assert db.load_analyst_selections(project, 1)["peak_uid"].tolist() == ["p1", "p3"]
    assert csv.read_bytes() == original_csv
    assert (first_run / "model_figures" / "RTAlignment-Summary__run__C18.pdf").read_bytes() == original_plot

    config["rt_alignment"]["C18"]["exclude_inchikeys"] = ["key"]
    assert align.run_reverse_rt_alignment("config", project, out, 1)[0] == 2
    second = pd.read_csv(annotation_dir / "rt_alignment_2" / csv.name)
    assert second["Compound Label"].tolist() == ["A (peak1)", "A (peak2)"]
    assert json.loads((first_run / "alignment_config.json").read_text())["rt_alignment"]["C18"] == {}
    assert json.loads((annotation_dir / "rt_alignment_2" / "alignment_config.json").read_text())[
        "rt_alignment"]["C18"]["exclude_inchikeys"] == ["key"]
    with pytest.raises(ValueError, match="draft"):
        align.run_reverse_rt_alignment("config", project, out, 2)

    summary.run_standards_annotation_summary(data, max_workers=1)
    assert align.run_reverse_rt_alignment("config", project, out, 2)[0] == 3
    result = pd.read_csv(tmp_path / "outputs" / "annotation_run_2" / "rt_alignment_3" / csv.name)
    assert result["Peak #"].tolist() == [1]
    assert result["Compound Label"].tolist() == ["A"]
    assert csv.read_bytes() == original_csv

    runs = db.list_alignment_runs(project)
    assert runs["alignment_run_id"].tolist() == [1, 2, 3]
    assert runs["annotation_run_id"].tolist() == [1, 1, 2]
    assert runs["n_aligned_peaks"].tolist() == [2, 2, 1]
    assert runs["tags"].tolist() == ["EMA;test"] * 3
    models = db.load_alignment_models(project, 2)
    assert json.loads(models.loc[0, "params"])["exclude_inchikeys"] == ["key"]
    aligned = db.load_aligned_peaks(project, 1)
    assert aligned["peak_uid"].tolist() == ["p1", "p3"]
    assert aligned["peak_label"].tolist() == ["A (peak1)", "A (peak2)"]
    assert db.list_annotation_runs(project)["n_alignment_runs"].tolist() == [2, 1]


@pytest.mark.parametrize("use_tags", [True, False])
def test_reverse_alignment_snapshots_resolved_atlas(project, tmp_path, monkeypatch, use_tags):
    from metatlas2 import atlas_sheet

    selector = {"tags": ["QC"], "polarity": "POS"} if use_tags else {"uid": "qc-uid", "polarity": "POS"}
    config = {
        "project": {"standards_run_name": "run", "owner": "jgi"},
        "rt_alignment": {"C18": {"ATLAS": selector}},
    }
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    atlas = SimpleNamespace(
        atlas_uid="qc-uid", chromatography="C18", polarity="POS",
        atlas_category=align.wfo.AtlasCategory.QC, compound_mzrts={"c": object()},
        to_dataframe=lambda: pd.DataFrame([{"x": 1}]),
    )
    monkeypatch.setattr(align.wfo.Atlas, "from_database", classmethod(lambda cls, db, uid: atlas))
    if use_tags:
        monkeypatch.setattr(atlas_sheet, "read_centralized_atlas_sheet", lambda: pd.DataFrame())
        monkeypatch.setattr(atlas_sheet, "select_centralized_atlas_rows", lambda *_args, **_kwargs: pd.DataFrame())
        monkeypatch.setattr(atlas_sheet, "_create_or_reuse_atlas", lambda *_args, **_kwargs: "qc-uid")
    load_qc_atlas = align.load_qc_atlas
    mock_alignment_model(monkeypatch)
    monkeypatch.setattr(align, "load_qc_atlas", load_qc_atlas)
    make_table = summary.make_standards_annotation_table
    captured = []
    def capture_table(data, *args, **kwargs):
        captured.append(data.copy())
        return make_table(data, *args, **kwargs)
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    monkeypatch.setattr(summary, "make_standards_annotation_table", capture_table)

    _, alignment_dir = align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)

    chrom = json.loads((alignment_dir / "alignment_config.json").read_text())["rt_alignment"]["C18"]
    assert chrom["ATLAS"] == selector
    assert chrom["resolved_qc_atlas_uid"] == "qc-uid"
    assert captured[-1]["qc_atlas_uid"].tolist() == ["qc-uid", "qc-uid"]
    model = db.load_alignment_models(project, 1).iloc[0]
    assert model["qc_atlas_uid"] == "qc-uid"
    assert json.loads(model["qc_atlas_selector"]) == selector
    assert "ATLAS" not in json.loads(model["params"])
    assert json.loads(db.get_alignment_run(project, 1)["config_snapshot"])["rt_alignment"]["C18"]["ATLAS"] == selector


def test_failed_summary_publishes_nothing_and_keeps_draft(project, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    def fail(**_):
        raise RuntimeError("figure failed")
    monkeypatch.setattr(summary, "make_standards_annotation_figures", fail)
    data = SimpleNamespace(config={"project": {"standards_run_name": "run", "owner": "jgi"}}, project_db_path=project)
    with pytest.raises(RuntimeError, match="figure failed"):
        summary.run_standards_annotation_summary(data)
    assert list((tmp_path / "outputs").iterdir()) == []
    assert db.get_draft_annotation_run(project) == 1
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr(db, "freeze_annotation_run", Mock(side_effect=RuntimeError("db failed")))
    with pytest.raises(RuntimeError, match="db failed"):
        summary.run_standards_annotation_summary(data)
    assert list((tmp_path / "outputs").iterdir()) == []
    assert db.get_draft_annotation_run(project) == 1


def test_alignment_rejects_draft_or_missing_run(project, tmp_path, monkeypatch):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}}
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    with pytest.raises(ValueError, match="draft"):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    with pytest.raises(ValueError, match="does not exist"):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 7)
    assert not (tmp_path / "outputs").exists()
    assert db.list_alignment_runs(project).empty


def test_alignment_rejects_missing_peak_results(project, tmp_path, monkeypatch):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    mock_alignment_model(monkeypatch)
    monkeypatch.setattr(align, "apply_reverse_rt_model", lambda df, _: df.iloc[:1].copy())
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    with pytest.raises(ValueError, match="identities do not match"):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    assert [p.name for p in (tmp_path / "outputs" / "annotation_run_1").iterdir()] == [
        "StandardsAnnotation_Summary_run.csv"]
    assert db.list_alignment_runs(project).empty


def test_alignment_rejects_nonfinite_results(project, tmp_path, monkeypatch):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    mock_alignment_model(monkeypatch)
    monkeypatch.setattr(align, "apply_reverse_rt_model", lambda df, _: df.assign(
        rt_peak_aligned=float("nan"), rt_min_aligned=df["rt_min"], rt_max_aligned=df["rt_max"],
    ))
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    with pytest.raises(ValueError, match="Aligned RT values must be finite"):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    assert not list((tmp_path / "outputs" / "annotation_run_1").glob("*rt_alignment*"))
    assert db.list_alignment_runs(project).empty


def test_real_review_figure_is_written_and_verified(project, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}}
    data = SimpleNamespace(
        config=config, project_db_path=project,
        ms1_df=pd.DataFrame(columns=["mz_rt_uid"]), tic_by_file={},
        expanded_table=pd.DataFrame([{
            "compound_name": "A", "mz_rt_uid": "t", "adduct": "[M+H]+",
            "chromatography": "C18", "polarity": "POS", "collision_energy_label": "CE",
        }]),
    )
    summary.run_standards_annotation_summary(data, max_workers=1)
    assert list((tmp_path / "outputs" / "annotation_run_1").glob("summary_figures/**/*.pdf"))


def test_figure_worker_failure_is_raised(project, tmp_path, monkeypatch):
    class FailedFuture:
        def result(self):
            raise RuntimeError("plot failure")

    class Executor:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def submit(self, *_):
            return FailedFuture()

    monkeypatch.setattr(summary, "ProcessPoolExecutor", Executor)
    selections = review.fix_peak_labels(db.load_selected_peaks_with_ms2(project))
    data = SimpleNamespace(
        ms1_df=pd.DataFrame(columns=["mz_rt_uid"]),
        tic_by_file={},
        expanded_table=pd.DataFrame([{
            "compound_name": "A", "mz_rt_uid": "t", "adduct": "[M+H]+",
            "chromatography": "C18", "polarity": "POS", "collision_energy_label": "CE",
        }]),
    )
    with pytest.raises(RuntimeError, match="plot failure"):
        summary.make_standards_annotation_figures(data, selections, tmp_path)


def test_manual_peak_and_ms2_are_inserted_together(project):
    peak = db.load_peaks(project).iloc[0].to_dict()
    peak.update(peak_uid="manual", rt_peak=4, rt_min=3.9, rt_max=4.1, is_manual=True)
    ms2 = {"peak_uid": "manual", "frag_mzs": [100.0], "frag_ints": [10.0], "filename": "file.h5"}
    db.save_manual_peak_to_db(project, peak, [ms2])
    assert "manual" in set(db.load_peak_ms2(project)["peak_uid"])
    duplicate = dict(peak, peak_uid="other")
    with pytest.raises(duckdb.ConstraintException):
        db.save_manual_peak_to_db(project, duplicate, [dict(ms2, peak_uid="other")])
    assert "other" not in set(db.load_peaks(project)["peak_uid"])
    assert "other" not in set(db.load_peak_ms2(project)["peak_uid"])


def test_real_model_applies_to_chosen_run_selections(project, tmp_path, monkeypatch):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    qc_runs = [SimpleNamespace(filename="qc1.h5"), SimpleNamespace(filename="qc2.h5")]
    monkeypatch.setattr(align, "get_qc_runs_for_chrom", lambda *_args, **_kwargs: qc_runs)
    monkeypatch.setattr(align, "load_qc_atlas", lambda *_: SimpleNamespace(atlas_uid="qc-uid", to_dataframe=lambda: pd.DataFrame([{"x": 1}])))
    monkeypatch.setattr(align, "extract_qc_ms1", lambda *_: pd.DataFrame([{"x": 1}]))
    modeling = pd.DataFrame({
        "exp_rt_peak": [1., 2., 3.], "atlas_rt_peak": [1.5, 2.5, 3.5],
        "predicted_rt": [1.5, 2.5, 3.5], "residual": [0., 0., 0.],
        "label": ["A", "B", "C"], "inchi_key": ["a", "b", "c"],
        "adduct": ["[M+H]+"] * 3,
    })
    monkeypatch.setattr(align, "build_reverse_rt_model", lambda *_: {
        "model_type": "median_offset", "offset": .5, "r2": 1., "rmse": 0., "degree": 0,
        "coefficients": align.np.array([]), "intercept": .5, "equation": "RT_aligned = RT_atlas + 0.5",
        "modeling_df": modeling,
    })
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    db.replace_compound_selections(project, "A", ["c"], [("p2", "changed")], {"C18": "changed"}, ["C18"])
    _, run = align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    aligned_summary = pd.read_csv(run / "StandardsAnnotation_Summary_Aligned_run.csv")
    assert aligned_summary["RT peak"].tolist() == [1., 3.]
    assert aligned_summary["Aligned RT peak"].tolist() == [1.5, 3.5]
    assert aligned_summary["RT diff"].tolist() == [.5, .5]
    assert aligned_summary["Analyst Notes"].tolist() == ["note", "note"]
    assert (run / "model_figures" / "RTAlignment-Summary__run__C18.pdf").is_file()
    assert (run / "alignment_config.json").is_file()
    assert not list(run.rglob("Atlas-Summary-Table__*.csv"))

    model = db.load_alignment_models(project, 1).iloc[0]
    assert (model["chromatography"], model["model_type"], model["offset"], model["r2"]) == (
        "C18", "median_offset", .5, 1.)
    assert model["equation"] == "RT_aligned = RT_atlas + 0.5"
    assert list(model["qc_files"]) == ["qc1.h5", "qc2.h5"]
    assert model["n_points"] == 3
    points = db.load_alignment_qc_points(project, 1)
    assert points["label"].tolist() == ["A", "B", "C"]
    assert points["atlas_rt_peak"].tolist() == [1.5, 2.5, 3.5]
    peaks = db.load_aligned_peaks(project, 1)
    assert peaks["rt_peak_aligned"].tolist() == [1.5, 3.5]
    assert peaks["rt_delta"].tolist() == [.5, .5]


@pytest.mark.parametrize("failure", ["publish", "record"])
def test_failed_alignment_publication_is_incomplete(project, tmp_path, monkeypatch, failure):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    mock_alignment_model(monkeypatch)
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    annotation_dir = tmp_path / "outputs" / "annotation_run_1"
    if failure == "publish":
        target = (align.os, "replace")
    else:
        target = (db, "record_alignment_run")
    original = getattr(*target)
    monkeypatch.setattr(*target, Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    assert [p.name for p in annotation_dir.iterdir()] == ["StandardsAnnotation_Summary_run.csv"]
    assert db.list_alignment_runs(project).empty
    monkeypatch.setattr(*target, original)
    assert align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)[0] == 1
    assert (annotation_dir / "rt_alignment_1" / "StandardsAnnotation_Summary_Aligned_run.csv").is_file()


def test_alignment_summary_preserves_rows_and_adds_aligned_rts(project, tmp_path, monkeypatch):
    template = ("20260720_JGI_PLM-RS_512633_SorghSwg_final_EXP120B_HILICZ_"
                "USHXG03401_POS_MS2_RefStd-1_FA-CoA_1_Rg70to1050-{}norm-RxnMix-S1_Run{}.h5")
    filenames = [template.format("CE102040", 135), template.format("CE205060", 136)]
    with duckdb.connect(project) as conn:
        conn.execute("UPDATE sa_extraction_targets SET annotation_notes='C18 note' WHERE mz_rt_uid='t'")
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('hilic', 'c', 'file.h5', 'HILIC', 0, 10, 'CE', 'HILIC note')"
        )
        conn.execute("INSERT INTO sa_compounds VALUES ('neg', 'A', 'key', '', '', '[M-H]-', 'NEG', 100)")
        conn.execute(
            "INSERT INTO sa_extraction_targets VALUES ('hilic-neg', 'neg', 'file.h5', 'HILIC', 0, 10, 'CE', 'HILIC note')"
        )
    peak = db.load_peaks(project).iloc[0].to_dict()
    peak.update(peak_uid="hilic-peak", chromatography="HILIC", mz_rt_uid="hilic",
                rt_peak=4.0, rt_min=3.8, rt_max=4.3, peak_num=5)
    scans = [
        dict(peak_uid="hilic-peak", filename=filename, scan_rt=4 + i * .1,
             precursor_mz=100 + i, precursor_intensity=200 + i, collision_energy=10 + i,
             frag_mzs=[50 + i], frag_ints=[100 + i])
        for i, filename in enumerate(filenames)
    ]
    db.save_manual_peak_to_db(project, peak, scans)
    neg_peak = dict(peak, peak_uid="hilic-neg-peak", compound_uid="neg", mz_rt_uid="hilic-neg",
                    polarity="NEG", adduct="[M-H]-", rt_peak=6., rt_min=5.8, rt_max=6.2)
    db.save_manual_peak_to_db(project, neg_peak, [])
    db.replace_compound_selections(
        project, "A", ["c", "neg"], [("p1", "c18 analyst"), ("p3", "c18 analyst"),
                                     ("hilic-peak", "hilic analyst"), ("hilic-neg-peak", "hilic analyst")],
        {"C18": "c18 analyst", "HILIC": "hilic analyst"}, ["C18", "HILIC"],
    )
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}, "HILIC": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    original_apply = align.apply_reverse_rt_model
    mock_alignment_model(monkeypatch)
    monkeypatch.setattr(align, "build_reverse_rt_model", lambda _ms1, _atlas, _config, chrom: {
        "model_type": "median_offset", "offset": .5 if chrom == "C18" else -1.,
        "r2": 1., "rmse": 0., "modeling_df": pd.DataFrame([{"x": 1}]),
    })
    monkeypatch.setattr(align, "apply_reverse_rt_model", original_apply)
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    annotation_dir = tmp_path / "outputs" / "annotation_run_1"
    measured = pd.read_csv(annotation_dir / "StandardsAnnotation_Summary_run.csv")
    _, run = align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    aligned = pd.read_csv(run / "StandardsAnnotation_Summary_Aligned_run.csv")
    columns = measured.columns.tolist()
    for name in ("RT peak", "RT min", "RT max"):
        columns.insert(columns.index(name) + 1, f"Aligned {name}")
    columns.insert(columns.index("Aligned RT peak") + 1, "RT diff")
    columns.append("tags")
    assert aligned.columns.tolist() == columns
    pd.testing.assert_frame_equal(aligned[measured.columns], measured)
    offsets = aligned["Chromatography"].map({"C18": .5, "HILIC": -1.})
    assert aligned["RT diff"].tolist() == pytest.approx(offsets.tolist())
    for name in ("RT peak", "RT min", "RT max"):
        assert aligned[f"Aligned {name}"].tolist() == pytest.approx((measured[name] + offsets).tolist())
    hilic = aligned.loc[(aligned["Chromatography"] == "HILIC") & (aligned["Polarity"] == "POS")].iloc[0]
    assert hilic["Annotation Notes"] == "HILIC note"
    assert json.loads(hilic["CE102040 spectrum"]) == [[50.0], [100.0]]
    assert json.loads(hilic["CE205060 spectrum"]) == [[51.0], [101.0]]
    assert aligned.loc[aligned["Chromatography"] == "C18", "CE102040 spectrum"].isna().all()
    assert aligned.loc[aligned["Polarity"] == "NEG", "Aligned RT peak"].tolist() == [5.]
    assert len(list((run / "model_figures").glob("*.pdf"))) == 2


@pytest.mark.parametrize("source", ["median", "file"])
def test_reverse_model_from_real_qc_hdf5(pos_atlas, lcmsrun, tmp_path, monkeypatch, source):
    from matplotlib.axes import Axes

    config = {"extraction": {"max_workers": 1},
              "rt_alignment": {"HILIC": {"model_type": "median_offset",
                                          "min_compounds_for_modeling": 2,
                                          "rt_peak_source": lcmsrun.filename if source == "file" else source}}}
    ms1 = align.extract_qc_ms1([lcmsrun], pos_atlas.to_dataframe(), config, "HILIC")
    assert set(ms1["mz_rt_uid"]) == set(pos_atlas.compound_mzrts)
    model = align.build_reverse_rt_model(ms1, pos_atlas, config, "HILIC")
    assert len(model["modeling_df"]) == 2
    assert model["offset"] == pytest.approx(.1, abs=1e-5)
    captured = {}
    original_table = Axes.table

    def capture_table(ax, *args, **kwargs):
        captured.update(kwargs)
        return original_table(ax, *args, **kwargs)

    monkeypatch.setattr(Axes, "table", capture_table)
    pdf = Path(align.make_rt_alignment_summary_plot(model["modeling_df"], model, tmp_path, "run"))
    assert pdf.read_bytes().startswith(b"%PDF")
    assert list(captured["colLabels"])[1] == "Compound Name"
    expected = model["modeling_df"].sort_values("exp_rt_peak")
    assert list(captured["cellText"][:, 1]) == [
        pos_atlas.compound_mzrts[uid].label for uid in expected["mz_rt_uid"]
    ]
    assert list(captured["cellText"][:, 0]) == [1, 2]
    selected = pd.DataFrame({"rt_peak": [2., 4.], "rt_min": [1.9, 3.8], "rt_max": [2.2, 4.3]})
    result = align.apply_reverse_rt_model(selected, model)
    assert result["rt_peak_aligned"].tolist() == pytest.approx([2.1, 4.1], abs=1e-5)
    assert result["rt_min_aligned"].tolist() == pytest.approx([2., 3.9], abs=1e-5)
    assert result["rt_max_aligned"].tolist() == pytest.approx([2.3, 4.4], abs=1e-5)


def test_reverse_model_respects_minimum_qc_observations(monkeypatch):
    def observation(rt):
        return dict(rt_peak=rt, intensity_peak=10, atlas_rt_peak=rt + .5,
                    atlas_rt_min=0, atlas_rt_max=10, atlas_mz=100,
                    compound_name=str(rt), inchi_key=str(rt), adduct="H", compound_uid=str(rt))

    data = {"a": {"one": observation(1), "two": observation(1)},
            "b": {"one": observation(2), "two": observation(2)},
            "c": {"one": observation(3)}}
    monkeypatch.setattr(align.rat, "_extract_per_file_peak_rts", lambda **_: data)
    config = {"rt_alignment": {"C18": {"model_type": "median_offset",
                                        "min_compounds_for_modeling": 2,
                                        "min_observations_per_compound": 2}}}
    model = align.build_reverse_rt_model(pd.DataFrame([{"x": 1}]),
                                         SimpleNamespace(compound_mzrts={}), config, "C18")
    assert set(model["modeling_df"]["mz_rt_uid"]) == {"a", "b"}
    assert model["offset"] == pytest.approx(.5)


@pytest.mark.parametrize("failure", ["no_config", "no_qc", "no_data", "no_model"])
def test_missing_model_never_publishes_alignment(project, tmp_path, monkeypatch, failure):
    config = {"project": {"standards_run_name": "run", "owner": "jgi"}, "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    mock_alignment_model(monkeypatch)
    if failure == "no_config":
        config["rt_alignment"] = {}
    elif failure == "no_qc":
        monkeypatch.setattr(align, "get_qc_runs_for_chrom", lambda *_args, **_kwargs: [])
    elif failure == "no_data":
        monkeypatch.setattr(align, "extract_qc_ms1", lambda *_: pd.DataFrame())
    else:
        monkeypatch.setattr(align, "build_reverse_rt_model", lambda *_: None)
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    with pytest.raises(ValueError):
        align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    assert not list((tmp_path / "outputs" / "annotation_run_1").glob("*rt_alignment*"))
    assert db.list_alignment_runs(project).empty


@pytest.mark.parametrize("force_add,expected_keys", [
    (False, ["new", "new"]),
    (True, ["existing", "new", "existing", "new", "new"]),
])
def test_deposit_sheet_skips_duplicate_row_identities(monkeypatch, force_add, expected_keys):
    atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS) for _ in range(5)], columns=deposit.ATLAS_COLUMNS)
    atlas["inchi_key"] = ["existing", "new", "existing", "new", ""]
    atlas["adduct"] = ["[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+", "[M+Na]+"]
    atlas["chromatography"] = "HILICZ"
    atlas["polarity"] = "positive"
    atlas["chem_source"] = "Vendor 123"
    atlas.loc[4, "inchi_key"] = "new"
    sheet = Mock(id=123)
    sheet.row_values.return_value = deposit.ATLAS_COLUMNS
    existing_row = [""] * len(deposit.ATLAS_COLUMNS)
    existing_row[deposit.ATLAS_COLUMNS.index("inchi_key")] = "existing"
    existing_row[deposit.ATLAS_COLUMNS.index("adduct")] = "[M+H]+"
    existing_row[deposit.ATLAS_COLUMNS.index("chromatography")] = "HILICZ"
    existing_row[deposit.ATLAS_COLUMNS.index("polarity")] = "positive"
    sheet.get_all_values.return_value = [deposit.ATLAS_COLUMNS, existing_row]
    spreadsheet = Mock()
    spreadsheet.worksheet.return_value = sheet
    monkeypatch.setattr(deposit, "open_centralized_atlas_worksheet", lambda **_: (spreadsheet, sheet))
    warning = Mock()
    monkeypatch.setattr(deposit.logger, "warning", warning)

    deposit.deposit_atlas_to_google_sheet(atlas, force_add=force_add)

    appended = sheet.append_rows.call_args.args[0]
    assert [row[2] for row in appended] == expected_keys
    assert all(row[15:17] == ["Vendor 123", ""] for row in appended)
    assert sheet.get_all_values.call_count == (0 if force_add else 1)
    assert warning.call_count == (0 if force_add else 2)
    spreadsheet.batch_update.assert_called_once()


def test_deposit_sheet_all_existing_does_not_append_or_sort(monkeypatch):
    atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS)], columns=deposit.ATLAS_COLUMNS)
    atlas["inchi_key"] = "existing"
    atlas["adduct"] = "[M+H]+"
    atlas["chromatography"] = "HILICZ"
    atlas["polarity"] = "positive"
    sheet = Mock()
    sheet.row_values.return_value = deposit.ATLAS_COLUMNS
    existing_row = [""] * len(deposit.ATLAS_COLUMNS)
    existing_row[deposit.ATLAS_COLUMNS.index("inchi_key")] = "existing"
    existing_row[deposit.ATLAS_COLUMNS.index("adduct")] = "[M+H]+"
    existing_row[deposit.ATLAS_COLUMNS.index("chromatography")] = "HILICZ"
    existing_row[deposit.ATLAS_COLUMNS.index("polarity")] = "positive"
    sheet.get_all_values.return_value = [deposit.ATLAS_COLUMNS, existing_row]
    spreadsheet = Mock()
    spreadsheet.worksheet.return_value = sheet
    monkeypatch.setattr(deposit, "open_centralized_atlas_worksheet", lambda **_: (spreadsheet, sheet))
    deposit.deposit_atlas_to_google_sheet(atlas)
    sheet.append_rows.assert_not_called()
    spreadsheet.batch_update.assert_not_called()


@pytest.mark.parametrize("force_add", [True, False])
def test_deposit_sheet_rejects_mismatched_header(monkeypatch, force_add):
    atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS)], columns=deposit.ATLAS_COLUMNS)
    sheet = Mock()
    sheet.row_values.return_value = [column for column in deposit.ATLAS_COLUMNS if column != "chem_source"]
    spreadsheet = Mock()
    monkeypatch.setattr(deposit, "open_centralized_atlas_worksheet", lambda **_: (spreadsheet, sheet))

    with pytest.raises(ValueError, match="header does not match"):
        deposit.deposit_atlas_to_google_sheet(atlas, force_add=force_add)
    sheet.append_rows.assert_not_called()
    spreadsheet.batch_update.assert_not_called()


MS2_TEMPLATE = ("20260720_JGI_PLM-RS_512633_SorghSwg_final_EXP120B_HILICZ_"
                "USHXG03401_POS_MS2_RefStd-1_FA-CoA_1_Rg70to1050-{}norm-RxnMix-S1_Run{}.h5")


@pytest.fixture
def deposit_project(project, tmp_path, monkeypatch):
    """A project with a frozen annotation run 1 and alignment run 1 (offset -0.5 min)."""
    import metatlas2.database_interact as dbi

    monkeypatch.setenv("METATLAS_DATA_DIR", str(tmp_path))
    main_db = tmp_path / "databases" / "main_db" / "metatlas.duckdb"
    dbi.create_metatlas_database(str(main_db))
    filenames = [MS2_TEMPLATE.format("CE102040", 135), MS2_TEMPLATE.format("CE205060", 136)]
    with duckdb.connect(project) as conn:
        conn.execute("UPDATE sa_compounds SET smiles='CCO' WHERE compound_uid='c'")
        conn.execute("UPDATE sa_peaks SET mz_observed=99 WHERE peak_uid='p1'")
        for i, filename in enumerate(filenames):
            conn.execute(
                "INSERT INTO sa_peak_ms2 VALUES ('p1', ?, ?, ?, ?, ?, ?, ?)",
                [1.0, 100 + i, 200 + i, 10 + i, json.dumps([50 + i]), json.dumps([300 + i]), filename],
            )
    config = {"project": {"standards_run_name": "run", "owner": "jgi", "chem_source": "Vendor 123"},
              "rt_alignment": {"C18": {}}}
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(db, "get_project_db_path", lambda _: project)
    monkeypatch.setattr(summary, "make_standards_annotation_figures", lambda **_: None)
    monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
    monkeypatch.setattr(align, "get_project_lcmsruns", lambda *_: [])
    original_apply = align.apply_reverse_rt_model
    mock_alignment_model(monkeypatch)
    monkeypatch.setattr(align, "build_reverse_rt_model", lambda *_: {
        "model_type": "median_offset", "offset": -.5, "r2": 1., "rmse": 0.,
        "modeling_df": pd.DataFrame([{"x": 1}]),
    })
    monkeypatch.setattr(align, "apply_reverse_rt_model", original_apply)
    sheet = Mock(side_effect=lambda atlas, force_add=False: ["appended"] * len(atlas))
    monkeypatch.setattr(deposit, "deposit_atlas_to_google_sheet", sheet)
    compounds = Mock()
    monkeypatch.setattr("metatlas2.workflow_objects.add_compounds_from_files", compounds)
    monkeypatch.setattr("metatlas2.database_interact.get_compound_uids_by_inchi_keys",
                        lambda _path, keys: dict.fromkeys(keys, "uid"))
    monkeypatch.setattr("IPython.display.display", lambda *_: None)
    summary.run_standards_annotation_summary(SimpleNamespace(config=config, project_db_path=project))
    alignment_id, alignment_dir = align.run_reverse_rt_alignment("config", project, tmp_path / "outputs", 1)
    return SimpleNamespace(
        db=project, config=config, main_db=main_db, sheet=sheet, compounds=compounds,
        alignment_id=alignment_id, alignment_dir=alignment_dir,
        deposit_dir=lambda n: alignment_dir / f"deposit_{n}",
    )


def _read_preview(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, sep="\t" if path.suffix == ".tab" else ",")


def _write_preview(frame, path):
    frame.to_csv(path, index=False, sep="\t" if path.suffix == ".tab" else ",")


def _msms_refs(main_db):
    with duckdb.connect(str(main_db), read_only=True) as conn:
        return conn.execute(
            "SELECT database, ref_id, inchi_key, precursor_mz, collision_energy, mz, intensities "
            "FROM msms_refs ORDER BY ref_id"
        ).fetchall()


def test_deposit_column_types_match_file_schemas():
    assert list(db.ATLAS_COLUMN_TYPES) == deposit.ATLAS_COLUMNS
    assert list(db.MSMS_COLUMN_TYPES) == deposit.MSMS_COLUMNS


def test_deposit_requires_alignment_id_and_preview(deposit_project):
    with pytest.raises(ValueError, match="ANNOTATION_RUN_ID and ALIGNMENT_RUN_ID"):
        run_atlas_and_msms_deposit("config", 1, None)
    with pytest.raises(ValueError, match="ANNOTATION_RUN_ID and ALIGNMENT_RUN_ID"):
        run_atlas_and_msms_deposit("config", None, 1)
    with pytest.raises(ValueError, match="aligned annotation run 1, not annotation run 2"):
        run_atlas_and_msms_deposit("config", 2, 1)
    with pytest.raises(ValueError, match="DRY_RUN = True first"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    with pytest.raises(ValueError, match="does not exist"):
        run_atlas_and_msms_deposit("config", 1, 9)
    assert db.list_deposits(deposit_project.db).empty


def test_deposit_dry_run_builds_previews_from_database(deposit_project, monkeypatch):
    p = deposit_project
    warnings = []
    monkeypatch.setattr(deposit.logger, "warning", lambda message, *args: warnings.append(message % args))
    reviewed = pd.read_csv(p.alignment_dir / "StandardsAnnotation_Summary_Aligned_run.csv")
    (p.alignment_dir / "StandardsAnnotation_Summary_Aligned_run.csv").unlink()  # not an input
    # A newer draft annotation run does not affect the deposit of alignment run 1.
    db.replace_compound_selections(p.db, "A", ["c"], [("p2", "changed")], {"C18": "changed"}, ["C18"])

    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)

    directory = p.deposit_dir(1)
    atlas = _read_preview(directory / "Atlas_Deposit_run.csv")
    msms = _read_preview(directory / "MsmsRefs_Deposit_run.tab")
    assert atlas.columns.tolist() == deposit.ATLAS_COLUMNS + ["sa_entry_id", "sa_peak_uid"]
    assert atlas["sa_entry_id"].tolist() == ["1", "2"]
    assert atlas["sa_peak_uid"].tolist() == ["p1", "p3"]
    assert atlas["label"].tolist() == reviewed["Compound Label"].tolist()
    assert atlas["rt_peak"].astype(float).tolist() == reviewed["Aligned RT peak"].tolist() == [.5, 2.5]
    assert atlas["rt_min"].astype(float).tolist() == pytest.approx(reviewed["Aligned RT min"].tolist())
    assert atlas["rt_max"].astype(float).tolist() == pytest.approx(reviewed["Aligned RT max"].tolist())
    assert atlas["mz"].astype(float).tolist() == [100, 100]
    assert atlas["mz_tolerance"].astype(float).tolist() == [5, 5]
    assert atlas["tags"].tolist() == ["EMA", "EMA"]
    assert atlas["polarity"].tolist() == ["positive", "positive"]
    assert atlas["identification_notes"].tolist() == ["note; annotation", "note; annotation"]
    assert atlas["formula"].tolist() == ["C2H6O", "C2H6O"]
    assert atlas["chem_source"].tolist() == ["Vendor 123", "Vendor 123"]
    assert atlas["lcmsrun_source"].tolist() == ["file.h5", "file.h5"]
    assert atlas["date_created"].str.fullmatch(r"\d{8}").all()
    assert msms.columns.tolist() == deposit.MSMS_COLUMNS + [
        "sa_entry_id", "sa_peak_uid", "sa_ce_label", "sa_measured_collision_energy"]
    assert msms["sa_peak_uid"].tolist() == ["p1", "p1"]
    assert msms["sa_ce_label"].tolist() == ["CE102040", "CE205060"]
    assert msms["sa_measured_collision_energy"].astype(float).tolist() == [10, 11]
    assert msms["id"].str.fullmatch(r"[0-9a-f]{32}").all() and msms["id"].is_unique
    assert msms["name"].tolist() == ["A (peak1)"] * 2
    assert msms["collision_energy"].tolist() == ["ramp-102040norm", "ramp-205060norm"]
    assert msms["instrument"].tolist() == ["EXP120B", "EXP120B"]
    assert msms["precursor_mz"].astype(float).tolist() == [100, 101]
    assert [json.loads(v) for v in msms["spectrum"]] == [[[50.0], [300.0]], [[51.0], [301.0]]]
    assert NewMsmsRefsConfig._parse_spectrum_column(msms.loc[0, "spectrum"]) == ([50.0], [300.0])
    assert any("No CE102040 spectrum for A (peak2)" in message for message in warnings)

    record = db.get_deposit(p.db, 1)
    assert (record["status"], record["alignment_run_id"], record["chem_source"]) == ("previewed", 1, "Vendor 123")
    generated_atlas, generated_msms = db.load_deposit_rows(p.db, 1, "generated")
    assert len(generated_atlas) == 2 and len(generated_msms) == 2
    p.sheet.assert_not_called()
    assert _msms_refs(p.main_db) == []

    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    assert db.get_deposit(p.db, 1)["status"] == "superseded"
    assert db.get_deposit(p.db, 2)["status"] == "previewed"
    assert _read_preview(p.deposit_dir(2) / "MsmsRefs_Deposit_run.tab")["id"].tolist() == msms["id"].tolist()
    with pytest.raises(ValueError, match="superseded"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False, deposit_id=1)


def test_deposit_reads_back_edits_and_records_exact_deposit(deposit_project):
    p = deposit_project
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    directory = p.deposit_dir(1)
    atlas_path, msms_path = directory / "Atlas_Deposit_run.csv", directory / "MsmsRefs_Deposit_run.tab"
    atlas = _read_preview(atlas_path)
    atlas.loc[0, "identification_notes"] = "edited note"
    atlas.loc[1, ["rt_min", "rt_peak", "rt_max"]] = ["2.3", "2.45", "2.6"]
    added = atlas.loc[0].copy()
    added[["label", "sa_entry_id", "sa_peak_uid", "adduct"]] = ["A manual", "", "", "[M+Na]+"]
    atlas = pd.concat([atlas, added.to_frame().T], ignore_index=True)
    _write_preview(atlas, atlas_path)
    msms = _read_preview(msms_path)
    kept_id, removed_id = msms["id"].tolist()
    _write_preview(msms.iloc[:1], msms_path)

    run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)

    record = db.get_deposit(p.db, 1)
    assert record["status"] == "deposited"
    assert (record["compounds_stage"], record["atlas_stage"], record["msms_stage"]) == ("done",) * 3
    assert record["atlas_final_sha256"] == hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    p.compounds.assert_called_once()
    assert p.compounds.call_args.args[0] == [str(directory / "Atlas_Deposit_run.final.csv")]
    sent = p.sheet.call_args.args[0]
    assert sent.columns.tolist() == deposit.ATLAS_COLUMNS
    assert sent["label"].tolist() == ["A (peak1)", "A (peak2)", "A manual"]
    assert sent["identification_notes"].tolist()[0] == "edited note"
    assert sent["rt_peak"].tolist() == [.5, 2.45, .5]
    final_atlas_file = pd.read_csv(directory / "Atlas_Deposit_run.final.csv")
    assert final_atlas_file.columns.tolist() == deposit.ATLAS_COLUMNS
    final_msms_file = pd.read_csv(directory / "MsmsRefs_Deposit_run.final.tab", sep="\t")
    assert final_msms_file.columns.tolist() == deposit.MSMS_COLUMNS
    assert final_msms_file["id"].tolist() == [kept_id]
    assert _msms_refs(p.main_db) == [("metatlas", kept_id, "key", 100.0, 10.0, [50.0], [300.0])]

    final_atlas, final_msms = db.load_deposit_rows(p.db, 1, "final")
    assert final_atlas["change_status"].tolist() == ["edited", "edited", "added"]
    assert final_atlas["sa_entry_id"].tolist() == [1, 2, 3]
    assert final_atlas["sheet_outcome"].tolist() == ["appended"] * 3
    assert final_msms["change_status"].tolist() == ["unchanged"]
    assert final_msms["import_outcome"].tolist() == ["inserted"]
    assert final_msms["main_db_ref_uid"].notna().all()
    changes = db.load_deposit_changes(p.db, 1)
    summary_rows = set(zip(changes["file"], changes["sa_entry_id"], changes["change_type"],
                           changes["column_name"].fillna("")))
    assert summary_rows == {
        ("atlas", 1, "edited", "identification_notes"), ("atlas", 2, "edited", "rt_min"),
        ("atlas", 2, "edited", "rt_peak"), ("atlas", 2, "edited", "rt_max"),
        ("atlas", 3, "added", ""), ("msms", 2, "removed", ""),
    }
    assert len(pd.read_csv(directory / "deposit_changes.csv")) == len(changes)

    # Edits are written back to the alignment run's deposit content.
    entries, msms_entries = db.load_deposit_entries(p.db, 1)
    assert entries["identification_notes"].tolist()[0] == "edited note"
    assert entries["rt_peak"].tolist() == [.5, 2.45, .5]
    assert entries["origin"].tolist() == ["generated", "generated", "manual"]
    assert entries["last_modified_by_deposit"].tolist() == [1, 1, 1]
    assert msms_entries["id"].tolist() == [kept_id]
    _, all_msms = db.load_deposit_entries(p.db, 1, include_excluded=True)
    assert all_msms.set_index("id").loc[removed_id, "is_excluded"]

    with pytest.raises(ValueError, match="already deposited"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    again = _read_preview(p.deposit_dir(2) / "Atlas_Deposit_run.csv")
    assert again["label"].tolist() == ["A (peak1)", "A (peak2)", "A manual"]
    assert again["sa_entry_id"].tolist() == ["1", "2", "3"]
    assert _read_preview(p.deposit_dir(2) / "MsmsRefs_Deposit_run.tab")["id"].tolist() == [kept_id]
    assert db.get_deposit(p.db, 1)["status"] == "deposited"


@pytest.mark.parametrize("target,edit,message", [
    ("msms", lambda f: f.assign(spectrum="[[1], []]"), "spectrum"),
    ("msms", lambda f: f.assign(spectrum="[[], []]"), "empty"),
    ("msms", lambda f: f.assign(id="same"), "duplicate id"),
    ("atlas", lambda f: f.assign(sa_entry_id="7"), "not in this preview"),
    ("atlas", lambda f: f.assign(sa_entry_id="1"), "more than once"),
    ("atlas", lambda f: f.drop(columns="tags"), "missing columns"),
    ("atlas", lambda f: f.assign(extra="x"), "unexpected columns"),
    ("atlas", lambda f: f.assign(rt_min="9"), "rt_min <= rt_peak"),
    ("atlas", lambda f: f.assign(polarity="POS"), "polarity"),
    ("atlas", lambda f: f.assign(inchi_key=""), "inchi_key is required"),
])
def test_deposit_rejects_invalid_edits_without_changing_db(deposit_project, target, edit, message):
    p = deposit_project
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    name = "Atlas_Deposit_run.csv" if target == "atlas" else "MsmsRefs_Deposit_run.tab"
    path = p.deposit_dir(1) / name
    _write_preview(edit(_read_preview(path)), path)
    with pytest.raises(ValueError, match=message):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    assert db.get_deposit(p.db, 1)["status"] == "previewed"
    assert db.load_deposit_rows(p.db, 1, "final")[0].empty
    assert db.load_deposit_changes(p.db, 1).empty
    p.sheet.assert_not_called()
    p.compounds.assert_not_called()


def test_deposit_retry_resumes_and_refuses_changed_files(deposit_project, monkeypatch):
    p = deposit_project
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    import_refs = Mock(side_effect=RuntimeError("import failed"))
    original_execute = NewMsmsRefsConfig.execute
    monkeypatch.setattr(NewMsmsRefsConfig, "execute", import_refs)
    with pytest.raises(RuntimeError, match="import failed"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    record = db.get_deposit(p.db, 1)
    assert (record["status"], record["atlas_stage"], record["msms_stage"]) == ("in_progress", "done", "pending")
    assert db.load_deposit_rows(p.db, 1, "final")[0]["sheet_outcome"].tolist() == ["appended"] * 2
    with pytest.raises(ValueError, match="partially complete"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)

    atlas_path = p.deposit_dir(1) / "Atlas_Deposit_run.csv"
    original = atlas_path.read_bytes()
    atlas_path.write_bytes(original.replace(b"note; annotation", b"late edit"))
    monkeypatch.setattr(NewMsmsRefsConfig, "execute", original_execute)
    with pytest.raises(ValueError, match="changed since"):
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    atlas_path.write_bytes(original)
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
    assert p.sheet.call_count == 1
    assert p.compounds.call_count == 1
    assert db.get_deposit(p.db, 1)["status"] == "deposited"
    assert len(_msms_refs(p.main_db)) == 2


@pytest.mark.parametrize("skip_atlas_upload", [True, False])
@pytest.mark.parametrize("skip_msms_refs_deposit", [True, False])
def test_deposit_skip_flags_are_recorded_and_can_be_resumed(
    deposit_project, skip_atlas_upload, skip_msms_refs_deposit,
):
    p = deposit_project
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    run_atlas_and_msms_deposit(
        "config", 1, 1, dry_run=False, skip_atlas_upload=skip_atlas_upload,
        skip_msms_refs_deposit=skip_msms_refs_deposit,
    )
    record = db.get_deposit(p.db, 1)
    both = skip_atlas_upload and skip_msms_refs_deposit
    assert record["status"] == "deposited"
    assert record["compounds_stage"] == ("skipped" if both else "done")
    assert record["atlas_stage"] == ("skipped" if skip_atlas_upload else "done")
    assert record["msms_stage"] == ("skipped" if skip_msms_refs_deposit else "done")
    assert p.sheet.call_count == int(not skip_atlas_upload)
    assert len(_msms_refs(p.main_db)) == (0 if skip_msms_refs_deposit else 2)
    final_atlas, final_msms = db.load_deposit_rows(p.db, 1, "final")
    assert set(final_atlas["sheet_outcome"]) == {"skipped_by_flag" if skip_atlas_upload else "appended"}
    assert set(final_msms["import_outcome"]) == {"skipped_by_flag" if skip_msms_refs_deposit else "inserted"}
    if skip_atlas_upload or skip_msms_refs_deposit:
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False)
        record = db.get_deposit(p.db, 1)
        assert (record["status"], record["atlas_stage"], record["msms_stage"]) == ("deposited", "done", "done")
        assert p.sheet.call_count == 1
        assert len(_msms_refs(p.main_db)) == 2


@pytest.mark.parametrize("fail", [False, True])
def test_deposit_uploads_files_and_records_url(deposit_project, mock_standards_drive_upload, fail):
    p = deposit_project
    mock_standards_drive_upload.reset_mock()
    if fail:
        mock_standards_drive_upload.side_effect = RuntimeError("Drive unavailable")
        with pytest.raises(RuntimeError, match="Drive unavailable"):
            run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
        assert db.get_deposit(p.db, 1)["status"] == "previewed"
        assert (p.deposit_dir(1) / "Atlas_Deposit_run.csv").is_file()
        p.sheet.assert_not_called()
        run_atlas_and_msms_deposit("config", 1, 1, dry_run=False, upload_to_gdrive=False)
        assert db.get_deposit(p.db, 1)["status"] == "deposited"
        return
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=True)
    mock_standards_drive_upload.assert_called_once_with(
        p.config, p.alignment_dir.parent, [p.deposit_dir(1)],
    )
    assert db.get_deposit(p.db, 1)["gdrive_url"] == mock_standards_drive_upload.return_value
    run_atlas_and_msms_deposit("config", 1, 1, dry_run=False, upload_to_gdrive=False)
    assert mock_standards_drive_upload.call_count == 1


def _source_row(**overrides):
    row = dict(
        peak_uid="p1", peak_label="A", compound_name="A", inchi_key="key", adduct="[M+H]+",
        polarity="POS", chromatography="HILIC", filename="/data/file.h5", mz_theoretical=100.,
        rt_peak_aligned=1., rt_min_aligned=.9, rt_max_aligned=1.1, smiles="", inchi="",
        analyst_notes="", annotation_notes="", tags="EMA", ms2_scans=[],
    )
    row.update(overrides)
    return row


def test_deposit_entries_negative_polarity_and_inchi_structure():
    scan = dict(filename=MS2_TEMPLATE.format("CE205060", 136).replace("_POS_", "_NEG_"),
                frag_mzs=[51.], frag_ints=[301.], precursor_mz=100., collision_energy=20.)
    atlas, msms = deposit.build_deposit_entries(pd.DataFrame([_source_row(
        polarity="NEG", inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", ms2_scans=[scan],
    )]))
    assert atlas.loc[0, "chem_source"] == ""
    assert atlas.loc[0, "polarity"] == msms.loc[0, "polarity"] == "negative"
    assert atlas.loc[0, "chromatography"] == "HILIC"
    assert atlas.loc[0, "lcmsrun_source"] == "file.h5"
    assert atlas.loc[0, "formula"] == msms.loc[0, "formula"] == "C2H6O"
    assert msms["sa_ce_label"].tolist() == ["CE205060"]
    assert msms["sa_measured_collision_energy"].tolist() == [20.]


@pytest.mark.parametrize("filenames,message", [
    (["invalid.h5"], "filename"),
    ([MS2_TEMPLATE.format("CE205060", 1).replace("CE205060norm", "CE205060")], "filename"),
    ([MS2_TEMPLATE.format("CE205060", 1), MS2_TEMPLATE.format("CE205060", 2)], "More than one CE205060"),
])
def test_deposit_entries_reject_bad_ms2_filenames(filenames, message):
    scans = [dict(filename=f, frag_mzs=[51.], frag_ints=[301.], precursor_mz=100., collision_energy=20.)
             for f in filenames]
    with pytest.raises(ValueError, match=message):
        deposit.build_deposit_entries(pd.DataFrame([_source_row(ms2_scans=scans)]))


def test_deposit_reports_empty_spectrum(monkeypatch):
    warnings = []
    monkeypatch.setattr(deposit.logger, "warning", lambda message, *args: warnings.append(message % args))
    assert deposit._spectrum("[[], []]", "CE102040", 2) is None
    assert any("Empty CE102040 spectrum" in message for message in warnings)
    scan = dict(filename=MS2_TEMPLATE.format("CE102040", 1), frag_mzs=[], frag_ints=[],
                precursor_mz=100., collision_energy=20.)
    _, msms = deposit.build_deposit_entries(pd.DataFrame([_source_row(ms2_scans=[scan])]))
    assert msms.empty


@pytest.mark.parametrize("force_add", [False, True])
def test_deposit_msms_import_records_existing_pairs(tmp_path, monkeypatch, force_add):
    import metatlas2.database_interact as dbi

    # The importer resolves the main database from METATLAS_DATA_DIR; never use the real one.
    monkeypatch.setenv("METATLAS_DATA_DIR", str(tmp_path))
    main_db = tmp_path / "databases" / "main_db" / "metatlas.duckdb"
    dbi.create_metatlas_database(str(main_db))
    dbi.batch_save_msms_refs(str(main_db), [{
        "database": "another_source", "inchi_key": "existing", "adduct": "[M+H]+",
        "ref_id": "previous", "mz": [50.0], "intensities": [300.0],
    }])
    msms = pd.DataFrame([
        {"database": "metatlas", "id": f"run_{i}", "name": "one", "inchi_key": key, "adduct": adduct,
         "spectrum": spectrum, "precursor_mz": 100, "polarity": "positive", "exact_mass": 46,
         "collision_energy": "ramp-102040norm", "sa_measured_collision_energy": 10.0 + i}
        for i, (key, adduct, spectrum) in enumerate([
            ("existing", "[M+H]+", "[[50, 105], [300, 400]]"),
            ("existing", "[M+Na]+", "[[50, 105], [300, 400]]"),
            ("new", "[M+H]+", "[[50, 105], [300, 400]]"),
            ("new", "[M+H]+", "[[105], [400]]"),
        ])
    ], columns=deposit.MSMS_COLUMNS + ["sa_measured_collision_energy"]).fillna("")
    path = tmp_path / "refs.tab"
    msms[deposit.MSMS_COLUMNS].to_csv(path, sep="\t", index=False)

    outcomes, ref_uids = deposit._import_msms(msms, path, str(main_db), force_add, NewMsmsRefsConfig)

    assert outcomes == ["inserted" if force_add else "skipped_existing", "inserted", "inserted", "filtered"]
    assert [uid is not None for uid in ref_uids] == [force_add, True, True, False]
    with duckdb.connect(str(main_db), read_only=True) as conn:
        refs = conn.execute(
            "SELECT inchi_key, adduct, ref_id, collision_energy, mz FROM msms_refs ORDER BY ref_id"
        ).fetchall()
    expected = [("existing", "[M+H]+", "previous", None, [50.0])]
    if force_add:
        expected.append(("existing", "[M+H]+", "run_0", 10.0, [50.0]))
    expected += [("existing", "[M+Na]+", "run_1", 11.0, [50.0]), ("new", "[M+H]+", "run_2", 12.0, [50.0])]
    assert refs == expected


def test_standalone_msms_export_uses_selected_scan_metadata(project, tmp_path):
    filename = ("20260720_JGI_PLM-RS_512633_SorghSwg_final_EXP120B_HILICZ_"
                "USHXG03401_POS_MS2_RefStd-1_FA-CoA_1_Rg70to1050-CE205060norm-RxnMix-S1_Run136.h5")
    with duckdb.connect(project) as conn:
        conn.execute(
            "INSERT INTO sa_peak_ms2 VALUES ('p1', ?, ?, ?, ?, ?, ?, ?)",
            [1.0, 100, 200, 30, json.dumps([50]), json.dumps([300]), filename],
        )
    tab = summary.export_msms_refs_tab(project, tmp_path, "run")
    refs = pd.read_csv(tab, sep="\t")
    assert refs["name"].tolist() == ["A (peak1)"]
    assert refs["database"].tolist() == ["metatlas"]
    assert refs["id"].str.fullmatch(r"[0-9a-f]{32}").all()
    assert refs["collision_energy"].tolist() == ["ramp-205060norm"]
    assert refs["instrument"].tolist() == ["EXP120B"]
    assert refs["instrument_type"].tolist() == ["Orbitrap"]





# def test_deposit_atlas_appends_and_sorts_whole_sheet(monkeypatch, capsys):
#     import gspread
#     from oauth2client.service_account import ServiceAccountCredentials

#     atlas = pd.DataFrame([
#         ["A", "A", "key", "", "", "", "[M+H]+", 100.0, 3.0, 2.9, 3.1,
#          5.0, "C18", "positive", "", "source", "20260101", "20260101", "EMA"],
#         ["B", "B", "key2", "", "", "", "[M+H]+", 101.0, 1.0, .9, 1.1,
#          5.0, "C18", "positive", "", "source2", "20260101", "20260101", "EMA"],
#     ], columns=deposit.ATLAS_COLUMNS)
#     sheet = Mock(id=123)
#     sheet.row_values.return_value = deposit.ATLAS_COLUMNS
#     sheet.col_values.return_value = ["inchi_key"]
#     spreadsheet = Mock()
#     spreadsheet.worksheet.return_value = sheet
#     credentials = object()
#     credential_factory = Mock(return_value=credentials)
#     authorize = Mock(return_value=Mock(open_by_key=Mock(return_value=spreadsheet)))
#     monkeypatch.setattr(ServiceAccountCredentials, "from_json_keyfile_name", credential_factory)
#     monkeypatch.setattr(gspread, "authorize", authorize)
#     warning = Mock()
#     monkeypatch.setattr(deposit.logger, "warning", warning)

#     deposit.deposit_atlas_to_google_sheet(atlas)

#     credential_factory.assert_called_once_with(
#         deposit.GOOGLE_SERVICE_ACCOUNT_FILE,
#         ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
#     )
#     authorize.assert_called_once_with(credentials)
#     authorize.return_value.open_by_key.assert_called_once_with(deposit.ATLAS_SPREADSHEET)
#     spreadsheet.worksheet.assert_called_once_with("complete_atlas")
#     sheet.col_values.assert_called_once_with(3)
#     warning.assert_not_called()
#     sheet.append_rows.assert_called_once_with(atlas.values.tolist(), value_input_option="RAW")
#     spreadsheet.batch_update.assert_called_once_with({"requests": [{
#         "sortRange": {
#             "range": {"sheetId": 123, "startRowIndex": 1},
#             "sortSpecs": [{"dimensionIndex": 8, "sortOrder": "ASCENDING"}],
#         },
#     }]})
#     assert "Successfully appended and sorted 2 atlas rows" in capsys.readouterr().out


# def test_deposit_atlas_rejects_mismatched_header_before_append(monkeypatch):
#     import gspread
#     from oauth2client.service_account import ServiceAccountCredentials

#     atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS)], columns=deposit.ATLAS_COLUMNS)
#     sheet = Mock()
#     sheet.row_values.return_value = list(reversed(deposit.ATLAS_COLUMNS))
#     spreadsheet = Mock()
#     spreadsheet.worksheet.return_value = sheet
#     monkeypatch.setattr(ServiceAccountCredentials, "from_json_keyfile_name", Mock())
#     monkeypatch.setattr(gspread, "authorize", Mock(return_value=Mock(open_by_key=Mock(return_value=spreadsheet))))

#     with pytest.raises(ValueError, match="header"):
#         deposit.deposit_atlas_to_google_sheet(atlas)
#     sheet.append_rows.assert_not_called()
#     spreadsheet.batch_update.assert_not_called()


# def test_deposit_atlas_rejects_empty_or_wrong_columns():
#     with pytest.raises(ValueError, match="empty"):
#         deposit.deposit_atlas_to_google_sheet(pd.DataFrame(columns=deposit.ATLAS_COLUMNS))
#     with pytest.raises(ValueError, match="columns"):
#         deposit.deposit_atlas_to_google_sheet(pd.DataFrame({"rt_peak": [1.0]}))


# def test_deposit_atlas_warns_once_per_existing_key_and_appends(monkeypatch):
#     import gspread
#     from oauth2client.service_account import ServiceAccountCredentials

#     atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS) for _ in range(5)], columns=deposit.ATLAS_COLUMNS)
#     atlas["inchi_key"] = ["existing", "existing", "new", "", "inchi_key"]
#     sheet = Mock(id=123)
#     sheet.row_values.return_value = deposit.ATLAS_COLUMNS
#     sheet.col_values.return_value = ["inchi_key", "existing", "", "existing"]
#     spreadsheet = Mock()
#     spreadsheet.worksheet.return_value = sheet
#     monkeypatch.setattr(ServiceAccountCredentials, "from_json_keyfile_name", Mock())
#     monkeypatch.setattr(gspread, "authorize", Mock(return_value=Mock(open_by_key=Mock(return_value=spreadsheet))))
#     def check_before_append(*_args):
#         sheet.append_rows.assert_not_called()

#     warning = Mock(side_effect=check_before_append)
#     monkeypatch.setattr(deposit.logger, "warning", warning)

#     deposit.deposit_atlas_to_google_sheet(atlas)

#     warning.assert_called_once_with("Adding new entries for existing inchi_key: %s", "existing")
#     sheet.append_rows.assert_called_once_with(atlas.values.tolist(), value_input_option="RAW")


# def test_deposit_atlas_reports_append_if_sort_fails(monkeypatch, capsys):
#     import gspread
#     from oauth2client.service_account import ServiceAccountCredentials

#     atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS)], columns=deposit.ATLAS_COLUMNS)
#     sheet = Mock(id=123)
#     sheet.row_values.return_value = deposit.ATLAS_COLUMNS
#     sheet.col_values.return_value = ["inchi_key"]
#     spreadsheet = Mock()
#     spreadsheet.worksheet.return_value = sheet
#     spreadsheet.batch_update.side_effect = RuntimeError("sort failed")
#     monkeypatch.setattr(ServiceAccountCredentials, "from_json_keyfile_name", Mock())
#     monkeypatch.setattr(gspread, "authorize", Mock(return_value=Mock(open_by_key=Mock(return_value=spreadsheet))))

#     with pytest.raises(RuntimeError, match="sort failed"):
#         deposit.deposit_atlas_to_google_sheet(atlas)
#     sheet.append_rows.assert_called_once()
#     output = capsys.readouterr().out
#     assert "if sorting fails, do not append them again" in output
#     assert "Successfully" not in output


# def test_deposit_atlas_does_not_sort_after_append_failure(monkeypatch):
#     import gspread
#     from oauth2client.service_account import ServiceAccountCredentials

#     atlas = pd.DataFrame([[""] * len(deposit.ATLAS_COLUMNS)], columns=deposit.ATLAS_COLUMNS)
#     sheet = Mock()
#     sheet.row_values.return_value = deposit.ATLAS_COLUMNS
#     sheet.col_values.return_value = ["inchi_key"]
#     sheet.append_rows.side_effect = RuntimeError("append failed")
#     spreadsheet = Mock()
#     spreadsheet.worksheet.return_value = sheet
#     monkeypatch.setattr(ServiceAccountCredentials, "from_json_keyfile_name", Mock())
#     monkeypatch.setattr(gspread, "authorize", Mock(return_value=Mock(open_by_key=Mock(return_value=spreadsheet))))

#     with pytest.raises(RuntimeError, match="append failed"):
#         deposit.deposit_atlas_to_google_sheet(atlas)
#     spreadsheet.batch_update.assert_not_called()


# @pytest.mark.parametrize("timestamp,alignment_timestamp", [
#     ("bad", "20260101T120000_000000Z"),
#     ("20260101T120000_000000Z", "../bad"),
# ])
# def test_deposit_rejects_invalid_timestamps(project, tmp_path, monkeypatch, timestamp, alignment_timestamp):
#     config = {"project": {"standards_run_name": "run"}}
#     monkeypatch.setattr("metatlas2.load_tools.load_standards_config", lambda _: config)
#     monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
#     monkeypatch.setattr(db, "get_project_db_path", lambda _: project)
#     with pytest.raises(ValueError, match="timestamp"):
#         run_atlas_and_msms_deposit("config", timestamp, alignment_timestamp)


@pytest.mark.parametrize("inline", [False, True])
def test_docker_jupyter_mounts_external_standards_config(tmp_path, inline):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for executable in ("dirname", "basename", "id", "cat"):
        (bin_dir / executable).symlink_to(shutil.which(executable))
    calls = tmp_path / "calls"
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_CALLS"\n')
    docker.chmod(0o755)
    data = tmp_path / "data"
    db_dir = data / "databases" / "main_db"
    db_dir.mkdir(parents=True)
    (db_dir / "metatlas.duckdb").touch()
    (db_dir / ".zenodo_version").write_text("https://doi.org/10.5281/zenodo.22968291\n")
    config = tmp_path / "config with spaces.yaml"
    config.touch()
    env = dict(os.environ, PATH=str(bin_dir), METATLAS_DATA_DIR=str(data), TEST_CALLS=str(calls))
    script = Path(__file__).resolve().parents[1] / "scripts" / "metatlas2.sh"
    args = [f"--config_path={config}"] if inline else ["--config_path", str(config)]
    subprocess.run(["/bin/bash", str(script), "jupyter", *args], env=env, capture_output=True, text=True, check=True)
    docker_args = calls.read_text().splitlines()
    assert f"{tmp_path}:{tmp_path}:ro" in docker_args
    key = Path("/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json")
    assert (f"{key}:{key}:ro" in docker_args) == key.is_file()


def _build_gui(project, monkeypatch, annotation_run_id=None):
    expanded = pd.DataFrame([{
        "compound_name": "A", "compound_uid": "c", "mz_rt_uid": "t", "adduct": "[M+H]+",
        "chromatography": "C18", "polarity": "POS", "filename": "file.h5",
        "mz_theoretical": 100, "collision_energy_label": "CE",
    }])
    data = SimpleNamespace(
        project_db_path=project, expanded_table=expanded,
        ms1_df=pd.DataFrame(columns=["mz_rt_uid"]), ms2_df=pd.DataFrame(), tic_by_file={},
    )
    apps = []
    real_dash = gui.dash.Dash
    def create_dash(*args, **kwargs):
        app = real_dash(*args, **kwargs)
        apps.append(app)
        return app
    monkeypatch.setattr(gui.dash, "Dash", create_dash)
    monkeypatch.setattr("werkzeug.serving.make_server", lambda *_: SimpleNamespace(
        shutdown=lambda: None, serve_forever=lambda: None,
    ))
    monkeypatch.setattr("IPython.display.display", lambda *_: None)
    monkeypatch.setattr(gui.time, "sleep", lambda _: None)
    gui.build_standards_annotation_gui(
        data, {"project": {"standards_run_name": "run", "owner": "jgi"}}, port=0,
        annotation_run_id=annotation_run_id,
    )
    callbacks = {}
    for callback in apps[0].callback_map.values():
        if "callback" in callback:
            func = callback["callback"].__wrapped__
            callbacks.setdefault(func.__name__, func)
    return apps[0], callbacks


def test_annotation_run_lifecycle_revert_and_discard(project):
    assert db.get_draft_annotation_run(project) == 1
    db.freeze_annotation_run(project, 1, "/outputs/annotation_run_1")
    with pytest.raises(ValueError, match="frozen"):
        db.replace_compound_selections(project, "A", ["c"], [("p2", "")], {}, ["C18"], annotation_run_id=1)
    assert db.open_or_create_draft_annotation_run(project) == 2
    assert db.open_or_create_draft_annotation_run(project) == 2
    assert db.load_analyst_selections(project, 2)["peak_uid"].tolist() == ["p1", "p3"]
    assert db.load_compound_notes(project, "A", 2) == {"C18": "note"}
    assert db.get_annotation_run(project, 2)["parent_run_id"] == 1

    manual = db.load_peaks(project).iloc[0].to_dict()
    manual.update(peak_uid="manual", rt_peak=4, rt_min=3.9, rt_max=4.1, is_manual=True,
                  created_in_annotation_run=2)
    db.save_manual_peak_to_db(project, manual, [{"peak_uid": "manual", "filename": "file.h5"}])
    with pytest.raises(ValueError, match="frozen"):
        db.save_manual_peak_to_db(project, dict(manual, peak_uid="m2", rt_peak=5, created_in_annotation_run=1), [])
    assert "manual" not in set(db.load_peaks(project, annotation_run_id=1)["peak_uid"])
    db.replace_compound_selections(project, "A", ["c"], [("manual", "new")], {"C18": "new"}, ["C18"])
    with pytest.raises(ValueError, match="Draft annotation run 2 already exists"):
        db.start_new_annotation_run(project, 1)
    with pytest.raises(ValueError, match="draft"):
        db.start_new_annotation_run(project, 2)

    assert db.start_new_annotation_run(project, 1, discard_draft=True) == 2
    assert "manual" not in set(db.load_peaks(project)["peak_uid"])
    ms2 = db.load_peak_ms2(project)
    assert ms2.empty or "manual" not in set(ms2["peak_uid"])
    assert db.load_analyst_selections(project, 2)["peak_uid"].tolist() == ["p1", "p3"]
    db.replace_compound_selections(project, "A", ["c"], [("p2", "")], {}, ["C18"])
    db.freeze_annotation_run(project, 2, "/outputs/annotation_run_2")
    with pytest.raises(ValueError, match="frozen annotation run"):
        db.delete_manual_peak_from_db(project, "p2")
    assert db.discard_draft_annotation_run(project) is None
    runs = db.list_annotation_runs(project)
    assert runs[["annotation_run_id", "status", "n_selected_peaks"]].values.tolist() == [
        [1, "frozen", 2], [2, "frozen", 1]]

    tables = review.show_standards_project_history(project)
    assert tables["annotation_runs"]["annotation_run_id"].tolist() == [1, 2]
    assert tables["alignment_runs"].empty and tables["deposits"].empty


def test_gui_edits_new_draft_and_views_frozen_run_read_only(project, monkeypatch):
    db.freeze_annotation_run(project, 1, "/outputs/annotation_run_1")
    app, callbacks = _build_gui(project, monkeypatch)
    assert db.get_draft_annotation_run(project) == 2
    session = app.layout.children[0].data
    assert {k for k, v in session["checked_peaks"].items() if v} == {"p1", "p3"}
    session = {"compound_idx": 0, "checked_peaks": {"p2": True}, "analyst_notes_by_chrom": {"C18": "run 2"}}
    callbacks["save_and_exit"](1, session)
    assert db.load_analyst_selections(project, 2)["peak_uid"].tolist() == ["p2"]
    assert db.load_analyst_selections(project, 1)["peak_uid"].tolist() == ["p1", "p3"]

    app, callbacks = _build_gui(project, monkeypatch, annotation_run_id=1)
    restored = app.layout.children[0].data
    assert {k for k, v in restored["checked_peaks"].items() if v} == {"p1", "p3"}
    assert restored["analyst_notes_by_chrom"] == {"C18": "note"}
    options = callbacks["update_checklists"](restored)
    assert options[1] == ["p1", "p3"]
    assert all(option.get("disabled") for option in options[0])
    _, message = callbacks["save_and_exit"](1, {**restored, "checked_peaks": {"p2": True}})
    assert "read-only" in message
    with pytest.raises(gui.dash.exceptions.PreventUpdate):
        callbacks["add_manual_peak"](1, 1.0, "[M+H]+", restored, 0)
    assert db.load_analyst_selections(project, 1)["peak_uid"].tolist() == ["p1", "p3"]
    assert db.load_analyst_selections(project, 2)["peak_uid"].tolist() == ["p2"]

    _, callbacks = _build_gui(project, monkeypatch, annotation_run_id=2)
    callbacks["save_and_exit"](1, {"compound_idx": 0, "checked_peaks": {"p3": True},
                                   "analyst_notes_by_chrom": {}})
    assert db.load_analyst_selections(project, 2)["peak_uid"].tolist() == ["p3"]
