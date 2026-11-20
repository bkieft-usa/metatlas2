from __future__ import annotations

from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest
from rdkit import Chem

from metatlas2 import standards_annotation_checks as checks
from metatlas2 import standards_annotation_db as db
from metatlas2 import standards_annotation_summary as summary
from metatlas2 import standards_annotation_tools as tools


def structure(smiles):
    inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
    return inchi, Chem.InchiToInchiKey(inchi)


@pytest.mark.parametrize("smiles", ["CCO", "C[N+](C)(C)C", "C[N+](=O)[O-]"])
def test_neutral_or_permanently_charged_structure_passes(smiles):
    checks.validate_neutral_inchi_key("compound", *structure(smiles))


@pytest.mark.parametrize("smiles", ["C[NH3+]", "CC(=O)[O-].[Na+]"])
def test_charged_or_salted_structure_fails(smiles):
    inchi, key = structure(smiles)
    with pytest.raises(ValueError, match="desalted and neutralized"):
        checks.validate_neutral_inchi_key("compound", inchi, key)


def test_provided_key_must_match_neutral_structure():
    inchi, _ = structure("CCO")
    _, different_key = structure("CC")
    with pytest.raises(ValueError, match="compound.*InChIKey"):
        checks.validate_neutral_inchi_key("compound", inchi, different_key)
    with pytest.raises(ValueError, match="valid InChI and InChIKey"):
        checks.validate_neutral_inchi_key("compound", "", different_key)


def test_input_expansion_checks_derived_and_provided_keys(tmp_path):
    path = tmp_path / "input.csv"
    config = {"project": {"standards_input_file": str(path)}}
    for row in [
        {"compound_name": "charged", "smiles": "C[NH3+]"},
        {"compound_name": "mismatch", "smiles": "CCO", "inchi_key": structure("CC")[1]},
        {"compound_name": "inconsistent", "smiles": "C[NH3+]",
         "inchi": structure("CN")[0], "inchi_key": structure("CN")[1]},
    ]:
        pd.DataFrame([row]).to_csv(path, index=False)
        with pytest.raises(ValueError, match=row["compound_name"]):
            tools.build_standards_lcmsrun_table(config, raw_data_root=str(tmp_path))


def selections(rows):
    return pd.DataFrame([
        {
            "peak_uid": str(i),
            "compound_name": name,
            "chromatography": chrom,
            "polarity": pol,
            "adduct": adduct,
            "rt_peak": rt,
            "inchi_key": key,
        }
        for i, (name, chrom, pol, adduct, rt, key) in enumerate(rows)
    ])


def targets(rows):
    return pd.DataFrame(rows, columns=["compound_name", "chromatography", "polarity", "mz_theoretical"])


def test_isomer_adducts_use_all_target_masses_and_block_disjoint_selections():
    df = selections([
        ("A", "C18", "POS", "H", 1, "AAA-X"),
        ("B", "C18", "POS", "Na", 1, "BBB-X"),
    ])
    target_df = targets([
        ("A", "C18", "POS", 100), ("A", "C18", "POS", 122),
        ("B", "C18", "POS", 100 * (1 + 4.99e-6 / 2) / (1 - 4.99e-6 / 2)),
        ("B", "C18", "POS", 122.01),
    ])
    with pytest.raises(ValueError, match="A.*B.*no selected adduct in common"):
        checks.check_isomer_adducts(df, target_df)
    target_df.loc[2, "mz_theoretical"] = 100 * (1 + 5.01e-6 / 2) / (1 - 5.01e-6 / 2)
    checks.check_isomer_adducts(df, target_df)
    target_df.loc[2, "mz_theoretical"] = 122.0004
    with pytest.raises(ValueError, match="no selected adduct"):
        checks.check_isomer_adducts(df, target_df)


def test_isomer_key_and_partial_adduct_overlap_warn(monkeypatch):
    messages = []
    monkeypatch.setattr(checks.logger, "warning", lambda fmt, *args: messages.append(fmt % args))
    df = selections([
        ("A", "C18", "POS", "H", 1, "SAME-X"),
        ("A", "C18", "POS", "Na", 1, "SAME-X"),
        ("B", "C18", "POS", "H", 1, "SAME-Y"),
        ("B", "C18", "POS", "K", 1, "SAME-Y"),
        ("B", "C18", "NEG", "-H", 1, "SAME-Y"),
    ])
    checks.check_isomer_adducts(df, targets([]))
    assert len(messages) == 2
    assert "'A' has extra selected adducts ['Na'] not in 'B'" in messages[0]
    assert "'B' has extra selected adducts ['K'] not in 'A'" in messages[1]


def test_retention_time_buckets_and_polarity_medians(monkeypatch):
    messages = []
    monkeypatch.setattr(checks.logger, "warning", lambda fmt, *args: messages.append(fmt % args))
    df = selections([
        ("A", "C18", "POS", "H", 1, "KEY"),
        ("A", "C18", "POS", "H", 5, "KEY"),
        ("A", "C18", "POS", "Na", 5.4, "KEY"),
        ("A", "C18", "NEG", "-H", 1.01, "KEY"),
        ("A", "C18", "NEG", "-H", 5.8, "KEY"),
        ("A", "HILIC", "POS", "H", 2, "KEY"),
    ])
    checks.check_retention_times(df)
    assert len(messages) == 2
    assert "C18/POS/A" in messages[0] and "5.4" in messages[0]
    assert "C18/A" in messages[1] and "median RT" in messages[1]


def test_retention_time_threshold_and_invalid_values(monkeypatch):
    messages = []
    monkeypatch.setattr(checks.logger, "warning", lambda fmt, *args: messages.append(fmt % args))
    df = selections([
        ("A", "C18", "POS", "H", 0, "KEY"),
        ("A", "C18", "POS", "Na", 0, "KEY"),
        ("B", "C18", "POS", "H", 1, "KEY"),
        ("B", "C18", "POS", "Na", (1 + .0499 / 2) / (1 - .0499 / 2), "KEY"),
        ("C", "C18", "POS", "H", float("nan"), "KEY"),
        ("D", "C18", "POS", "H", 1, "KEY"),
        ("D", "C18", "POS", "Na", (1 + .0501 / 2) / (1 - .0501 / 2), "KEY"),
    ])
    checks.check_retention_times(df)
    assert len(messages) == 2
    assert "invalid RT" in messages[0]
    assert "C18/POS/D" in messages[1]


def test_summary_checks_before_creating_output(tmp_path, monkeypatch):
    df = selections([
        ("A", "C18", "POS", "H", 1, "SAME-X"),
        ("B", "C18", "POS", "Na", 1, "SAME-Y"),
    ])
    monkeypatch.setattr(db, "get_draft_annotation_run", lambda _: 1)
    monkeypatch.setattr(db, "validate_annotation_choices", lambda _, annotation_run_id: None)
    monkeypatch.setattr(db, "load_selected_peaks_with_ms2", lambda _, annotation_run_id: df)
    monkeypatch.setattr(db, "load_isomer_check_targets", lambda _: targets([]))
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    monkeypatch.setattr(summary, "fix_peak_labels", lambda selections: selections)
    data = SimpleNamespace(config={"project": {"standards_run_name": "run"}}, project_db_path="db")
    with pytest.raises(ValueError, match="no selected adduct"):
        summary.run_standards_annotation_summary(data)
    assert not (tmp_path / "outputs").exists()


def test_summary_detects_isomers_using_unselected_database_target(tmp_path, monkeypatch):
    path = str(tmp_path / "project.duckdb")
    db.create_project_db(path)
    with duckdb.connect(path) as conn:
        for uid, name, key, adduct, mz in [
            ("a-h", "A", "AAA-X", "H", 100.0),
            ("b-h", "B", "BBB-X", "H", 100.0004),
            ("b-na", "B", "BBB-X", "Na", 122.0),
        ]:
            conn.execute("INSERT INTO sa_compounds VALUES (?, ?, ?, '', '', ?, 'POS', ?)",
                         [uid, name, key, adduct, mz])
            conn.execute("INSERT INTO sa_extraction_targets VALUES (?, ?, 'file.h5', 'C18', 0, 10, '', '')",
                         ["target-" + uid, uid])
    peaks = [
        dict(peak_uid=uid, compound_uid=uid, compound_name=name, inchi_key=key,
             adduct=adduct, polarity="POS", chromatography="C18",
             filename="file.h5", mz_rt_uid="target-" + uid, mz_theoretical=mz,
             rt_peak=1, rt_min=.9, rt_max=1.1, peak_intensity=100,
             mz_observed=mz, peak_num=1, is_manual=False, collision_energy_label="")
        for uid, name, key, adduct, mz in [
            ("a-h", "A", "AAA-X", "H", 100.0),
            ("b-na", "B", "BBB-X", "Na", 122.0),
        ]
    ]
    db.save_peaks_to_db(path, peaks)
    db.replace_compound_selections(path, "A", ["a-h"], [("a-h", "")], {}, ["C18"])
    db.replace_compound_selections(path, "B", ["b-h", "b-na"], [("b-na", "")], {}, ["C18"])
    monkeypatch.setattr(db, "get_project_output_root", lambda _: tmp_path)
    data = SimpleNamespace(config={"project": {"standards_run_name": "run"}}, project_db_path=path)
    with pytest.raises(ValueError, match="A.*B.*no selected adduct"):
        summary.run_standards_annotation_summary(data)
    assert not (tmp_path / "outputs").exists()
