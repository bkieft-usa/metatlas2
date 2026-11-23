from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest

import metatlas2.shared_tools.atlas_sheet as atlas_sheet
import metatlas2.standards_annotation.standards_annotation_rt_align as standards_align
from metatlas2.shared_tools.load_tools import _build_metatlas2_config
from metatlas2.shared_tools.workflow_objects import (
    Atlas,
    AtlasCategory,
    Metatlas2Config,
    TargetedAnalysis,
    _validate_existing_atlas_selections,
)


def _row(**overrides) -> dict[str, str]:
    row = {column: "" for column in atlas_sheet.ATLAS_COLUMNS}
    row.update({
        "label": "Adenine",
        "compound_name": "vendor adenine",
        "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N",
        "adduct": "[M+H]+",
        "mz": "136.062",
        "rt_peak": "2.68",
        "rt_min": "2.0",
        "rt_max": "3.2",
        "mz_tolerance": "5.0",
        "chromatography": "HILICZ",
        "polarity": "positive",
        "tags": "EMA",
    })
    row.update(overrides)
    return row


def test_read_sheet_preserves_chem_source_with_reordered_columns(monkeypatch):
    row = _row(chem_source="Vendor 123", lcmsrun_source="run.h5")
    header = list(reversed(atlas_sheet.ATLAS_COLUMNS))
    sheet = SimpleNamespace(get_all_values=lambda: [header, [row[column] for column in header]])
    monkeypatch.setattr(atlas_sheet, "open_centralized_atlas_worksheet", lambda **_: (None, sheet))

    result = atlas_sheet.read_centralized_atlas_sheet()

    assert result.loc[0, "chem_source"] == "Vendor 123"
    assert result.loc[0, "lcmsrun_source"] == "run.h5"


def test_select_centralized_atlas_rows_uses_exact_tag_tokens_and_or_semantics():
    rows = pd.DataFrame([
        _row(tags="EMA; ISTD"),
        _row(inchi_key="other", tags="EMAX"),
        _row(inchi_key="negative", polarity="negative"),
        _row(inchi_key="c18", chromatography="C18"),
    ], columns=atlas_sheet.ATLAS_COLUMNS)

    selected = atlas_sheet.select_centralized_atlas_rows(
        rows,
        tags=["EMA", "QC"],
        chromatography="HILIC",
        polarity="POS",
        atlas_category="REF",
    )

    assert selected["inchi_key"].tolist() == ["GFFGJBXGBJISGV-UHFFFAOYSA-N"]
    assert selected.iloc[0]["polarity"] == "POS"


def test_select_centralized_atlas_rows_rejects_duplicate_analysis_rows():
    rows = pd.DataFrame([
        _row(rt_peak="2.68"),
        _row(rt_peak="2.75"),
    ], columns=atlas_sheet.ATLAS_COLUMNS)

    with pytest.raises(ValueError, match="Duplicate centralized-atlas row identities"):
        atlas_sheet.select_centralized_atlas_rows(
            rows, ["EMA"], "HILICZ", "POS", "REF"
        )


def test_select_centralized_atlas_rows_rejects_empty_selection():
    rows = pd.DataFrame([_row(tags="OTHER")], columns=atlas_sheet.ATLAS_COLUMNS)

    with pytest.raises(ValueError, match="No centralized-atlas rows found"):
        atlas_sheet.select_centralized_atlas_rows(
            rows, ["EMA"], "HILICZ", "POS", "REF"
        )


def test_resolve_config_atlas_inputs_persists_sheet_source_atlas_uids(monkeypatch):
    rows = pd.DataFrame([
        _row(),
        _row(inchi_key="QC-KEY", tags="QCv7"),
    ], columns=atlas_sheet.ATLAS_COLUMNS)
    monkeypatch.setattr(atlas_sheet, "read_centralized_atlas_sheet", lambda: rows)

    created = []
    monkeypatch.setattr(
        "metatlas2.shared_tools.database_interact.get_atlas_uid_by_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "metatlas2.shared_tools.database_interact.get_compound_uids_by_inchi_keys",
        lambda *_args: {
            "GFFGJBXGBJISGV-UHFFFAOYSA-N": "compound-uid",
            "QC-KEY": "qc-compound-uid",
        },
    )

    def create_atlas(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(atlas_uid=f"atlas-{len(created)}")

    monkeypatch.setattr(
        "metatlas2.shared_tools.database_interact.create_new_atlas_from_dataframe",
        create_atlas,
    )
    monkeypatch.setattr(
        "metatlas2.shared_tools.database_interact.save_atlas_to_database",
        lambda *_args: None,
    )
    config = Metatlas2Config(
        rt_alignment_config={
            "HILIC": {
                "ATLAS": {"tags": ["QCv7"], "polarity": "POS", "uid": None},
                "PARAMS": {},
            },
        },
        targeted_analyses=[
            TargetedAnalysis(
                chromatography="HILIC",
                polarity="POS",
                atlas_category=AtlasCategory.REF,
                atlas_name="EMA",
                tags=["EMA"],
            ),
        ],
    )

    atlas_sheet.resolve_config_atlas_inputs(config, "main.duckdb")

    assert config.rt_alignment_config["HILIC"]["ATLAS"]["uid"] == "atlas-1"
    assert config.targeted_analyses[0].atlas_uid == "atlas-2"
    assert len(created) == 2
    assert created[0]["atlas_category"] is AtlasCategory.QC
    assert created[1]["atlas_category"] is AtlasCategory.REF


def _selector_config(rt_atlas: dict, targeted_atlas: dict) -> dict:
    return {
        "RT_ALIGNMENT": {
            "HILICZ": {"ATLAS": rt_atlas, "PARAMS": {}},
        },
        "TARGETED_ANALYSES": {
            "HILICZ": {
                "POS": {
                    "REF": {
                        "EMA": {"ATLAS": targeted_atlas, "PARAMS": {}},
                    },
                },
            },
        },
    }


def test_config_parser_accepts_uid_selectors():
    config = _build_metatlas2_config(
        _selector_config(
            {"uid": "qc-atlas", "polarity": "POS"},
            {"uid": "ref-atlas"},
        ),
        "test",
    )

    rt_atlas = config.rt_alignment_config["HILIC"]["ATLAS"]
    assert rt_atlas["uid"] == "qc-atlas"
    assert rt_atlas["tags"] == []
    assert config.targeted_analyses[0].atlas_uid == "ref-atlas"
    assert config.targeted_analyses[0].tags == []


def test_config_parser_skips_validation_outside_analysis_subset():
    raw = _selector_config({"uid": "qc-atlas", "polarity": "POS"}, {})
    raw["TARGETED_ANALYSES"]["HILICZ"]["POS"]["REF"]["OTHER"] = {
        "ATLAS": {"uid": "other-atlas"},
        "PARAMS": {},
    }

    with pytest.raises(ValueError, match="must specify exactly one"):
        _build_metatlas2_config(copy.deepcopy(raw), "test")

    config = _build_metatlas2_config(raw, "test", ["HILIC-POS-REF-OTHER"])
    assert [ta.atlas_name for ta in config.targeted_analyses] == ["OTHER"]


def test_existing_snapshot_validation_preserves_tag_selector_with_resolved_uid():
    requested = _build_metatlas2_config(
        _selector_config(
            {"tags": ["QCv7"], "polarity": "POS"},
            {"tags": ["EMA"]},
        ),
        "test",
    )
    stored = _build_metatlas2_config(
        _selector_config(
            {"tags": ["QCv7"], "polarity": "POS"},
            {"tags": ["EMA"]},
        ),
        "test",
    )
    stored.rt_alignment_config["HILIC"]["ATLAS"]["uid"] = "resolved-qc-atlas"
    stored.targeted_analyses[0].atlas_uid = "resolved-ref-atlas"

    _validate_existing_atlas_selections(requested, stored)

    stored.targeted_analyses[0].tags = []
    with pytest.raises(ValueError, match="different atlas selectors"):
        _validate_existing_atlas_selections(requested, stored)


def test_resolve_config_atlas_inputs_validates_uids_without_reading_sheet(monkeypatch):
    loaded_uids = []

    def load_atlas(_database_path, atlas_uid):
        loaded_uids.append(atlas_uid)
        category = AtlasCategory.QC if atlas_uid == "qc-atlas" else AtlasCategory.REF
        return SimpleNamespace(
            atlas_uid=atlas_uid,
            chromatography="HILICZ",
            polarity="POS",
            atlas_category=category,
        )

    monkeypatch.setattr(Atlas, "from_database", classmethod(lambda cls, db, uid: load_atlas(db, uid)))
    monkeypatch.setattr(
        atlas_sheet,
        "read_centralized_atlas_sheet",
        lambda: pytest.fail("UID selectors should not read the master sheet"),
    )
    config = Metatlas2Config(
        rt_alignment_config={
            "HILICZ": {
                "ATLAS": {"uid": "qc-atlas", "tags": [], "polarity": "POS"},
                "PARAMS": {},
            },
        },
        targeted_analyses=[
            TargetedAnalysis(
                chromatography="HILICZ",
                polarity="POS",
                atlas_category=AtlasCategory.REF,
                atlas_name="EMA",
                atlas_uid="ref-atlas",
            ),
        ],
    )

    atlas_sheet.resolve_config_atlas_inputs(config, "main.duckdb")

    assert loaded_uids == ["qc-atlas", "ref-atlas"]


def test_resolve_config_atlas_inputs_rejects_incompatible_uid(monkeypatch):
    monkeypatch.setattr(
        Atlas,
        "from_database",
        classmethod(
            lambda cls, _db, uid: SimpleNamespace(
                atlas_uid=uid,
                chromatography="HILICZ",
                polarity="NEG",
                atlas_category=AtlasCategory.REF,
            )
        ),
    )
    config = Metatlas2Config(
        rt_alignment_config={
            "HILICZ": {
                "ATLAS": {"uid": "wrong-atlas", "tags": [], "polarity": "POS"},
                "PARAMS": {},
            },
        },
    )

    with pytest.raises(ValueError, match="config expects"):
        atlas_sheet.resolve_config_atlas_inputs(config, "main.duckdb")


def test_resolve_config_atlas_inputs_rejects_uid_missing_from_database(monkeypatch):
    def missing_atlas(_cls, _db, _uid):
        raise ValueError("No compounds found")

    monkeypatch.setattr(Atlas, "from_database", classmethod(missing_atlas))
    config = Metatlas2Config(
        rt_alignment_config={
            "HILICZ": {
                "ATLAS": {"uid": "missing-atlas", "tags": [], "polarity": "POS"},
                "PARAMS": {},
            },
        },
    )

    with pytest.raises(ValueError, match="not a valid atlas in the main database"):
        atlas_sheet.resolve_config_atlas_inputs(config, "main.duckdb")


@pytest.mark.parametrize(
    "selector",
    [{}, {"tags": ["QC"], "uid": "qc-atlas"}, {"tags": []}, {"uid": ""}],
)
def test_standards_qc_atlas_rejects_invalid_selectors(selector):
    config = {"rt_alignment": {"HILIC": {"ATLAS": {**selector, "polarity": "POS"}}}}
    with pytest.raises(ValueError, match="RT_ALIGNMENT HILIC ATLAS"):
        standards_align.load_qc_atlas(config, "HILIC")


def test_standards_qc_atlas_requires_polarity():
    config = {"rt_alignment": {"HILIC": {"ATLAS": {"tags": ["QC"]}}}}
    with pytest.raises(ValueError, match="missing polarity"):
        standards_align.load_qc_atlas(config, "HILIC")


@pytest.mark.parametrize("polarity", [None, "", "  "])
def test_standards_qc_atlas_rejects_empty_polarity(polarity):
    config = {"rt_alignment": {"HILIC": {"ATLAS": {"uid": "qc", "polarity": polarity}}}}
    with pytest.raises(ValueError, match="polarity must be a non-empty string"):
        standards_align.load_qc_atlas(config, "HILIC")


@pytest.mark.parametrize("use_tags", [True, False])
def test_standards_qc_atlas_resolves_and_records_uid(monkeypatch, use_tags):
    atlas = SimpleNamespace(
        atlas_uid="qc-atlas", chromatography="HILICZ", polarity="POS",
        atlas_category=AtlasCategory.QC, compound_mzrts={"compound": object()},
    )
    monkeypatch.setattr(Atlas, "from_database", classmethod(lambda cls, db, uid: atlas))
    if use_tags:
        rows = pd.DataFrame([
            _row(inchi_key="selected", tags="QC"),
            _row(inchi_key="wrong-polarity", tags="QC", polarity="negative"),
            _row(inchi_key="wrong-chrom", tags="QC", chromatography="C18"),
        ])
        monkeypatch.setattr(atlas_sheet, "read_centralized_atlas_sheet", lambda: rows)
        def create(selected, **kwargs):
            assert selected["inchi_key"].tolist() == ["selected"]
            assert kwargs["tags"] == ["QC"]
            assert kwargs["atlas_category"] is AtlasCategory.QC
            return "qc-atlas"
        monkeypatch.setattr(atlas_sheet, "_create_or_reuse_atlas", create)
        selector = {"tags": [" QC "], "polarity": "positive"}
    else:
        monkeypatch.setattr(
            atlas_sheet, "read_centralized_atlas_sheet",
            lambda: pytest.fail("UID selector should not read the master sheet"),
        )
        selector = {"uid": "qc-atlas", "polarity": "POS"}
    config = {"rt_alignment": {"hilic": {"ATLAS": selector.copy()}}}

    assert standards_align.load_qc_atlas(config, "HILIC") is atlas
    assert config["rt_alignment"]["hilic"]["ATLAS"] == selector
    assert config["rt_alignment"]["hilic"]["resolved_qc_atlas_uid"] == "qc-atlas"


def test_standards_qc_atlas_rejects_incompatible_uid(monkeypatch):
    monkeypatch.setattr(
        Atlas, "from_database",
        classmethod(lambda cls, db, uid: SimpleNamespace(
            atlas_uid=uid, chromatography="C18", polarity="POS",
            atlas_category=AtlasCategory.QC,
        )),
    )
    config = {"rt_alignment": {"HILIC": {"ATLAS": {"uid": "qc", "polarity": "POS"}}}}
    with pytest.raises(ValueError, match="config expects"):
        standards_align.load_qc_atlas(config, "HILIC")


def test_standards_qc_atlas_rejects_unmatched_tags(monkeypatch):
    monkeypatch.setattr(
        atlas_sheet, "read_centralized_atlas_sheet",
        lambda: pd.DataFrame([_row(tags="OTHER")]),
    )
    config = {"rt_alignment": {"HILIC": {"ATLAS": {"tags": ["QC"], "polarity": "POS"}}}}
    with pytest.raises(ValueError, match="No centralized-atlas rows found"):
        standards_align.load_qc_atlas(config, "HILIC")
