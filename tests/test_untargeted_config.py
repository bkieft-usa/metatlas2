"""Tests for metatlas2.untargeted.config, the CLI flag mapping, and lenient name helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from metatlas2.shared_tools.file_and_project_format import get_name_field, parent_dir_num_fields, validate_file_name_minimal
from metatlas2.untargeted.run_untargeted_pipeline import build_parser, config_overrides
from metatlas2.untargeted.config import DEFAULT_CONFIG, load_untargeted_config, normalize_number

REPO = Path(__file__).resolve().parents[1]
PROJECT = "20260101_JGI_AB_123456_Test_final_EXP120B_C18_USDAY00001"


def test_defaults_match_legacy_values():
    cfg = load_untargeted_config()
    assert cfg["paths"]["output_dir"] == "/global/cfs/cdirs/metatlas/projects/untargeted_tasks"
    assert cfg["lims"]["tasks_table"] == "untargeted_tasks"
    assert cfg["filtering"]["background_ratio"] == 3 and isinstance(cfg["filtering"]["background_ratio"], int)
    assert cfg["filtering"]["zero_value"] == pytest.approx(2 / 3)
    assert cfg["project_selection"]["background_designator"] == ["ExCtrl"]
    assert cfg["mzmine"]["slurm"]["time"] == "3:00:00"


def test_shipped_yaml_equals_builtin_defaults():
    path = REPO / "configs" / "untargeted" / "untargeted_pipeline.yaml"
    assert load_untargeted_config(path) == load_untargeted_config()
    raw = yaml.safe_load(path.read_text())
    assert {k.lower() for k in raw} == set(DEFAULT_CONFIG)


def test_yaml_partial_override_and_cli_override(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("FILTERING:\n  background_ratio: 5\nMZMINE:\n  slurm:\n    qos: debug\n")
    cfg = load_untargeted_config(path, {"filtering": {"zero_value": 0.5, "background_ratio": None}})
    assert cfg["filtering"]["background_ratio"] == 5
    assert cfg["filtering"]["zero_value"] == 0.5
    assert cfg["mzmine"]["slurm"]["qos"] == "debug"
    assert cfg["mzmine"]["slurm"]["account"] == "m2650"  # untouched nested key survives


def test_unknown_section_rejected(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("NOPE:\n  a: 1\n")
    with pytest.raises(ValueError, match="Unknown section"):
        load_untargeted_config(path)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_untargeted_config("/nonexistent/cfg.yaml")


def test_normalize_number_keeps_legacy_file_names():
    assert normalize_number(3.0) == 3 and isinstance(normalize_number(3.0), int)
    assert normalize_number(2.5) == 2.5


def test_cli_flags_map_to_config():
    args = build_parser().parse_args([
        "run", "--background_ratio", "3", "--no_gdrive_upload", "--skip_mirror_raw_data",
        "--direct_input", "A,B", "--background_designator", "ExCtrl,Blank", "--output_dir", "/x",
    ])
    cfg = load_untargeted_config(None, config_overrides(args))
    assert cfg["filtering"]["background_ratio"] == 3 and isinstance(cfg["filtering"]["background_ratio"], int)
    assert cfg["gdrive"]["upload"] is False
    assert cfg["gdrive"]["add_documentation"] is True  # unset flag keeps default
    assert cfg["gnps2"]["mirror_raw_data"] is False
    assert cfg["gnps2"]["mirror_mzmine_results"] is True
    assert cfg["project_selection"]["background_designator"] == ["ExCtrl", "Blank"]
    assert cfg["paths"]["output_dir"] == "/x"
    assert args.direct_input == ["A", "B"]


# -- lenient name helpers (file_and_project_format) ---------------------------------

def test_get_name_field():
    assert get_name_field(PROJECT, 1) == "JGI"
    assert get_name_field(PROJECT, 6) == "EXP120B"
    assert get_name_field(PROJECT, 20) is None
    assert get_name_field("/a/b/x_y.mzML", 1) == "y"


def test_validate_file_name_minimal():
    good = f"{PROJECT}_POS_MS2_1_ExCtrl_1_meta_Run1.mzML"
    assert validate_file_name_minimal(good) == []
    assert validate_file_name_minimal(f"{PROJECT}_XXX_MS2_1_ExCtrl_1_meta_Run1.mzML")
    assert validate_file_name_minimal("a_b_c.mzML")


def test_parent_dir_num_fields():
    assert parent_dir_num_fields(f"/raw/{PROJECT}/file.mzML") == []
    assert parent_dir_num_fields(f"/raw/{PROJECT}_extra/file.mzML") == []
    assert parent_dir_num_fields("/raw/a_b_c/file.mzML")
    assert parent_dir_num_fields(f"/raw/{PROJECT}_x_y/file.mzML")
