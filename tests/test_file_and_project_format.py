"""Unit tests for metatlas2/file_and_project_format.py.

All functions are pure (regex matching + string manipulation, no I/O).

Tested functions
----------------
* :func:`normalize_chromatography`
* :func:`parse_file_name`
* :func:`parse_project_name`
* :func:`get_project_chromatography`
* :func:`get_file_parts`

Real filename and project-name examples are taken from the dev data package
defined in ``local/prepare_local_data_package.sh``.
"""

from __future__ import annotations

import pytest

from metatlas2.file_and_project_format import (
    get_file_parts,
    get_project_chromatography,
    normalize_chromatography,
    parse_file_name,
    parse_project_name,
)


# ---------------------------------------------------------------------------
# Representative filenames and project names from the dev data package
# ---------------------------------------------------------------------------

# Positive-mode MS2 file from the HILICZ project
_HILICZ_POS_FILE = (
    "20260311_JGI_AE_511825_SorghAnth_final_EXP120B_HILICZ_USHXG03401_"
    "POS_MS2_56_T1-256534-8-Tr-RE_1__Run158.h5"
)

# Negative-mode MS2 file
_HILICZ_NEG_FILE = (
    "20260311_JGI_AE_511825_SorghAnth_final_EXP120B_HILICZ_USHXG03401_"
    "NEG_MS2_56_T1-256534-8-Tr-RE_1__Run159.h5"
)

# Standalone dev project name
_STANDALONE_PROJECT = "20260101_JGI_XX_000000_STANDALONE-DEV_test_EXP000_HILICZ_TESTXXXX"

# C18-EP project name
_C18_PROJECT = "20260805_EB_MdR_109570-002_WaveStab6_20260724-20260731_EXP120A_C18-EP_USDAY99655"


# ===========================================================================
# normalize_chromatography
# ===========================================================================

class TestNormalizeChromatography:

    def test_hilicz_normalized_to_hilic(self):
        assert normalize_chromatography("HILICZ") == "hilic"

    def test_hilicz_lowercase_normalized(self):
        assert normalize_chromatography("hilicz") == "hilic"

    def test_c18_ep_normalized_to_c18(self):
        assert normalize_chromatography("C18-EP") == "c18"

    def test_c18ep_no_dash_normalized_to_c18(self):
        assert normalize_chromatography("C18EP") == "c18"

    def test_hilic_lowercased_unchanged(self):
        assert normalize_chromatography("HILIC") == "hilic"

    def test_c18_lowercased_unchanged(self):
        assert normalize_chromatography("C18") == "c18"

    def test_unknown_value_lowercased(self):
        assert normalize_chromatography("RPLC") == "rplc"

    def test_empty_string_returned_unchanged(self):
        assert normalize_chromatography("") == ""

    def test_none_like_falsy_returned_unchanged(self):
        # The function guards with `if not chrom: return chrom`
        assert normalize_chromatography(None) is None

    def test_mixed_case_c18_ep(self):
        assert normalize_chromatography("c18-EP") == "c18"

    def test_already_canonical_hilic(self):
        assert normalize_chromatography("hilic") == "hilic"


# ===========================================================================
# parse_file_name
# ===========================================================================

class TestParseFileName:

    def test_hilicz_pos_file_parsed(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert fields["polarity"] == "POS"
        assert fields["ms_level"] == "MS2"
        assert fields["chromatography"] == "hilic"   # normalized from HILICZ
        assert fields["ext"] == "h5"

    def test_hilicz_neg_file_parsed(self):
        fields = parse_file_name(_HILICZ_NEG_FILE)
        assert fields["polarity"] == "NEG"
        assert fields["ms_level"] == "MS2"

    def test_run_number_extracted(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert fields["run_number"] == "Run158"

    def test_owner_extracted(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert fields["owner"] == "JGI"

    def test_date_extracted(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert fields["date"] == "20260311"

    def test_sample_name_extracted(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert "T1-256534-8-Tr-RE" in fields["sample_name"]

    def test_invalid_filename_raises(self):
        with pytest.raises(Exception, match="does not match"):
            parse_file_name("not_a_valid_filename.h5")

    def test_returns_dict(self):
        result = parse_file_name(_HILICZ_POS_FILE)
        assert isinstance(result, dict)

    def test_all_expected_keys_present(self):
        fields = parse_file_name(_HILICZ_POS_FILE)
        for key in ("date", "owner", "pi", "project_id", "project_shortname",
                    "experiment", "instrument", "chromatography", "run_id",
                    "polarity", "ms_level", "sample_number", "sample_name",
                    "replicate", "run_metadata", "run_number", "ext"):
            assert key in fields, f"Missing key: {key}"

    def test_chromatography_normalized_in_output(self):
        # HILICZ in the filename → "hilic" in the parsed dict
        fields = parse_file_name(_HILICZ_POS_FILE)
        assert fields["chromatography"] == "hilic"

    def test_mzml_extension_accepted(self):
        fname = _HILICZ_POS_FILE.replace(".h5", ".mzML")
        fields = parse_file_name(fname)
        assert fields["ext"] == "mzML"

    def test_raw_extension_accepted(self):
        fname = _HILICZ_POS_FILE.replace(".h5", ".raw")
        fields = parse_file_name(fname)
        assert fields["ext"] == "raw"


# ===========================================================================
# parse_project_name
# ===========================================================================

class TestParseProjectName:

    def test_standalone_project_accepted(self):
        result = parse_project_name(_STANDALONE_PROJECT)
        assert result == _STANDALONE_PROJECT

    def test_c18_project_accepted(self):
        result = parse_project_name(_C18_PROJECT)
        assert result == _C18_PROJECT

    def test_invalid_project_name_raises(self):
        with pytest.raises(Exception, match="does not match"):
            parse_project_name("not_a_valid_project")

    def test_returns_project_name_string(self):
        result = parse_project_name(_STANDALONE_PROJECT)
        assert isinstance(result, str)

    def test_jgi_owner_accepted(self):
        # _STANDALONE_PROJECT has owner JGI
        result = parse_project_name(_STANDALONE_PROJECT)
        assert result == _STANDALONE_PROJECT

    def test_eb_owner_accepted(self):
        # _C18_PROJECT has owner EB
        result = parse_project_name(_C18_PROJECT)
        assert result == _C18_PROJECT


# ===========================================================================
# get_project_chromatography
# ===========================================================================

class TestGetProjectChromatography:

    def test_hilicz_project_returns_hilic(self):
        result = get_project_chromatography(_STANDALONE_PROJECT)
        assert result == "hilic"

    def test_c18_ep_project_returns_c18(self):
        result = get_project_chromatography(_C18_PROJECT)
        assert result == "c18"

    def test_invalid_project_raises_value_error(self):
        with pytest.raises(ValueError, match="does not match"):
            get_project_chromatography("bad_project_name")

    def test_returns_lowercase_string(self):
        result = get_project_chromatography(_STANDALONE_PROJECT)
        assert result == result.lower()


# ===========================================================================
# get_file_parts
# ===========================================================================

class TestGetFileParts:

    def test_polarity_extracted(self):
        assert get_file_parts(_HILICZ_POS_FILE, "polarity") == "POS"

    def test_polarity_neg_extracted(self):
        assert get_file_parts(_HILICZ_NEG_FILE, "polarity") == "NEG"

    def test_ms_level_extracted(self):
        assert get_file_parts(_HILICZ_POS_FILE, "ms_level") == "MS2"

    def test_chromatography_normalized(self):
        # HILICZ in filename → "hilic" returned
        assert get_file_parts(_HILICZ_POS_FILE, "chromatography") == "hilic"

    def test_run_number_extracted(self):
        assert get_file_parts(_HILICZ_POS_FILE, "run_number") == "Run158"

    def test_owner_extracted(self):
        assert get_file_parts(_HILICZ_POS_FILE, "owner") == "JGI"

    def test_date_extracted(self):
        assert get_file_parts(_HILICZ_POS_FILE, "date") == "20260311"

    def test_h5_extension_added_automatically(self):
        # Strip the .h5 extension — the function should add it back
        stem = _HILICZ_POS_FILE.replace(".h5", "")
        assert get_file_parts(stem, "polarity") == "POS"

    def test_invalid_filename_raises_value_error(self):
        with pytest.raises(ValueError, match="does not match"):
            get_file_parts("not_a_valid_filename.h5", "polarity")

    def test_sample_name_extracted(self):
        result = get_file_parts(_HILICZ_POS_FILE, "sample_name")
        assert "T1-256534-8-Tr-RE" in result

    def test_instrument_extracted(self):
        result = get_file_parts(_HILICZ_POS_FILE, "instrument")
        assert result == "EXP120B"
