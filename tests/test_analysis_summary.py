"""Unit tests for metatlas2/analysis_summary.py.

Only pure calculation functions are tested — no database, no filesystem
(except ``apply_auto_curation_defaults`` which operates on an in-memory
DataFrame), no subprocess, and no network calls.

Tested functions
----------------
* :func:`mz_quality`
* :func:`rt_quality`
* :func:`total_score_and_msi`
* :func:`_compute_all_overlapping_compounds`
* :func:`apply_auto_curation_defaults`
* :func:`_strip_non_chars`
* :func:`_display_compound_idx`
* :func:`_get_file_color`
* :func:`_short_fname`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metatlas2.analysis_summary import (
    _compute_all_overlapping_compounds,
    _display_compound_idx,
    _get_file_color,
    _short_fname,
    _strip_non_chars,
    apply_auto_curation_defaults,
    mz_quality,
    rt_quality,
    total_score_and_msi,
)


# ===========================================================================
# mz_quality
# ===========================================================================

class TestMzQuality:

    def test_perfect_match_returns_1(self):
        assert mz_quality(0.0, 0.0) == 1.0

    def test_within_5ppm_returns_1(self):
        assert mz_quality(4.9, 0.002) == 1.0

    def test_exactly_5ppm_returns_1(self):
        assert mz_quality(5.0, 0.002) == 1.0

    def test_within_da_threshold_returns_1_even_if_ppm_high(self):
        # ppm > 5 but Da <= 0.0015 → score = 1
        assert mz_quality(8.0, 0.001) == 1.0

    def test_between_5_and_10ppm_returns_0_5(self):
        assert mz_quality(7.5, 0.002) == 0.5

    def test_exactly_10ppm_returns_0_5(self):
        assert mz_quality(10.0, 0.002) == 0.5

    def test_above_10ppm_returns_0(self):
        assert mz_quality(15.0, 0.005) == 0.0

    def test_nan_ppm_returns_nan(self):
        result = mz_quality(float("nan"), 0.001)
        assert np.isnan(result)

    def test_nan_da_returns_nan(self):
        result = mz_quality(3.0, float("nan"))
        assert np.isnan(result)

    def test_negative_ppm_uses_absolute_value(self):
        # Negative ppm within 5 → score = 1
        assert mz_quality(-4.0, 0.002) == 1.0

    def test_negative_ppm_between_5_and_10_returns_0_5(self):
        assert mz_quality(-7.0, 0.002) == 0.5

    def test_negative_ppm_above_10_returns_0(self):
        assert mz_quality(-12.0, 0.005) == 0.0


# ===========================================================================
# rt_quality
# ===========================================================================

class TestRtQuality:

    # --- C18 thresholds: ≤0.25 → 1, ≤0.5 → 0.5, >0.5 → 0
    # The production code checks: "c18" in chromatography.lower() and "lipid" not in it.
    # "HILICZ" / "HILIC" do NOT contain "c18" → non-C18 thresholds.
    # "C18-EP" and "C18" contain "c18" → C18 thresholds.

    def test_c18ep_perfect_match_returns_1(self):
        assert rt_quality(0.0, "C18-EP") == 1.0

    def test_c18ep_within_0_25_returns_1(self):
        assert rt_quality(0.20, "C18-EP") == 1.0

    def test_c18ep_exactly_0_25_returns_1(self):
        assert rt_quality(0.25, "C18-EP") == 1.0

    def test_c18ep_between_0_25_and_0_5_returns_0_5(self):
        assert rt_quality(0.35, "C18-EP") == 0.5

    def test_c18ep_exactly_0_5_returns_0_5(self):
        assert rt_quality(0.5, "C18-EP") == 0.5

    def test_c18ep_above_0_5_returns_0(self):
        assert rt_quality(0.6, "C18-EP") == 0.0

    def test_c18ep_negative_error_uses_absolute_value(self):
        assert rt_quality(-0.20, "C18-EP") == 1.0

    # --- Non-C18 thresholds: ≤0.5 → 1, ≤2.0 → 0.5, >2.0 → 0
    # "HILICZ" and "HILIC" are synonyms and do not contain "c18" → non-C18 thresholds.

    def test_hilicz_within_0_5_returns_1(self):
        assert rt_quality(0.4, "HILICZ") == 1.0

    def test_hilicz_exactly_0_5_returns_1(self):
        assert rt_quality(0.5, "HILICZ") == 1.0

    def test_hilicz_between_0_5_and_2_returns_0_5(self):
        assert rt_quality(1.0, "HILICZ") == 0.5

    def test_hilicz_exactly_2_returns_0_5(self):
        assert rt_quality(2.0, "HILICZ") == 0.5

    def test_hilicz_above_2_returns_0(self):
        assert rt_quality(3.0, "HILICZ") == 0.0

    def test_hilic_synonym_same_as_hilicz(self):
        # HILIC and HILICZ are synonyms — both use non-C18 thresholds
        assert rt_quality(0.4, "HILIC") == rt_quality(0.4, "HILICZ")
        assert rt_quality(1.0, "HILIC") == rt_quality(1.0, "HILICZ")

    def test_nan_error_returns_nan(self):
        result = rt_quality(float("nan"), "C18-EP")
        assert np.isnan(result)

    def test_c18_lipid_uses_non_c18_thresholds(self):
        # "C18-LIPID" contains "c18" but also "lipid" → non-C18 thresholds
        assert rt_quality(0.4, "C18-LIPID") == 1.0
        assert rt_quality(1.0, "C18-LIPID") == 0.5

    def test_case_insensitive_c18_detection(self):
        assert rt_quality(0.20, "c18-ep") == 1.0
        assert rt_quality(0.35, "c18-ep") == 0.5


# ===========================================================================
# total_score_and_msi
# ===========================================================================

class TestTotalScoreAndMsi:

    def test_all_ones_returns_3_and_exceeds_level_1(self):
        total, msi = total_score_and_msi(1.0, 1.0, 1.0)
        assert total == 3.0
        assert msi == "Exceeds Level 1"

    def test_two_ones_one_half_returns_2_5_and_level_1(self):
        total, msi = total_score_and_msi(1.0, 1.0, 0.5)
        assert total == 2.5
        assert msi == "Level 1"

    def test_median_below_1_returns_putative(self):
        # median([0.5, 0.5, 0.5]) = 0.5 < 1 → putative
        total, msi = total_score_and_msi(0.5, 0.5, 0.5)
        assert msi == "putative"

    def test_msms_minus_1_returns_remove(self):
        total, msi = total_score_and_msi(-1.0, 1.0, 1.0)
        assert "REMOVE" in msi

    def test_all_nan_returns_putative(self):
        total, msi = total_score_and_msi(float("nan"), float("nan"), float("nan"))
        assert msi == "putative"

    def test_total_is_nansum_of_scores(self):
        total, _ = total_score_and_msi(0.5, 1.0, 0.5)
        assert abs(total - 2.0) < 1e-9

    def test_nan_scores_excluded_from_total(self):
        # nansum([1.0, nan, 1.0]) = 2.0
        total, _ = total_score_and_msi(1.0, float("nan"), 1.0)
        assert abs(total - 2.0) < 1e-9

    def test_zero_scores_returns_putative(self):
        total, msi = total_score_and_msi(0.0, 0.0, 0.0)
        assert msi == "putative"
        assert total == 0.0

    def test_mixed_nan_and_ones_level_1(self):
        # scores = [1.0, 1.0]; median = 1.0; total = 2.0 → Level 1 (not 3)
        total, msi = total_score_and_msi(1.0, 1.0, float("nan"))
        assert msi == "Level 1"
        assert abs(total - 2.0) < 1e-9


# ===========================================================================
# _compute_all_overlapping_compounds
# ===========================================================================

class TestComputeAllOverlappingCompounds:

    def _make_mc(self, rows: list[dict]) -> pd.DataFrame:
        """Build a minimal manual_curation_df from a list of row dicts."""
        defaults = {
            "compound_name": "compound",
            "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N",
            "atlas_mz": 136.062,
            "rt_min": 1.0,
            "rt_max": 3.0,
        }
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def _mass_map(self, mc: pd.DataFrame, mass: float = 135.054) -> dict:
        return {ik: mass for ik in mc["inchi_key"].unique()}

    def test_single_compound_no_overlap(self):
        mc = self._make_mc([{"compound_name": "adenine"}])
        result = _compute_all_overlapping_compounds(mc, self._mass_map(mc))
        assert result[0] == ("", "")

    def test_two_distinct_compounds_no_overlap(self):
        mc = self._make_mc([
            {"compound_name": "adenine",   "atlas_mz": 136.062, "rt_min": 1.0, "rt_max": 3.0,
             "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N"},
            {"compound_name": "riboflavin","atlas_mz": 377.146, "rt_min": 4.0, "rt_max": 6.0,
             "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N"},
        ])
        mass_map = {
            "GFFGJBXGBJISGV-UHFFFAOYSA-N": 135.054,
            "AUNGANRZJHBGPY-SCRDCRAPSA-N": 376.138,
        }
        result = _compute_all_overlapping_compounds(mc, mass_map)
        assert result[0] == ("", "")
        assert result[1] == ("", "")

    def test_two_compounds_same_mz_overlapping_rt_detected(self):
        mc = self._make_mc([
            {"compound_name": "leucine",    "atlas_mz": 132.102, "rt_min": 8.5, "rt_max": 10.5,
             "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N"},
            {"compound_name": "isoleucine", "atlas_mz": 132.102, "rt_min": 9.0, "rt_max": 11.0,
             "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N"},
        ])
        mass_map = {
            "ROHFNLRQFUQHCH-UHFFFAOYSA-N": 131.094,
            "AGPKZVBTJJNPAG-WHFBIAKZSA-N": 131.094,
        }
        result = _compute_all_overlapping_compounds(mc, mass_map)
        # Both should list each other as overlapping
        assert result[0] != ("", "")
        assert result[1] != ("", "")

    def test_non_overlapping_rt_not_flagged(self):
        mc = self._make_mc([
            {"compound_name": "leucine",    "atlas_mz": 132.102, "rt_min": 1.0, "rt_max": 3.0,
             "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N"},
            {"compound_name": "isoleucine", "atlas_mz": 132.102, "rt_min": 5.0, "rt_max": 7.0,
             "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N"},
        ])
        mass_map = {
            "ROHFNLRQFUQHCH-UHFFFAOYSA-N": 131.094,
            "AGPKZVBTJJNPAG-WHFBIAKZSA-N": 131.094,
        }
        result = _compute_all_overlapping_compounds(mc, mass_map)
        assert result[0] == ("", "")
        assert result[1] == ("", "")

    def test_result_keys_cover_all_compounds(self):
        mc = self._make_mc([
            {"compound_name": "a", "atlas_mz": 100.0, "rt_min": 1.0, "rt_max": 2.0},
            {"compound_name": "b", "atlas_mz": 200.0, "rt_min": 3.0, "rt_max": 4.0},
            {"compound_name": "c", "atlas_mz": 300.0, "rt_min": 5.0, "rt_max": 6.0},
        ])
        result = _compute_all_overlapping_compounds(mc, {})
        assert set(result.keys()) == {0, 1, 2}

    def test_overlapping_names_joined_with_double_slash(self):
        mc = self._make_mc([
            {"compound_name": "leucine",    "atlas_mz": 132.102, "rt_min": 8.5, "rt_max": 10.5,
             "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N"},
            {"compound_name": "isoleucine", "atlas_mz": 132.102, "rt_min": 9.0, "rt_max": 11.0,
             "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N"},
        ])
        mass_map = {
            "ROHFNLRQFUQHCH-UHFFFAOYSA-N": 131.094,
            "AGPKZVBTJJNPAG-WHFBIAKZSA-N": 131.094,
        }
        result = _compute_all_overlapping_compounds(mc, mass_map)
        names_str = result[0][0]
        assert "//" in names_str


# ===========================================================================
# apply_auto_curation_defaults
# ===========================================================================

class TestApplyAutoCurationDefaults:

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {"ms1_notes": "", "ms2_notes": "", "analyst_notes": ""}
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def test_empty_ms1_notes_set_to_keep(self):
        df = self._make_df([{"ms1_notes": ""}])
        result = apply_auto_curation_defaults(df)
        assert result["ms1_notes"].iloc[0] == "keep"

    def test_existing_ms1_notes_not_overwritten(self):
        df = self._make_df([{"ms1_notes": "remove"}])
        result = apply_auto_curation_defaults(df)
        assert result["ms1_notes"].iloc[0] == "remove"

    def test_empty_ms2_notes_set_to_default(self):
        df = self._make_df([{"ms2_notes": ""}])
        result = apply_auto_curation_defaults(df)
        assert "curation skipped" in result["ms2_notes"].iloc[0]

    def test_existing_ms2_notes_not_overwritten(self):
        df = self._make_df([{"ms2_notes": "1.0, confirmed"}])
        result = apply_auto_curation_defaults(df)
        assert result["ms2_notes"].iloc[0] == "1.0, confirmed"

    def test_empty_analyst_notes_set_to_suffix(self):
        df = self._make_df([{"analyst_notes": ""}])
        result = apply_auto_curation_defaults(df)
        assert "manual curation skipped" in result["analyst_notes"].iloc[0]

    def test_existing_analyst_notes_get_suffix_appended(self):
        df = self._make_df([{"analyst_notes": "reviewed"}])
        result = apply_auto_curation_defaults(df)
        assert result["analyst_notes"].iloc[0].startswith("reviewed")
        assert "manual curation skipped" in result["analyst_notes"].iloc[0]

    def test_none_df_returned_unchanged(self):
        result = apply_auto_curation_defaults(None)
        assert result is None

    def test_empty_df_returned_unchanged(self):
        result = apply_auto_curation_defaults(pd.DataFrame())
        assert result.empty

    def test_multiple_rows_all_updated(self):
        df = self._make_df([
            {"ms1_notes": "", "ms2_notes": "", "analyst_notes": ""},
            {"ms1_notes": "", "ms2_notes": "", "analyst_notes": ""},
        ])
        result = apply_auto_curation_defaults(df)
        assert (result["ms1_notes"] == "keep").all()

    def test_whitespace_only_ms1_notes_treated_as_empty(self):
        df = self._make_df([{"ms1_notes": "   "}])
        result = apply_auto_curation_defaults(df)
        assert result["ms1_notes"].iloc[0] == "keep"


# ===========================================================================
# _strip_non_chars
# ===========================================================================

class TestStripNonChars:

    def test_normal_string_unchanged(self):
        assert _strip_non_chars("adenine") == "adenine"

    def test_empty_string_unchanged(self):
        assert _strip_non_chars("") == ""

    def test_unicode_non_character_removed(self):
        # U+FFFE is a non-character
        s = "hello\uFFFEworld"
        result = _strip_non_chars(s)
        assert "\uFFFE" not in result
        assert "helloworld" == result

    def test_u_ffff_removed(self):
        s = "test\uFFFF"
        result = _strip_non_chars(s)
        assert "\uFFFF" not in result

    def test_fdd0_range_removed(self):
        # U+FDD0 is in the non-character range FDD0-FDEF
        s = "a\uFDD0b"
        result = _strip_non_chars(s)
        assert "\uFDD0" not in result
        assert result == "ab"

    def test_regular_unicode_preserved(self):
        s = "café"
        assert _strip_non_chars(s) == "café"


# ===========================================================================
# _display_compound_idx
# ===========================================================================

class TestDisplayCompoundIdx:

    def test_zero_returns_one(self):
        assert _display_compound_idx(0) == 1

    def test_one_returns_two(self):
        assert _display_compound_idx(1) == 2

    def test_large_index(self):
        assert _display_compound_idx(99) == 100

    def test_returns_int(self):
        assert isinstance(_display_compound_idx(0), int)

    def test_float_input_converted(self):
        # The function casts to int internally
        assert _display_compound_idx(4.0) == 5


# ===========================================================================
# _get_file_color
# ===========================================================================

class TestGetFileColor:

    def test_none_color_map_returns_gray(self):
        assert _get_file_color("some_QC_file.h5", None) == "gray"

    def test_matching_key_returns_mapped_color(self):
        color_map = {"QC": "orange", "ISTD": "blue"}
        assert _get_file_color("run_QC_001.h5", color_map) == "orange"

    def test_case_insensitive_match(self):
        color_map = {"qc": "orange"}
        assert _get_file_color("run_QC_001.h5", color_map) == "orange"

    def test_no_matching_key_returns_gray(self):
        color_map = {"ISTD": "blue"}
        assert _get_file_color("run_QC_001.h5", color_map) == "gray"

    def test_empty_color_map_returns_gray(self):
        assert _get_file_color("run_QC_001.h5", {}) == "gray"

    def test_multiple_keys_first_match_wins(self):
        # Both "QC" and "run" appear in the filename; whichever is first in the dict wins
        color_map = {"QC": "orange", "run": "green"}
        result = _get_file_color("run_QC_001.h5", color_map)
        assert result in ("orange", "green")  # one of the two matches

    def test_istd_key_matched(self):
        color_map = {"ISTD": "blue", "QC": "orange"}
        assert _get_file_color("sample_ISTD_001.h5", color_map) == "blue"


# ===========================================================================
# _short_fname
# ===========================================================================

class TestShortFname:

    def _make_fname(self, parts: list[str], ext: str = ".h5") -> str:
        return "_".join(parts) + ext

    def test_empty_string_returns_no_data(self):
        assert _short_fname("") == "no data"

    def test_short_filename_returns_stem(self):
        # Fewer than 13 parts → return the full stem
        fname = "short_name.h5"
        result = _short_fname(fname)
        assert result == "short_name"

    def test_13_parts_returns_part_12(self):
        # 13 parts (indices 0-12) → return parts[12]
        parts = [f"p{i:02d}" for i in range(13)]
        fname = self._make_fname(parts)
        result = _short_fname(fname)
        assert result == "p12"

    def test_16_plus_parts_returns_parts_12_and_15(self):
        # 16+ parts → return f"{parts[12]}_{parts[15]}"
        parts = [f"p{i:02d}" for i in range(16)]
        fname = self._make_fname(parts)
        result = _short_fname(fname)
        assert result == "p12_p15"

    def test_path_prefix_stripped(self):
        # Full path should be reduced to basename stem
        fname = "/some/long/path/to/p00_p01_p02_p03_p04_p05_p06_p07_p08_p09_p10_p11_p12_p13_p14_p15.h5"
        result = _short_fname(fname)
        assert result == "p12_p15"

    def test_no_extension_handled(self):
        # No extension — splitext returns the whole name as stem
        fname = "a_b_c"
        result = _short_fname(fname)
        assert result == "a_b_c"
