"""Unit tests for metatlas2/lcmsruns_tools.py.

Tested functions
----------------
* :func:`_classify_file_type`
* :func:`filter_lcmsruns_list`

Regression cases covered
------------------------
1. Compound ``sample_name`` values such as ``QC-SOPv7`` (hyphen + suffix after
   a known key) must be classified by their first hyphen-segment, not by exact
   match.
2. ``ExCtrl-NA-NA-NA-NA`` and ``TxCtrl-NA-NA-NA-NA`` must be classified as
   ``'exctrl'`` via the first segment.
3. ``run_metadata`` fields that embed a type token as a hyphen-segment (e.g.
   ``Rg80to1200-CE102040norm-filtrate-QC``) must be classified correctly.
4. ``QC-C18QU`` embedded in ``run_metadata`` must be **skipped** so that the
   next available classification (from ``sample_name``) is used instead.
5. ``SAMPLE_NAME_TO_FILE_TYPE`` insertion order determines priority: ``qc``
   is checked before ``istd``, ``exctrl``, etc.
"""

from __future__ import annotations

import pytest

from metatlas2.lcmsruns_tools import _classify_file_type, filter_lcmsruns_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fields(sample_name: str, run_metadata: str = "") -> dict:
    """Return a minimal fields dict as produced by ``fpf.parse_file_name``."""
    return {"sample_name": sample_name, "run_metadata": run_metadata}


def _run(file_type: str, polarity: str = "fps", chromatography: str = "hilic",
         ms_level: str = "ms1", file_format: str = "h5") -> dict:
    """Return a minimal LCMS run dict."""
    return {
        "file_type": file_type,
        "polarity": polarity,
        "chromatography": chromatography,
        "ms_level": ms_level,
        "file_format": file_format,
        "filename": f"dummy_{file_type}_{polarity}.h5",
        "file_path": f"/data/dummy_{file_type}_{polarity}.h5",
    }


# ===========================================================================
# _classify_file_type — exact / single-segment sample_name
# ===========================================================================

class TestClassifyFileTypeExact:

    def test_exact_qc(self):
        assert _classify_file_type(_fields("QC")) == "qc"

    def test_exact_qc_lowercase(self):
        assert _classify_file_type(_fields("qc")) == "qc"

    def test_exact_istd(self):
        assert _classify_file_type(_fields("ISTD")) == "istd"

    def test_exact_exctrl(self):
        assert _classify_file_type(_fields("EXCTRL")) == "exctrl"

    def test_exact_txctrl_maps_to_exctrl(self):
        assert _classify_file_type(_fields("TXCTRL")) == "exctrl"

    def test_exact_injbl(self):
        assert _classify_file_type(_fields("INJBL")) == "injbl"

    def test_exact_blank_maps_to_injbl(self):
        assert _classify_file_type(_fields("BLANK")) == "injbl"

    def test_exact_refstd(self):
        assert _classify_file_type(_fields("REFSTD")) == "refstd"

    def test_exact_standard_maps_to_refstd(self):
        assert _classify_file_type(_fields("STANDARD")) == "refstd"


# ===========================================================================
# _classify_file_type — compound sample_name (first-segment match)
# ===========================================================================

class TestClassifyFileTypeCompoundSampleName:

    def test_qc_sopv7_classified_as_qc(self):
        """QC-SOPv7: first segment 'qc' → 'qc'."""
        assert _classify_file_type(_fields("QC-SOPv7")) == "qc"

    def test_qc_sopv7_lowercase(self):
        assert _classify_file_type(_fields("qc-sopv7")) == "qc"

    def test_qc_v2_classified_as_qc(self):
        assert _classify_file_type(_fields("QC-v2")) == "qc"

    def test_istd_with_suffix_classified_as_istd(self):
        assert _classify_file_type(_fields("ISTD-v2")) == "istd"

    def test_exctrl_with_suffix_classified_as_exctrl(self):
        assert _classify_file_type(_fields("EXCTRL-batch1")) == "exctrl"

    def test_refstd_with_suffix_classified_as_refstd(self):
        assert _classify_file_type(_fields("REFSTD-pooled")) == "refstd"

    # C18 dataset patterns
    def test_exctrl_lcms_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-LCMS-PPL")) == "exctrl"

    def test_exctrl_meoh_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-MeOH-PPL")) == "exctrl"

    def test_exctrl_vial_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-Vial-PPL")) == "exctrl"

    def test_exctrl_full_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-Full-PPL")) == "exctrl"

    def test_exctrl_n2_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-N2-PPL")) == "exctrl"

    def test_exctrl_meohhcl_ppl_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-MeOHHCl-PPL")) == "exctrl"

    def test_exctrl_vial_endo_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-Vial-ENDO")) == "exctrl"

    def test_exctrl_meoh_endo_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-MeOH-ENDO")) == "exctrl"

    def test_exctrl_teftube_endo_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-TefTube-ENDO")) == "exctrl"

    def test_exctrl_filter_endo_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-Filter-ENDO")) == "exctrl"

    def test_exctrl_lcms_endo_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-LCMS-ENDO")) == "exctrl"

    # HILICZ dataset patterns
    def test_txctrl_na_classified_as_exctrl(self):
        assert _classify_file_type(_fields("TxCtrl-NA-NA-NA-NA")) == "exctrl"

    def test_exctrl_na_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-NA-NA-NA-NA")) == "exctrl"

    # Experimental samples — must NOT match any key
    def test_ctrl_media_ppl_classified_as_experimental(self):
        assert _classify_file_type(_fields("Ctrl-Media-PPL")) == "experimental"

    def test_wc_t0_ppl_classified_as_experimental(self):
        assert _classify_file_type(_fields("WC-T0-PPL")) == "experimental"

    def test_ab_t1_endo_classified_as_experimental(self):
        assert _classify_file_type(_fields("AB-T1-ENDO")) == "experimental"

    def test_soil_sample_classified_as_experimental(self):
        assert _classify_file_type(_fields("soil-Slyc-OMTmb-inf-35dag")) == "experimental"

    def test_rttis_sample_classified_as_experimental(self):
        assert _classify_file_type(_fields("RtTis-Slyc-OMTmb-inf-35dag")) == "experimental"


# ===========================================================================
# _classify_file_type — run_metadata segment scanning
# ===========================================================================

class TestClassifyFileTypeRunMetadata:

    def test_run_metadata_trailing_qc_classified_as_qc(self):
        """run_metadata ending in '-QC' must be classified as 'qc'."""
        assert _classify_file_type(
            _fields("AB-T1-ENDO", "Rg80to1200-CE102040norm-cells-QC")
        ) == "qc"

    def test_run_metadata_trailing_qc_filtrate_classified_as_qc(self):
        assert _classify_file_type(
            _fields("ExCtrl-MeOH-PPL", "Rg80to1200-CE102040norm-filtrate-QC")
        ) == "qc"

    def test_run_metadata_qc_c18qu_skipped_sample_name_exctrl(self):
        """QC-C18QU in run_metadata must be skipped; ExCtrl in sample_name wins."""
        assert _classify_file_type(
            _fields("ExCtrl-MeOH-PPL", "Rg80to1200-CE102040norm-filtrate-QC-C18QU")
        ) == "exctrl"

    def test_run_metadata_qc_c18qu_skipped_sample_name_experimental(self):
        """QC-C18QU in run_metadata skipped; sample_name has no key → experimental."""
        assert _classify_file_type(
            _fields("AB-T1-ENDO", "Rg80to1200-CE102040norm-cells-QC-C18QU")
        ) == "experimental"

    def test_run_metadata_istd_classified_as_istd(self):
        assert _classify_file_type(
            _fields("AB-T0-PPL", "Rg80to1200-CE102040norm-filtrate-ISTD")
        ) == "istd"

    def test_run_metadata_injbl_meoh_classified_as_injbl(self):
        assert _classify_file_type(
            _fields("WC-T0-PPL", "Rg80to1200-CE102040norm-filtrate-InjBL-MeOH")
        ) == "injbl"

    def test_run_metadata_exact_qc_standalone(self):
        """Standalone 'QC' in run_metadata (no other segments) → 'qc'."""
        assert _classify_file_type(_fields("unknown", "QC")) == "qc"

    def test_run_metadata_exact_istd_standalone(self):
        assert _classify_file_type(_fields("unknown", "ISTD")) == "istd"

    # Priority: sample_name scanned first
    def test_sample_name_wins_over_run_metadata(self):
        """sample_name 'QC' must win even if run_metadata also has a key."""
        assert _classify_file_type(
            _fields("QC-SOPv7", "Rg80to1200-CE102040norm-filtrate-ISTD")
        ) == "qc"

    def test_qc_key_order_priority_over_exctrl_in_run_metadata(self):
        """'qc' appears before 'exctrl' in SAMPLE_NAME_TO_FILE_TYPE, so a
        run_metadata segment 'qc' wins over an 'exctrl' segment in the same
        field."""
        assert _classify_file_type(
            _fields("unknown", "Rg80to1200-CE102040norm-filtrate-QC")
        ) == "qc"


# ===========================================================================
# _classify_file_type — case insensitivity and edge cases
# ===========================================================================

class TestClassifyFileTypeEdgeCases:

    def test_mixed_case_qc_sopv7(self):
        assert _classify_file_type(_fields("Qc-SopV7")) == "qc"

    def test_empty_sample_name_returns_experimental(self):
        assert _classify_file_type(_fields("")) == "experimental"

    def test_both_fields_empty_returns_experimental(self):
        assert _classify_file_type(_fields("", "")) == "experimental"

    def test_unknown_sample_name_returns_experimental(self):
        assert _classify_file_type(_fields("T1-256534-8-Tr-RE")) == "experimental"

    def test_qc_c18qu_alone_in_sample_name_is_not_qc(self):
        """QC-C18QU as the entire sample_name: 'qc' segment is followed by
        'c18qu' → skip → no other key → experimental."""
        assert _classify_file_type(_fields("QC-C18QU")) == "experimental"


# ===========================================================================
# filter_lcmsruns_list
# ===========================================================================

class TestFilterLcmsrunsList:

    _QC_FPS  = _run("qc",           polarity="fps")
    _QC_POS  = _run("qc",           polarity="pos")
    _QC_NEG  = _run("qc",           polarity="neg")
    _EXP_FPS = _run("experimental", polarity="fps")
    _EXP_POS = _run("experimental", polarity="pos")
    _EXP_NEG = _run("experimental", polarity="neg")
    _ISTD    = _run("istd",         polarity="fps")
    _EXCTRL  = _run("exctrl",       polarity="fps")

    _ALL = [_QC_FPS, _QC_POS, _QC_NEG, _EXP_FPS, _EXP_POS, _EXP_NEG, _ISTD, _EXCTRL]

    # --- RT alignment scenario: include QC, exclude NEG ---

    def test_rt_alignment_include_qc_exclude_neg_returns_fps_and_pos_qc(self):
        """Mirrors the RT alignment filter call in workflows.run_rt_alignment."""
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            include_file_type=["QC"],
            exclude_file_type=["NEG"],
            ms_level="ms1",
        )
        file_types = {r["file_type"] for r in result}
        polarities = {r["polarity"] for r in result}
        assert file_types == {"qc"}
        assert "neg" not in polarities

    def test_rt_alignment_none_include_defaults_to_all_file_types(self):
        """When include_file_type is None (YAML null), no file_type filter is applied."""
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            include_file_type=None,
            exclude_file_type=["NEG"],
            ms_level="ms1",
        )
        polarities = {r["polarity"] for r in result}
        assert "neg" not in polarities
        file_types = {r["file_type"] for r in result}
        assert "qc" in file_types
        assert "experimental" in file_types

    # --- exclude polarity only ---

    def test_exclude_neg_removes_neg_runs(self):
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            exclude_file_type=["NEG"],
        )
        assert all(r["polarity"] != "neg" for r in result)

    def test_exclude_pos_removes_pos_runs(self):
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            exclude_file_type=["POS"],
        )
        assert all(r["polarity"] != "pos" for r in result)

    # --- include file_type only ---

    def test_include_qc_returns_only_qc_runs(self):
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            include_file_type=["QC"],
        )
        assert all(r["file_type"] == "qc" for r in result)

    def test_include_istd_returns_only_istd_runs(self):
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            include_file_type=["ISTD"],
        )
        assert all(r["file_type"] == "istd" for r in result)

    def test_include_exctrl_returns_only_exctrl_runs(self):
        result = filter_lcmsruns_list(
            lcmsruns=self._ALL,
            include_file_type=["exctrl"],
        )
        assert all(r["file_type"] == "exctrl" for r in result)

    # --- no matching runs raises ValueError ---

    def test_no_matching_runs_raises_value_error(self):
        with pytest.raises(ValueError, match="No LCMS runs matched"):
            filter_lcmsruns_list(
                lcmsruns=self._ALL,
                include_file_type=["refstd"],
            )
