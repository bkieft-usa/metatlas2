"""Unit tests for metatlas2/lcmsruns_tools.py.

Tested functions
----------------
* :func:`_classify_file_type`
* :func:`filter_lcmsruns_list`

Algorithm overview (two-phase)
-------------------------------
Phase 1 — ``run_metadata`` (and ``sample_name`` for QC) for high-priority types:
    ``qc``, ``istd``, ``injbl`` / ``injblk`` / ``blank`` — in that order.
    ``qc`` is checked in both fields; ``istd`` / ``injbl`` are checked in
    ``run_metadata`` only (they rarely appear in ``sample_name``).

Phase 2 — ``sample_name`` then ``run_metadata`` for lower-priority types:
    ``exctrl`` / ``txctrl``, ``refstd`` / ``standard``.

Phase 3 — fallback: ``'experimental'``.

Special case: ``QC-C18QU`` in either field is **not** a real QC run; the
``qc`` segment is skipped when immediately followed by ``c18qu``.

Regression cases covered
------------------------
1. ``QC-SOPv7`` in ``sample_name`` → ``'qc'`` (first-segment match).
2. ``ExCtrl-NA-NA-NA-NA`` / ``TxCtrl-NA-NA-NA-NA`` → ``'exctrl'`` via Phase 2.
3. ``run_metadata`` ending in ``-QC`` → ``'qc'`` via Phase 1.
4. ``QC-C18QU`` in ``run_metadata`` is skipped; next available key wins.
5. ``ISTD`` / ``InjBL`` in ``run_metadata`` → ``'istd'`` / ``'injbl'`` via Phase 1,
   even when ``ExCtrl`` is present in ``sample_name``.
6. Real-world HILICZ filenames from the SorghSwg project.
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
# _classify_file_type — Phase 1: QC detection (sample_name and run_metadata)
# ===========================================================================

class TestClassifyFileTypeQC:

    def test_exact_qc_in_sample_name(self):
        assert _classify_file_type(_fields("QC")) == "qc"

    def test_exact_qc_lowercase_in_sample_name(self):
        assert _classify_file_type(_fields("qc")) == "qc"

    def test_qc_sopv7_in_sample_name(self):
        """QC-SOPv7: first segment 'qc' → 'qc'."""
        assert _classify_file_type(_fields("QC-SOPv7")) == "qc"

    def test_qc_sopv7_lowercase_in_sample_name(self):
        assert _classify_file_type(_fields("qc-sopv7")) == "qc"

    def test_qc_v2_in_sample_name(self):
        assert _classify_file_type(_fields("QC-v2")) == "qc"

    def test_qc_in_run_metadata_trailing(self):
        """run_metadata ending in '-QC' must be classified as 'qc'."""
        assert _classify_file_type(
            _fields("AB-T1-ENDO", "Rg80to1200-CE102040norm-cells-QC")
        ) == "qc"

    def test_qc_in_run_metadata_with_exctrl_sample_name(self):
        assert _classify_file_type(
            _fields("ExCtrl-MeOH-PPL", "Rg80to1200-CE102040norm-filtrate-QC")
        ) == "qc"

    def test_qc_standalone_in_run_metadata(self):
        """Standalone 'QC' in run_metadata (no other segments) → 'qc'."""
        assert _classify_file_type(_fields("unknown", "QC")) == "qc"

    def test_qc_in_sample_name_beats_istd_in_run_metadata(self):
        """Phase 1 checks 'qc' before 'istd'; QC in sample_name wins."""
        assert _classify_file_type(
            _fields("QC-SOPv7", "Rg80to1200-CE102040norm-filtrate-ISTD")
        ) == "qc"

    def test_qc_c18qu_in_run_metadata_skipped(self):
        """QC-C18QU in run_metadata must be skipped; ExCtrl in sample_name wins."""
        assert _classify_file_type(
            _fields("ExCtrl-MeOH-PPL", "Rg80to1200-CE102040norm-filtrate-QC-C18QU")
        ) == "exctrl"

    def test_qc_c18qu_in_run_metadata_skipped_experimental_fallback(self):
        """QC-C18QU in run_metadata skipped; sample_name has no key → experimental."""
        assert _classify_file_type(
            _fields("AB-T1-ENDO", "Rg80to1200-CE102040norm-cells-QC-C18QU")
        ) == "experimental"

    def test_qc_c18qu_alone_in_sample_name_is_not_qc(self):
        """QC-C18QU as the entire sample_name: 'qc' segment is followed by
        'c18qu' → skip → no other key → experimental."""
        assert _classify_file_type(_fields("QC-C18QU")) == "experimental"

    def test_mixed_case_qc(self):
        assert _classify_file_type(_fields("Qc-SopV7")) == "qc"


# ===========================================================================
# _classify_file_type — Phase 1: ISTD detection (run_metadata only)
# ===========================================================================

class TestClassifyFileTypeISTD:

    def test_istd_in_run_metadata_trailing(self):
        """ISTD as last segment of run_metadata → 'istd'."""
        assert _classify_file_type(
            _fields("AB-T0-PPL", "Rg80to1200-CE102040norm-filtrate-ISTD")
        ) == "istd"

    def test_istd_in_run_metadata_standalone(self):
        assert _classify_file_type(_fields("unknown", "ISTD")) == "istd"

    def test_istd_in_run_metadata_with_exctrl_sample_name(self):
        """ISTD in run_metadata beats ExCtrl in sample_name (Phase 1 wins)."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE205060norm-ExCtrl-ISTD")
        ) == "istd"

    def test_istd_in_run_metadata_with_experimental_sample_name(self):
        """ISTD in run_metadata; experimental sample_name → 'istd'."""
        assert _classify_file_type(
            _fields("WT-P1-S", "Rg70to1050-CE102040norm-Stem-ISTD")
        ) == "istd"

    def test_istd_in_run_metadata_lowercase(self):
        assert _classify_file_type(
            _fields("WT-P1-L", "Rg70to1050-CE102040norm-Leaf-istd")
        ) == "istd"

    def test_istd_only_in_sample_name_no_run_metadata_is_experimental(self):
        """ISTD in sample_name only (no run_metadata) → 'experimental' under
        the new algorithm (Phase 1 checks run_metadata only for ISTD)."""
        assert _classify_file_type(_fields("ISTD")) == "experimental"

    def test_istd_with_suffix_in_sample_name_only_is_experimental(self):
        """ISTD-v2 in sample_name only → 'experimental'."""
        assert _classify_file_type(_fields("ISTD-v2")) == "experimental"


# ===========================================================================
# _classify_file_type — Phase 1: InjBl detection (run_metadata only)
# ===========================================================================

class TestClassifyFileTypeInjBl:

    def test_injbl_in_run_metadata(self):
        assert _classify_file_type(
            _fields("WC-T0-PPL", "Rg80to1200-CE102040norm-filtrate-InjBL-MeOH")
        ) == "injbl"

    def test_injbl_in_run_metadata_with_exctrl_sample_name(self):
        """InjBL in run_metadata beats ExCtrl in sample_name (Phase 1 wins)."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH")
        ) == "injbl"

    def test_injblk_alias_in_run_metadata(self):
        assert _classify_file_type(
            _fields("WC-T0-PPL", "Rg80to1200-CE102040norm-filtrate-InjBlk-MeOH")
        ) == "injbl"

    def test_blank_alias_in_run_metadata(self):
        assert _classify_file_type(
            _fields("WC-T0-PPL", "Rg80to1200-CE102040norm-filtrate-Blank")
        ) == "injbl"

    def test_injbl_only_in_sample_name_no_run_metadata_is_experimental(self):
        """InjBL in sample_name only → 'experimental' (Phase 1 checks run_metadata only)."""
        assert _classify_file_type(_fields("INJBL")) == "experimental"

    def test_blank_only_in_sample_name_no_run_metadata_is_experimental(self):
        assert _classify_file_type(_fields("BLANK")) == "experimental"

    def test_injbl_priority_over_exctrl_in_run_metadata(self):
        """'injbl' is checked before 'exctrl' in Phase 1; InjBL wins."""
        assert _classify_file_type(
            _fields("unknown", "Rg70to1050-ExCtrl-InjBL-MeOH")
        ) == "injbl"


# ===========================================================================
# _classify_file_type — Phase 2: ExCtrl / RefStd (sample_name then run_metadata)
# ===========================================================================

class TestClassifyFileTypePhase2:

    def test_exctrl_in_sample_name(self):
        assert _classify_file_type(_fields("EXCTRL")) == "exctrl"

    def test_exctrl_with_suffix_in_sample_name(self):
        assert _classify_file_type(_fields("EXCTRL-batch1")) == "exctrl"

    def test_txctrl_maps_to_exctrl(self):
        assert _classify_file_type(_fields("TXCTRL")) == "exctrl"

    def test_txctrl_na_classified_as_exctrl(self):
        assert _classify_file_type(_fields("TxCtrl-NA-NA-NA-NA")) == "exctrl"

    def test_exctrl_na_classified_as_exctrl(self):
        assert _classify_file_type(_fields("ExCtrl-NA-NA-NA-NA")) == "exctrl"

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

    def test_refstd_in_sample_name(self):
        assert _classify_file_type(_fields("REFSTD")) == "refstd"

    def test_refstd_with_suffix_in_sample_name(self):
        assert _classify_file_type(_fields("REFSTD-pooled")) == "refstd"

    def test_standard_maps_to_refstd(self):
        assert _classify_file_type(_fields("STANDARD")) == "refstd"

    def test_exctrl_in_run_metadata_only_no_sample_name_match(self):
        """ExCtrl in run_metadata only (no sample_name match) → 'exctrl' via Phase 2."""
        assert _classify_file_type(
            _fields("AB-T1-ENDO", "Rg70to1050-CE102040norm-ExCtrl-S1")
        ) == "exctrl"

    def test_exctrl_sample_name_no_high_priority_run_metadata(self):
        """ExCtrl in sample_name, run_metadata has no high-priority token → 'exctrl'."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE102040norm-ExCtrl-S1")
        ) == "exctrl"


# ===========================================================================
# _classify_file_type — Phase 3: experimental fallback
# ===========================================================================

class TestClassifyFileTypeExperimental:

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

    def test_empty_sample_name_returns_experimental(self):
        assert _classify_file_type(_fields("")) == "experimental"

    def test_both_fields_empty_returns_experimental(self):
        assert _classify_file_type(_fields("", "")) == "experimental"

    def test_unknown_sample_name_returns_experimental(self):
        assert _classify_file_type(_fields("T1-256534-8-Tr-RE")) == "experimental"

    def test_wt_p1_s_with_stem_istd_run_metadata_is_istd(self):
        """Experimental-looking sample_name but ISTD in run_metadata → 'istd'."""
        assert _classify_file_type(
            _fields("WT-P1-S", "Rg70to1050-CE102040norm-Stem-ISTD")
        ) == "istd"


# ===========================================================================
# _classify_file_type — real-world HILICZ SorghSwg filenames
# ===========================================================================

class TestClassifyFileTypeRealWorldFilenames:
    """Verify the four ambiguous SorghSwg filenames from the design discussion."""

    def test_exctrl_4_with_istd_run_metadata_is_istd(self):
        """..._36_ExCtrl_4_Rg70to1050-CE205060norm-ExCtrl-ISTD_Run114
        ExCtrl in sample_name, ISTD in run_metadata → Phase 1 wins → 'istd'."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE205060norm-ExCtrl-ISTD")
        ) == "istd"

    def test_wt_p1_s_2_with_stem_istd_is_istd(self):
        """..._2_WT-P1-S_2_Rg70to1050-CE102040norm-Stem-ISTD_Run49
        Experimental sample_name, ISTD in run_metadata → Phase 1 wins → 'istd'."""
        assert _classify_file_type(
            _fields("WT-P1-S", "Rg70to1050-CE102040norm-Stem-ISTD")
        ) == "istd"

    def test_exctrl_1_with_injbl_meoh_run_metadata_is_injbl(self):
        """..._33_ExCtrl_1_Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH_Run16
        ExCtrl in sample_name, InjBL in run_metadata → Phase 1 wins → 'injbl'."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH")
        ) == "injbl"

    def test_exctrl_1_neg_ms2_with_s1_run_metadata_is_exctrl(self):
        """..._NEG_MS2_33_ExCtrl_1_Rg70to1050-CE102040norm-ExCtrl-S1_Run15
        ExCtrl in sample_name, no high-priority token in run_metadata → Phase 2 → 'exctrl'."""
        assert _classify_file_type(
            _fields("ExCtrl", "Rg70to1050-CE102040norm-ExCtrl-S1")
        ) == "exctrl"


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
