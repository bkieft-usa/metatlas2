"""Unit tests for metatlas2/shared_tools/lcmsruns_tools.py.

Tested functions
----------------
* :func:`_classify_file_type`
* :func:`filter_lcmsruns_list`

Classification rules (first match wins; no match -> ``'experimental'``)
------------------------------------------------------------------------
Each field is lowercased and split on ``-``; keywords must match whole segments.

owner="jgi"
    1. ``qc-c18qu``  -- segments ``qc``, ``c18qu`` (consecutive) in run_metadata
    2. ``qc``        -- ``qc`` in run_metadata
    3. ``istd``      -- ``istd`` in run_metadata
    4. ``injbl``     -- ``injbl`` in run_metadata

owner="egsb"
    1. ``qc``, 
    2. ``istd``, 
    3. ``injbl``
    run_metadata is never used.
"""

from __future__ import annotations

import pytest

from metatlas2.shared_tools.lcmsruns_tools import _classify_file_type, filter_lcmsruns_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fields(sample_name: str = "", run_metadata: str = "", sample_number: str = "") -> dict:
    """Return a minimal fields dict as produced by ``fpf.parse_file_name``."""
    return {
        "sample_number": sample_number,
        "sample_name": sample_name,
        "run_metadata": run_metadata,
    }


def jgi(**kw) -> str:
    return _classify_file_type(_fields(**kw), "jgi")


def egsb(**kw) -> str:
    return _classify_file_type(_fields(**kw), "egsb")


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


OWNERS = ["jgi", "egsb"]


# ===========================================================================
# Owner handling
# ===========================================================================

class TestOwnerHandling:

    def test_owner_is_required(self):
        with pytest.raises(TypeError):
            _classify_file_type(_fields("QC"))  # type: ignore[call-arg]

    @pytest.mark.parametrize("owner", ["", None, "esbg", "jig", "egbs", "unknown"])
    def test_unknown_owner_raises(self, owner):
        with pytest.raises(ValueError, match="Unknown owner"):
            _classify_file_type(_fields("QC"), owner)

    @pytest.mark.parametrize("owner", ["JGI", " jgi ", "Jgi"])
    def test_jgi_owner_is_case_and_whitespace_insensitive(self, owner):
        assert _classify_file_type(_fields(run_metadata="Rg80-QC"), owner) == "qc"

    @pytest.mark.parametrize("owner", ["EGSB", " egsb ", "Egsb"])
    def test_egsb_owner_is_case_and_whitespace_insensitive(self, owner):
        assert _classify_file_type(_fields(sample_name="QC"), owner) == "qc"

    def test_same_fields_classify_differently_per_owner(self):
        f = _fields(sample_name="QC-SOPv7", run_metadata="Rg80-ISTD")
        assert _classify_file_type(f, "jgi") == "istd"   # run_metadata only
        assert _classify_file_type(f, "egsb") == "qc"    # sample_name only


# ===========================================================================
# JGI: QC and QC-C18QU (run_metadata only)
# ===========================================================================

class TestJgiQC:

    def test_qc_trailing_in_run_metadata(self):
        assert jgi(sample_name="AB-T1-ENDO",
                   run_metadata="Rg80to1200-CE102040norm-cells-QC") == "qc"

    def test_qc_standalone_in_run_metadata(self):
        assert jgi(sample_name="unknown", run_metadata="QC") == "qc"

    def test_qc_lowercase_in_run_metadata(self):
        assert jgi(run_metadata="Rg80-qc") == "qc"

    def test_qc_in_run_metadata_with_exctrl_sample_name(self):
        assert jgi(sample_name="ExCtrl-MeOH-PPL",
                   run_metadata="Rg80to1200-CE102040norm-filtrate-QC") == "qc"

    @pytest.mark.parametrize("sample_name", ["QC", "QC-SOPv7", "qc-v2", "QC-C18QU"])
    def test_qc_in_sample_name_only_is_experimental(self, sample_name):
        assert jgi(sample_name=sample_name) == "experimental"

    def test_qc_requires_whole_segment(self):
        assert jgi(run_metadata="Rg80-QC1") == "experimental"
        assert jgi(run_metadata="Rg80-myQC") == "experimental"

    def test_qc_c18qu_in_run_metadata(self):
        assert jgi(sample_name="AB-T1-ENDO",
                   run_metadata="Rg80to1200-CE102040norm-cells-QC-C18QU") == "qc-c18qu"

    def test_qc_c18qu_standalone_in_run_metadata(self):
        assert jgi(run_metadata="QC-C18QU") == "qc-c18qu"

    def test_qc_c18qu_lowercase(self):
        assert jgi(run_metadata="Rg80-qc-c18qu") == "qc-c18qu"

    def test_qc_c18qu_beats_exctrl_sample_name(self):
        """Old behaviour (skip QC-C18QU, fall through to exctrl) is gone."""
        assert jgi(sample_name="ExCtrl-MeOH-PPL",
                   run_metadata="Rg80to1200-CE102040norm-filtrate-QC-C18QU") == "qc-c18qu"

    def test_qc_c18qu_is_not_plain_qc(self):
        assert jgi(run_metadata="Rg80-QC-C18QU") != "qc"

    def test_qc_and_c18qu_must_be_adjacent(self):
        """Non-adjacent 'qc' ... 'c18qu' is plain qc."""
        assert jgi(run_metadata="Rg80-QC-S1-C18QU") == "qc"

    def test_c18qu_without_qc_is_experimental(self):
        assert jgi(run_metadata="Rg80-C18QU") == "experimental"

    def test_qc_beats_istd_in_run_metadata(self):
        assert jgi(run_metadata="Rg80-QC-ISTD") == "qc"

    def test_mixed_case_qc(self):
        assert jgi(run_metadata="Rg80-Qc") == "qc"


# ===========================================================================
# JGI: ISTD and InjBl (run_metadata only)
# ===========================================================================

class TestJgiISTDAndInjBl:

    def test_istd_trailing(self):
        assert jgi(sample_name="AB-T0-PPL",
                   run_metadata="Rg80to1200-CE102040norm-filtrate-ISTD") == "istd"

    def test_istd_standalone(self):
        assert jgi(sample_name="unknown", run_metadata="ISTD") == "istd"

    def test_istd_lowercase(self):
        assert jgi(sample_name="WT-P1-L",
                   run_metadata="Rg70to1050-CE102040norm-Leaf-istd") == "istd"

    def test_istd_beats_exctrl_sample_name(self):
        assert jgi(sample_name="ExCtrl",
                   run_metadata="Rg70to1050-CE205060norm-ExCtrl-ISTD") == "istd"

    def test_istd_only_in_sample_name_is_experimental(self):
        assert jgi(sample_name="ISTD") == "experimental"
        assert jgi(sample_name="ISTD-v2") == "experimental"

    def test_injbl_in_run_metadata(self):
        assert jgi(sample_name="WC-T0-PPL",
                   run_metadata="Rg80to1200-CE102040norm-filtrate-InjBL-MeOH") == "injbl"

    def test_injbl_beats_exctrl_sample_name(self):
        assert jgi(sample_name="ExCtrl",
                   run_metadata="Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH") == "injbl"

    def test_injbl_only_in_sample_name_is_experimental(self):
        assert jgi(sample_name="INJBL") == "experimental"

    def test_istd_beats_injbl(self):
        assert jgi(run_metadata="Rg80-ISTD-InjBl") == "istd"

    @pytest.mark.parametrize("alias", ["InjBlk", "Blank"])
    def test_old_injbl_aliases_no_longer_match(self, alias):
        assert jgi(sample_name="WC-T0-PPL",
                   run_metadata=f"Rg80to1200-CE102040norm-filtrate-{alias}") == "experimental"


# ===========================================================================
# ExCtrl / TxCtrl are not types
# ===========================================================================

class TestExCtrlTxCtrlAreExperimental:

    @pytest.mark.parametrize("owner", OWNERS)
    @pytest.mark.parametrize("sample_name", [
        "ExCtrl", "EXCTRL-batch1", "TxCtrl", "TxCtrl-NA-NA-NA-NA", "ExCtrl-NA-NA-NA-NA",
        "ExCtrl-LCMS-PPL", "ExCtrl-MeOH-PPL", "ExCtrl-Vial-PPL", "ExCtrl-Full-PPL",
        "ExCtrl-N2-PPL", "ExCtrl-MeOHHCl-PPL", "ExCtrl-Vial-ENDO", "ExCtrl-MeOH-ENDO",
        "ExCtrl-TefTube-ENDO", "ExCtrl-Filter-ENDO", "ExCtrl-LCMS-ENDO",
    ])
    def test_sample_names(self, owner, sample_name):
        if "txctrl" in sample_name.lower():
            if owner == "egsb":
                assert _classify_file_type(_fields(sample_name=sample_name), owner) == "txctrl"
            if owner == "jgi":
                assert _classify_file_type(_fields(sample_name=sample_name), owner) == "experimental"
        else:
            assert _classify_file_type(_fields(sample_name=sample_name), owner) == "exctrl"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_exctrl_in_run_metadata(self, owner):
        assert _classify_file_type(
            _fields(sample_name="AB-T1-ENDO", run_metadata="Rg70to1050-CE102040norm-ExCtrl-S1"),
            owner,
        ) == "experimental"


# ===========================================================================
# EGSB (sample_name only)
# ===========================================================================

class TestEgsbSampleNameTypes:

    @pytest.mark.parametrize("sample_name, expected", [
        ("QC", "qc"), ("qc", "qc"), ("QC-SOPv7", "qc"), ("qc-sopv7", "qc"),
        ("QC-v2", "qc"), ("Qc-SopV7", "qc"),
        ("ISTD", "istd"), ("ISTD-v2", "istd"), ("istd", "istd"),
        ("InjBl", "injbl"), ("INJBL", "injbl"), ("injbl-MeOH", "injbl"),
    ])
    def test_keyword_in_sample_name(self, sample_name, expected):
        assert egsb(sample_name=sample_name) == expected

    def test_keyword_in_segment_middle(self):
        assert egsb(sample_name="AB-QC-1") == "qc"

    def test_qc_requires_whole_segment(self):
        assert egsb(sample_name="QC1") == "experimental"

    def test_qc_c18qu_has_no_special_type(self):
        assert egsb(sample_name="QC-C18QU") == "qc"

    @pytest.mark.parametrize("run_metadata", [
        "Rg80-QC", "Rg80-ISTD", "Rg80-InjBl", "Rg80-ExCtrl", "Rg80-TxCtrl", "Rg80-QC-C18QU",
    ])
    def test_run_metadata_is_ignored(self, run_metadata):
        assert egsb(sample_name="AB-T1-ENDO", run_metadata=run_metadata) == "experimental"

    @pytest.mark.parametrize("sample_name", ["INJBLK", "BLANK", "STANDARD"])
    def test_old_aliases_no_longer_match(self, sample_name):
        assert egsb(sample_name=sample_name) == "experimental"

    def test_priority_qc_beats_istd(self):
        assert egsb(sample_name="QC-ISTD") == "qc"

    def test_priority_istd_beats_injbl(self):
        assert egsb(sample_name="ISTD-InjBl") == "istd"

    def test_exctrl_sample_name_does_not_block_refstd(self):
        assert egsb(sample_name="ExCtrl", sample_number="RefStd-1") == "exctrl"


# ===========================================================================
# RefStd (sample_number, both owners)
# ===========================================================================

class TestRefStd:

    @pytest.mark.parametrize("owner", OWNERS)
    @pytest.mark.parametrize("sample_number", ["RefStd", "REFSTD", "refstd", "RefStd-1", "RefStd-pooled"])
    def test_refstd_in_sample_number(self, owner, sample_number):
        assert _classify_file_type(_fields(sample_number=sample_number), owner) == "refstd"

    @pytest.mark.parametrize("owner", OWNERS)
    @pytest.mark.parametrize("sample_name", ["REFSTD", "REFSTD-pooled", "STANDARD"])
    def test_refstd_in_sample_name_only_is_experimental(self, owner, sample_name):
        assert _classify_file_type(_fields(sample_name=sample_name), owner) == "experimental"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_refstd_in_run_metadata_only_is_experimental(self, owner):
        assert _classify_file_type(_fields(run_metadata="Rg80-RefStd"), owner) == "experimental"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_standard_in_sample_number_is_experimental(self, owner):
        assert _classify_file_type(_fields(sample_number="Standard"), owner) == "experimental"

    def test_jgi_refstd_is_lowest_priority(self):
        assert jgi(sample_number="RefStd-1", run_metadata="Rg80-InjBl") == "injbl"


# ===========================================================================
# Experimental fallback
# ===========================================================================

class TestExperimental:

    @pytest.mark.parametrize("owner", OWNERS)
    @pytest.mark.parametrize("sample_name", [
        "Ctrl-Media-PPL", "WC-T0-PPL", "AB-T1-ENDO", "soil-Slyc-OMTmb-inf-35dag",
        "RtTis-Slyc-OMTmb-inf-35dag", "T1-256534-8-Tr-RE", "",
    ])
    def test_no_keyword_is_experimental(self, owner, sample_name):
        assert _classify_file_type(_fields(sample_name=sample_name), owner) == "experimental"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_all_fields_empty(self, owner):
        assert _classify_file_type(_fields(), owner) == "experimental"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_missing_keys_and_none_values(self, owner):
        assert _classify_file_type({}, owner) == "experimental"
        assert _classify_file_type(
            {"sample_name": None, "run_metadata": None, "sample_number": None}, owner
        ) == "experimental"

    @pytest.mark.parametrize("owner", OWNERS)
    def test_documented_example_filename_is_experimental(self, owner):
        """20260925_JGI_TS_..._POS_MS2_40_J319-A-Lowland-653-Progeny_1_Rg80to1200-CE102040norm-G00031908-S1_Run123.h5"""
        f = _fields(
            sample_number="40",
            sample_name="J319-A-Lowland-653-Progeny",
            run_metadata="Rg80to1200-CE102040norm-G00031908-S1",
        )
        assert _classify_file_type(f, owner) == "experimental"


# ===========================================================================
# Real-world HILICZ SorghSwg filenames
# ===========================================================================

class TestRealWorldFilenames:
    """The four ambiguous SorghSwg filenames, classified with JGI rules."""

    def test_exctrl_4_with_istd_run_metadata_is_istd(self):
        """..._36_ExCtrl_4_Rg70to1050-CE205060norm-ExCtrl-ISTD_Run114"""
        assert jgi(sample_number="36", sample_name="ExCtrl",
                   run_metadata="Rg70to1050-CE205060norm-ExCtrl-ISTD") == "istd"

    def test_wt_p1_s_2_with_stem_istd_is_istd(self):
        """..._2_WT-P1-S_2_Rg70to1050-CE102040norm-Stem-ISTD_Run49"""
        assert jgi(sample_number="2", sample_name="WT-P1-S",
                   run_metadata="Rg70to1050-CE102040norm-Stem-ISTD") == "istd"

    def test_exctrl_1_with_injbl_meoh_run_metadata_is_injbl(self):
        """..._33_ExCtrl_1_Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH_Run16"""
        assert jgi(sample_number="33", sample_name="ExCtrl",
                   run_metadata="Rg70to1050-CE102040norm-ExCtrl-InjBL-MeOH") == "injbl"


    @pytest.mark.parametrize("owner", OWNERS)
    def test_refstd_sample_number_filename(self, owner):
        """..._POS_MS2_RefStd-1_FA-CoA_1_Rg70to1050-CE205060norm-RxnMix-S1_Run135"""
        f = _fields(sample_number="RefStd-1", sample_name="FA-CoA",
                    run_metadata="Rg70to1050-CE205060norm-RxnMix-S1")
        assert _classify_file_type(f, owner) == "refstd"


# ===========================================================================
# filter_lcmsruns_list  (function unchanged)
# ===========================================================================

class TestFilterLcmsrunsList:

    _QC_FPS  = _run("qc",           polarity="fps")
    _QC_POS  = _run("qc",           polarity="pos")
    _QC_NEG  = _run("qc",           polarity="neg")
    _EXP_FPS = _run("experimental", polarity="fps")
    _EXP_POS = _run("experimental", polarity="pos")
    _EXP_NEG = _run("experimental", polarity="neg")
    _ISTD    = _run("istd",         polarity="fps")

    _ALL = [_QC_FPS, _QC_POS, _QC_NEG, _EXP_FPS, _EXP_POS, _EXP_NEG, _ISTD]

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
        result = filter_lcmsruns_list(lcmsruns=self._ALL, exclude_file_type=["NEG"])
        assert all(r["polarity"] != "neg" for r in result)

    def test_exclude_pos_removes_pos_runs(self):
        result = filter_lcmsruns_list(lcmsruns=self._ALL, exclude_file_type=["POS"])
        assert all(r["polarity"] != "pos" for r in result)

    # --- include file_type only ---

    def test_include_qc_returns_only_qc_runs(self):
        result = filter_lcmsruns_list(lcmsruns=self._ALL, include_file_type=["QC"])
        assert all(r["file_type"] == "qc" for r in result)

    def test_include_istd_returns_only_istd_runs(self):
        result = filter_lcmsruns_list(lcmsruns=self._ALL, include_file_type=["ISTD"])
        assert all(r["file_type"] == "istd" for r in result)

    # --- new file types ---

    def test_include_qc_c18qu_returns_only_qc_c18qu_runs(self):
        runs = [_run("qc"), _run("qc-c18qu"), _run("experimental")]
        result = filter_lcmsruns_list(lcmsruns=runs, include_file_type=["QC-C18QU"])
        assert [r["file_type"] for r in result] == ["qc-c18qu"]

    def test_include_qc_excludes_qc_c18qu_runs(self):
        """RT alignment ('qc' only) must not pick up QC-C18QU runs."""
        runs = [_run("qc"), _run("qc-c18qu")]
        result = filter_lcmsruns_list(lcmsruns=runs, include_file_type=["QC"])
        assert [r["file_type"] for r in result] == ["qc"]

    # --- no matching runs raises ValueError ---

    def test_no_matching_runs_raises_value_error(self):
        with pytest.raises(ValueError, match="No LCMS runs matched"):
            filter_lcmsruns_list(lcmsruns=self._ALL, include_file_type=["InjBl"])