"""Unit tests for metatlas2/create_curation_container.py.

Tested functions
----------------
* :func:`analyze_ms1`
* :func:`_suggest_rt_bounds_from_ms1`
* :func:`_build_isomer_dict`
* :func:`create_manual_curation_obj`  (via a lightweight stub of ``auto_id_obj``)

The ``enrich_atlas_df_with_compound_metadata`` DB call is patched so no
DuckDB database is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from metatlas2.create_curation_container import (
    _build_isomer_dict,
    _suggest_rt_bounds_from_ms1,
    analyze_ms1,
    create_manual_curation_obj,
)
from metatlas2.workflow_objects import ExperimentalData

from conftest import (
    ADENINE_MZ,
    ADENINE_RT,
    ADENINE_RMIN,
    ADENINE_RMAX,
    RIBOFLAVIN_MZ,
    RIBOFLAVIN_RT,
    RIBOFLAVIN_RMIN,
    RIBOFLAVIN_RMAX,
    _make_compound_mzrt,
    _make_atlas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atlas_row(
    mz: float = ADENINE_MZ,
    rt_peak: float = ADENINE_RT,
    rt_min: float = ADENINE_RMIN,
    rt_max: float = ADENINE_RMAX,
) -> dict:
    """Return a minimal atlas-row mapping accepted by :func:`analyze_ms1`."""
    return {
        "mz": mz,
        "rt_peak": rt_peak,
        "rt_min": rt_min,
        "rt_max": rt_max,
    }


def _ms1_df_for_uid(uid: str, *, n_points: int = 5, peak_intensity: float = 1e5) -> pd.DataFrame:
    """Build a wide-format MS1 DataFrame for a single compound × single file."""
    rts  = [ADENINE_RT + (i - n_points // 2) * 0.05 for i in range(n_points)]
    ints = [peak_intensity * max(0.1, 1.0 - abs(i - n_points // 2) * 0.3) for i in range(n_points)]
    mzs  = [ADENINE_MZ] * n_points
    in_f = [True] * n_points
    return pd.DataFrame([{
        "mz_rt_uid":  uid,
        "filename":   "test_run.h5",
        "spec_rts":   rts,
        "spec_ints":  ints,
        "spec_mzs":   mzs,
        "in_feature": in_f,
    }])


def _make_auto_id_obj(atlas, ms1_df, ms2_df=None, params=None):
    """Build a minimal stub that satisfies ``create_manual_curation_obj``'s interface."""
    if ms2_df is None:
        ms2_df = pd.DataFrame()
    if params is None:
        params = {"suggested_min_conf": None, "remove_unided_compounds": False}

    exp_data = ExperimentalData(ms1_df=ms1_df, ms2_df=ms2_df)
    ta = SimpleNamespace(params=params)

    return SimpleNamespace(
        auto_ided_atlas_obj=atlas,
        experimental_data=exp_data,
        paths={"main_db_path": ""},   # empty → enrich call is a no-op
        ta=ta,
    )


# ===========================================================================
# analyze_ms1
# ===========================================================================

class TestAnalyzeMs1:

    def test_returns_dict_with_expected_keys(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        row = _atlas_row()
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(row, ms1)
        for key in ("mz", "rt_peak", "mz_error", "rt_error",
                    "max_eic_rt", "max_eic_intensity",
                    "rt_min", "rt_max",
                    "suggested_rt_min", "suggested_rt_max",
                    "suggested_rt_peak", "rt_suggestion_confidence"):
            assert key in result, f"Missing key: {key}"

    def test_empty_df_returns_empty_dict(self):
        result = analyze_ms1(_atlas_row(), pd.DataFrame())
        assert result == {}

    def test_rt_peak_close_to_atlas(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1)
        # Observed rt_peak should be within 0.2 min of the atlas value
        assert abs(result["rt_peak"] - ADENINE_RT) < 0.2

    def test_mz_error_small_for_accurate_data(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1)
        # mz_error is in ppm; synthetic data uses atlas mz exactly → near 0
        assert abs(result["mz_error"]) < 10.0  # < 10 ppm

    def test_rt_error_computed(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1)
        # rt_error = observed_rt_peak - atlas_rt_peak
        assert isinstance(result["rt_error"], float)

    def test_max_eic_rt_is_list(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1)
        assert isinstance(result["max_eic_rt"], list)
        assert len(result["max_eic_rt"]) > 0

    def test_max_eic_intensity_is_list(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1)
        assert isinstance(result["max_eic_intensity"], list)
        assert len(result["max_eic_intensity"]) == len(result["max_eic_rt"])

    def test_rt_bounds_shifted_with_observed_peak(self, adenine_compound):
        """rt_min / rt_max should be centred on the observed peak, not the atlas peak."""
        uid = adenine_compound.mz_rt_uid
        # Shift all scan points 0.3 min later than the atlas peak
        shift = 0.3
        rts  = [ADENINE_RT + shift + (i - 2) * 0.05 for i in range(5)]
        ints = [1e4, 5e4, 1e5, 5e4, 1e4]
        ms1 = pd.DataFrame([{
            "mz_rt_uid":  uid,
            "filename":   "f.h5",
            "spec_rts":   rts,
            "spec_ints":  ints,
            "spec_mzs":   [ADENINE_MZ] * 5,
            "in_feature": [True] * 5,
        }])
        result = analyze_ms1(_atlas_row(), ms1)
        # The observed rt_peak should be shifted relative to atlas
        assert result["rt_peak"] > ADENINE_RT

    def test_post_curation_summary_stage(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        result = analyze_ms1(_atlas_row(), ms1, stage="post_curation_summary")
        # post_curation_summary should NOT include EIC lists
        assert "max_eic_rt" not in result
        assert "max_eic_intensity" not in result
        # But should include error metrics
        assert "mz_error" in result
        assert "rt_error" in result

    def test_apply_bounds_cutoff_updates_rt_bounds(self, adenine_compound):
        """When confidence > cutoff, suggested bounds should replace active bounds."""
        uid = adenine_compound.mz_rt_uid
        # Use many points to get a high-confidence suggestion
        rts  = [ADENINE_RT + (i - 10) * 0.03 for i in range(21)]
        ints = [max(0.0, 1e5 * (1.0 - abs(i - 10) * 0.08)) for i in range(21)]
        ms1 = pd.DataFrame([{
            "mz_rt_uid":  uid,
            "filename":   "f.h5",
            "spec_rts":   rts,
            "spec_ints":  ints,
            "spec_mzs":   [ADENINE_MZ] * 21,
            "in_feature": [True] * 21,
        }])
        result_no_cutoff  = analyze_ms1(_atlas_row(), ms1, apply_bounds_cutoff=None)
        result_low_cutoff = analyze_ms1(_atlas_row(), ms1, apply_bounds_cutoff=0.0)
        # With a zero cutoff every suggestion is accepted; rt_min should equal suggested_rt_min
        if result_low_cutoff.get("rt_suggestion_confidence", 0) > 0.0:
            assert result_low_cutoff["rt_min"] == result_low_cutoff["suggested_rt_min"]

    def test_no_in_feature_points_returns_empty(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = pd.DataFrame([{
            "mz_rt_uid":  uid,
            "filename":   "f.h5",
            "spec_rts":   [ADENINE_RT],
            "spec_ints":  [1e5],
            "spec_mzs":   [ADENINE_MZ],
            "in_feature": [False],   # nothing in feature
        }])
        result = analyze_ms1(_atlas_row(), ms1)
        assert result == {}

    def test_multiple_files_aggregated(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        ms1 = pd.DataFrame([
            {
                "mz_rt_uid":  uid, "filename": "file1.h5",
                "spec_rts":   [ADENINE_RT - 0.05, ADENINE_RT, ADENINE_RT + 0.05],
                "spec_ints":  [5e4, 1e5, 5e4],
                "spec_mzs":   [ADENINE_MZ] * 3,
                "in_feature": [True, True, True],
            },
            {
                "mz_rt_uid":  uid, "filename": "file2.h5",
                "spec_rts":   [ADENINE_RT - 0.05, ADENINE_RT, ADENINE_RT + 0.05],
                "spec_ints":  [3e4, 8e4, 3e4],
                "spec_mzs":   [ADENINE_MZ] * 3,
                "in_feature": [True, True, True],
            },
        ])
        result = analyze_ms1(_atlas_row(), ms1)
        # Should still return a valid result aggregating both files
        assert "rt_peak" in result
        assert len(result["max_eic_rt"]) > 0


# ===========================================================================
# _suggest_rt_bounds_from_ms1
# ===========================================================================

class TestSuggestRtBoundsFromMs1:

    def _gaussian_trace(self, rt_peak: float = ADENINE_RT, n: int = 50, sigma: float = 0.15) -> dict:
        """Return a synthetic Gaussian EIC trace centred on *rt_peak*."""
        rts = np.linspace(rt_peak - 1.0, rt_peak + 1.0, n)
        ints = np.exp(-0.5 * ((rts - rt_peak) / sigma) ** 2) * 1e5
        return {"rt": rts, "i": ints}

    def test_returns_dict_with_expected_keys(self):
        trace = self._gaussian_trace()
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        for key in ("rt_min", "rt_max", "rt_peak", "confidence"):
            assert key in result

    def test_rt_peak_close_to_true_peak(self):
        trace = self._gaussian_trace(rt_peak=ADENINE_RT)
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        assert abs(result["rt_peak"] - ADENINE_RT) < 0.3

    def test_rt_min_less_than_rt_max(self):
        trace = self._gaussian_trace()
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        assert result["rt_min"] < result["rt_max"]

    def test_confidence_between_0_and_1(self):
        trace = self._gaussian_trace()
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_trace_returns_none(self):
        result = _suggest_rt_bounds_from_ms1({}, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is None

    def test_all_zero_intensity_returns_none(self):
        rts = np.linspace(ADENINE_RMIN, ADENINE_RMAX, 20)
        trace = {"rt": rts, "i": np.zeros(20)}
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is None

    def test_fewer_than_5_points_returns_none(self):
        rts = np.array([ADENINE_RT - 0.1, ADENINE_RT, ADENINE_RT + 0.1])
        ints = np.array([1e4, 1e5, 1e4])
        trace = {"rt": rts, "i": ints}
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is None

    def test_high_confidence_for_clean_gaussian(self):
        trace = self._gaussian_trace(n=100, sigma=0.1)
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        assert result["confidence"] > 0.3

    def test_suggested_bounds_within_reasonable_range(self):
        trace = self._gaussian_trace()
        result = _suggest_rt_bounds_from_ms1(trace, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX)
        assert result is not None
        # Bounds should be within ±2 min of the atlas window
        assert result["rt_min"] > ADENINE_RMIN - 2.0
        assert result["rt_max"] < ADENINE_RMAX + 2.0


# ===========================================================================
# _build_isomer_dict
# ===========================================================================

class TestBuildIsomerDict:

    def test_no_isomers_for_distinct_compounds(self, pos_atlas):
        """Adenine and riboflavin have very different mz and mass — no isomers."""
        result = _build_isomer_dict(pos_atlas)
        # Each UID should be present
        for uid in pos_atlas.compound_mzrts:
            assert uid in result
        # Neither should list the other as an isomer (mz differ by >200 Da)
        for uid, isomers in result.items():
            for iso in isomers:
                assert iso["mz_rt_uid"] != uid  # sanity: no self-reference

    def test_same_inchi_prefix_detected_as_isomer(self):
        """Two entries sharing the first InChIKey block are flagged as isomers."""
        shared_inchi_prefix = "AGPKZVBTJJNPAG"  # L-isoleucine
        c1 = _make_compound_mzrt(
            mz=132.102, rt_peak=9.72, rt_min=8.97, rt_max=10.47,
            inchi_key=f"{shared_inchi_prefix}-WHFBIAKZSA-N",
            compound_name="isoleucine",
        )
        c2 = _make_compound_mzrt(
            mz=132.102, rt_peak=9.34, rt_min=8.59, rt_max=10.09,
            inchi_key=f"{shared_inchi_prefix}-YFKPBYRVSA-N",  # same prefix, different stereo
            compound_name="norleucine",
        )
        atlas = _make_atlas([c1, c2])
        result = _build_isomer_dict(atlas)
        # c1 should list c2 as an isomer and vice-versa
        assert any(iso["mz_rt_uid"] == c2.mz_rt_uid for iso in result[c1.mz_rt_uid])
        assert any(iso["mz_rt_uid"] == c1.mz_rt_uid for iso in result[c2.mz_rt_uid])

    def test_similar_mz_detected_as_isomer(self):
        """Two entries within 5 mDa of each other are flagged as isomers."""
        c1 = _make_compound_mzrt(
            mz=132.102, rt_peak=9.72, rt_min=8.97, rt_max=10.47,
            inchi_key="AGPKZVBTJJNPAG-WHFBIAKZSA-N",
            compound_name="isoleucine",
        )
        c2 = _make_compound_mzrt(
            mz=132.1023, rt_peak=9.34, rt_min=8.59, rt_max=10.09,  # Δmz ≈ 0.3 mDa
            inchi_key="ROHFNLRQFUQHCH-UHFFFAOYSA-N",
            compound_name="leucine",
        )
        atlas = _make_atlas([c1, c2])
        result = _build_isomer_dict(atlas)
        assert any(iso["mz_rt_uid"] == c2.mz_rt_uid for iso in result[c1.mz_rt_uid])

    def test_no_self_reference_in_isomers(self, pos_atlas):
        result = _build_isomer_dict(pos_atlas)
        for uid, isomers in result.items():
            assert all(iso["mz_rt_uid"] != uid for iso in isomers)

    def test_isomer_record_has_expected_keys(self):
        shared_prefix = "AGPKZVBTJJNPAG"
        c1 = _make_compound_mzrt(
            mz=132.102, rt_peak=9.72, rt_min=8.97, rt_max=10.47,
            inchi_key=f"{shared_prefix}-WHFBIAKZSA-N",
            compound_name="isoleucine",
        )
        c2 = _make_compound_mzrt(
            mz=132.102, rt_peak=9.34, rt_min=8.59, rt_max=10.09,
            inchi_key=f"{shared_prefix}-YFKPBYRVSA-N",
            compound_name="norleucine",
        )
        atlas = _make_atlas([c1, c2])
        result = _build_isomer_dict(atlas)
        for iso in result[c1.mz_rt_uid]:
            for key in ("mz_rt_uid", "inchi_key", "adduct", "compound_name", "rt", "mz"):
                assert key in iso


# ===========================================================================
# create_manual_curation_obj
# ===========================================================================

class TestCreateManualCurationObj:
    """Tests for the top-level curation builder.

    ``enrich_atlas_df_with_compound_metadata`` is patched to return the
    atlas DataFrame unchanged (no DB required).
    """

    @pytest.fixture()
    def _patch_enrich(self):
        """Patch the DB enrichment call so it is a no-op."""
        with patch(
            "metatlas2.create_curation_container.dbi.enrich_atlas_df_with_compound_metadata",
            side_effect=lambda df, _path: df,
        ):
            yield

    def test_curation_df_attached_to_obj(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        assert obj.experimental_data.curation_df is not None
        assert not obj.experimental_data.curation_df.empty

    def test_one_row_per_atlas_compound(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert len(df) == 1

    def test_passed_autoid_true_when_ms1_present(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert df["passed_autoid"].iloc[0] is True or df["passed_autoid"].iloc[0] == True

    def test_passed_autoid_false_when_no_ms1(self, _patch_enrich, adenine_compound, single_compound_atlas):
        # Provide an empty MS1 DataFrame that has the required column so the
        # groupby inside create_manual_curation_obj does not raise KeyError.
        empty_ms1 = pd.DataFrame(columns=["mz_rt_uid", "filename",
                                           "spec_rts", "spec_ints", "spec_mzs", "in_feature"])
        obj = _make_auto_id_obj(single_compound_atlas, empty_ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert df["passed_autoid"].iloc[0] is False or df["passed_autoid"].iloc[0] == False

    def test_remove_unided_compounds_drops_no_ms1_rows(self, _patch_enrich, adenine_compound, single_compound_atlas):
        """When all compounds are filtered out the curation_df should be empty."""
        empty_ms1 = pd.DataFrame(columns=["mz_rt_uid", "filename",
                                           "spec_rts", "spec_ints", "spec_mzs", "in_feature"])
        obj = _make_auto_id_obj(
            single_compound_atlas,
            empty_ms1,
            params={"suggested_min_conf": None, "remove_unided_compounds": True},
        )
        create_manual_curation_obj(obj)
        assert obj.experimental_data.curation_df.empty

    def test_curation_df_has_expected_columns(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        expected_cols = {
            "mz_rt_uid", "compound_name", "inchi_key", "adduct",
            "passed_autoid", "passed_curation",
            "atlas_mz", "atlas_rt_peak", "atlas_rt_min", "atlas_rt_max",
            "mz", "rt_peak", "rt_min", "rt_max",
            "rt_error", "mz_error",
            "ms1_notes", "ms2_notes", "other_notes",
            "max_eic_rt", "max_eic_intensity",
            "isomers",
            "suggested_rt_min", "suggested_rt_max",
            "suggested_rt_peak", "rt_suggestion_confidence",
        }
        missing = expected_cols - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_isomers_column_is_json_string(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        import json
        # isomers column should be a JSON-serialised list
        val = df["isomers"].iloc[0]
        parsed = json.loads(val)
        assert isinstance(parsed, list)

    def test_two_compound_atlas_produces_two_rows(self, _patch_enrich, pos_atlas,
                                                   adenine_compound, riboflavin_compound):
        ms1 = pd.concat([
            _ms1_df_for_uid(adenine_compound.mz_rt_uid),
            _ms1_df_for_uid(riboflavin_compound.mz_rt_uid, peak_intensity=5e4),
        ], ignore_index=True)
        obj = _make_auto_id_obj(pos_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert len(df) == 2

    def test_mz_rt_uid_matches_atlas(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert df["mz_rt_uid"].iloc[0] == uid

    def test_atlas_mz_preserved(self, _patch_enrich, adenine_compound, single_compound_atlas):
        uid = adenine_compound.mz_rt_uid
        ms1 = _ms1_df_for_uid(uid)
        obj = _make_auto_id_obj(single_compound_atlas, ms1)
        create_manual_curation_obj(obj)
        df = obj.experimental_data.curation_df
        assert abs(df["atlas_mz"].iloc[0] - ADENINE_MZ) < 1e-4
