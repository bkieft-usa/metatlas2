"""Unit tests for metatlas2/extract_data_from_h5.py.

Every test exercises a *pure calculation* function — no database, no
subprocess, no network.  HDF5 I/O is covered by the ``synthetic_h5_file``
fixture in conftest.py which writes a real (tiny) file to ``tmp_path``.

Tested functions
----------------
* :func:`_expand_atlas_windows`
* :func:`_interval_join_mz`
* :func:`_join_ms1_to_atlas`
* :func:`_join_ms2_to_atlas`
* :func:`_widen_one_file_ms1`
* :func:`_widen_one_file_ms2`
* :func:`_sort_frags`
* :func:`_sort_ms1_lists_by_rts`
* :func:`_filter_ms1_points`
* :func:`_filter_ms2_points`
* :func:`_merge_wide_ms1` / :func:`_merge_wide_ms2`
* :func:`_join_metadata`
* :func:`_load_h5_table`  (round-trip through the synthetic HDF5 file)
* :func:`_process_one_file` (integration of load + join for one file)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metatlas2.extract_data_from_h5 import (
    _expand_atlas_windows,
    _filter_ms1_points,
    _filter_ms2_points,
    _interval_join_mz,
    _join_metadata,
    _join_ms1_to_atlas,
    _join_ms2_to_atlas,
    _load_h5_table,
    _merge_wide_ms1,
    _merge_wide_ms2,
    _process_one_file,
    _sort_frags,
    _sort_ms1_lists_by_rts,
    _widen_one_file_ms1,
    _widen_one_file_ms2,
)

# Pull constants from conftest so tests stay in sync with fixture data.
from conftest import ADENINE_MZ, ADENINE_RT, ADENINE_RMIN, ADENINE_RMAX
from conftest import RIBOFLAVIN_MZ, RIBOFLAVIN_RT, RIBOFLAVIN_RMIN, RIBOFLAVIN_RMAX


# ===========================================================================
# _expand_atlas_windows
# ===========================================================================

class TestExpandAtlasWindows:

    def test_polarity_column_added(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        out = _expand_atlas_windows(df, extra_time=0.0, ms1_mz_tolerance_ppm=5.0, polarity="positive")
        assert "polarity" in out.columns
        assert (out["polarity"] == "positive").all()

    def test_mz_bounds_computed_correctly(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        ppm = 5.0
        out = _expand_atlas_windows(df, extra_time=0.0, ms1_mz_tolerance_ppm=ppm, polarity="positive")
        for _, row in out.iterrows():
            expected_delta = row["mz"] * ppm * 1e-6
            assert abs(row["mz_min"] - (row["mz"] - expected_delta)) < 1e-4
            assert abs(row["mz_max"] - (row["mz"] + expected_delta)) < 1e-4

    def test_rt_padding_applied(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        extra = 0.5
        out = _expand_atlas_windows(df, extra_time=extra, ms1_mz_tolerance_ppm=5.0, polarity="positive")
        for _, row in out.iterrows():
            assert abs(row["rt_min_pad"] - (row["rt_min"] - extra)) < 1e-6
            assert abs(row["rt_max_pad"] - (row["rt_max"] + extra)) < 1e-6

    def test_zero_extra_time_leaves_rt_unchanged(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        out = _expand_atlas_windows(df, extra_time=0.0, ms1_mz_tolerance_ppm=5.0, polarity="positive")
        for _, row in out.iterrows():
            assert abs(row["rt_min_pad"] - row["rt_min"]) < 1e-6
            assert abs(row["rt_max_pad"] - row["rt_max"]) < 1e-6

    def test_original_dataframe_not_mutated(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        original_cols = set(df.columns)
        _expand_atlas_windows(df, extra_time=0.1, ms1_mz_tolerance_ppm=5.0, polarity="positive")
        assert set(df.columns) == original_cols


# ===========================================================================
# _interval_join_mz
# ===========================================================================

class TestIntervalJoinMz:

    def test_exact_match(self):
        query_mz   = np.array([136.062])
        atlas_min  = np.array([136.061])
        atlas_max  = np.array([136.063])
        q_idx, a_idx = _interval_join_mz(query_mz, atlas_min, atlas_max)
        assert len(q_idx) == 1
        assert q_idx[0] == 0
        assert a_idx[0] == 0

    def test_no_match_outside_window(self):
        query_mz  = np.array([200.0])
        atlas_min = np.array([136.061])
        atlas_max = np.array([136.063])
        q_idx, a_idx = _interval_join_mz(query_mz, atlas_min, atlas_max)
        assert len(q_idx) == 0
        assert len(a_idx) == 0

    def test_multiple_queries_multiple_atlas(self):
        # Two query points, two atlas windows; each query matches exactly one window.
        query_mz  = np.array([136.062, 377.146])
        atlas_min = np.array([136.060, 377.140])
        atlas_max = np.array([136.064, 377.152])
        q_idx, a_idx = _interval_join_mz(query_mz, atlas_min, atlas_max)
        assert len(q_idx) == 2
        # Each query should match its own atlas window
        pairs = set(zip(q_idx.tolist(), a_idx.tolist()))
        assert (0, 0) in pairs
        assert (1, 1) in pairs

    def test_one_query_matches_two_atlas_windows(self):
        # Overlapping atlas windows — query should match both.
        query_mz  = np.array([136.062])
        atlas_min = np.array([136.060, 136.061])
        atlas_max = np.array([136.064, 136.063])
        q_idx, a_idx = _interval_join_mz(query_mz, atlas_min, atlas_max)
        assert len(q_idx) == 2

    def test_nan_query_ignored(self):
        query_mz  = np.array([np.nan, 136.062])
        atlas_min = np.array([136.060])
        atlas_max = np.array([136.064])
        q_idx, a_idx = _interval_join_mz(query_mz, atlas_min, atlas_max)
        # Only the non-NaN query should match
        assert len(q_idx) == 1

    def test_empty_inputs_return_empty(self):
        q_idx, a_idx = _interval_join_mz(np.array([]), np.array([136.0]), np.array([137.0]))
        assert len(q_idx) == 0
        q_idx, a_idx = _interval_join_mz(np.array([136.0]), np.array([]), np.array([]))
        assert len(q_idx) == 0


# ===========================================================================
# _join_ms1_to_atlas
# ===========================================================================

class TestJoinMs1ToAtlas:

    def _make_atlas_expanded(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        return _expand_atlas_windows(df, extra_time=0.0, ms1_mz_tolerance_ppm=5.0, polarity="positive")

    def test_in_window_point_matched(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms1 = pd.DataFrame([{"mz": ADENINE_MZ, "rt": ADENINE_RT, "i": 1e5}])
        result = _join_ms1_to_atlas(ms1, atlas_exp, only_in_feature=False)
        assert len(result) == 1
        assert result["in_feature"].iloc[0]

    def test_out_of_rt_window_not_in_feature(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        # mz matches adenine but rt is far outside the window
        ms1 = pd.DataFrame([{"mz": ADENINE_MZ, "rt": 99.0, "i": 1e5}])
        result = _join_ms1_to_atlas(ms1, atlas_exp, only_in_feature=False)
        assert len(result) == 1
        assert not result["in_feature"].iloc[0]

    def test_only_in_feature_drops_out_of_window(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms1 = pd.DataFrame([
            {"mz": ADENINE_MZ, "rt": ADENINE_RT, "i": 1e5},   # in feature
            {"mz": ADENINE_MZ, "rt": 99.0,        "i": 1e5},   # out of feature
        ])
        result = _join_ms1_to_atlas(ms1, atlas_exp, only_in_feature=True)
        assert len(result) == 1
        assert result["in_feature"].iloc[0]

    def test_empty_ms1_returns_empty(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        result = _join_ms1_to_atlas(pd.DataFrame(), atlas_exp, only_in_feature=False)
        assert result.empty

    def test_empty_atlas_returns_empty(self, pos_atlas):
        ms1 = pd.DataFrame([{"mz": ADENINE_MZ, "rt": ADENINE_RT, "i": 1e5}])
        result = _join_ms1_to_atlas(ms1, pd.DataFrame(), only_in_feature=False)
        assert result.empty

    def test_mz_rt_uid_column_present(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms1 = pd.DataFrame([{"mz": ADENINE_MZ, "rt": ADENINE_RT, "i": 1e5}])
        result = _join_ms1_to_atlas(ms1, atlas_exp, only_in_feature=False)
        assert "mz_rt_uid" in result.columns


# ===========================================================================
# _join_ms2_to_atlas
# ===========================================================================

class TestJoinMs2ToAtlas:

    def _make_atlas_expanded(self, pos_atlas):
        df = pos_atlas.to_dataframe()
        return _expand_atlas_windows(df, extra_time=0.0, ms1_mz_tolerance_ppm=20.0, polarity="positive")

    def _ms2_row(self, precursor_mz, rt, mz=94.04, intensity=1e4):
        return {
            "mz": mz, "i": intensity, "rt": rt,
            "precursor_MZ": precursor_mz,
            "precursor_intensity": 1e5,
            "collision_energy": 35.0,
        }

    def test_in_window_scan_matched(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms2 = pd.DataFrame([self._ms2_row(ADENINE_MZ, ADENINE_RT)])
        result = _join_ms2_to_atlas(ms2, atlas_exp, only_in_feature=False)
        assert len(result) == 1
        assert result["in_feature"].iloc[0]

    def test_zero_precursor_mz_dropped(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms2 = pd.DataFrame([self._ms2_row(0.0, ADENINE_RT)])
        result = _join_ms2_to_atlas(ms2, atlas_exp, only_in_feature=False)
        assert result.empty

    def test_nan_precursor_mz_dropped(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms2 = pd.DataFrame([self._ms2_row(float("nan"), ADENINE_RT)])
        result = _join_ms2_to_atlas(ms2, atlas_exp, only_in_feature=False)
        assert result.empty

    def test_only_in_feature_filters_out_of_rt(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        ms2 = pd.DataFrame([
            self._ms2_row(ADENINE_MZ, ADENINE_RT),   # in feature
            self._ms2_row(ADENINE_MZ, 99.0),          # out of feature
        ])
        result = _join_ms2_to_atlas(ms2, atlas_exp, only_in_feature=True)
        assert len(result) == 1

    def test_empty_ms2_returns_empty(self, pos_atlas):
        atlas_exp = self._make_atlas_expanded(pos_atlas)
        result = _join_ms2_to_atlas(pd.DataFrame(), atlas_exp, only_in_feature=False)
        assert result.empty


# ===========================================================================
# _widen_one_file_ms1
# ===========================================================================

class TestWidenOneFileMs1:

    def _long_df(self, uid: str) -> pd.DataFrame:
        return pd.DataFrame([
            {"mz": ADENINE_MZ, "rt": ADENINE_RT - 0.05, "i": 5e4, "in_feature": True,  "mz_rt_uid": uid, "filename": "f.h5"},
            {"mz": ADENINE_MZ, "rt": ADENINE_RT,         "i": 1e5, "in_feature": True,  "mz_rt_uid": uid, "filename": "f.h5"},
            {"mz": ADENINE_MZ, "rt": ADENINE_RT + 0.05, "i": 6e4, "in_feature": True,  "mz_rt_uid": uid, "filename": "f.h5"},
        ])

    def test_one_row_per_uid(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        wide = _widen_one_file_ms1(self._long_df(uid))
        assert len(wide) == 1
        assert wide["mz_rt_uid"].iloc[0] == uid

    def test_list_columns_have_correct_length(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        wide = _widen_one_file_ms1(self._long_df(uid))
        assert len(wide["spec_rts"].iloc[0]) == 3
        assert len(wide["spec_ints"].iloc[0]) == 3
        assert len(wide["spec_mzs"].iloc[0]) == 3
        assert len(wide["in_feature"].iloc[0]) == 3

    def test_filename_preserved(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        wide = _widen_one_file_ms1(self._long_df(uid))
        assert wide["filename"].iloc[0] == "f.h5"

    def test_empty_input_returns_empty(self):
        assert _widen_one_file_ms1(pd.DataFrame()).empty


# ===========================================================================
# _widen_one_file_ms2
# ===========================================================================

class TestWidenOneFileMs2:

    def _long_df(self, uid: str) -> pd.DataFrame:
        return pd.DataFrame([
            {"mz": 94.04,  "i": 1e4, "rt": ADENINE_RT, "precursor_MZ": ADENINE_MZ,
             "precursor_intensity": 1e5, "collision_energy": 35.0,
             "in_feature": True, "mz_rt_uid": uid, "filename": "f.h5"},
            {"mz": 119.04, "i": 5e4, "rt": ADENINE_RT, "precursor_MZ": ADENINE_MZ,
             "precursor_intensity": 1e5, "collision_energy": 35.0,
             "in_feature": True, "mz_rt_uid": uid, "filename": "f.h5"},
        ])

    def test_one_row_per_scan(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        wide = _widen_one_file_ms2(self._long_df(uid))
        assert len(wide) == 1  # both fragments belong to the same scan (same rt)

    def test_frag_lists_aggregated(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        wide = _widen_one_file_ms2(self._long_df(uid))
        assert len(wide["frag_mzs"].iloc[0]) == 2
        assert len(wide["frag_ints"].iloc[0]) == 2

    def test_frags_sorted_by_mz(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        # Provide fragments in reverse mz order
        df = pd.DataFrame([
            {"mz": 200.0, "i": 1e4, "rt": ADENINE_RT, "precursor_MZ": ADENINE_MZ,
             "precursor_intensity": 1e5, "collision_energy": 35.0,
             "in_feature": True, "mz_rt_uid": uid, "filename": "f.h5"},
            {"mz": 100.0, "i": 5e4, "rt": ADENINE_RT, "precursor_MZ": ADENINE_MZ,
             "precursor_intensity": 1e5, "collision_energy": 35.0,
             "in_feature": True, "mz_rt_uid": uid, "filename": "f.h5"},
        ])
        wide = _widen_one_file_ms2(df)
        mzs = wide["frag_mzs"].iloc[0]
        assert mzs[0] < mzs[1]

    def test_empty_input_returns_empty(self):
        assert _widen_one_file_ms2(pd.DataFrame()).empty


# ===========================================================================
# _sort_frags
# ===========================================================================

class TestSortFrags:

    def test_unsorted_frags_sorted(self):
        df = pd.DataFrame([{"frag_mzs": [300.0, 100.0, 200.0], "frag_ints": [3.0, 1.0, 2.0]}])
        out = _sort_frags(df)
        assert out["frag_mzs"].iloc[0] == [100.0, 200.0, 300.0]
        assert out["frag_ints"].iloc[0] == [1.0, 2.0, 3.0]

    def test_already_sorted_unchanged(self):
        df = pd.DataFrame([{"frag_mzs": [100.0, 200.0, 300.0], "frag_ints": [1.0, 2.0, 3.0]}])
        out = _sort_frags(df)
        assert out["frag_mzs"].iloc[0] == [100.0, 200.0, 300.0]

    def test_missing_columns_returns_unchanged(self):
        df = pd.DataFrame([{"other_col": 1}])
        out = _sort_frags(df)
        assert "frag_mzs" not in out.columns


# ===========================================================================
# _sort_ms1_lists_by_rts
# ===========================================================================

class TestSortMs1ListsByRts:

    def test_unsorted_rts_sorted(self):
        df = pd.DataFrame([{
            "spec_rts":   [3.0, 1.0, 2.0],
            "spec_ints":  [30.0, 10.0, 20.0],
            "spec_mzs":   [136.0, 136.0, 136.0],
            "in_feature": [True, False, True],
        }])
        out = _sort_ms1_lists_by_rts(df)
        assert out["spec_rts"].iloc[0] == [1.0, 2.0, 3.0]
        assert out["spec_ints"].iloc[0] == [10.0, 20.0, 30.0]
        assert out["in_feature"].iloc[0] == [False, True, True]

    def test_already_sorted_unchanged(self):
        df = pd.DataFrame([{
            "spec_rts":   [1.0, 2.0, 3.0],
            "spec_ints":  [10.0, 20.0, 30.0],
            "spec_mzs":   [136.0, 136.0, 136.0],
            "in_feature": [True, True, False],
        }])
        out = _sort_ms1_lists_by_rts(df)
        assert out["spec_rts"].iloc[0] == [1.0, 2.0, 3.0]


# ===========================================================================
# _filter_ms1_points
# ===========================================================================

class TestFilterMs1Points:

    def _make_ms1(self, uid: str, in_feature: list[bool], ints: list[float]) -> pd.DataFrame:
        return pd.DataFrame([{
            "mz_rt_uid":  uid,
            "filename":   "f.h5",
            "spec_rts":   list(range(len(ints))),
            "spec_ints":  ints,
            "spec_mzs":   [136.0] * len(ints),
            "in_feature": in_feature,
        }])

    def test_no_filters_returns_all(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms1(uid, [True, True, True], [1e4, 1e5, 5e4])
        out = _filter_ms1_points(df, min_pts=None, min_int=None)
        assert len(out) == 1

    def test_min_pts_filter_removes_row(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        # Only 1 in-feature point; require 3 → should be dropped
        df = self._make_ms1(uid, [True, False, False], [1e5, 0.0, 0.0])
        out = _filter_ms1_points(df, min_pts=3, min_int=None)
        assert out.empty

    def test_min_pts_filter_keeps_row(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms1(uid, [True, True, True], [1e5, 5e4, 3e4])
        out = _filter_ms1_points(df, min_pts=3, min_int=None)
        assert len(out) == 1

    def test_min_int_filter_removes_row(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        # Max in-feature intensity is 1e3; require 1e5 → dropped
        df = self._make_ms1(uid, [True, True], [1e3, 5e2])
        out = _filter_ms1_points(df, min_pts=None, min_int=1e5)
        assert out.empty

    def test_min_int_filter_keeps_row(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms1(uid, [True, True], [1e5, 5e4])
        out = _filter_ms1_points(df, min_pts=None, min_int=1e4)
        assert len(out) == 1

    def test_empty_df_returns_empty(self):
        out = _filter_ms1_points(pd.DataFrame(), min_pts=1, min_int=1e4)
        assert out.empty

    def test_row_with_no_in_feature_points_dropped(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms1(uid, [False, False], [1e5, 5e4])
        out = _filter_ms1_points(df, min_pts=1, min_int=None)
        assert out.empty


# ===========================================================================
# _filter_ms2_points
# ===========================================================================

class TestFilterMs2Points:

    def _make_ms2(self, uid: str, in_feature: bool, precursor_intensity: float) -> pd.DataFrame:
        return pd.DataFrame([{
            "mz_rt_uid":           uid,
            "filename":            "f.h5",
            "scan_rt":             ADENINE_RT,
            "precursor_MZ":        ADENINE_MZ,
            "precursor_intensity": precursor_intensity,
            "frag_mzs":            [94.04],
            "frag_ints":           [1e4],
            "in_feature":          in_feature,
        }])

    def test_no_filters_returns_all(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms2(uid, True, 1e5)
        out = _filter_ms2_points(df, pd.DataFrame(), min_scans=None, min_int=None)
        assert len(out) == 1

    def test_min_scans_removes_insufficient_group(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        # Only 1 in-feature scan; require 2 → dropped
        df = self._make_ms2(uid, True, 1e5)
        out = _filter_ms2_points(df, pd.DataFrame(), min_scans=2, min_int=None)
        assert out.empty

    def test_min_int_removes_low_intensity(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        df = self._make_ms2(uid, True, 1e3)
        out = _filter_ms2_points(df, pd.DataFrame(), min_scans=None, min_int=1e5)
        assert out.empty

    def test_out_of_feature_scan_dropped_after_group_filter(self, adenine_compound):
        uid = adenine_compound.mz_rt_uid
        # One in-feature scan (passes min_scans=1) and one out-of-feature scan
        df = pd.DataFrame([
            {"mz_rt_uid": uid, "filename": "f.h5", "scan_rt": ADENINE_RT,
             "precursor_MZ": ADENINE_MZ, "precursor_intensity": 1e5,
             "frag_mzs": [94.04], "frag_ints": [1e4], "in_feature": True},
            {"mz_rt_uid": uid, "filename": "f.h5", "scan_rt": ADENINE_RT + 5.0,
             "precursor_MZ": ADENINE_MZ, "precursor_intensity": 1e5,
             "frag_mzs": [94.04], "frag_ints": [1e4], "in_feature": False},
        ])
        out = _filter_ms2_points(df, pd.DataFrame(), min_scans=1, min_int=None)
        # Only the in-feature scan should survive
        assert len(out) == 1
        assert bool(out["in_feature"].iloc[0])

    def test_empty_df_returns_empty(self):
        out = _filter_ms2_points(pd.DataFrame(), pd.DataFrame(), min_scans=1, min_int=None)
        assert out.empty


# ===========================================================================
# _merge_wide_ms1 / _merge_wide_ms2
# ===========================================================================

class TestMergeWide:

    def test_merge_ms1_empty_accumulator(self, ms1_wide_df):
        result = _merge_wide_ms1(pd.DataFrame(), ms1_wide_df)
        assert len(result) == len(ms1_wide_df)

    def test_merge_ms1_accumulates_rows(self, ms1_wide_df):
        chunk2 = ms1_wide_df.copy()
        chunk2["filename"] = "second_file.h5"
        acc = _merge_wide_ms1(ms1_wide_df, chunk2)
        assert len(acc) == 2

    def test_merge_ms2_empty_accumulator(self, ms2_wide_df):
        result = _merge_wide_ms2(pd.DataFrame(), ms2_wide_df)
        assert len(result) == len(ms2_wide_df)

    def test_merge_ms2_accumulates_rows(self, ms2_wide_df):
        chunk2 = ms2_wide_df.copy()
        chunk2["filename"] = "second_file.h5"
        acc = _merge_wide_ms2(ms2_wide_df, chunk2)
        assert len(acc) == 2


# ===========================================================================
# _join_metadata
# ===========================================================================

class TestJoinMetadata:

    def test_inchi_key_added_to_ms1(self, pos_atlas, ms1_wide_df):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1_out, _ = _join_metadata(ms1_wide_df, pd.DataFrame(), atlas_exp)
        assert "inchi_key" in ms1_out.columns

    def test_adduct_added_to_ms1(self, pos_atlas, ms1_wide_df):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1_out, _ = _join_metadata(ms1_wide_df, pd.DataFrame(), atlas_exp)
        assert "adduct" in ms1_out.columns

    def test_empty_ms1_stays_empty(self, pos_atlas):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1_out, _ = _join_metadata(pd.DataFrame(), pd.DataFrame(), atlas_exp)
        assert ms1_out.empty


# ===========================================================================
# _load_h5_table  (round-trip through the synthetic HDF5 file)
# ===========================================================================

class TestLoadH5Table:

    def test_ms1_pos_loaded(self, synthetic_h5_file):
        df = _load_h5_table(str(synthetic_h5_file), "ms1_pos")
        assert not df.empty
        assert "mz" in df.columns
        assert "rt" in df.columns
        assert "i" in df.columns

    def test_ms2_pos_loaded(self, synthetic_h5_file):
        df = _load_h5_table(str(synthetic_h5_file), "ms2_pos")
        assert not df.empty
        assert "precursor_MZ" in df.columns

    def test_mz_bounds_prefilter_works(self, synthetic_h5_file):
        # Request only adenine's mz range — riboflavin rows should be excluded.
        ppm = 5.0
        mz_min = ADENINE_MZ * (1 - ppm * 1e-6)
        mz_max = ADENINE_MZ * (1 + ppm * 1e-6)
        df = _load_h5_table(str(synthetic_h5_file), "ms1_pos", mz_bounds=(mz_min, mz_max))
        assert not df.empty
        # All returned rows must be within the requested mz window
        assert (df["mz"] >= mz_min).all()
        assert (df["mz"] <= mz_max).all()
        # Riboflavin mz should not appear
        assert not (df["mz"] > 300.0).any()

    def test_missing_key_returns_empty(self, synthetic_h5_file):
        df = _load_h5_table(str(synthetic_h5_file), "ms1_neg")
        assert df.empty

    def test_missing_file_returns_empty(self, tmp_path):
        df = _load_h5_table(str(tmp_path / "nonexistent.h5"), "ms1_pos")
        assert df.empty


# ===========================================================================
# _process_one_file  (integration: load + join for one LCMSRun)
# ===========================================================================

class TestProcessOneFile:

    def test_ms1_extracted_for_both_compounds(self, lcmsrun, pos_atlas):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1, ms2 = _process_one_file(lcmsrun, atlas_exp, only_in_feature=False)
        assert not ms1.empty
        # Both adenine and riboflavin should appear
        uids_found = set(ms1["mz_rt_uid"].unique())
        atlas_uids = set(pos_atlas.compound_mzrts.keys())
        assert uids_found == atlas_uids

    def test_ms2_extracted(self, lcmsrun, pos_atlas):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=20.0, polarity="positive"
        )
        ms1, ms2 = _process_one_file(lcmsrun, atlas_exp, only_in_feature=False)
        assert not ms2.empty

    def test_filename_column_added(self, lcmsrun, pos_atlas):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1, ms2 = _process_one_file(lcmsrun, atlas_exp, only_in_feature=False)
        assert "filename" in ms1.columns
        assert ms1["filename"].iloc[0] == lcmsrun.filename

    def test_only_in_feature_restricts_ms1(self, lcmsrun, pos_atlas):
        atlas_exp = _expand_atlas_windows(
            pos_atlas.to_dataframe(), extra_time=0.0,
            ms1_mz_tolerance_ppm=5.0, polarity="positive"
        )
        ms1_all, _ = _process_one_file(lcmsrun, atlas_exp, only_in_feature=False)
        ms1_feat, _ = _process_one_file(lcmsrun, atlas_exp, only_in_feature=True)
        # With only_in_feature=True we should get ≤ the number of rows from False
        assert len(ms1_feat) <= len(ms1_all)
