"""Unit tests for metatlas2/targeted/ms2_hit_detection.py.

All tests exercise pure calculation functions — no database, no filesystem,
no subprocess, and no network calls.  The ``matchms`` library is used directly
to build reference :class:`~matchms.Spectrum` objects so the scoring path is
exercised end-to-end without mocking.

Tested functions
----------------
* :func:`_no_match_alignment`
* :func:`_align_spectra_for_plotting`
* :func:`_assign_hits`
* :func:`_filter_out_ms2_data`
* :func:`_process_compound_batch`
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from matchms.similarity import CosineHungarian

from metatlas2.targeted.ms2_hit_detection import (
    _add_precursor_to_ref,
    _align_spectra_for_plotting,
    _assign_hits,
    _filter_out_ms2_data,
    _no_match_alignment,
    _process_compound_batch,
    find_ms2_hits,
)

from conftest import ADENINE_MZ, ADENINE_RT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_spectrum(
    precursor_mz: float,
    mz: list[float],
    intensities: list[float],
    inchi_key: str = "GFFGJBXGBJISGV-UHFFFAOYSA-N",
    name: str = "adenine",
    database: str = "metatlas",
    ref_id: str = "ref001",
) -> Spectrum:
    """Build a minimal matchms Spectrum suitable for use as a reference."""
    return Spectrum(
        mz=np.array(mz, dtype=np.float32),
        intensities=np.array(intensities, dtype=np.float32),
        metadata={
            "precursor_mz": precursor_mz,
            "inchi_key": inchi_key,
            "name": name,
            "database": database,
            "id": ref_id,
        },
    )


def _scan_dict(
    frag_mzs: list[float],
    frag_ints: list[float],
    precursor_mz: float = ADENINE_MZ,
    precursor_intensity: float = 1e5,
) -> dict:
    return {
        "frag_mzs": frag_mzs,
        "frag_ints": frag_ints,
        "precursor_MZ": precursor_mz,
        "precursor_intensity": precursor_intensity,
    }


def _ms2_df(uid: str, filename: str = "f.h5", n_scans: int = 1,
            in_feature: bool = True, hits=None) -> pd.DataFrame:
    """Build a minimal wide-format MS2 DataFrame.

    ``hits`` is a list of per-scan hit lists.  Element ``hits[i]`` is the list
    of hit dicts for scan ``i``.  When ``hits`` is ``None`` every scan gets an
    empty hits list.  When ``hits`` is a plain list (not a list-of-lists, e.g.
    ``[]``) every scan also gets an empty hits list.
    """
    rows = []
    for i in range(n_scans):
        if hits is None:
            scan_hits = []
        elif isinstance(hits, list) and i < len(hits):
            scan_hits = hits[i]
        else:
            scan_hits = []
        rows.append({
            "mz_rt_uid":           uid,
            "filename":            filename,
            "inchi_key":           "GFFGJBXGBJISGV-UHFFFAOYSA-N",
            "adduct":              "[M+H]+",
            "scan_rt":             ADENINE_RT + i * 0.1,
            "precursor_MZ":        ADENINE_MZ,
            "precursor_intensity": 1e5,
            "frag_mzs":            [94.04, 119.04, 136.06],
            "frag_ints":           [1e4, 5e4, 2e5],
            "collision_energy":    35.0,
            "in_feature":          in_feature,
            "hits":                scan_hits,
        })
    return pd.DataFrame(rows)


def _ms1_df(uid: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "mz_rt_uid":  uid,
        "filename":   "f.h5",
        "inchi_key":  "GFFGJBXGBJISGV-UHFFFAOYSA-N",
        "adduct":     "[M+H]+",
        "spec_rts":   [ADENINE_RT],
        "spec_ints":  [1e5],
        "spec_mzs":   [ADENINE_MZ],
        "in_feature": [True],
    }])


def test_keyless_ms2_scans_skip_reference_lookup(monkeypatch):
    import metatlas2.targeted.ms2_hit_detection as ms2_detection

    def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("Keyless scans must not look up MS/MS references.")

    monkeypatch.setattr(ms2_detection.ldt, "load_msms_refs_from_db", unexpected_lookup)
    monkeypatch.setattr(ms2_detection.ldt, "load_msms_refs_file", unexpected_lookup)
    monkeypatch.setattr(ms2_detection, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(ms2_detection, "get_max_workers", lambda _workers: 1)

    dataset = SimpleNamespace(
        ms2_df=pd.DataFrame([{
            "mz_rt_uid": "keyless-mzrt",
            "filename": "sample.h5",
            "inchi_key": "",
            "adduct": "[M+H]+",
            "scan_rt": 1.0,
            "frag_mzs": [50.0],
            "frag_ints": [100.0],
            "precursor_MZ": 100.0,
            "precursor_intensity": 1000.0,
            "collision_energy": 20.0,
            "in_feature": True,
            "hits": [],
        }]),
        ms1_df=pd.DataFrame(),
    )
    auto_id_obj = SimpleNamespace(
        experimental_data=dataset,
        ta=SimpleNamespace(
            polarity="POS",
            params={
                "ms2_min_score": 0,
                "ms2_min_matching_frags": 0,
                "keep_top_scan_per_compound_file": False,
            },
        ),
        polarity="POS",
        config=SimpleNamespace(
            msms_refs_path=None,
            msms_refs_db_filter=None,
            max_workers=1,
        ),
        paths={"main_db_path": None},
    )

    find_ms2_hits(auto_id_obj)

    assert len(dataset.ms2_df) == 1
    assert dataset.ms2_df.loc[0, "inchi_key"] == ""
    assert dataset.ms2_df.loc[0, "hits"] == []


# ===========================================================================
# _no_match_alignment
# ===========================================================================

class TestNoMatchAlignment:

    def test_returns_dict_with_expected_keys(self):
        q_mz = np.array([94.04, 119.04], dtype=np.float32)
        q_int = np.array([1e4, 5e4], dtype=np.float32)
        r_mz = np.array([94.04, 136.06], dtype=np.float32)
        r_int = np.array([2e4, 1e5], dtype=np.float32)
        result = _no_match_alignment(q_mz, q_int, r_mz, r_int)
        for key in ("matched_fragments", "fragment_colors", "query_aligned", "ref_aligned", "num_matched"):
            assert key in result

    def test_matched_fragments_empty(self):
        q_mz = np.array([94.04], dtype=np.float32)
        q_int = np.array([1e4], dtype=np.float32)
        r_mz = np.array([136.06], dtype=np.float32)
        r_int = np.array([1e5], dtype=np.float32)
        result = _no_match_alignment(q_mz, q_int, r_mz, r_int)
        assert result["matched_fragments"] == []
        assert result["num_matched"] == 0

    def test_all_fragment_colors_are_red(self):
        q_mz = np.array([94.04, 119.04], dtype=np.float32)
        q_int = np.array([1e4, 5e4], dtype=np.float32)
        r_mz = np.array([136.06], dtype=np.float32)
        r_int = np.array([1e5], dtype=np.float32)
        result = _no_match_alignment(q_mz, q_int, r_mz, r_int)
        assert all(c == "red" for c in result["fragment_colors"])
        assert len(result["fragment_colors"]) == len(q_mz)

    def test_query_aligned_contains_all_query_peaks(self):
        q_mz = np.array([94.04, 119.04], dtype=np.float32)
        q_int = np.array([1e4, 5e4], dtype=np.float32)
        r_mz = np.array([136.06], dtype=np.float32)
        r_int = np.array([1e5], dtype=np.float32)
        result = _no_match_alignment(q_mz, q_int, r_mz, r_int)
        assert len(result["query_aligned"][0]) == len(q_mz)

    def test_ref_aligned_contains_all_ref_peaks(self):
        q_mz = np.array([94.04], dtype=np.float32)
        q_int = np.array([1e4], dtype=np.float32)
        r_mz = np.array([136.06, 200.0], dtype=np.float32)
        r_int = np.array([1e5, 5e4], dtype=np.float32)
        result = _no_match_alignment(q_mz, q_int, r_mz, r_int)
        assert len(result["ref_aligned"][0]) == len(r_mz)


# ===========================================================================
# _align_spectra_for_plotting
# ===========================================================================

class TestAlignSpectraForPlotting:

    def _adenine_query(self):
        return (
            np.array([94.04, 119.04, 136.06], dtype=np.float32),
            np.array([1e4,   5e4,    2e5],    dtype=np.float32),
        )

    def _adenine_ref(self):
        return (
            np.array([94.04, 119.04, 136.06], dtype=np.float32),
            np.array([2e4,   6e4,    3e5],    dtype=np.float32),
        )

    def test_returns_dict_with_expected_keys(self):
        q_mz, q_int = self._adenine_query()
        r_mz, r_int = self._adenine_ref()
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        for key in ("matched_fragments", "fragment_colors", "query_aligned", "ref_aligned", "num_matched"):
            assert key in result

    def test_identical_spectra_all_matched(self):
        q_mz, q_int = self._adenine_query()
        r_mz, r_int = self._adenine_ref()
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert result["num_matched"] == 3
        assert all(c == "green" for c in result["fragment_colors"] if c != "red")

    def test_no_overlap_returns_no_match(self):
        q_mz = np.array([50.0, 60.0], dtype=np.float32)
        q_int = np.array([1e4, 2e4], dtype=np.float32)
        r_mz = np.array([200.0, 300.0], dtype=np.float32)
        r_int = np.array([1e5, 5e4], dtype=np.float32)
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert result["num_matched"] == 0
        assert result["matched_fragments"] == []

    def test_partial_overlap_correct_count(self):
        # Query has 3 peaks; only 1 matches the reference
        q_mz = np.array([94.04, 119.04, 136.06], dtype=np.float32)
        q_int = np.array([1e4,   5e4,    2e5],    dtype=np.float32)
        r_mz = np.array([136.06, 500.0], dtype=np.float32)
        r_int = np.array([3e5,   1e4],   dtype=np.float32)
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert result["num_matched"] == 1

    def test_tight_tolerance_misses_slightly_shifted_peak(self):
        q_mz = np.array([136.06], dtype=np.float32)
        q_int = np.array([2e5],   dtype=np.float32)
        r_mz = np.array([136.12], dtype=np.float32)   # 0.06 Da away
        r_int = np.array([3e5],   dtype=np.float32)
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert result["num_matched"] == 0

    def test_loose_tolerance_catches_shifted_peak(self):
        q_mz = np.array([136.06], dtype=np.float32)
        q_int = np.array([2e5],   dtype=np.float32)
        r_mz = np.array([136.12], dtype=np.float32)   # 0.06 Da away
        r_int = np.array([3e5],   dtype=np.float32)
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.1)
        assert result["num_matched"] == 1

    def test_matched_fragments_are_query_mz_values(self):
        q_mz, q_int = self._adenine_query()
        r_mz, r_int = self._adenine_ref()
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        for mz in result["matched_fragments"]:
            assert any(abs(mz - qm) < 0.1 for qm in q_mz)

    def test_query_aligned_and_ref_aligned_same_length(self):
        q_mz, q_int = self._adenine_query()
        r_mz, r_int = self._adenine_ref()
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert len(result["query_aligned"][0]) == len(result["ref_aligned"][0])

    def test_fragment_colors_length_matches_aligned_arrays(self):
        q_mz, q_int = self._adenine_query()
        r_mz, r_int = self._adenine_ref()
        result = _align_spectra_for_plotting(q_mz, q_int, r_mz, r_int, frag_mz_tolerance=0.05)
        assert len(result["fragment_colors"]) == len(result["query_aligned"][0])


# ===========================================================================
# _assign_hits
# ===========================================================================

class TestAssignHits:

    def test_hits_assigned_to_correct_rows(self):
        uid = "uid-001"
        df = _ms2_df(uid, n_scans=2)
        fake_hit = [{"score": 0.9, "ref_name": "adenine"}]
        results_map = {(uid, "f.h5"): [[fake_hit], []]}
        out = _assign_hits(df, results_map)
        assert out["hits"].iloc[0] == [fake_hit]
        assert out["hits"].iloc[1] == []

    def test_missing_key_in_results_map_gives_empty_hits(self):
        uid = "uid-001"
        df = _ms2_df(uid, n_scans=1)
        out = _assign_hits(df, {})
        assert out["hits"].iloc[0] == []

    def test_multiple_compounds_assigned_independently(self):
        uid1, uid2 = "uid-001", "uid-002"
        df = pd.concat([_ms2_df(uid1), _ms2_df(uid2)], ignore_index=True)
        hit1 = [{"score": 0.9}]
        hit2 = [{"score": 0.5}]
        results_map = {
            (uid1, "f.h5"): [[hit1]],
            (uid2, "f.h5"): [[hit2]],
        }
        out = _assign_hits(df, results_map)
        row1 = out[out["mz_rt_uid"] == uid1].iloc[0]
        row2 = out[out["mz_rt_uid"] == uid2].iloc[0]
        assert row1["hits"] == [hit1]
        assert row2["hits"] == [hit2]

    def test_scan_order_preserved_within_group(self):
        uid = "uid-001"
        df = _ms2_df(uid, n_scans=3)
        hits_list = [[{"score": 0.9}], [{"score": 0.7}], [{"score": 0.5}]]
        results_map = {(uid, "f.h5"): hits_list}
        out = _assign_hits(df, results_map)
        for i, expected in enumerate(hits_list):
            assert out["hits"].iloc[i] == expected

    def test_output_has_same_number_of_rows(self):
        uid = "uid-001"
        df = _ms2_df(uid, n_scans=4)
        results_map = {(uid, "f.h5"): [[], [], [], []]}
        out = _assign_hits(df, results_map)
        assert len(out) == 4

    def test_hits_column_added_to_output(self):
        uid = "uid-001"
        df = _ms2_df(uid)
        out = _assign_hits(df, {})
        assert "hits" in out.columns


# ===========================================================================
# _filter_out_ms2_data
# ===========================================================================

class TestFilterOutMs2Data:

    def test_zero_thresholds_returns_all_scans(self):
        uid = "uid-001"
        ms2 = _ms2_df(uid, hits=[[{"score": 0.9}]])
        ms1 = _ms1_df(uid)
        out_ms2, out_ms1 = _filter_out_ms2_data(ms2, ms1, min_score=0, min_frags=0)
        assert len(out_ms2) == 1

    def test_compound_with_hits_retained(self):
        uid = "uid-001"
        ms2 = _ms2_df(uid, hits=[[{"score": 0.9, "num_matches": 2}]])
        ms1 = _ms1_df(uid)
        out_ms2, _ = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert uid in out_ms2["mz_rt_uid"].values

    def test_compound_without_hits_removed(self):
        uid = "uid-001"
        ms2 = _ms2_df(uid, hits=[])   # empty hits list
        ms1 = _ms1_df(uid)
        out_ms2, _ = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert uid not in out_ms2["mz_rt_uid"].values

    def test_ms1_synced_to_ms2_uids(self):
        uid_with_hits = "uid-001"
        uid_no_hits   = "uid-002"
        ms2 = pd.concat([
            _ms2_df(uid_with_hits, hits=[[{"score": 0.9, "num_matches": 2}]]),
            _ms2_df(uid_no_hits,   hits=[]),
        ], ignore_index=True)
        ms1 = pd.concat([_ms1_df(uid_with_hits), _ms1_df(uid_no_hits)], ignore_index=True)
        _, out_ms1 = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert uid_with_hits in out_ms1["mz_rt_uid"].values
        assert uid_no_hits not in out_ms1["mz_rt_uid"].values

    def test_out_of_feature_scans_do_not_count_as_hits(self):
        uid = "uid-001"
        # Scan has a hit but is NOT in_feature → should not count
        ms2 = _ms2_df(uid, in_feature=False, hits=[[{"score": 0.9, "num_matches": 2}]])
        ms1 = _ms1_df(uid)
        out_ms2, _ = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert uid not in out_ms2["mz_rt_uid"].values

    def test_output_has_final_columns(self):
        uid = "uid-001"
        ms2 = _ms2_df(uid, hits=[[{"score": 0.9, "num_matches": 2}]])
        ms1 = _ms1_df(uid)
        out_ms2, _ = _filter_out_ms2_data(ms2, ms1, min_score=0, min_frags=0)
        expected_cols = {"mz_rt_uid", "filename", "inchi_key", "adduct", "scan_rt",
                         "frag_mzs", "frag_ints", "precursor_MZ", "precursor_intensity",
                         "collision_energy", "in_feature", "hits"}
        assert expected_cols.issubset(set(out_ms2.columns))

    def test_empty_ms2_returns_empty(self):
        ms2 = pd.DataFrame(columns=["mz_rt_uid", "filename", "inchi_key", "adduct",
                                     "scan_rt", "frag_mzs", "frag_ints", "precursor_MZ",
                                     "precursor_intensity", "collision_energy", "in_feature", "hits"])
        ms1 = pd.DataFrame()
        # Should not raise even with an empty DataFrame
        out_ms2, out_ms1 = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert out_ms2.empty

    def test_multiple_scans_one_with_hit_retains_compound(self):
        uid = "uid-001"
        ms2 = pd.concat([
            _ms2_df(uid, hits=[[{"score": 0.9, "num_matches": 2}]]),   # scan 1: has hit
            _ms2_df(uid, hits=[]),                                       # scan 2: no hit
        ], ignore_index=True)
        ms1 = _ms1_df(uid)
        out_ms2, _ = _filter_out_ms2_data(ms2, ms1, min_score=0.5, min_frags=1)
        assert uid in out_ms2["mz_rt_uid"].values


# ===========================================================================
# _process_compound_batch
# ===========================================================================

class TestProcessCompoundBatch:
    """Tests for the per-(compound, file) scoring worker.

    Uses real matchms Spectrum objects so the CosineHungarian scoring path
    is exercised without any mocking.
    """

    # Adenine fragment pattern used in both query and reference
    _ADENINE_MZ  = [94.04, 119.04, 136.06]
    _ADENINE_INT = [1e4,   5e4,    2e5]

    def _ref(self, precursor_mz=ADENINE_MZ):
        return _ref_spectrum(
            precursor_mz=precursor_mz,
            mz=self._ADENINE_MZ,
            intensities=self._ADENINE_INT,
        )

    def _job(self, scans, refs, *, frag_tol=0.05, min_score=0.0,
             min_frags=0, ppm=20.0, limit=10):
        return ("uid-001", "f.h5", scans, refs, frag_tol, min_score, min_frags, ppm, limit, "GFFGJBXGBJISGV-UHFFFAOYSA-N")

    def test_returns_uid_filename_results_tuple(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        uid, fname, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        assert uid == "uid-001"
        assert fname == "f.h5"
        assert isinstance(results, list)

    def test_results_length_matches_scans(self):
        scans = [_scan_dict(self._ADENINE_MZ, self._ADENINE_INT)] * 3
        _, _, results, _ = _process_compound_batch(self._job(scans, [self._ref()]))
        assert len(results) == 3

    def test_matching_scan_produces_hit(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        assert len(results[0]) > 0

    def test_hit_has_expected_keys(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        hit = results[0][0]
        for key in ("score", "num_matches", "mz_measured", "mz_theoretical",
                    "ppm_error", "ref_frags", "data_frags",
                    "matched_fragments", "fragment_colors",
                    "query_aligned", "ref_aligned"):
            assert key in hit, f"Missing key: {key}"

    def test_score_between_0_and_1(self):
        # matchms CosineHungarian can return values marginally above 1.0 due to
        # float32 rounding; allow a small epsilon rather than asserting strict ≤ 1.
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        assert 0.0 <= results[0][0]["score"] <= 1.0 + 1e-5

    def test_identical_spectra_score_near_1(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        assert results[0][0]["score"] > 0.9

    def test_no_refs_returns_empty_hits(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], []))
        assert results[0] == []

    def test_empty_scan_frag_mzs_skipped(self):
        scan = _scan_dict([], [], precursor_mz=ADENINE_MZ)
        _, _, results, _ = _process_compound_batch(self._job([scan], [self._ref()]))
        assert results[0] == []

    def test_min_score_threshold_does_not_filter_hits_in_process_batch(self):
        # _process_compound_batch now stores ALL candidate hits regardless of
        # score — the min_score threshold is enforced only by
        # _filter_out_ms2_data at the compound level.  A low-scoring hit
        # against a completely different reference should still be stored.
        bad_ref = _ref_spectrum(
            precursor_mz=ADENINE_MZ,
            mz=[500.0, 600.0, 700.0],
            intensities=[1e4, 2e4, 3e4],
        )
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(
            self._job([scan], [bad_ref], min_score=0.9)
        )
        # Hit is stored even though score < min_score=0.9
        assert len(results[0]) == 1
        assert results[0][0]["score"] < 0.9

    def test_ppm_filter_excludes_mismatched_precursor(self):
        # Reference precursor is 10 Da away from query precursor
        far_ref = _ref_spectrum(
            precursor_mz=ADENINE_MZ + 10.0,
            mz=self._ADENINE_MZ,
            intensities=self._ADENINE_INT,
        )
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT, precursor_mz=ADENINE_MZ)
        _, _, results, _ = _process_compound_batch(
            self._job([scan], [far_ref], ppm=5.0)
        )
        assert results[0] == []

    def test_limit_to_n_hits_respected(self):
        # Build 5 identical references; with limit=2 only 2 hits should be returned
        refs = [self._ref() for _ in range(5)]
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        _, _, results, _ = _process_compound_batch(self._job([scan], refs, limit=2))
        assert len(results[0]) <= 2

    def test_ppm_error_computed_correctly(self):
        scan = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT, precursor_mz=ADENINE_MZ)
        ref = self._ref(precursor_mz=ADENINE_MZ)
        _, _, results, _ = _process_compound_batch(self._job([scan], [ref]))
        hit = results[0][0]
        expected_ppm = (ADENINE_MZ - ADENINE_MZ) / ADENINE_MZ * 1e6
        assert abs(hit["ppm_error"] - expected_ppm) < 0.01

    def test_multiple_scans_scored_independently(self):
        # Two scans: one matches, one has empty fragments
        scan_good  = _scan_dict(self._ADENINE_MZ, self._ADENINE_INT)
        scan_empty = _scan_dict([], [])
        _, _, results, _ = _process_compound_batch(
            self._job([scan_good, scan_empty], [self._ref()])
        )
        assert len(results[0]) > 0   # good scan has hits
        assert results[1] == []      # empty scan has no hits

# ===========================================================================
# _add_precursor_to_ref
# ===========================================================================

class TestAddPrecursorToRef:
    """Direct unit tests for the reference-augmentation helper."""

    TOL = 0.05

    def _ref(self, mz, ints):
        return _ref_spectrum(precursor_mz=ADENINE_MZ, mz=mz, intensities=ints)

    def _q(self, mz, ints):
        return np.array(mz, dtype=np.float32), np.array(ints, dtype=np.float32)

    def test_returns_none_when_ref_already_has_precursor(self):
        ref = self._ref([94.04, ADENINE_MZ], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        assert _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL) is None

    def test_returns_none_when_ref_peak_within_tolerance(self):
        ref = self._ref([94.04, ADENINE_MZ + 0.03], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        assert _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL) is None

    def test_adds_peak_when_ref_peak_just_outside_tolerance(self):
        ref = self._ref([94.04, ADENINE_MZ + 0.08], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        out = _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL)
        assert out is not None
        assert len(out.mz) == 3

    def test_returns_none_when_query_has_no_precursor_peak(self):
        ref = self._ref([94.04, 119.04], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, 119.04], [1e5, 5e4])
        assert _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL) is None

    def test_added_intensity_scaled_to_reference(self):
        # Query precursor peak = 80% of query base peak -> 80% of ref base peak
        ref = self._ref([94.04, 119.04], [2e5, 1e5])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        out = _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL)
        idx = int(np.argmin(np.abs(out.mz - ADENINE_MZ)))
        assert out.mz[idx] == pytest.approx(ADENINE_MZ, abs=1e-3)
        assert out.intensities[idx] == pytest.approx(0.8 * 2e5, rel=1e-4)

    def test_output_mz_sorted_and_peak_count_incremented(self):
        # Precursor falls in the middle of the ref peaks
        ref = self._ref([50.0, ADENINE_MZ + 20.0], [1e5, 5e4])
        q_mz, q_int = self._q([50.0, ADENINE_MZ], [1e5, 8e4])
        out = _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL)
        assert len(out.mz) == 3
        assert np.all(np.diff(out.mz) > 0)

    def test_original_reference_not_mutated(self):
        ref = self._ref([94.04, 119.04], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL)
        assert len(ref.mz) == 2

    def test_metadata_preserved(self):
        ref = self._ref([94.04, 119.04], [1e5, 5e4])
        q_mz, q_int = self._q([94.04, ADENINE_MZ], [1e5, 8e4])
        out = _add_precursor_to_ref(ref, q_mz, q_int, ADENINE_MZ, self.TOL)
        assert out.metadata["id"] == ref.metadata["id"]
        #assert out.metadata["name"] == ref.metadata["name"]
        assert out.metadata["database"] == ref.metadata["database"]


# ===========================================================================
# _process_compound_batch: precursor augmentation
# ===========================================================================

class TestProcessCompoundBatchPrecursorAddition:
    """End-to-end tests of the add-missing-precursor behaviour in the worker."""

    INCHI = "GFFGJBXGBJISGV-UHFFFAOYSA-N"

    # Query: two fragments plus a precursor peak at 80% of the *ref-equivalent* scale.
    # A reference with the identical pattern (incl. precursor) is a perfect match.
    _Q_MZ  = [94.04, 119.04, ADENINE_MZ]
    _Q_INT = [1e5,   5e4,    8e4]

    # Reference that lacks the precursor peak entirely
    _REF_NO_PREC_MZ  = [94.04, 119.04]
    _REF_NO_PREC_INT = [1e5,   5e4]

    def _job(self, scans, refs, *, frag_tol=0.05, ppm=20.0, limit=10):
        return ("uid-001", "f.h5", scans, refs, frag_tol, 0.0, 0, ppm, limit, self.INCHI)

    def _ref_no_prec(self, ref_id="ref_no_prec"):
        return _ref_spectrum(ADENINE_MZ, self._REF_NO_PREC_MZ, self._REF_NO_PREC_INT, ref_id=ref_id)

    def _ref_with_prec(self, ref_id="ref_with_prec"):
        return _ref_spectrum(ADENINE_MZ, self._Q_MZ, self._Q_INT, ref_id=ref_id)

    def _scan(self, mz=None, ints=None):
        return _scan_dict(mz or self._Q_MZ, ints or self._Q_INT, precursor_mz=ADENINE_MZ)

    # ---- precursor missing -> added ----------------------------------------

    def test_missing_precursor_returns_inchi_key(self):
        *_, added_key = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        assert added_key == self.INCHI

    def test_missing_precursor_hit_is_flagged(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        assert results[0][0]["precursor_added"] is True

    def test_missing_precursor_hit_still_returned(self):
        # Regression for the original draft: the hit must NOT be dropped.
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        assert len(results[0]) == 1

    def test_missing_precursor_ref_frags_incremented(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        assert results[0][0]["ref_frags"] == len(self._REF_NO_PREC_MZ) + 1

    def test_missing_precursor_counts_as_match(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        assert results[0][0]["num_matches"] == 3

    def test_missing_precursor_score_improves_vs_unaugmented(self):
        # Baseline: score the query against the reference *without* augmentation
        scan = self._scan()
        ref = self._ref_no_prec()
        query = Spectrum(
            mz=np.array(self._Q_MZ, dtype=np.float32),
            intensities=np.array(self._Q_INT, dtype=np.float32),
            metadata={"precursor_mz": ADENINE_MZ},
        )
        baseline = float(CosineHungarian(tolerance=0.05).pair(ref, query)["score"])

        _, _, results, _ = _process_compound_batch(self._job([scan], [ref]))
        augmented = results[0][0]["score"]

        assert augmented > baseline + 0.1
        # Scaled precursor intensity makes ref identical to query -> ~perfect score
        assert augmented > 0.99

    def test_aligned_ref_contains_added_precursor_peak(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_no_prec()]))
        ref_mz_aligned = np.array(results[0][0]["ref_aligned"][0], dtype=float)
        ref_mz_aligned = ref_mz_aligned[~np.isnan(ref_mz_aligned)]
        assert np.any(np.abs(ref_mz_aligned - ADENINE_MZ) <= 0.05)

    def test_input_reference_not_mutated(self):
        ref = self._ref_no_prec()
        _process_compound_batch(self._job([self._scan()], [ref]))
        assert len(ref.mz) == len(self._REF_NO_PREC_MZ)

    # ---- precursor present -> unchanged ------------------------------------

    def test_present_precursor_returns_none_key(self):
        *_, added_key = _process_compound_batch(self._job([self._scan()], [self._ref_with_prec()]))
        assert added_key is None

    def test_present_precursor_hit_not_flagged(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_with_prec()]))
        assert results[0][0]["precursor_added"] is False

    def test_present_precursor_ref_frags_unchanged(self):
        _, _, results, _ = _process_compound_batch(self._job([self._scan()], [self._ref_with_prec()]))
        assert results[0][0]["ref_frags"] == len(self._Q_MZ)

    def test_ref_precursor_within_tolerance_not_augmented(self):
        ref = _ref_spectrum(ADENINE_MZ, [94.04, 119.04, ADENINE_MZ + 0.03], self._Q_INT)
        *_, added_key = _process_compound_batch(self._job([self._scan()], [ref], frag_tol=0.05))
        assert added_key is None

    def test_ref_precursor_outside_tolerance_is_augmented(self):
        ref = _ref_spectrum(ADENINE_MZ, [94.04, 119.04, ADENINE_MZ + 0.08], self._Q_INT)
        *_, added_key = _process_compound_batch(self._job([self._scan()], [ref], frag_tol=0.05))
        assert added_key == self.INCHI

    # ---- query has no precursor peak -> nothing to add ---------------------

    def test_query_without_precursor_peak_not_augmented(self):
        scan = self._scan(mz=[94.04, 119.04], ints=[1e5, 5e4])
        _, _, results, added_key = _process_compound_batch(self._job([scan], [self._ref_no_prec()]))
        assert added_key is None
        assert results[0][0]["precursor_added"] is False
        assert results[0][0]["ref_frags"] == len(self._REF_NO_PREC_MZ)

    # ---- mixed refs / multiple scans ---------------------------------------

    def test_only_refs_missing_precursor_are_flagged(self):
        refs = [self._ref_no_prec("A_missing"), self._ref_with_prec("B_present")]
        _, _, results, added_key = _process_compound_batch(self._job([self._scan()], refs))
        by_id = {h["ref_id"]: h for h in results[0]}
        assert by_id["A_missing"]["precursor_added"] is True
        assert by_id["B_present"]["precursor_added"] is False
        assert added_key == self.INCHI

    def test_augmentation_is_per_scan(self):
        scan_with = self._scan()
        scan_without = self._scan(mz=[94.04, 119.04], ints=[1e5, 5e4])
        _, _, results, added_key = _process_compound_batch(
            self._job([scan_with, scan_without], [self._ref_no_prec()])
        )
        assert results[0][0]["precursor_added"] is True
        assert results[1][0]["precursor_added"] is False
        assert added_key == self.INCHI

    def test_no_refs_returns_none_key(self):
        *_, added_key = _process_compound_batch(self._job([self._scan()], []))
        assert added_key is None