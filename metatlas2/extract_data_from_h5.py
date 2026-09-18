from __future__ import annotations

import pandas as pd
import numpy as np
import os
import sys
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf
import metatlas2.load_tools as ldt
from metatlas2.utils import should_disable_tqdm, get_max_workers
logger = lcf.get_logger('extract_data_from_h5')

def _load_h5_table(file_path, key, columns=None, mz_bounds=None):
    """Load a PyTables/HDF5 table from *file_path* into a DataFrame.

    When *mz_bounds* is provided the ``<key>_mz`` sorted variant is read and
    a binary-search pre-filter is applied so only rows within the m/z range
    are returned.

    Args:
        file_path: Path to the HDF5 file.
        key:       HDF5 table key (e.g. ``"ms1_pos"``).
        columns:   Optional list of column names to load; ``None`` loads all.
        mz_bounds: Optional ``(mz_min, mz_max)`` tuple.  When given, the
                   ``<key>_mz`` sorted variant is read and rows outside the
                   range are dropped via ``searchsorted``.

    Returns:
        DataFrame with the requested rows, or an empty DataFrame on error.
    """
    read_key = key + "_mz" if mz_bounds is not None else key
    try:
        df = pd.read_hdf(file_path, key=read_key, columns=columns)
    except (KeyError, ValueError, OSError) as exc:
        logger.warning(f"Could not read key {read_key} from {file_path}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return df
    logger.debug(f"Loaded {len(df)} rows from {file_path} key {read_key}")
    if mz_bounds is not None:
        mz_min, mz_max = mz_bounds
        mz = df["mz"].to_numpy()
        lo = np.searchsorted(mz, mz_min, side="left")
        hi = np.searchsorted(mz, mz_max, side="right")
        df = df.iloc[lo:hi]
        logger.debug(f"Filtered to {len(df)} rows within atlas mz bounds (to remove out-of-scope data points) [{mz_min:f}, {mz_max:f}]")
    float_cols = df.select_dtypes(include=['float64']).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].astype(np.float32, copy=False)
    return df

def _expand_atlas_windows(
    atlas: pd.DataFrame,
    extra_time: float,
    ms1_mz_tolerance_ppm: float,
    polarity: str,
    extract_extra_time: float | None = None,
) -> pd.DataFrame:
    """Add padded m/z and RT bound columns to the atlas DataFrame.

    Computes ``mz_min``/``mz_max`` from the ppm tolerance and
    ``rt_min_pad``/``rt_max_pad`` by subtracting/adding *extra_time* to the
    atlas RT bounds.  These columns are consumed by the interval-join helpers
    to determine which scan points are tagged ``in_feature``.

    When *extract_extra_time* is provided (and larger than *extra_time*), two
    additional columns ``rt_min_extract`` / ``rt_max_extract`` are added.
    These wider windows are used by :func:`_process_one_file` to pre-filter
    the raw HDF5 data before the join, so that ``only_keep_data_in_feature``
    can be set to ``False`` (retaining the full EIC shape) while still
    discarding data that is far outside any atlas feature.  When
    *extract_extra_time* is ``None`` the extract columns are set equal to the
    pad columns (no additional pre-filtering).

    Args:
        atlas:                 Atlas compound DataFrame (must have ``mz``,
                               ``rt_min``, ``rt_max`` columns).
        extra_time:            Extra time (minutes) added to each RT window to
                               define the ``in_feature`` tag boundary.
        ms1_mz_tolerance_ppm:  m/z tolerance in ppm used to compute the
                               ``mz_min``/``mz_max`` search window.
        polarity:              Polarity string (``"positive"`` or
                               ``"negative"``) written into the output.
        extract_extra_time:    Optional wider RT padding (minutes) used only
                               for the HDF5 pre-filter step.  Must be ≥
                               *extra_time*.  ``None`` means no extra
                               pre-filtering beyond the ``in_feature`` window.

    Returns:
        Copy of *atlas* with ``polarity``, ``mz_min``, ``mz_max``,
        ``rt_min_pad``, ``rt_max_pad``, ``rt_min_extract``, and
        ``rt_max_extract`` columns added.
    """
    eet = extra_time if extract_extra_time is None else max(extract_extra_time, extra_time)
    logger.info(
        f"Expanding atlas windows for {len(atlas)} compounds with "
        f"extra_time={extra_time}, extract_extra_time={eet}, "
        f"mz_tolerance_ppm={ms1_mz_tolerance_ppm}"
    )

    out = atlas.copy()
    out["polarity"] = polarity
    mz = out["mz"].to_numpy(dtype=np.float64)
    tol = mz * ms1_mz_tolerance_ppm * 1e-6
    out["mz_min"] = (mz - tol).astype(np.float32)
    out["mz_max"] = (mz + tol).astype(np.float32)
    rt_min = out["rt_min"].to_numpy(dtype=np.float64)
    rt_max = out["rt_max"].to_numpy(dtype=np.float64)
    out["rt_min_pad"] = (rt_min - extra_time).astype(np.float32)
    out["rt_max_pad"] = (rt_max + extra_time).astype(np.float32)
    out["rt_min_extract"] = (rt_min - eet).astype(np.float32)
    out["rt_max_extract"] = (rt_max + eet).astype(np.float32)
    return out

def _interval_join_mz(query_mz, atlas_mz_min, atlas_mz_max, chunk_size=50_000):
    """Return index pairs (query_idx, atlas_idx) for all overlapping m/z intervals.

    Implements a vectorised sweep-line interval join: for each query m/z value
    find every atlas feature whose ``[mz_min, mz_max]`` window contains it.
    Processing is done in chunks to bound peak memory usage.

    Args:
        query_mz:     1-D numpy array of query m/z values (may contain NaN).
        atlas_mz_min: 1-D numpy array of atlas lower m/z bounds.
        atlas_mz_max: 1-D numpy array of atlas upper m/z bounds.
        chunk_size:   Number of sorted query points processed per iteration.

    Returns:
        Tuple ``(query_indices, atlas_indices)`` — parallel integer arrays
        giving the row index in the original query and atlas arrays for each
        matching pair.  Both arrays are empty when there are no matches.
    """
    n, m = len(atlas_mz_min), len(query_mz)
    if n == 0 or m == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    order_min = np.argsort(atlas_mz_min, kind="stable")
    sorted_min, sorted_max = atlas_mz_min[order_min], atlas_mz_max[order_min]
    mz_order = np.argsort(query_mz, kind="stable").astype(np.int64, copy=False)
    q_sorted_full = query_mz[mz_order]
    valid_count = int(np.count_nonzero(~np.isnan(q_sorted_full)))
    if valid_count == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    mz_order, q_sorted = mz_order[:valid_count], q_sorted_full[:valid_count]
    q_chunks_q, q_chunks_a = [], []
    for start in range(0, valid_count, chunk_size):
        stop = min(start + chunk_size, valid_count)
        q = q_sorted[start:stop]
        hi = np.searchsorted(sorted_min, q.max(), side="right")
        live_mask = sorted_max[:hi] >= q.min()
        live_pos = np.nonzero(live_mask)[0]
        if live_pos.size == 0: 
            continue
        live_min, live_max = sorted_min[live_pos], sorted_max[live_pos]
        n_open_local = np.searchsorted(live_min, q, side="right")
        total = int(n_open_local.sum())
        if total == 0: 
            continue
        q_local = np.repeat(np.arange(q.size, dtype=np.int64), n_open_local)
        starts = np.zeros(q.size, dtype=np.int64)
        np.cumsum(n_open_local[:-1], out=starts[1:])
        cand_pos = np.arange(total, dtype=np.int64) - np.repeat(starts, n_open_local)
        keep = live_max[cand_pos] >= q[q_local]
        if not keep.any(): 
            continue
        q_chunks_q.append(mz_order[start + q_local[keep]])
        q_chunks_a.append(order_min[live_pos[cand_pos[keep]]])
    if not q_chunks_q:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(q_chunks_q), np.concatenate(q_chunks_a)

def _join_ms1_to_atlas(ms1_df: pd.DataFrame, atlas: pd.DataFrame, only_in_feature: bool) -> pd.DataFrame:
    """Join raw MS1 scan points to atlas features via m/z interval + RT window.

    Each scan point is matched to every atlas feature whose m/z window
    contains the scan m/z.  An ``in_feature`` boolean column is added
    indicating whether the scan RT also falls within the padded RT window.

    Args:
        ms1_df:          Long-format MS1 DataFrame with ``mz``, ``rt``, ``i``
                         columns (one row per scan point).
        atlas:           Expanded atlas DataFrame (output of
                         :func:`_expand_atlas_windows`).
        only_in_feature: When ``True``, rows where ``in_feature`` is ``False``
                         are dropped before returning.

    Returns:
        DataFrame with columns ``mz``, ``rt``, ``i``, ``mz_rt_uid``,
        ``in_feature``, or an empty DataFrame when there are no matches.
    """
    if ms1_df.empty or atlas.empty:
        return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_feature"])
    logger.debug(f"Joining {len(ms1_df)} MS1 points to {len(atlas)} atlas features")
    scans = ms1_df[["mz", "rt", "i"]].reset_index(drop=True)
    atlas_r = atlas.reset_index(drop=True)
    q_idx, a_idx = _interval_join_mz(scans["mz"].to_numpy(), atlas_r["mz_min"].to_numpy(), atlas_r["mz_max"].to_numpy())
    if len(q_idx) == 0:
        return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_feature"])
    
    scan_rt = scans["rt"].to_numpy()
    in_feature = (scan_rt[q_idx] >= atlas_r["rt_min_pad"].to_numpy()[a_idx]) & \
                 (scan_rt[q_idx] <= atlas_r["rt_max_pad"].to_numpy()[a_idx])

    # remove any rows that are not in a feature if only_in_feature is True
    if only_in_feature:
        keep_mask = in_feature
        q_idx = q_idx[keep_mask]
        a_idx = a_idx[keep_mask]
        in_feature = in_feature[keep_mask]
        logger.debug(f"Filtered to {len(q_idx)} MS1 points that are in features (only_in_feature=True)")

    return pd.DataFrame({
        "mz": scans["mz"].to_numpy()[q_idx],
        "rt": scans["rt"].to_numpy()[q_idx],
        "i": scans["i"].to_numpy()[q_idx],
        "mz_rt_uid": atlas_r["mz_rt_uid"].to_numpy()[a_idx],
        "in_feature": in_feature,
    })

def _join_ms2_to_atlas(ms2_df: pd.DataFrame, atlas: pd.DataFrame, only_in_feature: bool) -> pd.DataFrame:
    """Join raw MS2 scan points to atlas features via precursor m/z + RT window.

    Matches each MS2 scan to atlas features using the precursor m/z interval
    join.  Scans with missing or zero precursor m/z are discarded before
    joining.

    Args:
        ms2_df:          Long-format MS2 DataFrame with ``mz``, ``i``, ``rt``,
                         ``precursor_MZ``, ``precursor_intensity``,
                         ``collision_energy`` columns.
        atlas:           Expanded atlas DataFrame (output of
                         :func:`_expand_atlas_windows`).
        only_in_feature: When ``True``, rows where ``in_feature`` is ``False``
                         are dropped before returning.

    Returns:
        DataFrame with the original MS2 columns plus ``mz_rt_uid`` and
        ``in_feature``, or an empty DataFrame when there are no matches.
    """
    if ms2_df.empty or atlas.empty:
        return pd.DataFrame(columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy", "mz_rt_uid", "in_feature"])
    logger.debug(f"Joining {len(ms2_df)} MS2 points to {len(atlas)} atlas features")
    needed = ("mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy")
    scans = ms2_df[[c for c in needed if c in ms2_df.columns]].reset_index(drop=True)
    precursor_raw = scans["precursor_MZ"].to_numpy()
    valid_precursor = ~np.isnan(precursor_raw) & (precursor_raw > 0)
    scans = scans.loc[valid_precursor].reset_index(drop=True)
    if scans.empty:
        return pd.DataFrame(columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy", "mz_rt_uid", "in_feature"])

    atlas_r = atlas.reset_index(drop=True)
    q_idx, a_idx = _interval_join_mz(scans["precursor_MZ"].to_numpy(), atlas_r["mz_min"].to_numpy(), atlas_r["mz_max"].to_numpy())
    if len(q_idx) == 0:
        return pd.DataFrame(columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy", "mz_rt_uid", "in_feature"])

    scan_rt = scans["rt"].to_numpy()
    in_feature = (scan_rt[q_idx] >= atlas_r["rt_min_pad"].to_numpy()[a_idx]) & \
                 (scan_rt[q_idx] <= atlas_r["rt_max_pad"].to_numpy()[a_idx])
    pts = scans.iloc[q_idx].reset_index(drop=True)
    pts["mz_rt_uid"] = atlas_r["mz_rt_uid"].to_numpy()[a_idx]
    pts["in_feature"] = in_feature

    # remove any rows that are not in a feature if only_in_feature is True
    if only_in_feature:
        pts = pts[pts["in_feature"]].reset_index(drop=True)
        logger.debug(f"Filtered to {len(pts)} MS2 points that are in features (only_in_feature=True)")

    return pts

def _process_one_file(run, atlas, only_in_feature):
    """Extract and atlas-join MS1 and MS2 data for a single HDF5 file.

    Designed to be called from a :class:`ProcessPoolExecutor` worker.  Reads
    the appropriate polarity table from *run.file_path*, applies the m/z
    pre-filter, and joins the result to *atlas* via
    :func:`_join_ms1_to_atlas` / :func:`_join_ms2_to_atlas`.

    When the atlas contains ``rt_min_extract`` / ``rt_max_extract`` columns
    (added by :func:`_expand_atlas_windows` when ``extract_extra_time`` is
    set), the raw HDF5 data is pre-filtered to those wider RT windows before
    the join.  This allows ``only_keep_data_in_feature=False`` (full EIC
    shape) while still discarding data that is far outside any atlas feature,
    reducing memory usage compared to loading the entire chromatogram.

    Args:
        run:             :class:`LCMSRun` object with ``file_path`` and
                         ``filename`` attributes.
        atlas:           Expanded atlas DataFrame (output of
                         :func:`_expand_atlas_windows`).
        only_in_feature: Passed through to the join helpers; when ``True``
                         only in-feature scan points are retained.

    Returns:
        Tuple ``(ms1_extracted, ms2_extracted)`` — long-format DataFrames
        with a ``filename`` column added.  Either may be empty.
    """
    polarity = atlas['polarity'].iloc[0] if 'polarity' in atlas.columns else 'unknown'
    ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
    ms2_key = {"positive": "ms2_pos", "negative": "ms2_neg"}.get(polarity)

    ms1_df = _load_h5_table(run.file_path, ms1_key, columns=["mz", "rt", "i"],
                            mz_bounds=(float(atlas["mz_min"].min()), float(atlas["mz_max"].max())))

    # Pre-filter MS1 rows to the extract RT windows when they are present and
    # wider than the in-feature windows (i.e. extract_extra_time was set).
    if (
        not ms1_df.empty
        and "rt_min_extract" in atlas.columns
        and "rt_max_extract" in atlas.columns
        and not only_in_feature
    ):
        rt_global_min = float(atlas["rt_min_extract"].min())
        rt_global_max = float(atlas["rt_max_extract"].max())
        rt_arr = ms1_df["rt"].to_numpy()
        rt_mask = (rt_arr >= rt_global_min) & (rt_arr <= rt_global_max)
        if not rt_mask.all():
            logger.debug(
                f"Pre-filtered MS1 from {len(ms1_df)} to {int(rt_mask.sum())} rows "
                f"using extract RT window [{rt_global_min:.3f}, {rt_global_max:.3f}] for {run.filename}"
            )
            ms1_df = ms1_df.loc[rt_mask].reset_index(drop=True)

    ms2_df = _load_h5_table(run.file_path, ms2_key, columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy"]) if ms2_key else pd.DataFrame()

    ms1_extracted = _join_ms1_to_atlas(ms1_df, atlas, only_in_feature)
    ms2_extracted = _join_ms2_to_atlas(ms2_df, atlas, only_in_feature)

    logger.debug(f"Extracted {len(ms1_extracted)} MS1 points and {len(ms2_extracted)} MS2 points for run {run.filename}")
    if not ms1_extracted.empty:
        ms1_extracted["filename"] = run.filename
    if not ms2_extracted.empty:
        ms2_extracted["filename"] = run.filename

    return ms1_extracted, ms2_extracted

def _sort_frags(wide):
    """Sort MS2 fragment m/z and intensity lists in ascending m/z order.

    Operates in-place on the ``frag_mzs`` and ``frag_ints`` list columns of
    a wide-format MS2 DataFrame.  Rows that are already sorted are left
    unchanged.

    Args:
        wide: Wide-format MS2 DataFrame with ``frag_mzs`` and ``frag_ints``
              list columns.

    Returns:
        The same DataFrame with sorted fragment lists (modified in-place).
    """
    if not 'frag_mzs' in wide.columns or not 'frag_ints' in wide.columns:
        return wide
    else:
        frag_mzs = wide['frag_mzs'].values
        frag_ints = wide['frag_ints'].values
        def sort_frags(mzs, ints):
            if mzs is None or ints is None or len(mzs) != len(ints) or len(mzs) == 0:
                return mzs, ints
            is_sorted = all(mzs[i] <= mzs[i+1] for i in range(len(mzs)-1))
            if is_sorted:
                return mzs, ints
            idx = np.argsort(mzs)
            return [np.array(mzs)[idx].tolist(), np.array(ints)[idx].tolist()]
        sorted_frags = [sort_frags(m, i) for m, i in zip(frag_mzs, frag_ints)]
        wide['frag_mzs'] = [x[0] for x in sorted_frags]
        wide['frag_ints'] = [x[1] for x in sorted_frags]
        return wide

def _sort_ms1_lists_by_rts(wide):
    """Sort MS1 list columns (spec_rts, spec_ints, spec_mzs, in_feature) by RT.

    Ensures that the per-file EIC lists are in ascending retention-time order,
    which is required for correct ``np.interp`` calls in downstream analysis.

    Args:
        wide: Wide-format MS1 DataFrame with ``spec_rts``, ``spec_ints``,
              ``spec_mzs``, and ``in_feature`` list columns.

    Returns:
        The same DataFrame with all list columns sorted by RT (modified
        in-place).
    """
    if not all(col in wide.columns for col in ['spec_rts', 'spec_ints', 'spec_mzs', 'in_feature']):
        return wide
    def sort_row(rts, ints, mzs, feats):
        if rts is None or ints is None or mzs is None or feats is None:
            return rts, ints, mzs, feats
        if len(rts) != len(ints) or len(rts) != len(mzs) or len(rts) != len(feats) or len(rts) == 0:
            return rts, ints, mzs, feats
        idx = np.argsort(rts)
        return [
            np.array(rts)[idx].tolist(),
            np.array(ints)[idx].tolist(),
            np.array(mzs)[idx].tolist(),
            np.array(feats)[idx].tolist()
        ]
    sorted_cols = [sort_row(rts, ints, mzs, feats) for rts, ints, mzs, feats in zip(wide['spec_rts'], wide['spec_ints'], wide['spec_mzs'], wide['in_feature'])]
    wide['spec_rts'] = [x[0] for x in sorted_cols]
    wide['spec_ints'] = [x[1] for x in sorted_cols]
    wide['spec_mzs'] = [x[2] for x in sorted_cols]
    wide['in_feature'] = [x[3] for x in sorted_cols]
    return wide

def _widen_one_file_ms1(ms1_long: pd.DataFrame) -> pd.DataFrame:
    """Widen a single file's long-format MS1 data into one row per compound.

    Because *ms1_long* contains data for exactly one ``filename``, the group
    key is just ``mz_rt_uid``.  Each compound's scan points are collected into
    ``spec_rts``, ``spec_ints``, and ``spec_mzs`` lists and the ``in_feature``
    flag list is preserved.  The ``filename`` column is carried through as a
    scalar (same value for every row).

    Args:
        ms1_long: Long-format MS1 DataFrame for a single file with columns
                  ``mz``, ``rt``, ``i``, ``in_feature``, ``filename``,
                  ``mz_rt_uid``.

    Returns:
        Wide-format MS1 DataFrame with one row per ``mz_rt_uid`` and list
        columns ``spec_rts``, ``spec_ints``, ``spec_mzs``, ``in_feature``.
        Returns an empty DataFrame when *ms1_long* is empty.
    """
    if ms1_long.empty:
        return pd.DataFrame()

    agg = (
        ms1_long
        .groupby("mz_rt_uid", sort=False)
        .agg(
            spec_rts=("rt", list),
            spec_ints=("i", list),
            spec_mzs=("mz", list),
            in_feature=("in_feature", list),
            filename=("filename", "first"),
        )
        .reset_index()
    )
    return agg


def _widen_one_file_ms2(ms2_long: pd.DataFrame) -> pd.DataFrame:
    """Widen a single file's long-format MS2 data into one row per scan.

    Because *ms2_long* contains data for exactly one ``filename``, the group
    key is ``(mz_rt_uid, rt)``.  Fragment m/z and intensity values are
    collected into ``frag_mzs`` and ``frag_ints`` lists.  The ``in_feature``
    flag is taken from the first row of each scan group (all rows in a scan
    share the same value).  Extra scalar columns (``precursor_MZ``,
    ``precursor_intensity``, ``collision_energy``) are carried through via
    ``first``.

    Args:
        ms2_long: Long-format MS2 DataFrame for a single file with columns
                  ``mz``, ``i``, ``rt``, ``precursor_MZ``,
                  ``precursor_intensity``, ``collision_energy``,
                  ``in_feature``, ``filename``, ``mz_rt_uid``.

    Returns:
        Wide-format MS2 DataFrame with one row per ``(mz_rt_uid, scan_rt)``
        and list columns ``frag_mzs``, ``frag_ints``.  Returns an empty
        DataFrame when *ms2_long* is empty.
    """
    if ms2_long.empty:
        return pd.DataFrame()

    scalar_cols = [c for c in ("precursor_MZ", "precursor_intensity", "collision_energy") if c in ms2_long.columns]
    agg_spec: dict = {
        "frag_mzs": ("mz", list),
        "frag_ints": ("i", list),
        "in_feature": ("in_feature", lambda x: bool(x.iloc[0]) if len(x) > 0 else False),
        "filename": ("filename", "first"),
    }
    for col in scalar_cols:
        agg_spec[col] = (col, "first")

    wide = (
        ms2_long
        .groupby(["mz_rt_uid", "rt"], sort=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"rt": "scan_rt"})
    )
    wide = _sort_frags(wide)
    return wide


def _merge_wide_ms1(
    accumulator: pd.DataFrame,
    new_chunk: pd.DataFrame,
) -> pd.DataFrame:
    """Append a per-file wide MS1 chunk to a running accumulator.

    This helper is retained for use in tests and any external callers.
    The main :func:`extract_data_from_raw` pipeline now collects all chunks
    in a list and calls :func:`pandas.concat` once at the end to avoid
    creating N-1 intermediate DataFrames.

    Args:
        accumulator: Current wide-format MS1 accumulator (may be empty on the
                     first call).
        new_chunk:   Wide-format MS1 DataFrame for one file (output of
                     :func:`_widen_one_file_ms1`).

    Returns:
        Updated accumulator with *new_chunk* appended.
    """
    if new_chunk.empty:
        return accumulator
    if accumulator.empty:
        return new_chunk
    return pd.concat([accumulator, new_chunk], ignore_index=True)


def _merge_wide_ms2(
    accumulator: pd.DataFrame,
    new_chunk: pd.DataFrame,
) -> pd.DataFrame:
    """Append a per-file wide MS2 chunk to a running accumulator.

    This helper is retained for use in tests and any external callers.
    The main :func:`extract_data_from_raw` pipeline now collects all chunks
    in a list and calls :func:`pandas.concat` once at the end to avoid
    creating N-1 intermediate DataFrames.

    Args:
        accumulator: Current wide-format MS2 accumulator (may be empty on the
                     first call).
        new_chunk:   Wide-format MS2 DataFrame for one file (output of
                     :func:`_widen_one_file_ms2`).

    Returns:
        Updated accumulator with *new_chunk* appended.
    """
    if new_chunk.empty:
        return accumulator
    if accumulator.empty:
        return new_chunk
    return pd.concat([accumulator, new_chunk], ignore_index=True)


def _log_ms_totals(ms1_df: pd.DataFrame, ms2_df: pd.DataFrame) -> None:
    """Log aggregate counts for the fully assembled wide MS1 and MS2 DataFrames.

    Counts total list elements across all rows to report the equivalent of the
    pre-widening long-format row counts that were previously logged inside
    :func:`_widen_ms_data`.

    Args:
        ms1_df: Wide-format MS1 DataFrame (post stream-widen).
        ms2_df: Wide-format MS2 DataFrame (post stream-widen).
    """
    if not ms1_df.empty:
        total_ms1 = int(ms1_df["spec_rts"].apply(len).sum()) if "spec_rts" in ms1_df.columns else 0
        in_feat_ms1 = int(
            ms1_df["in_feature"].apply(lambda x: sum(1 for v in x if v)).sum()
        ) if "in_feature" in ms1_df.columns else 0
        pct = in_feat_ms1 / max(total_ms1, 1) * 100
        logger.info(f"Total ms1 data points: {total_ms1}")
        logger.info(f"Total ms1 data points in atlas feature windows: {in_feat_ms1} ({pct:.2f}%)")
        logger.info(
            f"Aggregated ms1 spectral data to {len(ms1_df)} unique feature compound+file entries."
        )
        logger.info(f"  Unique files: {ms1_df['filename'].nunique()}")
        logger.info(f"  Unique compounds (mz_rt_uid): {ms1_df['mz_rt_uid'].nunique()}")
    if not ms2_df.empty:
        total_ms2 = int(ms2_df["frag_mzs"].apply(len).sum()) if "frag_mzs" in ms2_df.columns else 0
        logger.info(f"Total ms2 data points: {total_ms2}")
        logger.info(
            f"Aggregated ms2 spectral data to {len(ms2_df)} unique scan compound+file+scan entries."
        )
        logger.info(f"  Unique files: {ms2_df['filename'].nunique()}")
        logger.info(f"  Unique compounds (mz_rt_uid): {ms2_df['mz_rt_uid'].nunique()}")

def _filter_ms2_points(ms2_df, ms1_df, min_scans=None, min_int=None):
    """Filter wide-format MS2 DataFrame by minimum scan count and precursor intensity.

    *ms2_df* is expected to be in the post-:func:`_widen_ms_data` format:
    one row per ``(mz_rt_uid, filename, scan_rt)`` with a scalar ``in_feature``
    bool and a scalar ``precursor_intensity`` float per row.

    Filters are applied at the ``(mz_rt_uid, filename)`` group level:

    * ``min_scans`` — keep groups that have at least this many in-feature scans.
    * ``min_int``   — keep groups whose maximum in-feature precursor intensity
      meets the threshold.

    After group-level filtering, any remaining rows with ``in_feature=False``
    are dropped so only in-feature scans survive.  Orphan MS2 entries (no
    corresponding MS1 compound) are also removed when *ms1_df* is non-empty.

    Args:
        ms2_df:     Wide-format MS2 DataFrame (one row per scan).
        ms1_df:     Wide-format MS1 DataFrame used to remove orphan MS2 entries.
        min_scans:  Minimum number of in-feature scans required per
                    ``(mz_rt_uid, filename)`` group.  ``None`` skips this filter.
        min_int:    Minimum precursor intensity required among in-feature scans.
                    ``None`` skips this filter.

    Returns:
        Filtered MS2 DataFrame.
    """
    if ms2_df.empty:
        logger.warning("No MS2 data found. Skipping point filtering.")
        return ms2_df
    if min_scans is None and min_int is None:
        logger.info(f"No MS2 point filters specified. Retaining all {len(ms2_df)} entries across {ms2_df['mz_rt_uid'].nunique()} compounds.")
        return ms2_df
    if min_scans == 0 and min_int == 0:
        logger.info(f"ms2_min_scans=0 and ms2_min_intensity=0: skipping MS2 point filter. Retaining all {len(ms2_df)} entries across {ms2_df['mz_rt_uid'].nunique()} compounds.")
        return ms2_df

    starting_scans = len(ms2_df)
    starting_uids = ms2_df['mz_rt_uid'].nunique()

    # Each row is one scan; in_feature is a scalar bool.
    # Build a per-(mz_rt_uid, filename) summary of in-feature scans.
    group_cols = ['mz_rt_uid', 'filename']
    in_feature_mask = ms2_df['in_feature'].astype(bool)

    # Collect per-step stats for the summary table: (step_label, scans_after, compounds_after)
    steps = [("extracted", starting_scans, starting_uids)]

    if min_scans is not None and min_scans > 0 and not ms2_df.empty:
        # Count in-feature rows per (mz_rt_uid, filename) group.
        in_feature_counts = (
            ms2_df[in_feature_mask]
            .groupby(group_cols)
            .size()
            .rename('_in_feature_count')
        )
        ms2_df = ms2_df.join(in_feature_counts, on=group_cols)
        ms2_df = ms2_df[ms2_df['_in_feature_count'].fillna(0) >= min_scans].drop(columns='_in_feature_count')
        in_feature_mask = ms2_df['in_feature'].astype(bool)  # refresh after filter
        steps.append((f"min_scans >= {min_scans}", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))
        if ms2_df.empty:
            ldt.log_filter_table(steps, starting_scans, starting_uids, entries_label="Scans", title="MS2 point filtering summary")
            return ms2_df

    if min_int is not None and min_int > 0 and not ms2_df.empty:
        # Max precursor_intensity among in-feature rows per (mz_rt_uid, filename) group.
        int_col = 'precursor_intensity'
        in_feature_max_int = (
            ms2_df.loc[in_feature_mask, group_cols + [int_col]]
            .groupby(group_cols)[int_col]
            .max()
            .rename('_max_in_feature_int')
        )
        ms2_df = ms2_df.join(in_feature_max_int, on=group_cols)
        ms2_df = ms2_df[ms2_df['_max_in_feature_int'].fillna(-float('inf')) >= min_int].drop(columns='_max_in_feature_int')
        in_feature_mask = ms2_df['in_feature'].astype(bool)  # refresh after filter
        steps.append((f"min_intensity >= {min_int}", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))
        if ms2_df.empty:
            ldt.log_filter_table(steps, starting_scans, starting_uids, entries_label="Scans", title="MS2 point filtering summary")
            return ms2_df

    if not ms2_df.empty:
        ms2_df = ms2_df[in_feature_mask].reset_index(drop=True)
        steps.append(("any in-feature", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))

    if not ms2_df.empty:
        if not ms1_df.empty:
            valid_uids = ms1_df['mz_rt_uid'].unique()
            ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(valid_uids)]
            steps.append((f"remove orphan MS2", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))

    ldt.log_filter_table(steps, starting_scans, starting_uids, entries_label="Scans", title="MS2 point filtering summary")

    return ms2_df

def _filter_ms1_points(ms1_df, min_pts, min_int):
    """Filter wide-format MS1 DataFrame by minimum in-feature point count and intensity.

    Each row of *ms1_df* represents one compound x one file with list columns
    ``spec_rts``, ``spec_ints``, and ``in_feature``.

    Filtering is applied at the **compound level** (``mz_rt_uid``).  Each
    threshold is evaluated independently across all files for a compound:

    * ``min_pts`` — a compound qualifies for this threshold if **any** of its
      ``(mz_rt_uid, filename)`` rows has at least ``min_pts`` in-feature scan
      points.
    * ``min_int`` — a compound qualifies for this threshold if **any** of its
      rows has at least one in-feature point with intensity ≥ ``min_int``
      (equivalently, the per-row max in-feature intensity ≥ ``min_int``).

    Both active thresholds must be satisfied (AND), but each is checked
    independently across all files — a single file does not need to satisfy
    both simultaneously.

    When a compound qualifies, **all** of its rows (across every file) are
    retained, including files that individually fall below the thresholds.
    This prevents the situation where a compound with strong signal in one
    file appears to have zero signal in all other files simply because those
    files had lower (but real) intensities.

    Args:
        ms1_df:   Wide-format MS1 DataFrame (one row per compound x file).
        min_pts:  Minimum number of in-feature scan points required in at
                  least one file per compound.  ``None`` or ``0`` skips this
                  filter.
        min_int:  Minimum peak intensity threshold: a compound qualifies if
                  at least one in-feature point in any file reaches this
                  value.  ``None`` or ``0`` skips this filter.

    Returns:
        Filtered MS1 DataFrame with all rows for qualifying compounds retained.
    """
    if ms1_df.empty:
        logger.warning("No MS1 data found. Skipping point filtering.")
        return ms1_df
    if min_pts is None and min_int is None:
        logger.info(f"No MS1 point filters specified. Retaining all {len(ms1_df)} entries across {ms1_df['mz_rt_uid'].nunique()} compounds.")
        return ms1_df
    if min_pts == 0 and min_int == 0:
        logger.info(f"ms1_min_pts=0 and ms1_min_intensity=0: skipping MS1 point filter. Retaining all {len(ms1_df)} entries across {ms1_df['mz_rt_uid'].nunique()} compounds.")
        return ms1_df

    starting_compounds = ms1_df['mz_rt_uid'].nunique()
    starting_entries = len(ms1_df)

    # Collect per-step stats for the summary table: (step_label, entries_after, compounds_after)
    steps = [("extracted", starting_entries, starting_compounds)]

    # compute per-row in-feature point count and max intensity
    def _in_feature_count(in_feature_col):
        """Count True values in each row's in_feature list."""
        return in_feature_col.apply(
            lambda x: int(np.sum(x)) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
        )

    def _in_feature_max_int(in_feature_col, spec_ints_col):
        """Max intensity among in-feature points per row."""
        def _row_max(pair):
            mask, ints = pair
            if not isinstance(mask, (list, np.ndarray)) or not isinstance(ints, (list, np.ndarray)):
                return -float('inf')
            arr_mask = np.asarray(mask, dtype=bool)
            arr_ints = np.asarray(ints, dtype=float)
            in_f = arr_ints[arr_mask]
            return float(in_f.max()) if in_f.size > 0 else -float('inf')
        return pd.Series(
            [_row_max(pair) for pair in zip(in_feature_col, spec_ints_col)],
            index=in_feature_col.index,
        )

    passing_uids = set(ms1_df['mz_rt_uid'].unique())

    if min_pts is not None and min_pts > 0:
        counts = _in_feature_count(ms1_df['in_feature'])
        pts_pass_uids = set(ms1_df.loc[counts >= min_pts, 'mz_rt_uid'].unique())
        passing_uids = passing_uids & pts_pass_uids
        steps.append((f"min_pts >= {min_pts}", len(ms1_df[ms1_df['mz_rt_uid'].isin(passing_uids)]), len(passing_uids)))
        if not passing_uids:
            ms1_df = ms1_df.iloc[0:0]
            ldt.log_filter_table(steps, starting_entries, starting_compounds, title="MS1 point filtering summary")
            return ms1_df

    if min_int is not None and min_int > 0:
        max_ints = _in_feature_max_int(ms1_df['in_feature'], ms1_df['spec_ints'])
        int_pass_uids = set(ms1_df.loc[max_ints >= min_int, 'mz_rt_uid'].unique())
        passing_uids = passing_uids & int_pass_uids
        steps.append((f"min_intensity >= {min_int}", len(ms1_df[ms1_df['mz_rt_uid'].isin(passing_uids)]), len(passing_uids)))
        if not passing_uids:
            ms1_df = ms1_df.iloc[0:0]
            ldt.log_filter_table(steps, starting_entries, starting_compounds, title="MS1 point filtering summary")
            return ms1_df

    ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(passing_uids)]

    no_feature_mask = ~ms1_df["in_feature"].apply(lambda x: isinstance(x, list) and any(x))
    ms1_df = ms1_df[~no_feature_mask]
    steps.append(("any in-feature", len(ms1_df), ms1_df['mz_rt_uid'].nunique()))

    ldt.log_filter_table(steps, starting_entries, starting_compounds, title="MS1 point filtering summary")
    return ms1_df

def _ensure_in_feature_list_of_bools(df, col="in_feature"):
    """Coerce the *col* column to a list of Python bools in every row.

    DuckDB REAL[] columns are read back as numpy arrays; this helper
    normalises them to plain ``list[bool]`` so downstream code can use
    standard Python list operations.

    Args:
        df:  DataFrame to modify in-place.
        col: Name of the list column to coerce (default ``"in_feature"``).

    Returns:
        The same DataFrame with *col* coerced (modified in-place).
    """
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: [bool(i) for i in x] if isinstance(x, (list, np.ndarray)) else []
        ).astype(object)
    return df

def _join_metadata(ms1_df, ms2_df, atlas):
    """Merge compound metadata (inchi_key, adduct) onto MS1 and MS2 DataFrames.

    Performs a left join on ``mz_rt_uid`` and reorders MS1 columns to the
    canonical wide-format column order.

    Args:
        ms1_df: Wide-format MS1 DataFrame.
        ms2_df: Wide-format MS2 DataFrame.
        atlas:  Expanded atlas DataFrame containing ``mz_rt_uid``,
                ``inchi_key``, and ``adduct`` columns.

    Returns:
        Tuple ``(ms1_df, ms2_df)`` with metadata columns added.
    """
    meta_df = atlas[["mz_rt_uid", "inchi_key", "adduct"]]
    if not ms1_df.empty:
        ms1_df = ms1_df.merge(meta_df, on="mz_rt_uid", how="left")
        ms1_df = _ensure_in_feature_list_of_bools(ms1_df, "in_feature")
        ms1_columns_order = ['mz_rt_uid', 'filename', 'inchi_key', 'adduct', 'spec_rts', 'spec_ints', 'spec_mzs', 'in_feature']
        ms1_df = ms1_df.reindex(columns=ms1_columns_order)
    if not ms2_df.empty:
        ms2_df = ms2_df.merge(meta_df, on="mz_rt_uid", how="left")
    return ms1_df, ms2_df

_VALID_STAGES = frozenset({"rt_alignment", "auto_identification"})

def extract_data_from_raw(
    obj: "RTAlign" | "AutoIdentification",
    stage: str,
) -> None:
    """Extract MS1 and MS2 data from raw HDF5 files and attach to *obj*.

    Args:
        obj:   Either an :class:`RTAlign` or :class:`AutoIdentification`
               workflow object that has already been set up (atlas and
               lcmsruns populated).
        stage: ``"rt_alignment"`` or ``"auto_identification"``.  Passed
               explicitly so the function does not need to inspect *obj*
               with fragile ``hasattr`` checks.

    Raises:
        ValueError: If *stage* is not one of the recognised values.
        FileNotFoundError: If any raw HDF5 file is missing from disk.
    """
    from metatlas2.workflow_objects import ExperimentalData

    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Invalid stage {stage!r}. Expected one of: {sorted(_VALID_STAGES)}"
        )

    if stage == "rt_alignment":
        atlas = obj.align_atlas_obj
        lcmsruns = obj.aligner_lcmsruns
        wp = obj.rt_alignment_params
    else:
        atlas = obj.auto_ided_atlas_obj
        lcmsruns = obj.autoid_lcmsruns
        wp = obj.ta.params

    _POL_TO_H5 = {"pos": "positive", "neg": "negative"}
    canonical_pol = fpf.normalize_polarity(atlas.polarity)
    if canonical_pol not in _POL_TO_H5:
        raise ValueError(
            f"Atlas polarity '{atlas.polarity}' (canonical: '{canonical_pol}') is not supported "
            f"for data extraction. Expected 'pos' or 'neg' — the atlas must have a single polarity."
        )
    polarity = _POL_TO_H5[canonical_pol]

    used_params = [
        "atlas_extra_time", "extract_extra_time", "ms1_mz_tolerance_ppm",
        "only_keep_data_in_feature", "ms1_min_num_points", "ms1_min_peak_intensity",
        "ms2_min_num_scans", "ms2_min_precursor_intensity",
    ]
    logger.info("Running extraction with the following workflow parameters:")
    for k in used_params:
        if k in wp:
            logger.info(f"  {k}: {wp[k]}")

    atlas_df = atlas.to_dataframe()
    atlas_expanded = _expand_atlas_windows(
        atlas_df,
        wp.get("atlas_extra_time", 0.0),
        wp.get("ms1_mz_tolerance_ppm", 5.0),
        polarity,
        extract_extra_time=wp.get("extract_extra_time", None),
    )
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]

    # check that all files exist on disk before starting extraction
    missing_files = [r.file_path for r in runs if not Path(r.file_path).is_file()]
    if missing_files:
        logger.error("The following files are missing and cannot be processed:")
        for f in missing_files:
            logger.error(f"  {f}")
        raise FileNotFoundError(f"{len(missing_files)} files are missing. Is the conversion finished?.")

    logger.info(f"Extracting data for {len(runs)} files in stage '{stage}' with polarity '{polarity}'...")

    only_in_feature = wp.get("only_keep_data_in_feature", False)
    max_workers = get_max_workers(obj.config.max_workers if obj.config else None)
    logger.info(f"Using {max_workers} worker processes for data extraction.")
    ms1_chunks: list[pd.DataFrame] = []
    ms2_chunks: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_file, run, atlas_expanded, only_in_feature): run
            for run in runs
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting MS data",
            disable=should_disable_tqdm(),
        ):
            m1_long, m2_long = fut.result()
            w1 = _widen_one_file_ms1(m1_long)
            w2 = _widen_one_file_ms2(m2_long)
            del m1_long, m2_long
            if not w1.empty:
                ms1_chunks.append(w1)
            if not w2.empty:
                ms2_chunks.append(w2)

    final_ms1_df = pd.concat(ms1_chunks, ignore_index=True) if ms1_chunks else pd.DataFrame()
    final_ms2_df = pd.concat(ms2_chunks, ignore_index=True) if ms2_chunks else pd.DataFrame()
    del ms1_chunks, ms2_chunks

    # Sort MS1 list columns by RT (required for correct np.interp downstream)
    final_ms1_df = _sort_ms1_lists_by_rts(final_ms1_df)
    _log_ms_totals(final_ms1_df, final_ms2_df)

    # Filter by minimum number of points in MS1 and remove compounds with no MS1 points "in_feature"
    final_ms1_df = _filter_ms1_points(
        final_ms1_df,
        wp.get("ms1_min_num_points", None),
        wp.get("ms1_min_peak_intensity", None),
    )
    final_ms2_df = _filter_ms2_points(
        final_ms2_df,
        final_ms1_df,
        min_scans=wp.get("ms2_min_num_scans", None),
        min_int=wp.get("ms2_min_precursor_intensity", None),
    )

    final_ms1_df, final_ms2_df = _join_metadata(final_ms1_df, final_ms2_df, atlas_expanded)

    logger.info(f"Data extraction complete for stage '{stage}'.")
    logger.info(f"  MS1 compounds+files extracted: {len(final_ms1_df)}")
    logger.info(f"  MS2 compounds+files+scans extracted: {len(final_ms2_df)}")
    if not final_ms1_df.empty:
        logger.info(f"  Unique compounds (mz_rt_uid) in MS1: {final_ms1_df['mz_rt_uid'].nunique()}")
        logger.info(f"  Unique files in MS1: {final_ms1_df['filename'].nunique()}")
    if not final_ms2_df.empty:
        logger.info(f"  Unique compounds (mz_rt_uid) in MS2: {final_ms2_df['mz_rt_uid'].nunique()}")
        logger.info(f"  Unique files in MS2: {final_ms2_df['filename'].nunique()}")

    obj.experimental_data = ExperimentalData()
    obj.experimental_data.ms1_df = final_ms1_df
    obj.experimental_data.ms2_df = final_ms2_df

    return