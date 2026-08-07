from __future__ import annotations

import pandas as pd
import numpy as np
import os
import sys
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import metatlas2.logging_config as lcf
import metatlas2.load_tools as ldt
from metatlas2.utils import should_disable_tqdm
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

def _expand_atlas_windows(atlas: pd.DataFrame, extra_time: float, ms1_mz_tolerance_ppm: float, polarity: str) -> pd.DataFrame:
    """Add padded m/z and RT bound columns to the atlas DataFrame.

    Computes ``mz_min``/``mz_max`` from the ppm tolerance and
    ``rt_min_pad``/``rt_max_pad`` by subtracting/adding *extra_time* to the
    atlas RT bounds.  These columns are consumed by the interval-join helpers.

    Args:
        atlas:                 Atlas compound DataFrame (must have ``mz``,
                               ``rt_min``, ``rt_max`` columns).
        extra_time:            Extra time (minutes) added to each RT window.
        ms1_mz_tolerance_ppm:  m/z tolerance in ppm used to compute the
                               ``mz_min``/``mz_max`` search window.
        polarity:              Polarity string (``"positive"`` or
                               ``"negative"``) written into the output.

    Returns:
        Copy of *atlas* with ``polarity``, ``mz_min``, ``mz_max``,
        ``rt_min_pad``, and ``rt_max_pad`` columns added.
    """
    logger.info(
        f"Expanding atlas windows for {len(atlas)} compounds with extra_time={extra_time} and mz_tolerance_ppm={ms1_mz_tolerance_ppm}"
    )

    out = atlas.copy()
    out["polarity"] = polarity
    mz = out["mz"].to_numpy(dtype=np.float64)
    tol = mz * ms1_mz_tolerance_ppm * 1e-6
    out["mz_min"] = (mz - tol).astype(np.float32)
    out["mz_max"] = (mz + tol).astype(np.float32)
    out["rt_min_pad"] = (out["rt_min"].to_numpy(dtype=np.float64) - extra_time).astype(np.float32)
    out["rt_max_pad"] = (out["rt_max"].to_numpy(dtype=np.float64) + extra_time).astype(np.float32)
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

def _widen_ms_data(df, type_name, group_cols, list_cols):
    """Aggregate long-format MS data into wide format (one row per group).

    Groups *df* by *group_cols* and collects *list_cols* values into Python
    lists.  Non-list metadata columns are taken from the first row of each
    group.  Column renaming and fragment sorting are applied after aggregation.

    Args:
        df:         Long-format MS DataFrame (output of the join helpers).
        type_name:  ``"ms1"`` or ``"ms2"`` — controls column renaming and
                    logging labels.
        group_cols: Columns to group by (e.g. ``["mz_rt_uid", "filename"]``).
        list_cols:  Columns whose values should be collected into lists.

    Returns:
        Wide-format DataFrame, or the original empty DataFrame unchanged.
    """
    if df.empty:
        return df

    _total = len(df) if not df.empty else 0
    _in_feat = len(df[df['in_feature']]) if not df.empty else 0
    _pct = _in_feat / max(_total, 1) * 100
    logger.info(f"Total {type_name} data points: {_total}")
    logger.info(f"Total {type_name} data points in atlas feature windows: {_in_feat} ({_pct:.2f}%)")

    logger.info(f"Aggregating {type_name} data to wide format by {group_cols}...")
    agg_dict = {col: list for col in list_cols}
    if type_name == "ms2" and "in_feature" in list_cols:
        agg_dict["in_feature"] = lambda x: bool(x.iloc[0]) if len(x) > 0 else False

    wide = df.groupby(group_cols).agg(agg_dict).reset_index()
    meta_cols = [c for c in df.columns if c not in group_cols + list_cols]
    if meta_cols:
        meta = df.groupby(group_cols)[meta_cols].first().reset_index()
        wide = wide.merge(meta, on=group_cols, how='left')

    _grain = "feature" if type_name == "ms1" else "scan"
    _suffix = "+scan" if type_name == "ms2" else ""
    logger.info(f"Aggregated {type_name} spectral data to {len(wide)} unique {_grain} compound+file{_suffix} entries.")
    logger.info(f"  Unique files: {wide['filename'].nunique()}")
    logger.info(f"  Unique compounds (mz_rt_uid): {wide['mz_rt_uid'].nunique()}")

    if type_name == "ms1":
        wide = wide.rename(columns={'mz': 'spec_mzs', 'i': 'spec_ints', 'rt': 'spec_rts'})
    elif type_name == "ms2":
        wide = wide.rename(columns={'rt': 'scan_rt','mz': 'frag_mzs', 'i': 'frag_ints'})
        wide = _sort_frags(wide)
        
    return wide

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
    ``spec_rts``, ``spec_ints``, and ``in_feature``.  Rows are kept only when
    the in-feature subset meets both thresholds.  Uses vectorised numpy helpers
    to avoid slow Python-level ``apply(lambda row: ...)`` loops.

    Args:
        ms1_df:   Wide-format MS1 DataFrame (one row per compound x file).
        min_pts:  Minimum number of in-feature scan points required per row.
                  ``None`` or ``0`` skips this filter.
        min_int:  Minimum peak intensity (max of in-feature intensities) required
                  per row.  ``None`` or ``0`` skips this filter.

    Returns:
        Filtered MS1 DataFrame.
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

    if min_pts is not None and min_pts > 0:
        counts = _in_feature_count(ms1_df['in_feature'])
        ms1_df = ms1_df[counts >= min_pts]
        steps.append((f"min_pts >= {min_pts}", len(ms1_df), ms1_df['mz_rt_uid'].nunique()))
        if ms1_df.empty:
            ldt.log_filter_table(steps, starting_entries, starting_compounds, title="MS1 point filtering summary")
            return ms1_df

    if min_int is not None and min_int > 0:
        max_ints = _in_feature_max_int(ms1_df['in_feature'], ms1_df['spec_ints'])
        ms1_df = ms1_df[max_ints >= min_int]
        steps.append((f"min_intensity >= {min_int}", len(ms1_df), ms1_df['mz_rt_uid'].nunique()))
        if ms1_df.empty:
            ldt.log_filter_table(steps, starting_entries, starting_compounds, title="MS1 point filtering summary")
            return ms1_df

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

    polarity = "positive" if atlas.polarity.lower() == "pos" else "negative" if atlas.polarity.lower() == "neg" else atlas.polarity.lower()
    
    used_params = [
        "atlas_extra_time", "ms1_mz_tolerance_ppm", "only_keep_data_in_feature",
        "ms1_min_num_points", "ms1_min_peak_intensity",
        "ms2_min_num_scans", "ms2_min_precursor_intensity",
    ]
    logger.info("Running extraction with the following workflow parameters:")
    for k in used_params:
        if k in wp:
            logger.info(f"  {k}: {wp[k]}")

    atlas_df = atlas.to_dataframe()
    atlas_expanded = _expand_atlas_windows(atlas_df, wp.get("atlas_extra_time", 0.0), wp.get("ms1_mz_tolerance_ppm", 5.0), polarity)
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]

    # check that all files exist on disk before starting extraction
    missing_files = [r.file_path for r in runs if not Path(r.file_path).is_file()]
    if missing_files:
        logger.error("The following files are missing and cannot be processed:")
        for f in missing_files:
            logger.error(f"  {f}")
        raise FileNotFoundError(f"{len(missing_files)} files are missing. Is the conversion finished?.")
    
    logger.info(f"Extracting data for {len(runs)} files in stage '{stage}' with polarity '{polarity}'...")

    # 1. Parallel Extraction (Purely loading and tagging)
    ms1_all, ms2_all = [], []
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 10)) as executor:
        futures = {executor.submit(_process_one_file, 
                                   run, 
                                   atlas_expanded, 
                                   wp.get("only_keep_data_in_feature", False)
                                ): run for run in runs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting MS data", disable=should_disable_tqdm()):
            m1, m2 = fut.result()
            if not m1.empty: 
                ms1_all.append(m1)
            if not m2.empty: 
                ms2_all.append(m2)

    final_ms1_df = pd.concat(ms1_all, ignore_index=True) if ms1_all else pd.DataFrame()
    final_ms2_df = pd.concat(ms2_all, ignore_index=True) if ms2_all else pd.DataFrame()

    # widen the dataframes from long
    final_ms1_df = _widen_ms_data(final_ms1_df, "ms1", ['mz_rt_uid', 'filename'], ['rt', 'i', 'mz', 'in_feature'])
    final_ms1_df = _sort_ms1_lists_by_rts(final_ms1_df)
    final_ms2_df = _widen_ms_data(final_ms2_df, "ms2", ['mz_rt_uid', 'filename', 'rt'], ['mz', 'i', 'in_feature'])

    # filter by minimum number of points in MS1 and remove compounds with no MS1 points "in_feature"
    final_ms1_df = _filter_ms1_points(
        final_ms1_df, 
        wp.get("ms1_min_num_points", None), 
        wp.get("ms1_min_peak_intensity", None)
    )
    final_ms2_df = _filter_ms2_points(
        final_ms2_df,
        final_ms1_df,
        min_scans=wp.get("ms2_min_num_scans", None),
        min_int=wp.get("ms2_min_precursor_intensity", None),
    )

    # Final Metadata Join
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