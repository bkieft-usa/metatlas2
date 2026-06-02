import pandas as pd
import numpy as np
import os
import sys
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import metatlas2.logging_config as lcf
logger = lcf.get_logger('extract_data_from_h5')

def should_disable_tqdm():
    return ("SLURM_JOB_ID" in os.environ or not sys.stdout.isatty())

def _load_h5_table(file_path, key, columns=None, mz_bounds=None):
    read_key = key + "_mz" if mz_bounds is not None else key
    try:
        df = pd.read_hdf(file_path, key=read_key, columns=columns)
    except (KeyError, ValueError, OSError) as exc:
        logger.warning("Could not read key %s from %s: %s", read_key, file_path, exc)
        return pd.DataFrame()
    if df.empty:
        return df
    logger.debug("Loaded %d rows from %s key %s", len(df), file_path, read_key)
    if mz_bounds is not None:
        mz_min, mz_max = mz_bounds
        mz = df["mz"].to_numpy()
        lo = np.searchsorted(mz, mz_min, side="left")
        hi = np.searchsorted(mz, mz_max, side="right")
        df = df.iloc[lo:hi]
        logger.debug("Filtered to %d rows within atlas mz bounds (to remove out-of-scope data points) [%f, %f]", len(df), mz_min, mz_max)
    float_cols = df.select_dtypes(include=['float64']).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].astype(np.float32, copy=False)
    return df

def _expand_atlas_windows(atlas: pd.DataFrame, extra_time: float, mz_tolerance_ppm: float, polarity: str) -> pd.DataFrame:

    logger.info(
        "Expanding atlas windows for %d compounds with extra_time=%s and mz_tolerance_ppm=%s",
        len(atlas),
        extra_time,
        mz_tolerance_ppm,
    )

    out = atlas.copy()
    out["polarity"] = polarity
    mz = out["mz"].to_numpy(dtype=np.float64)
    tol = mz * mz_tolerance_ppm * 1e-6
    out["mz_min"] = (mz - tol).astype(np.float32)
    out["mz_max"] = (mz + tol).astype(np.float32)
    out["rt_min_pad"] = (out["rt_min"].to_numpy(dtype=np.float64) - extra_time).astype(np.float32)
    out["rt_max_pad"] = (out["rt_max"].to_numpy(dtype=np.float64) + extra_time).astype(np.float32)
    return out

def _interval_join_mz(query_mz, atlas_mz_min, atlas_mz_max, chunk_size=50_000):
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
        if live_pos.size == 0: continue
        live_min, live_max = sorted_min[live_pos], sorted_max[live_pos]
        n_open_local = np.searchsorted(live_min, q, side="right")
        total = int(n_open_local.sum())
        if total == 0: continue
        q_local = np.repeat(np.arange(q.size, dtype=np.int64), n_open_local)
        starts = np.zeros(q.size, dtype=np.int64)
        np.cumsum(n_open_local[:-1], out=starts[1:])
        cand_pos = np.arange(total, dtype=np.int64) - np.repeat(starts, n_open_local)
        keep = live_max[cand_pos] >= q[q_local]
        if not keep.any(): continue
        q_chunks_q.append(mz_order[start + q_local[keep]])
        q_chunks_a.append(order_min[live_pos[cand_pos[keep]]])
    return np.concatenate(q_chunks_q), np.concatenate(q_chunks_a)

def _join_ms1_to_atlas(ms1_df: pd.DataFrame, atlas: pd.DataFrame, only_in_feature: bool) -> pd.DataFrame:
    if ms1_df.empty or atlas.empty:
        return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_feature"])
    logger.debug("Joining %d MS1 points to %d atlas features", len(ms1_df), len(atlas))
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
        logger.debug("Filtered to %d MS1 points that are in features (only_in_feature=True)", len(q_idx))

    return pd.DataFrame({
        "mz": scans["mz"].to_numpy()[q_idx],
        "rt": scans["rt"].to_numpy()[q_idx],
        "i": scans["i"].to_numpy()[q_idx],
        "mz_rt_uid": atlas_r["mz_rt_uid"].to_numpy()[a_idx],
        "in_feature": in_feature,
    })

def _join_ms2_to_atlas(ms2_df: pd.DataFrame, atlas: pd.DataFrame, only_in_feature: bool) -> pd.DataFrame:
    if ms2_df.empty or atlas.empty:
        return pd.DataFrame(columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy", "mz_rt_uid", "in_feature"])
    logger.debug("Joining %d MS2 points to %d atlas features", len(ms2_df), len(atlas))
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
        logger.debug("Filtered to %d MS2 points that are in features (only_in_feature=True)", len(pts))

    return pts

def _process_one_file(run, atlas, only_in_feature):
    polarity = atlas['polarity'].iloc[0] if 'polarity' in atlas.columns else 'unknown'
    ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
    ms2_key = {"positive": "ms2_pos", "negative": "ms2_neg"}.get(polarity)
    
    ms1_df = _load_h5_table(run.file_path, ms1_key, columns=["mz", "rt", "i"], 
                            mz_bounds=(float(atlas["mz_min"].min()), float(atlas["mz_max"].max())))

    ms2_df = _load_h5_table(run.file_path, ms2_key, columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy"]) if ms2_key else pd.DataFrame()

    ms1_extracted = _join_ms1_to_atlas(ms1_df, atlas, only_in_feature)
    ms2_extracted = _join_ms2_to_atlas(ms2_df, atlas, only_in_feature)

    logger.debug("Extracted %d MS1 points and %d MS2 points for run %s", len(ms1_extracted), len(ms2_extracted), run.filename)
    if not ms1_extracted.empty:
        ms1_extracted["filename"] = run.filename
    if not ms2_extracted.empty:
        ms2_extracted["filename"] = run.filename

    return ms1_extracted, ms2_extracted

def _sort_frags(wide):
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
    if df.empty: 
        return df

    logger.info("Total %s data points: %d", type_name, len(df) if not df.empty else 0)
    logger.info("Total %s data points in atlas feature windows: %d (%.2f%%)", type_name, len(df[df['in_feature']]) if not df.empty else 0, (len(df[df['in_feature']]) if not df.empty else 0) / (len(df) if not df.empty else 1) * 100)

    logger.info("Aggregating %s data to wide format by %s...", type_name, group_cols)
    agg_dict = {col: list for col in list_cols}
    if type_name == "ms2" and "in_feature" in list_cols:
        agg_dict["in_feature"] = lambda x: bool(x.iloc[0]) if len(x) > 0 else False

    wide = df.groupby(group_cols).agg(agg_dict).reset_index()
    meta_cols = [c for c in df.columns if c not in group_cols + list_cols]
    if meta_cols:
        meta = df.groupby(group_cols)[meta_cols].first().reset_index()
        wide = wide.merge(meta, on=group_cols, how='left')

    logger.info("Aggregated %s spectral data to %d unique %s compound+file%s entries.", type_name, len(wide), "feature" if type_name=="ms1" else "scan", "+scan" if type_name=="ms2" else "")
    logger.info(f"  Unique files: {wide['filename'].nunique()}")
    logger.info(f"  Unique compounds (mz_rt_uid): {wide['mz_rt_uid'].nunique()}")

    if type_name == "ms1":
        wide = wide.rename(columns={'mz': 'spec_mzs', 'i': 'spec_ints', 'rt': 'spec_rts'})
    elif type_name == "ms2":
        wide = wide.rename(columns={'rt': 'scan_rt','mz': 'frag_mzs', 'i': 'frag_ints'})
        wide = _sort_frags(wide)
        
    return wide

def _filter_ms2_points(ms2_df):
    if ms2_df.empty:
        return ms2_df
    
    starting_scans = len(ms2_df)
    starting_uids = ms2_df['mz_rt_uid'].nunique()
    logger.info("Filtering out compounds from MS2 data where no scans have in_feature=True. Starting with %d scans across %d compounds.", starting_scans, starting_uids)
    
    # Group by mz_rt_uid and keep only compounds that have at least one scan with in_feature=True
    compounds_with_in_feature = ms2_df.groupby('mz_rt_uid')['in_feature'].apply(
        lambda x: any(val is True or val == True for val in x)
    )
    keep_uids = set(compounds_with_in_feature[compounds_with_in_feature].index)
    ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(keep_uids)].reset_index(drop=True)
    
    ending_scans = len(ms2_df)
    ending_uids = ms2_df['mz_rt_uid'].nunique()
    scan_pct = 100.0 * ending_scans / starting_scans if starting_scans > 0 else 0.0
    uid_pct = 100.0 * ending_uids / starting_uids if starting_uids > 0 else 0.0
    logger.info(
        "MS2 data after filtering out compounds with no in-feature scans: %d scans remain (%.1f%% retained) across %d compounds (%.1f%% retained).",
        ending_scans, scan_pct, ending_uids, uid_pct
    )
    
    return ms2_df

def _filter_ms1_points(ms1_df, min_pts, min_int):
    if ms1_df.empty:
        return ms1_df
    
    starting_compounds = ms1_df['mz_rt_uid'].nunique()
    starting_entries = len(ms1_df)
    if min_pts is not None and min_pts > 0:
        logger.info(f"Filtering out compounds+files with fewer than {min_pts} points in their extracted ion chromatograms. Starting with {starting_entries} entries and {starting_compounds} unique compounds.")
        ms1_df = ms1_df[ms1_df.apply(lambda row: sum(1 for in_f, rt in zip(row['in_feature'], row['spec_rts']) if in_f) >= min_pts, axis=1)]
        pct_remaining_entries = (len(ms1_df) / starting_entries * 100) if starting_entries > 0 else 0.0
        pct_remaining_compounds = (ms1_df['mz_rt_uid'].nunique() / starting_compounds * 100) if starting_compounds > 0 else 0.0
        logger.info(f"  Completed 'points' filter. Ending with {len(ms1_df)} entries ({pct_remaining_entries:.2f}% of initial) and {ms1_df['mz_rt_uid'].nunique()} unique compounds ({pct_remaining_compounds:.2f}% of initial).")
        if ms1_df.empty:
            return ms1_df

    if min_int is not None and min_int > 0:
        logger.info(f"Filtering out compounds+files with peak intensity less than {min_int}. Starting with {len(ms1_df)} entries and {ms1_df['mz_rt_uid'].nunique()} unique compounds.")
        ms1_df = ms1_df[ms1_df.apply(lambda row: max((i for in_f, i in zip(row['in_feature'], row['spec_ints']) if in_f), default=-float('inf')) >= min_int, axis=1)]
        pct_remaining_entries = (len(ms1_df) / starting_entries * 100) if starting_entries > 0 else 0.0
        pct_remaining_compounds = (ms1_df['mz_rt_uid'].nunique() / starting_compounds * 100) if starting_compounds > 0 else 0.0
        logger.info(f"  Completed 'intensity' filter. Ending with {len(ms1_df)} entries ({pct_remaining_entries:.2f}% of initial) and {ms1_df['mz_rt_uid'].nunique()} unique compounds ({pct_remaining_compounds:.2f}% of initial).")
        if ms1_df.empty:
            return ms1_df
    
    logger.info(f"Filtering out compounds+files with no MS1 points in the feature. Starting with {len(ms1_df)} entries and {ms1_df['mz_rt_uid'].nunique()} unique compounds.")
    no_feature_mask = ~ms1_df["in_feature"].apply(lambda x: isinstance(x, list) and any(x))
    ms1_df = ms1_df[~(no_feature_mask)]
    pct_remaining_entries = (len(ms1_df) / starting_entries * 100) if starting_entries > 0 else 0.0
    pct_remaining_compounds = (ms1_df['mz_rt_uid'].nunique() / starting_compounds * 100) if starting_compounds > 0 else 0.0
    logger.info(f"  Completed 'in-feature' filter. Ending with {len(ms1_df)} entries ({pct_remaining_entries:.2f}% of initial) and {ms1_df['mz_rt_uid'].nunique()} unique compounds ({pct_remaining_compounds:.2f}% of initial).")

    return ms1_df

def _synchronize_ms2_with_ms1(ms1_df, ms2_df):
    if not ms2_df.empty:
        if not ms1_df.empty:
            logger.info(f"Synchronizing MS2 data with MS1 data to remove stranded MS2 points. Starting with {len(ms2_df)} MS2 entries and {ms2_df['mz_rt_uid'].nunique()} unique compounds.")
            valid_uids = ms1_df['mz_rt_uid'].unique()
            ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(valid_uids)].copy()
            logger.info(f"  Completed synchronization. Ending with {len(ms2_df)} MS2 entries and {ms2_df['mz_rt_uid'].nunique()} unique compounds.")
    return ms2_df

def _synchronize_ms1_with_ms2(ms1_df, ms2_df):
    if not ms1_df.empty:
        if not ms2_df.empty:
            logger.info(f"Synchronizing MS1 data with MS2 data to remove compounds with no MS2 hits that passed filters. Starting with {len(ms1_df)} MS1 entries and {ms1_df['mz_rt_uid'].nunique()} unique compounds.")
            valid_uids = ms2_df['mz_rt_uid'].unique()
            ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(valid_uids)].copy()
            logger.info(f"  Completed synchronization. Ending with {len(ms1_df)} MS1 entries and {ms1_df['mz_rt_uid'].nunique()} unique compounds.")
    return ms1_df

def _ensure_in_feature_list_of_bools(df, col="in_feature"):
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: [bool(i) for i in x] if isinstance(x, (list, np.ndarray)) else []
        ).astype(object)
    return df

def _join_metadata(ms1_df, ms2_df, atlas):
    meta_df = atlas[["mz_rt_uid", "inchi_key", "adduct"]]
    if not ms1_df.empty:
        ms1_df = ms1_df.merge(meta_df, on="mz_rt_uid", how="left")
        ms1_df = _ensure_in_feature_list_of_bools(ms1_df, "in_feature")
        ms1_columns_order = ['mz_rt_uid', 'filename', 'inchi_key', 'adduct', 'spec_rts', 'spec_ints', 'spec_mzs', 'in_feature']
        ms1_df = ms1_df.reindex(columns=ms1_columns_order)
    if not ms2_df.empty:
        ms2_df = ms2_df.merge(meta_df, on="mz_rt_uid", how="left")
    return ms1_df, ms2_df

def extract_data_from_raw(obj):
    from metatlas2.workflow_objects import ExperimentalData
    
    stage = "rt_alignment" if hasattr(obj, "rt_alignment_params") else "auto_identification"
    atlas = obj.align_atlas_obj if stage == "rt_alignment" else obj.pre_autoid_atlas_obj
    polarity = "positive" if atlas.polarity.lower() == "pos" else "negative" if atlas.polarity.lower() == "neg" else atlas.polarity.lower()
    lcmsruns = obj.aligner_lcmsruns if stage == "rt_alignment" else obj.autoid_lcmsruns
    wp = obj.rt_alignment_params if stage == "rt_alignment" else obj.workflow_params
    
    used_params = [
        "extra_time", "mz_tolerance_ppm", "only_keep_data_in_feature",
        "ms1_min_num_points", "ms1_min_peak_intensity"
    ]
    logger.info("Running extraction with the following workflow parameters (used in this script):")
    for k in used_params:
        if k in wp:
            logger.info(f"  {k}: {wp[k]}")

    atlas_df = atlas.to_dataframe()
    atlas_expanded = _expand_atlas_windows(atlas_df, wp.get("extra_time", 0.0), wp.get("mz_tolerance_ppm", 5.0), polarity)
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]
    
    logger.info(f"Extracting data for {len(runs)} files in stage '{stage}' with polarity '{polarity}'...")

    # 1. Parallel Extraction (Purely loading and tagging)
    ms1_all, ms2_all = [], []
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 10)) as executor:
        futures = {executor.submit(_process_one_file, 
                                   run, 
                                   atlas_expanded, 
                                   wp.get("only_keep_data_in_feature", False)
                                ): run for run in runs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting MS data"):
            m1, m2 = fut.result()
            if not m1.empty: ms1_all.append(m1)
            if not m2.empty: ms2_all.append(m2)

    final_ms1_df = pd.concat(ms1_all, ignore_index=True) if ms1_all else pd.DataFrame()
    final_ms2_df = pd.concat(ms2_all, ignore_index=True) if ms2_all else pd.DataFrame()

    # widen the dataframes from long
    final_ms1_df = _widen_ms_data(final_ms1_df, "ms1", ['mz_rt_uid', 'filename'], ['rt', 'i', 'mz', 'in_feature'])
    final_ms1_df = _sort_ms1_lists_by_rts(final_ms1_df)
    final_ms2_df = _widen_ms_data(final_ms2_df, "ms2", ['mz_rt_uid', 'filename', 'rt'], ['mz', 'i', 'in_feature'])

    # filter by minimum number of points in MS1 and remove compounds with no MS1 points "in_feature"
    final_ms1_df = _filter_ms1_points(final_ms1_df, wp.get("ms1_min_num_points", None), wp.get("ms1_min_peak_intensity", None))
    final_ms2_df = _filter_ms2_points(final_ms2_df)

    # remove any "stranded" ms2 data points that don't have a corresponding ms1 feature
    final_ms2_df = _synchronize_ms2_with_ms1(final_ms1_df, final_ms2_df)
    final_ms1_df = _synchronize_ms1_with_ms2(final_ms1_df, final_ms2_df)

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

