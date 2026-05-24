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

_EMPTY_MS1 = pd.DataFrame(columns=["mz", "i", "rt"]).astype(np.float32)
_EMPTY_MS2 = pd.DataFrame(columns=["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy"]).astype(np.float32)

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

def _load_h5_table(file_path, key, columns=None, mz_bounds=None):
    read_key = key + "_mz" if mz_bounds is not None else key
    try:
        df = pd.read_hdf(file_path, key=read_key, columns=columns)
    except (KeyError, ValueError, OSError) as exc:
        logger.warning("Could not read key %s from %s: %s", read_key, file_path, exc)
        return pd.DataFrame()
    
    if df.empty:
        return df

    if mz_bounds is not None:
        mz_min, mz_max = mz_bounds
        mz = df["mz"].to_numpy()
        lo = np.searchsorted(mz, mz_min, side="left")
        hi = np.searchsorted(mz, mz_max, side="right")
        df = df.iloc[lo:hi]
    
    float_cols = df.select_dtypes(include=['float64']).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].astype(np.float32, copy=False)
    
    return df

def _expand_atlas_windows(atlas: pd.DataFrame, workflow_params: dict, polarity: str) -> pd.DataFrame:
    """
    Pre-compute mz_min/mz_max and rt_min_pad/rt_max_pad columns for the atlas
    so that the merge step can do simple interval lookups.
    """
    logger.info(
        "Expanding atlas windows for %d entries with extra_time=%s and ppm_error=%s",
        len(atlas),
        workflow_params.get("extra_time", 0.0),
        workflow_params.get("ppm_error", 5.0),
    )
    extra_time = float(workflow_params.get("extra_time", 0.0))
    mz_tolerance = float(workflow_params.get("ppm_error", 5.0))
    out = atlas.copy()
    out["polarity"] = polarity

    # Compute in float64 for precision, store as float32 to match scan data dtype.
    mz = out["mz"].to_numpy(dtype=np.float64)
    tol = mz * mz_tolerance * 1e-6
    out["mz_min"] = (mz - tol).astype(np.float32)
    out["mz_max"] = (mz + tol).astype(np.float32)
    out["rt_min_pad"] = (out["rt_min"].to_numpy(dtype=np.float64) - extra_time).astype(np.float32)
    out["rt_max_pad"] = (out["rt_max"].to_numpy(dtype=np.float64) + extra_time).astype(np.float32)
    return out

def _interval_join_mz(query_mz, atlas_mz_min, atlas_mz_max, chunk_size=50_000):
    """Find all (query, atlas) index pairs where a query m/z falls inside an
    atlas interval.

    Returns two parallel int64 arrays ``(query_idx, atlas_idx)`` such that
    ``atlas_mz_min[atlas_idx[k]] <= query_mz[query_idx[k]] <= atlas_mz_max[atlas_idx[k]]``
    for every k. Every containing pair is emitted, so overlapping atlas
    intervals (e.g. isomers) all receive their share of each matching query.

    Queries are internally sorted by m/z so that each processing chunk spans
    a narrow m/z window, bounding the per-chunk candidate set. This matters
    for inputs like MS2 precursor_MZ where consecutive scans can jump across
    the full m/z range. Output indices are remapped back to the caller's
    original query order.

    NaN values in ``query_mz`` are silently dropped (no atlas interval can
    contain a NaN). Callers that care should filter upstream.
    """
    n, m = len(atlas_mz_min), len(query_mz)
    if n == 0 or m == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    order_min = np.argsort(atlas_mz_min, kind="stable")
    sorted_min = atlas_mz_min[order_min]
    sorted_max = atlas_mz_max[order_min]

    # Sort queries by m/z so chunks span a narrow m/z window. argsort places
    # NaNs at the end; we slice them off so q.max()/q.min() are well-defined.
    mz_order = np.argsort(query_mz, kind="stable").astype(np.int64, copy=False)
    q_sorted_full = query_mz[mz_order]
    valid_count = int(np.count_nonzero(~np.isnan(q_sorted_full)))
    if valid_count == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    mz_order = mz_order[:valid_count]
    q_sorted = q_sorted_full[:valid_count]

    q_chunks_q, q_chunks_a = [], []
    for start in range(0, valid_count, chunk_size):
        stop = min(start + chunk_size, valid_count)
        q = q_sorted[start:stop]

        # Which intervals have opened by the max q in this chunk?
        # Which of those are still possibly live at the min q in this chunk?
        hi = np.searchsorted(sorted_min, q.max(), side="right")
        live_mask = sorted_max[:hi] >= q.min()
        live_pos = np.nonzero(live_mask)[0]
        if live_pos.size == 0:
            continue

        live_min = sorted_min[live_pos]
        live_max = sorted_max[live_pos]

        # For each q in chunk, candidates are live intervals with
        # live_min <= q <= live_max. Materialize candidate pairs.
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

def _join_ms1_to_atlas(
    ms1_df: pd.DataFrame,
    atlas: pd.DataFrame,
    keep_outside_of_feature: bool = True
) -> pd.DataFrame:
    if ms1_df.empty or atlas.empty:
        return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_feature"])

    scans = ms1_df[["mz", "rt", "i"]].reset_index(drop=True)
    atlas_r = atlas.reset_index(drop=True)

    scan_mz = scans["mz"].to_numpy()
    scan_rt = scans["rt"].to_numpy()
    scan_i = scans["i"].to_numpy()

    atlas_mz_min = atlas_r["mz_min"].to_numpy()
    atlas_mz_max = atlas_r["mz_max"].to_numpy()
    atlas_rt_min = atlas_r["rt_min_pad"].to_numpy()
    atlas_rt_max = atlas_r["rt_max_pad"].to_numpy()
    atlas_uid = atlas_r["mz_rt_uid"].to_numpy()

    q_idx, a_idx = _interval_join_mz(scan_mz, atlas_mz_min, atlas_mz_max)
    if len(q_idx) == 0:
        return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_window"])

    scan_rt_paired = scan_rt[q_idx]
    in_window = (scan_rt_paired >= atlas_rt_min[a_idx]) & (scan_rt_paired <= atlas_rt_max[a_idx])

    if not keep_outside_of_feature:
        q_idx = q_idx[in_window]
        a_idx = a_idx[in_window]
        if len(q_idx) == 0:
            return pd.DataFrame(columns=["mz", "rt", "i", "mz_rt_uid", "in_window"])
        in_window_val = np.ones(len(q_idx), dtype=bool)
    else:
        in_window_val = in_window

    return pd.DataFrame({
        "mz":        scan_mz[q_idx],
        "rt":        scan_rt[q_idx],
        "i":         scan_i[q_idx],
        "mz_rt_uid": atlas_uid[a_idx],
        "in_window": in_window_val,
    })

def _join_ms2_to_atlas(
    ms2_df: pd.DataFrame,
    atlas: pd.DataFrame,
) -> pd.DataFrame:
    if ms2_df.empty or atlas.empty:
        return pd.DataFrame()

    needed = ("mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy")
    keep_cols = [c for c in needed if c in ms2_df.columns]
    if "precursor_MZ" not in keep_cols or "rt" not in keep_cols:
        logger.warning("MS2 table missing precursor_MZ or rt; skipping")
        return pd.DataFrame()

    scans = ms2_df[keep_cols].reset_index(drop=True)
    precursor_raw = scans["precursor_MZ"].to_numpy()
    valid_precursor = ~np.isnan(precursor_raw) & (precursor_raw > 0)
    if not valid_precursor.all():
        scans = scans.loc[valid_precursor].reset_index(drop=True)
    if scans.empty:
        return pd.DataFrame()

    atlas_r = atlas.reset_index(drop=True)
    precursor_MZ = scans["precursor_MZ"].to_numpy()
    scan_rt = scans["rt"].to_numpy()
    atlas_mz_min = atlas_r["mz_min"].to_numpy()
    atlas_mz_max = atlas_r["mz_max"].to_numpy()
    atlas_rt_min = atlas_r["rt_min_pad"].to_numpy()
    atlas_rt_max = atlas_r["rt_max_pad"].to_numpy()
    atlas_uid = atlas_r["mz_rt_uid"].to_numpy()

    q_idx, a_idx = _interval_join_mz(precursor_MZ, atlas_mz_min, atlas_mz_max)
    if len(q_idx) == 0:
        return pd.DataFrame()

    scan_rt_paired = scan_rt[q_idx]
    in_feature = (scan_rt_paired >= atlas_rt_min[a_idx]) & (scan_rt_paired <= atlas_rt_max[a_idx])

    hits = scans.iloc[q_idx].reset_index(drop=True)
    hits["mz_rt_uid"] = atlas_uid[a_idx]
    hits["in_feature"] = in_feature
    return hits

def _process_one_file(
    run,
    atlas: pd.DataFrame,
    atlas_info: dict,
    remove_unided: bool = True,
    ms1_only: bool = False,
    keep_outside_of_feature: bool = True,
    ms1_min_pts=None, 
    ms1_min_int=None
) -> tuple[list, list]:
    """Load one .h5, extract MS1 + MS2 results for all atlas entries."""
    polarity = atlas['polarity'].iloc[0] if 'polarity' in atlas.columns else 'unknown'
    ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
    ms2_key = None if ms1_only else {"positive": "ms2_pos", "negative": "ms2_neg"}.get(polarity)
    if ms1_key is None:
        return [], []

    ms1_mz_lo = float(atlas["mz_min"].min())
    ms1_mz_hi = float(atlas["mz_max"].max())
    ms1_cols = ["mz", "rt", "i"]
    ms2_cols = ["mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy"]

    # Load
    ms1_df = _load_h5_table(run.file_path, ms1_key, columns=ms1_cols, mz_bounds=(ms1_mz_lo, ms1_mz_hi))
    ms2_df = _load_h5_table(run.file_path, ms2_key, columns=ms2_cols) if ms2_key else _EMPTY_MS2

    # Join
    ms1_hits = _join_ms1_to_atlas(ms1_df, atlas, keep_outside_of_feature)
    ms2_hits = _join_ms2_to_atlas(ms2_df, atlas)

    # Build
    return _build_per_file_lists(atlas, ms1_hits, ms2_hits, atlas_info, run.filename, remove_unided, ms1_min_pts, ms1_min_int)

def _group_hits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty: return {}
    df = df.sort_values("rt", kind="mergesort")
    return {uid: g.drop(columns="mz_rt_uid").reset_index(drop=True) 
            for uid, g in df.groupby("mz_rt_uid", sort=False)}

def _build_per_file_lists(
    atlas: pd.DataFrame,
    ms1_hits: pd.DataFrame,
    ms2_hits: pd.DataFrame,
    atlas_info: dict,
    filename: str,
    remove_unided: bool = True,
    ms1_min_pts: int = None,
    ms1_min_int: float = None,
) -> tuple[list, list, dict]:
    from metatlas2.workflow_objects import MS1Data, MS2Data

    ms1_groups = _group_hits(ms1_hits)
    ms2_groups = _group_hits(ms2_hits)

    ms1_list, ms2_list = [], []

    mz_rt_uid_removed = {}
    for uid in atlas["mz_rt_uid"]:
        info = atlas_info.get(uid, {})
        ik, add = info.get("inchi_key", ""), info.get("adduct", "")
        
        # ms1 data and filter
        m1 = ms1_groups.get(uid)
        if m1 is not None:
            pts_ok = (ms1_min_pts is None) or (len(m1) >= ms1_min_pts)
            int_ok = (ms1_min_int is None) or (m1["i"].max() >= ms1_min_int)
            if pts_ok and int_ok:
                ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=m1))
            elif not remove_unided:
                ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=_EMPTY_MS1))
            else:
                if not pts_ok and not int_ok:
                    mz_rt_uid_removed[uid] = f"ms1 points {len(m1)} < {ms1_min_pts} and max intensity {m1['i'].max()} < {ms1_min_int}"
                elif not pts_ok:
                    mz_rt_uid_removed[uid] = f"ms1 points {len(m1)} < {ms1_min_pts}"
                elif not int_ok:
                    mz_rt_uid_removed[uid] = f"ms1 max intensity {m1['i'].max()} < {ms1_min_int}"
        else:
            if not remove_unided:
                ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=_EMPTY_MS1))
                ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=_EMPTY_MS2))
            else:
                mz_rt_uid_removed[uid] = "no ms1 data"
            continue

        # ms2 data
        m2 = ms2_groups.get(uid)
        if m2 is not None:
            ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=m2))
        elif not remove_unided:
            ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid, inchi_key=ik, adduct=add, data=_EMPTY_MS2))

    return ms1_list, ms2_list, mz_rt_uid_removed

def _export_removal_reasons(removal_records: list[tuple], output_file: Path = None):
    if removal_records:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'mz_rt_uid', 'removal_reason'])
            for row in removal_records:
                writer.writerow(row)

def _log_extraction_summary(exp_data, atlas_expanded,stage):
    logger.info("Extraction summary for stage '%s':", stage)
    logger.info("  Total MS1 entries extracted: %d", len(exp_data.ms1_data))
    logger.info("  Total MS2 entries extracted: %d", len(exp_data.ms2_data))

    # Total MS1 and MS2 data points (rows in all dataframes)
    total_ms1_points = sum(len(ms1.data) for ms1 in exp_data.ms1_data if ms1.data is not None)
    total_ms2_points = sum(len(ms2.data) for ms2 in exp_data.ms2_data if ms2.data is not None)
    logger.info("  Total MS1 data points (all files, all compounds): %d", total_ms1_points)
    logger.info("  Total MS2 data points (all files, all compounds): %d", total_ms2_points)

    # Compounds with any MS1 or MS2 data (non-empty)
    ms1_compounds = set(ms1.mz_rt_uid for ms1 in exp_data.ms1_data if ms1.data is not None and not ms1.data.empty)
    ms2_compounds = set(ms2.mz_rt_uid for ms2 in exp_data.ms2_data if ms2.data is not None and not ms2.data.empty)
    compounds_with_data = ms1_compounds.union(ms2_compounds)
    logger.info("  Total compounds with any MS1 or MS2 data: %d/%d", len(compounds_with_data), len(atlas_expanded["mz_rt_uid"]))

def extract_data_from_raw(
    obj: "RTAlignment" or "AutoIdentification",
    stage: str,
) -> "ExperimentalData":
    """
    Extract MS1 and MS2 data for all atlas entries across a set of HDF5
    files, one worker per file.

    Parameters
    ----------

    obj: RTAlignment or AutoIdentification
        The workflow object containing the aligned atlas and the list of files to process.
    stage: str
        The workflow stage, either "rt_alignment" or "autoid", which determines if ms1 or ms1+ms2 data is extracted.
    max_workers
        Number of parallel worker processes. Defaults to `os.cpu_count()`.

    Returns
    -------
    ExperimentalData
        Object containing the extracted MS1 and MS2 data.
    """

    from metatlas2.workflow_objects import ExperimentalData

    # set up starting conditions from object
    atlas = obj.align_atlas_obj if stage == "rt_alignment" else obj.pre_autoid_atlas_obj
    polarity = "positive" if atlas.polarity.lower() == "pos" else "negative" if atlas.polarity.lower() == "neg" else atlas.polarity.lower()
    lcmsruns = obj.aligner_lcmsruns if stage == "rt_alignment" else obj.autoid_lcmsruns
    workflow_params = obj.rt_alignment_params if stage == "rt_alignment" else obj.workflow_params
    remove_unided = workflow_params.get("remove_unided_compounds", True)
    ms1_min_pts = workflow_params.get("ms1_min_num_points", None)
    ms1_min_int = workflow_params.get("ms1_min_peak_intensity", None)
    ms1_only = (stage == "rt_alignment")
    keep_outside_of_feature = (stage != "rt_alignment")
    logger.info(f"Extracting raw data based on atlas: {atlas.polarity}, {atlas.chromatography}, {atlas.analysis_type}")

    # create expanded atlas and get lcmsrun files
    atlas_df = atlas.to_dataframe()
    atlas_expanded = _expand_atlas_windows(atlas_df, workflow_params, polarity)
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]
    ms1_only = (stage == "rt_alignment")

    # compute Atlas dictionary once
    logger.info("Pre-computing atlas metadata dictionary...")
    atlas_info = atlas_expanded.set_index("mz_rt_uid")[["inchi_key", "adduct"]].to_dict(orient="index")

    worker_func = partial(
        _process_one_file, 
        atlas=atlas_expanded, 
        atlas_info=atlas_info, 
        remove_unided=remove_unided, 
        ms1_only=ms1_only,
        keep_outside_of_feature=keep_outside_of_feature,
        ms1_min_pts=ms1_min_pts,
        ms1_min_int=ms1_min_int
    )

    n_workers = min(mp.cpu_count(), len(runs), 10)
    ms1_all, ms2_all = [], []
    removal_records = [] 
    ctx = mp.get_context("spawn")

    def process_result(ms1, ms2, mz_rt_uid_removed, filename):
        ms1_all.extend(ms1)
        ms2_all.extend(ms2)
        if mz_rt_uid_removed:
            for uid, reason in mz_rt_uid_removed.items():
                removal_records.append((filename, uid, reason))
            output_dir = obj.paths.get("rt_alignment_output_dir" if stage == "rt_alignment" else "analysis_output_dir", None)
            if output_dir is not None:
                output_file = Path(output_dir) / f"{stage}_compound_removal_reasons.csv"
            else:
                raise ValueError(f"Output directory not specified in obj.paths: {obj.paths}")
            _export_removal_reasons(removal_records, output_file)

    if n_workers <= 10:  # process in serial
        for run in tqdm(runs, desc="Processing HDF5 files", disable=should_disable_tqdm()):
            ms1, ms2, mz_rt_uid_removed = worker_func(run)
            filename = getattr(run, 'filename', 'unknown')
            process_result(ms1, ms2, mz_rt_uid_removed, filename)
    else:  # process in parallel
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {ex.submit(worker_func, run): run for run in runs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing HDF5 files", disable=should_disable_tqdm()):
                ms1, ms2, mz_rt_uid_removed = fut.result()
                filename = getattr(futures[fut], 'filename', 'unknown')
                process_result(ms1, ms2, mz_rt_uid_removed, filename)

    exp_data = ExperimentalData()
    exp_data.ms1_data = ms1_all
    exp_data.ms2_data = ms2_all

    _log_extraction_summary(exp_data, atlas_expanded, stage)

    return exp_data