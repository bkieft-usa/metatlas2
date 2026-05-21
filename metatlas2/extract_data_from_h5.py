
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any
from tqdm.auto import tqdm
import time
import tracemalloc

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import metatlas2.logging_config as lcf
logger = lcf.get_logger('extract_data_from_h5')

def _load_h5_table(file_path: str, key: str) -> pd.DataFrame:
    """Load a single PyTables table from an HDF5 file; empty df if missing."""
    try:
        df = pd.read_hdf(file_path, key=key)
    except (KeyError, ValueError, OSError) as exc:
        logger.warning("Could not read key %s from %s: %s", key, file_path, exc)
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df

def _expand_atlas_windows(atlas: pd.DataFrame, workflow_params: dict) -> pd.DataFrame:
    """
    Pre-compute mz_min/mz_max and rt_min_pad/rt_max_pad columns for the atlas
    so that the merge step can do simple interval lookups.
    """
    extra_time = float(workflow_params.get("extra_time", 0.0))
    mz_tolerance = float(workflow_params.get("ppm_error", 5.0))
    out = atlas.copy()
    out["mz_min"] = out["mz"] - out["mz"] * mz_tolerance * 1e-6
    out["mz_max"] = out["mz"] + out["mz"] * mz_tolerance * 1e-6
    out["rt_min_pad"] = out["rt_min"] - extra_time
    out["rt_max_pad"] = out["rt_max"] + extra_time
    return out

def _filter_atlas_for_polarity(atlas: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Return only atlas rows matching the file's polarity (if column present)."""
    if "polarity" in atlas.columns:
        return atlas[atlas["polarity"] == polarity].reset_index(drop=True)
    return atlas

def _interval_join_mz(query_mz, atlas_mz_min, atlas_mz_max, chunk_size=50_000):
    """Find all (query, atlas) index pairs where a query m/z falls inside an
    atlas interval.

    Returns two parallel int64 arrays ``(query_idx, atlas_idx)`` such that
    ``atlas_mz_min[atlas_idx[k]] <= query_mz[query_idx[k]] <= atlas_mz_max[atlas_idx[k]]``
    for every k. Every containing pair is emitted, so overlapping atlas
    intervals (e.g. isomers) all receive their share of each matching query.

    Queries are internally sorted by m/z so that each processing chunk spans
    a narrow m/z window, bounding the per-chunk candidate set. This matters
    for inputs like MS2 precursor_mz where consecutive scans can jump across
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
        hi = np.searchsorted(sorted_min, q.max(), side="right")
        # Which of those are still possibly live at the min q in this chunk?
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

        # q_local[keep] indexes into the sorted chunk; remap to original
        # query indices via mz_order. cand_pos[keep] indexes into the live
        # arrays; remap to original atlas indices via live_pos then order_min.
        q_chunks_q.append(mz_order[start + q_local[keep]])
        q_chunks_a.append(order_min[live_pos[cand_pos[keep]]])

    if not q_chunks_q:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    return np.concatenate(q_chunks_q), np.concatenate(q_chunks_a)


def _join_ms1_to_atlas(
    ms1_df: pd.DataFrame,
    atlas: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Vectorized interval join of an MS1 scan table against all atlas entries.
    Returns {mz_rt_uid: subset_df}.

    Every scan point is reported against every atlas entry whose mz window
    contains it AND whose padded rt window contains it. Overlapping atlas
    windows (e.g. isomers sharing m/z) create shared data - a scan point
    inside K overlapping windows produces K hits.
    """
    if ms1_df.empty or atlas.empty:
        return {}

    scans = ms1_df[["mz", "rt", "i"]].reset_index(drop=True)
    atlas_r = atlas.reset_index(drop=True)

    scan_mz = scans["mz"].to_numpy()
    scan_rt = scans["rt"].to_numpy()
    scan_i = scans["i"].to_numpy()

    q_idx, a_idx = _interval_join_mz(
        scan_mz,
        atlas_r["mz_min"].to_numpy(),
        atlas_r["mz_max"].to_numpy(),
    )
    if len(q_idx) == 0:
        return {}

    # Apply the rt window filter on the candidate pairs.
    rt_min_pad = atlas_r["rt_min_pad"].to_numpy()[a_idx]
    rt_max_pad = atlas_r["rt_max_pad"].to_numpy()[a_idx]
    scan_rt_paired = scan_rt[q_idx]
    rt_mask = (scan_rt_paired >= rt_min_pad) & (scan_rt_paired <= rt_max_pad)

    if not rt_mask.any():
        return {}

    q_idx = q_idx[rt_mask]
    a_idx = a_idx[rt_mask]

    hits = pd.DataFrame({
        "mz":        scan_mz[q_idx],
        "rt":        scan_rt[q_idx],
        "i":         scan_i[q_idx],
        "mz_rt_uid": atlas_r["mz_rt_uid"].to_numpy()[a_idx],
    })

    return {uid: g.drop(columns="mz_rt_uid").reset_index(drop=True)
            for uid, g in hits.groupby("mz_rt_uid", sort=False)}


def _join_ms2_to_atlas(
    ms2_df: pd.DataFrame,
    atlas: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Vectorized interval join of an MS2 fragment table against all atlas
    entries, matched on precursor_mz vs. atlas mz window and filtered by
    the atlas rt window. Fragment rows inherit the precursor's compound
    assignment, so all (mz, i, precursor_mz/i, collision_energy) columns are
    preserved per assigned compound.
    """
    if ms2_df.empty or atlas.empty:
        return {}

    needed = ("mz", "i", "rt", "precursor_mz", "precursor_intensity",
              "collision_energy")
    keep_cols = [c for c in needed if c in ms2_df.columns]
    if "precursor_mz" not in keep_cols or "rt" not in keep_cols:
        logger.warning("MS2 table missing precursor_mz or rt; skipping")
        return {}

    scans = ms2_df[keep_cols].reset_index(drop=True)

    # Drop rows with no usable precursor assignment. Some vendors write NaN,
    # some write 0, when the instrument failed to isolate a precursor.
    precursor_raw = scans["precursor_mz"].to_numpy()
    valid_precursor = ~np.isnan(precursor_raw) & (precursor_raw > 0)
    if not valid_precursor.all():
        scans = scans.loc[valid_precursor].reset_index(drop=True)
    if scans.empty:
        return {}

    atlas_r = atlas.reset_index(drop=True)

    precursor_mz = scans["precursor_mz"].to_numpy()
    scan_rt = scans["rt"].to_numpy()

    q_idx, a_idx = _interval_join_mz(
        precursor_mz,
        atlas_r["mz_min"].to_numpy(),
        atlas_r["mz_max"].to_numpy(),
    )
    if len(q_idx) == 0:
        return {}

    rt_min_pad = atlas_r["rt_min_pad"].to_numpy()[a_idx]
    rt_max_pad = atlas_r["rt_max_pad"].to_numpy()[a_idx]
    scan_rt_paired = scan_rt[q_idx]
    rt_mask = (scan_rt_paired >= rt_min_pad) & (scan_rt_paired <= rt_max_pad)

    if not rt_mask.any():
        return {}

    q_idx = q_idx[rt_mask]
    a_idx = a_idx[rt_mask]

    # Build the result by taking the paired rows from `scans` and attaching the atlas assignment.
    hits = scans.iloc[q_idx].reset_index(drop=True)
    hits["mz_rt_uid"] = atlas_r["mz_rt_uid"].to_numpy()[a_idx]

    return {uid: g.drop(columns="mz_rt_uid").reset_index(drop=True)
            for uid, g in hits.groupby("mz_rt_uid", sort=False)}

# Per-file worker
def _process_one_file(
    run,
    atlas: pd.DataFrame,
    remove_unided: bool = True,
    ms1_only: bool = False,
) -> tuple[list["MS1Data"], list["MS2Data"]]:
    """Load one .h5, extract MS1 + MS2 results for all atlas entries."""
    t0 = time.perf_counter()
    m0, _ = tracemalloc.get_traced_memory()
    polarity = run.polarity
    atlas_pol = _filter_atlas_for_polarity(atlas, polarity)
    t1 = time.perf_counter()
    m1, _ = tracemalloc.get_traced_memory()
    if atlas_pol.empty:
        logger.info("No atlas entries for polarity=%s; skipping %s",
                    polarity, run.filename)
        return [], []

    if ms1_only:
        ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
        ms2_key = None
    else:
        ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
        ms2_key = {"positive": "ms2_pos", "negative": "ms2_neg"}.get(polarity)
    if ms1_key is None:
        logger.error("Unknown polarity %r on %s", polarity, run.file_path)
        return [], []

    t2 = time.perf_counter()
    m2, _ = tracemalloc.get_traced_memory()
    ms1_df = _load_h5_table(run.file_path, ms1_key)
    ms2_df = _load_h5_table(run.file_path, ms2_key)
    t3 = time.perf_counter()
    m3, _ = tracemalloc.get_traced_memory()

    ms1_hits = _join_ms1_to_atlas(ms1_df, atlas_pol)
    ms2_hits = _join_ms2_to_atlas(ms2_df, atlas_pol)
    t4 = time.perf_counter()
    m4, _ = tracemalloc.get_traced_memory()

    ms1_results, ms2_results = _build_per_file_lists(
        atlas_pol, ms1_hits, ms2_hits, run.filename, remove_unided
    )
    t5 = time.perf_counter()
    m5, _ = tracemalloc.get_traced_memory()

    logger.info("%s timings (s): filter_atlas: %.2f, load_tables: %.2f, join: %.2f, build_lists: %.2f", run.filename, t1-t0, t3-t2, t4-t3, t5-t4)
    logger.info("%s memory (kB): start: %.1f, filter_atlas: %.1f, load_tables: %.1f, join: %.1f, build_lists: %.1f",
                 run.filename, m0/1024, m1/1024, m2/1024, m3/1024, m4/1024, m5/1024)
    logger.info("%s: %d MS1 entries (%d non-empty), %d MS2 entries (%d non-empty)",
                 run.filename,
                 len(ms1_results), sum(1 for r in ms1_results if not r.data.empty),
                 len(ms2_results), sum(1 for r in ms2_results if not r.data.empty))
    return ms1_results, ms2_results

def _build_per_file_lists(
    atlas: pd.DataFrame,
    ms1_by_uid: dict[str, pd.DataFrame],
    ms2_by_uid: dict[str, pd.DataFrame],
    filename: str,
    remove_unided: bool = True,
) -> tuple[list["MS1Data"], list["MS2Data"]]:
    """
    Build MS1Data / MS2Data objects for this file.

    Add empty dfs if an atlas entry has no hits but all atlas compound should be kept

    In both cases, non-empty frames are sorted by ascending rt.
    """

    from metatlas2.workflow_objects import MS1Data, MS2Data

    ms1_list: list[MS1Data] = []
    ms2_list: list[MS2Data] = []

    # Build lookup for inchi_key and adduct by mz_rt_uid
    mzrt_info = atlas.set_index("mz_rt_uid")[["inchi_key", "adduct"]].to_dict(orient="index")

    for uid in tqdm(atlas["mz_rt_uid"], desc=f"Building per-file lists for {filename}"):
        info = mzrt_info.get(uid, {})
        inchi_key = info.get("inchi_key", "")
        adduct = info.get("adduct", "")

        ms1_df = ms1_by_uid.get(uid)
        if ms1_df is None or ms1_df.empty:
            if remove_unided:
                pass  # skip — don't emit an MS1Data for this uid
            else:
                ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid, inchi_key=inchi_key, adduct=adduct,
                                        data=_empty_ms1_frame()))
        else:
            ms1_df = ms1_df.sort_values("rt", kind="mergesort").reset_index(drop=True)
            ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid, inchi_key=inchi_key, adduct=adduct, data=ms1_df))

        ms2_df = ms2_by_uid.get(uid)
        if ms2_df is None or ms2_df.empty:
            if remove_unided:
                pass
            else:
                ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid, inchi_key=inchi_key, adduct=adduct,
                                        data=_empty_ms2_frame()))
        else:
            ms2_df = ms2_df.sort_values("rt", kind="mergesort").reset_index(drop=True)
            ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid, inchi_key=inchi_key, adduct=adduct, data=ms2_df))

    return ms1_list, ms2_list

def _empty_ms1_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "mz": pd.Series([], dtype="float64"),
        "i":  pd.Series([], dtype="float64"),
        "rt": pd.Series([], dtype="float64"),
    })

def _empty_ms2_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "mz":                 pd.Series([], dtype="float64"),
        "i":                  pd.Series([], dtype="float64"),
        "rt":                 pd.Series([], dtype="float64"),
        "precursor_mz":       pd.Series([], dtype="float64"),
        "precursor_intensity":pd.Series([], dtype="float64"),
        "collision_energy":   pd.Series([], dtype="float64"),
    })

# Top-level entry point
def extract_data_from_raw(
    obj: "RTAlignment" or "AutoIdentification",
    stage: str,
    max_workers: int | None = None,
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


    from metatlas2.workflow_objects import ExperimentalData, MS1Data, MS2Data
    # Start memory tracking
    tracemalloc.start()

    # set up starting conditions from object
    t0 = time.perf_counter()
    atlas = obj.align_atlas_obj if stage == "rt_alignment" else obj.pre_autoid_atlas_obj
    lcmsruns = obj.aligner_lcmsruns if stage == "rt_alignment" else obj.autoid_lcmsruns
    workflow_params = obj.rt_alignment_params if stage == "rt_alignment" else obj.workflow_params
    remove_unided = workflow_params.get("remove_unided_compounds", True)
    ms1_only = True if stage == "rt_alignment" else False
    t1 = time.perf_counter()

    # create expanded atlas and get lcmsrun files
    atlas_df = atlas.compound_mzrts.to_dataframe().reset_index(drop=True)
    atlas_expanded = _expand_atlas_windows(atlas_df, workflow_params)
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]
    t2 = time.perf_counter()

    # set up containers
    experimental_data_obj = ExperimentalData()
    ms1_all: list[MS1Data] = []
    ms2_all: list[MS2Data] = []

    n_workers = max_workers or mp.cpu_count()
    n_workers = max(1, min(n_workers, len(runs)))
    t3 = time.perf_counter()
    if n_workers == 1:
        worker = partial(_process_one_file, atlas=atlas_expanded, remove_unided=remove_unided, ms1_only=ms1_only)
        for run in tqdm(runs, desc="Processing HDF5 files"):
            ms1, ms2 = worker(run)
            ms1_all.extend(ms1)
            ms2_all.extend(ms2)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_process_one_file, run, atlas=atlas_expanded, remove_unided=remove_unided, ms1_only=ms1_only): run for run in runs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing HDF5 files (parallel)"):
                ms1, ms2 = fut.result()
                ms1_all.extend(ms1)
                ms2_all.extend(ms2)
    t4 = time.perf_counter()

    logger.info("Step timings (seconds): parse input: %.2f, expand atlas: %.2f, setup: %.2f, process files: %.2f", t1-t0, t2-t1, t3-t2, t4-t3)
    logger.info("Extracted %d MS1 and %d MS2 result groups from %d files",
                len(ms1_all), len(ms2_all), len(runs))

    experimental_data_obj.ms1_data = ms1_all
    experimental_data_obj.ms2_data = ms2_all

    return experimental_data_obj