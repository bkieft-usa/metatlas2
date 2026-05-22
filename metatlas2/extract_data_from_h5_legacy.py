"""
Legacy-algorithm H5 extractor.

Drop-in replacement for the vectorized interval-join extractor. Public entry
point `extract_from_h5` and the per-file worker contract are identical; the
MS1/MS2 extraction core uses the original metatlas algorithm:

    1. group_consecutive  - bucket atlas m/z by ppm-adjacent neighborhoods
    2. map_mzgroups_to_data - nearest-neighbor assign scan m/z to a group
    3. outer-merge atlas + scans on group_index, then ppm + rt filter

Inputs (workflow object, stage), outputs (ExperimentalData with MS1Data /
MS2Data lists), and side effects (logging, tqdm bars) all match the
reference implementation.
"""

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm.auto import tqdm

import metatlas2.logging_config as lcf
logger = lcf.get_logger('extract_data_from_h5_legacy')


# ---------------------------------------------------------------------------
# H5 IO (verbatim behavior of the reference loader)
# ---------------------------------------------------------------------------

def _load_h5_table(file_path: str, key: str) -> pd.DataFrame:
    """Load a single PyTables table; empty df if missing or unreadable."""
    if key is None:
        return pd.DataFrame()
    try:
        df = pd.read_hdf(file_path, key=key)
    except (KeyError, ValueError, OSError) as exc:
        logger.warning("Could not read key %s from %s: %s", key, file_path, exc)
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


# ---------------------------------------------------------------------------
# Atlas prep — identical signatures to the reference
# ---------------------------------------------------------------------------

def _expand_atlas_windows(atlas: pd.DataFrame, workflow_params: dict, polarity: str) -> pd.DataFrame:
    """Same as the reference: compute mz_min/mz_max and padded rt columns."""
    logger.info(f"Expanding atlas windows for {len(atlas)} entries with extra_time={workflow_params.get('extra_time', 0.0)} and ppm_error={workflow_params.get('ppm_error', 5.0)}")
    extra_time = float(workflow_params.get("extra_time", 0.0))
    mz_tolerance = float(workflow_params.get("ppm_error", 5.0))
    out = atlas.copy()
    out["polarity"] = polarity
    out["mz_min"] = out["mz"] - out["mz"] * mz_tolerance * 1e-6
    out["mz_max"] = out["mz"] + out["mz"] * mz_tolerance * 1e-6
    out["rt_min_pad"] = out["rt_min"] - extra_time
    out["rt_max_pad"] = out["rt_max"] + extra_time
    # Stash the ppm tolerance so the legacy filter can use it directly,
    # avoiding any drift between mz_min/mz_max width and the ppm test.
    out["_ppm_tolerance"] = mz_tolerance
    return out

# ---------------------------------------------------------------------------
# Legacy primitives (verbatim)
# ---------------------------------------------------------------------------

def _group_consecutive(data: np.ndarray, stepsize: float = 10.0, do_ppm: bool = True) -> np.ndarray:
    """Bucket sorted values such that neighbors closer than 2*stepsize join a group."""
    if not isinstance(data, np.ndarray):
        raise TypeError("group_consecutive requires a numpy array")

    idx_sorted = data.argsort()
    sort_w_unsort = np.column_stack((np.arange(idx_sorted.size), idx_sorted))
    data_sorted = data[sort_w_unsort[:, 1]]

    if do_ppm:
        d = np.diff(data_sorted) / data_sorted[:-1] * 1e6
    else:
        d = np.diff(data_sorted)

    data_groups = np.split(data_sorted, np.where(d > 2.0 * stepsize)[0] + 1)
    for i, _ in enumerate(data_groups):
        data_groups[i] = data_groups[i] * 0 + i
    group_indices = np.concatenate(data_groups)
    group_indices = group_indices[np.argsort(sort_w_unsort[:, 1])]
    return group_indices.astype(int)


def _map_mzgroups_to_data(
    mz_atlas: np.ndarray,
    mz_group_indices: np.ndarray,
    mz_data: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor map each scan m/z to its closest atlas row's group."""
    f = interpolate.interp1d(
        mz_atlas,
        np.arange(mz_atlas.size),
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    idx = f(mz_data).astype("int")
    return mz_group_indices[idx]


def _build_legacy_atlas(atlas_pol: pd.DataFrame) -> pd.DataFrame:
    """
    Translate the expanded new-schema atlas into the columns the legacy
    filter expects: mz, rt_min, rt_max, ppm_tolerance, extra_time, label,
    group_index. Padding is folded into rt_min/rt_max with extra_time=0 so
    the legacy filter doesn't pad a second time.
    """
    a = atlas_pol.reset_index(drop=True).copy()
    # `mz` already exists in the new atlas; just use it directly so the
    # group bucketing operates on exactly the values the new pipeline uses
    # to build mz_min/mz_max.
    a["rt_min"] = a["rt_min_pad"]
    a["rt_max"] = a["rt_max_pad"]
    a["extra_time"] = 0.0
    a["ppm_tolerance"] = a["_ppm_tolerance"]
    a["label"] = a["mz_rt_uid"]
    a["group_index"] = _group_consecutive(
        a["mz"].to_numpy(),
        stepsize=float(a["_ppm_tolerance"].iloc[0]),
        do_ppm=True,
    )
    return a


def _legacy_filter(atlas_legacy: pd.DataFrame, msdata: pd.DataFrame) -> pd.DataFrame:
    """
    The original outer-merge + ppm/rt filter. `msdata` must already carry a
    `group_index` column produced by _map_mzgroups_to_data.
    """
    merged = pd.merge(
        atlas_legacy, msdata,
        left_on="group_index", right_on="group_index",
        how="outer", suffixes=("_atlas", "_data"),
    )
    #logger.info("[LEGACY] _legacy_filter merged candidate pairs: %d", len(merged))

    mz_cond = (
        np.abs(merged["mz_data"] - merged["mz_atlas"])
        / merged["mz_atlas"] * 1e6
        < merged["ppm_tolerance"]
    )
    rt_min_cond = merged["rt"] >= (merged["rt_min"] - merged["extra_time"])
    rt_max_cond = merged["rt"] <= (merged["rt_max"] + merged["extra_time"])

    out = merged[mz_cond & rt_min_cond & rt_max_cond].reset_index(drop=True)
    return out.rename(columns={"mz_data": "mz"})


# ---------------------------------------------------------------------------
# MS1 / MS2 joins — same signatures and return shapes as the reference
# ---------------------------------------------------------------------------

def _join_ms1_to_atlas(
    ms1_df: pd.DataFrame,
    atlas: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Legacy MS1 join. Returns {mz_rt_uid: DataFrame[mz, rt, i]}."""
    if ms1_df.empty or atlas.empty:
        return {}
    if not {"mz", "rt", "i"}.issubset(ms1_df.columns):
        logger.warning("MS1 table missing required columns; skipping")
        return {}

    import time
    t0 = time.perf_counter()

    atlas_legacy = _build_legacy_atlas(atlas)

    msdata = ms1_df[["mz", "rt", "i"]].copy()
    msdata["group_index"] = _map_mzgroups_to_data(
        atlas_legacy["mz"].to_numpy(),
        atlas_legacy["group_index"].to_numpy(),
        msdata["mz"].to_numpy(),
    )

    merged = _legacy_filter(atlas_legacy, msdata)
    t1 = time.perf_counter()
    #logger.info("[LEGACY] _join_ms1_to_atlas join time: %.4fs", t1-t0)
    if merged.empty:
        return {}

    out = merged[["label", "mz", "rt", "i"]]

    logger.info("Joined %d MS1 hits to %d atlas entries", len(out), len(atlas_legacy))
    return {
        uid: g.drop(columns="label").reset_index(drop=True)
        for uid, g in out.groupby("label", sort=False)
    }


def _join_ms2_to_atlas(
    ms2_df: pd.DataFrame,
    atlas: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Legacy MS2 join. Returns
    {mz_rt_uid: DataFrame[mz, i, rt, precursor_MZ, precursor_intensity,
                          collision_energy]}.

    The legacy trick: collapse fragments to one row per (rt) precursor for
    the m/z+rt filter, then re-attach fragment rows by rt. This keeps the
    outer-merge size proportional to #precursors, not #fragments.
    """
    if ms2_df.empty or atlas.empty:
        return {}

    import time
    t0 = time.perf_counter()

    needed = {"mz", "i", "rt", "precursor_MZ", "precursor_intensity", "collision_energy"}
    missing = needed - set(ms2_df.columns)
    if missing:
        logger.warning("MS2 table missing columns %s; skipping", missing)
        return {}

    # Drop rows with no usable precursor — same rule as the reference.
    precursor_raw = ms2_df["precursor_MZ"].to_numpy()
    valid = ~np.isnan(precursor_raw) & (precursor_raw > 0)
    if not valid.all():
        ms2_df = ms2_df.loc[valid].reset_index(drop=True)
    if ms2_df.empty:
        t1 = time.perf_counter()
        #logger.info("[LEGACY] _join_ms2_to_atlas join time: %.4fs (empty)", t1-t0)
        return {}

    atlas_legacy = _build_legacy_atlas(atlas)

    # Step 1: precursor-level dedup + legacy filter.
    precursors = (
        ms2_df[["rt", "precursor_MZ"]]
        .drop_duplicates("rt")
        .rename(columns={"precursor_MZ": "mz"})
        .reset_index(drop=True)
    )
    precursors["group_index"] = _map_mzgroups_to_data(
        atlas_legacy["mz"].to_numpy(),
        atlas_legacy["group_index"].to_numpy(),
        precursors["mz"].to_numpy(),
    )

    #logger.info("[LEGACY] _legacy_filter (MS2) precursor candidate pairs: %d", len(precursors))
    matched = _legacy_filter(atlas_legacy, precursors)
    t1 = time.perf_counter()
    #logger.info("[LEGACY] _join_ms2_to_atlas join time: %.4fs", t1-t0)
    if matched.empty:
        return {}

    precursor_hits = matched[["label", "rt", "mz"]].rename(columns={"mz": "precursor_MZ"})

    # Step 2: re-attach fragments by rt.
    fragments = ms2_df[["rt", "mz", "i", "precursor_intensity", "collision_energy"]]
    hits = pd.merge(precursor_hits, fragments, on="rt", how="inner")
    if hits.empty:
        return {}

    # Match the reference output column order.
    out = hits[[
        "label", "mz", "i", "rt",
        "precursor_MZ", "precursor_intensity", "collision_energy",
    ]]
    logger.info("Joined %d MS2 hits to %d atlas entries", len(out), len(atlas_legacy))
    return {
        uid: g.drop(columns="label").reset_index(drop=True)
        for uid, g in out.groupby("label", sort=False)
    }


# ---------------------------------------------------------------------------
# Empty-frame factories — identical schemas to the reference
# ---------------------------------------------------------------------------

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
        "precursor_MZ":       pd.Series([], dtype="float64"),
        "precursor_intensity":pd.Series([], dtype="float64"),
        "collision_energy":   pd.Series([], dtype="float64"),
    })


# ---------------------------------------------------------------------------
# Per-file worker — same signature, same return contract as the reference
# ---------------------------------------------------------------------------

def _process_one_file(
    run,
    atlas: pd.DataFrame,
    remove_unided: bool = True,
    ms1_only: bool = False,
) -> tuple[list["MS1Data"], list["MS2Data"]]:
    """Load one .h5, extract MS1 (+optionally MS2) for all atlas entries."""
    t0 = time.perf_counter()
    polarity = atlas['polarity'].iloc[0] if 'polarity' in atlas.columns else 'unknown'
    t1 = time.perf_counter()
    if atlas.empty:
        logger.info("No atlas entries for polarity=%s; skipping %s",
                    polarity, run.filename)
        return [], []

    ms1_key = {"positive": "ms1_pos", "negative": "ms1_neg"}.get(polarity)
    if ms1_only:
        ms2_key = None
    else:
        ms2_key = {"positive": "ms2_pos", "negative": "ms2_neg"}.get(polarity)
    if ms1_key is None:
        logger.error("Unknown polarity %r on %s", polarity, run.file_path)
        return [], []
    #logger.info("Processing %s: ms1_key=%s, ms2_key=%s", run.filename, ms1_key, ms2_key)

    t2 = time.perf_counter()
    ms1_df = _load_h5_table(run.file_path, ms1_key) if ms1_key else _empty_ms1_frame()
    ms2_df = _load_h5_table(run.file_path, ms2_key) if ms2_key else _empty_ms2_frame()
    if not ms1_key and ms2_key:
        logger.warning(f"MS1 key is missing but MS2 key is present for {run.file_path}, not expected behavior.")
    elif not ms1_key and not ms2_key:
        logger.warning(f"Both MS1 and MS2 keys are None for {run.file_path}, returning empty dataframes.")
    t3 = time.perf_counter()

    ms1_hits = _join_ms1_to_atlas(ms1_df, atlas)
    ms2_hits = _join_ms2_to_atlas(ms2_df, atlas) if not ms1_only else {}
    t4 = time.perf_counter()

    ms1_results, ms2_results = _build_per_file_lists(
        atlas, ms1_hits, ms2_hits, run.filename, remove_unided
    )
    t5 = time.perf_counter()

    return ms1_results, ms2_results


def _build_per_file_lists(
    atlas: pd.DataFrame,
    ms1_by_uid: dict[str, pd.DataFrame],
    ms2_by_uid: dict[str, pd.DataFrame],
    filename: str,
    remove_unided: bool = True,
) -> tuple[list["MS1Data"], list["MS2Data"]]:
    """Same construction logic as the reference."""
    from metatlas2.workflow_objects import MS1Data, MS2Data

    ms1_list: list[MS1Data] = []
    ms2_list: list[MS2Data] = []

    mzrt_info = atlas.set_index("mz_rt_uid")[["inchi_key", "adduct"]].to_dict(orient="index")

    # Efficient groupby for MS1 and MS2
    ms1_groups = {}
    ms2_groups = {}
    if ms1_by_uid:
        ms1_concat = []
        for uid, df in ms1_by_uid.items():
            if df is not None and not df.empty:
                df = df.copy()
                df["label"] = uid
                ms1_concat.append(df)
        if ms1_concat:
            ms1_all = pd.concat(ms1_concat, ignore_index=True)
            ms1_sorted = ms1_all.sort_values(["label", "rt"], kind="mergesort")
            ms1_groups = dict(iter(ms1_sorted.groupby("label", sort=False)))
    if ms2_by_uid:
        ms2_concat = []
        for uid, df in ms2_by_uid.items():
            if df is not None and not df.empty:
                df = df.copy()
                df["label"] = uid
                ms2_concat.append(df)
        if ms2_concat:
            ms2_all = pd.concat(ms2_concat, ignore_index=True)
            ms2_sorted = ms2_all.sort_values(["label", "rt"], kind="mergesort")
            ms2_groups = dict(iter(ms2_sorted.groupby("label", sort=False)))

    for uid in atlas["mz_rt_uid"]:
        info = mzrt_info.get(uid, {})
        inchi_key = info.get("inchi_key", "")
        adduct = info.get("adduct", "")

        ms1_df = ms1_groups.get(uid)
        if ms1_df is None or ms1_df.empty:
            if not remove_unided:
                ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid,
                                        inchi_key=inchi_key, adduct=adduct,
                                        data=_empty_ms1_frame()))
        else:
            ms1_df = ms1_df.reset_index(drop=True)
            ms1_list.append(MS1Data(filename=filename, mz_rt_uid=uid,
                                    inchi_key=inchi_key, adduct=adduct, data=ms1_df))

        ms2_df = ms2_groups.get(uid)
        if ms2_df is None or ms2_df.empty:
            if not remove_unided:
                ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid,
                                        inchi_key=inchi_key, adduct=adduct,
                                        data=_empty_ms2_frame()))
        else:
            ms2_df = ms2_df.reset_index(drop=True)
            ms2_list.append(MS2Data(filename=filename, mz_rt_uid=uid,
                                    inchi_key=inchi_key, adduct=adduct, data=ms2_df))

    return ms1_list, ms2_list


# ---------------------------------------------------------------------------
# Top-level entry point — identical signature and behavior to the reference
# ---------------------------------------------------------------------------

def extract_data_from_raw(
    obj: "RTAlignment" or "AutoIdentification",
    stage: str,
) -> "ExperimentalData":
    """
    Extract MS1 (+ optionally MS2) for all atlas entries across a set of
    HDF5 files using the legacy algorithm. Inputs and outputs match the
    vectorized reference implementation exactly.
    """
    from metatlas2.workflow_objects import ExperimentalData, MS1Data, MS2Data
    # ...existing code...

    # set up starting conditions from object
    t0 = time.perf_counter()
    atlas = obj.align_atlas_obj if stage == "rt_alignment" else obj.pre_autoid_atlas_obj
    polarity = "positive" if atlas.polarity.lower() == "pos" else "negative" if atlas.polarity.lower() == "neg" else atlas.polarity.lower()
    lcmsruns = obj.aligner_lcmsruns if stage == "rt_alignment" else obj.autoid_lcmsruns
    workflow_params = obj.rt_alignment_params if stage == "rt_alignment" else obj.workflow_params
    remove_unided = workflow_params.get("remove_unided_compounds", True)
    ms1_only = True if stage == "rt_alignment" else False
    t1 = time.perf_counter()
    logger.info(f"Extracting raw data based on atlas: {atlas.polarity}, {atlas.chromatography}, {atlas.analysis_type}")

    # create expanded atlas and get lcmsrun files
    atlas_df = atlas.to_dataframe()
    atlas_expanded = _expand_atlas_windows(atlas_df, workflow_params, polarity)
    runs = [r for r in lcmsruns if getattr(r, "file_format", "h5") == "h5"]
    logger.info("Extracting data for stage %s (ms1 only = %s) from %d HDF5 files for %d atlas entries", stage, ms1_only, len(runs), len(atlas_df))
    t2 = time.perf_counter()

    # set up containers
    experimental_data_obj = ExperimentalData()
    ms1_all: list[MS1Data] = []
    ms2_all: list[MS2Data] = []

    # Chunked parallelism: process files in batches of size max_workers
    t3 = time.perf_counter()
    n_workers = max(1, min(1, len(runs)))
    logger.info("Processing files in batches of up to %d workers", n_workers)
    worker = partial(_process_one_file, atlas=atlas_expanded, remove_unided=remove_unided, ms1_only=ms1_only)
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    for batch in tqdm(list(chunked(runs, n_workers)), desc="Processing HDF5 files (batched)"):
        if len(batch) == 1:
            ms1, ms2 = worker(batch[0])
            ms1_all.extend(ms1)
            ms2_all.extend(ms2)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(batch), mp_context=ctx) as ex:
                futures = {ex.submit(worker, run): run for run in batch}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Batch (parallel)", leave=False):
                    ms1, ms2 = fut.result()
                    ms1_all.extend(ms1)
                    ms2_all.extend(ms2)
    t4 = time.perf_counter()

    logger.info("Step timings (seconds): parse input: %.2f, expand atlas: %.2f, setup: %.2f, process files: %.2f", t1-t0, t2-t1, t3-t2, t4-t3)
    logger.info("Extracted %d MS1 and %d MS2 result groups from %d files", len(ms1_all), len(ms2_all), len(runs))

    logger.info("Adding MS1 and MS2 data to ExperimentalData object...")
    experimental_data_obj.ms1_data = ms1_all
    experimental_data_obj.ms2_data = ms2_all

    t5 = time.perf_counter()

    logger.info(
        "Step timings (seconds): parse input: %.4f, expand atlas: %.4f, setup: %.4f, process files: %.4f, finalize: %.4f",
        t1-t0, t2-t1, t3-t2, t4-t3, t5-t4
    )
    return experimental_data_obj