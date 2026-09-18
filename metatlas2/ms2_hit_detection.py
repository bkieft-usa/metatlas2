from __future__ import annotations

import pandas as pd
import numpy as np
import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from matchms.similarity import CosineHungarian
from matchms import Spectrum
from pathlib import Path
from contextlib import contextmanager
import tqdm as tqdm_module
from scipy.optimize import linear_sum_assignment

import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
from metatlas2.utils import should_disable_tqdm, get_max_workers

logger = lcf.get_logger('ms2_hit_detection')

@contextmanager
def _suppress_tqdm():
    original_init = tqdm_module.tqdm.__init__
    def _disabled_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)
    tqdm_module.tqdm.__init__ = _disabled_init
    try:
        yield
    finally:
        tqdm_module.tqdm.__init__ = original_init

def _no_match_alignment(query_mz, query_int, ref_mz, ref_int) -> dict:
    """Fallback alignment when no peaks match (or on error)."""
    return {
        'matched_fragments': [],
        'fragment_colors': ['red'] * len(query_mz),
        'query_aligned': [query_mz.tolist(), query_int.tolist()],
        'ref_aligned': [ref_mz.tolist(), ref_int.tolist()],
        'num_matched': 0,
    }

def _align_spectra_for_plotting(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    ref_mz: np.ndarray,
    ref_int: np.ndarray,
    frag_mz_tolerance: float,
) -> dict:
    """
    Align query and reference spectra for mirror plotting.
    """
    try:
        # Build cost matrix and solve assignment in one shot
        mz_diff = np.abs(query_mz[:, None] - ref_mz[None, :])
        within = mz_diff <= frag_mz_tolerance
        cost = np.where(within, query_int[:, None] * ref_int[None, :], 0.0)

        if not cost.any():
            return _no_match_alignment(query_mz, query_int, ref_mz, ref_int)

        row_idx, col_idx = linear_sum_assignment(cost, maximize=True)
        valid = cost[row_idx, col_idx] > 0
        matched_q = row_idx[valid]
        matched_r = col_idx[valid]

        # Partition ref peaks into matched and unmatched
        ref_matched_mask = np.zeros(len(ref_mz), dtype=bool)
        ref_matched_mask[matched_r] = True
        unmatched_r = ~ref_matched_mask

        # Build aligned arrays
        n_unmatched_r = int(unmatched_r.sum())
        n_query = len(query_mz)

        q_aligned_mz = np.concatenate([np.full(n_unmatched_r, np.nan), query_mz])
        q_aligned_int = np.concatenate([np.full(n_unmatched_r, np.nan), query_int])

        r_slot_mz = np.full(n_query, np.nan)
        r_slot_int = np.full(n_query, np.nan)
        r_slot_mz[matched_q] = ref_mz[matched_r]
        r_slot_int[matched_q] = ref_int[matched_r]
        r_aligned_mz = np.concatenate([ref_mz[unmatched_r], r_slot_mz])
        r_aligned_int = np.concatenate([ref_int[unmatched_r], r_slot_int])

        both_present = ~np.isnan(q_aligned_mz) & ~np.isnan(r_aligned_mz)
        fragment_colors = np.where(both_present, "green", "red").tolist()
        matched_fragments = q_aligned_mz[both_present].tolist()

        return {
            'matched_fragments': matched_fragments,
            'fragment_colors': fragment_colors,
            'query_aligned': [q_aligned_mz.tolist(), q_aligned_int.tolist()],
            'ref_aligned': [r_aligned_mz.tolist(), r_aligned_int.tolist()],
            'num_matched': len(matched_fragments),
        }

    except Exception:
        logger.exception("Error in spectrum alignment")
        return _no_match_alignment(query_mz, query_int, ref_mz, ref_int)

def _process_compound_batch(job):
    """Score all MS2 scans for one (compound, file) group against reference spectra.

    Designed to run in a worker process via :class:`concurrent.futures.ProcessPoolExecutor`.
    Builds :class:`matchms.Spectrum` objects from raw scan data, applies a vectorised
    precursor m/z PPM filter, computes CosineHungarian scores against every candidate
    reference, and calls :func:`_align_spectra_for_plotting` for each hit.

    **All** candidate hits (those passing the precursor PPM filter) are stored,
    regardless of score or fragment-match count.  The ``min_score`` and
    ``min_frags`` thresholds are used only by :func:`_filter_out_ms2_data` to
    decide which **compounds** to retain — not to suppress individual hits.
    This ensures that, for a compound with at least one high-scoring scan, all
    lower-scoring hits are still visible in the output (they provide context
    about whether the good hit is trustworthy).

    Args:
        job: A tuple of
            ``(uid, filename, scans_data, ref_spectra, frag_mz_tolerance,
            min_score, min_frags, ms2_mz_tolerance_ppm, limit_to_n_hits)`` where

            * ``uid`` - compound ``mz_rt_uid`` string.
            * ``filename`` - sample file identifier.
            * ``scans_data`` - list of dicts with keys ``frag_mzs``, ``frag_ints``,
              ``precursor_MZ``, and ``precursor_intensity``.
            * ``ref_spectra`` - list of :class:`matchms.Spectrum` reference objects.
            * ``frag_mz_tolerance`` - fragment m/z tolerance in Da for scoring/alignment.
            * ``min_score`` - minimum cosine score threshold (used only for compound
              retention in :func:`_filter_out_ms2_data`, not for hit suppression here).
            * ``min_frags`` - minimum matched-fragment count threshold (same usage as
              ``min_score``).
            * ``ms2_mz_tolerance_ppm`` - precursor PPM filter (``None`` disables it).
            * ``limit_to_n_hits`` - maximum hits to return per scan (``None`` = unlimited).
              Applied after sorting by score descending so the top-N hits are kept.

    Returns:
        A 3-tuple ``(uid, filename, all_scan_results)`` where ``all_scan_results``
        is a list (one entry per input scan) of hit-record lists.  Each hit record
        is a ``dict`` containing score, alignment, and metadata fields.  Scans with
        no candidate references (after the PPM filter) get an empty list.
    """
    (uid, filename, scans_data, ref_spectra,
     frag_mz_tolerance, min_score, min_frags,
     ms2_mz_tolerance_ppm, limit_to_n_hits) = job
    
    if not ref_spectra:
        return uid, filename, [[] for _ in range(len(scans_data))]

    queries = []
    q_mzs = []
    valid_scans = []

    for i, scan in enumerate(scans_data):
        f_mz = scan['frag_mzs']
        f_int = scan['frag_ints']
        p_mz = scan['precursor_MZ']
        if f_mz is None or len(f_mz) == 0:
            continue
        mz_arr = np.array(f_mz, dtype=np.float32)
        int_arr = np.array(f_int, dtype=np.float32)
        qry = Spectrum(mz=mz_arr, intensities=int_arr, metadata={'precursor_mz': p_mz})
        queries.append(qry)
        q_mzs.append(p_mz)
        valid_scans.append(i)

    if not queries:
        return uid, filename, [[] for _ in range(len(scans_data))]

    ref_precursor_mzs = np.array([float(r.get('precursor_mz', 0.0) or 0.0) for r in ref_spectra])
    q_mzs_np = np.array(q_mzs)[:, None]
    
    if ms2_mz_tolerance_ppm is None:
        candidate_mask = np.ones((len(queries), len(ref_spectra)), dtype=bool)
    else:
        tol_matrix = ref_precursor_mzs * (ms2_mz_tolerance_ppm * 1e-6)
        candidate_mask = np.abs(q_mzs_np - ref_precursor_mzs) <= tol_matrix

    # always use the vectorised matrix method
    cosine_hungarian = CosineHungarian(tolerance=frag_mz_tolerance)
    with _suppress_tqdm():
        score_matrix = cosine_hungarian.matrix(references=ref_spectra, queries=queries)
        scores = score_matrix['score'].T
        matches = score_matrix['matches'].T

    all_scan_results = [[] for _ in range(len(scans_data))]
    
    for q_idx in range(len(queries)):
        ref_indices = np.where(candidate_mask[q_idx])[0]
        if ref_indices.size == 0:
            continue
            
        # Sort all candidates by score descending; apply limit_to_n_hits cap
        sorted_ref_indices = ref_indices[np.argsort(-scores[q_idx, ref_indices])]
        if limit_to_n_hits:
            sorted_ref_indices = sorted_ref_indices[:limit_to_n_hits]
            
        scan_meta = scans_data[valid_scans[q_idx]]
        
        # Cache query arrays for alignment to avoid redundant casting in the inner loop
        q_mz_np = np.array(scan_meta['frag_mzs'], dtype=np.float32)
        q_int_np = np.array(scan_meta['frag_ints'], dtype=np.float32)

        scan_hits = []
        for r_idx in sorted_ref_indices:
            ref = ref_spectra[r_idx]
            
            align_res = _align_spectra_for_plotting(
                q_mz_np,
                q_int_np,
                ref.mz,
                ref.intensities,
                frag_mz_tolerance
            )

            scan_hits.append({
                'mz_rt_uid': uid,
                'database': ref.metadata.get('database', 'unknown'),
                'ref_id': ref.metadata.get('id', ''),
                'ref_name': ref.metadata.get('name') or ref.metadata.get('compound_name') or 'Unknown',
                'score': float(scores[q_idx, r_idx]),
                'num_matches': int(matches[q_idx, r_idx]),
                'mz_theoretical': float(ref_precursor_mzs[r_idx]),
                'mz_measured': float(scan_meta['precursor_MZ']),
                'ppm_error': float((scan_meta['precursor_MZ'] - ref_precursor_mzs[r_idx]) / ref_precursor_mzs[r_idx] * 1e6),
                'qry_intensity_peak': float(scan_meta.get('precursor_intensity', 0.0)),
                'ref_frags': len(ref.mz),
                'data_frags': len(scan_meta['frag_mzs']),
                'matched_fragments': align_res['matched_fragments'],
                'fragment_colors': align_res['fragment_colors'],
                'query_aligned': align_res['query_aligned'],
                'ref_aligned': align_res['ref_aligned'],
            })
        
        all_scan_results[valid_scans[q_idx]] = scan_hits
        
    return uid, filename, all_scan_results

def _filter_out_ms2_data(ms2_df, ms1_df, min_score, min_frags):
    """Remove compounds that have no passing MS2 hits and synchronise the MS1 DataFrame.

    When both ``min_score`` and ``min_frags`` are 0 the filter is skipped and all
    scans are retained.  Otherwise a compound is kept if **at least one**
    ``in_feature`` scan across **any** file for that compound has at least one
    hit whose ``score >= min_score`` AND ``num_matches >= min_frags``.

    Because :func:`_process_compound_batch` now stores **all** candidate hits
    (not just those above the thresholds), the score/frags check is applied
    here by inspecting each hit dict.  When a compound qualifies, **all** of
    its scans are retained — including scans from other files and scans whose
    hits fall below the thresholds — so that low-scoring hits remain visible
    as context alongside the good hit.

    The MS1 DataFrame is then trimmed to the same set of compound UIDs.

    Args:
        ms2_df: :class:`pandas.DataFrame` of MS2 scans with a ``hits`` column
            populated by :func:`_assign_hits`.  Each element of ``hits`` is a
            list of dicts with at least ``score`` and ``num_matches`` keys.
        ms1_df: :class:`pandas.DataFrame` of MS1 data to synchronise.
        min_score: Minimum cosine score a hit must have to count as "passing".
        min_frags: Minimum matched-fragment count a hit must have to count as
            "passing".

    Returns:
        A 2-tuple ``(filtered_ms2_df, filtered_ms1_df)`` with standardised columns.
    """
    starting_scans = len(ms2_df)
    starting_uids = ms2_df['mz_rt_uid'].nunique()
    final_columns = ['mz_rt_uid', 'filename', 'inchi_key', 'adduct', 'scan_rt', 'frag_mzs', 'frag_ints', 'precursor_MZ', 'precursor_intensity', 'collision_energy', 'in_feature', 'hits']

    # When both thresholds are 0, treat as "no filter": retain all compounds
    if min_score == 0 and min_frags == 0:
        logger.info(
            "ms2_min_score=0 and ms2_min_matching_frags=0: skipping MS2 compound filter. "
            "Retaining all %d scans across %d compounds.",
            starting_scans, starting_uids
        )
        return ms2_df.reindex(columns=final_columns), ms1_df

    ms2_steps = [("all ms2 scans", starting_scans, starting_uids)]

    def _scan_has_passing_hit(hits_list, min_score, min_frags):
        """Return True if any hit in hits_list meets both score and frags thresholds."""
        if not isinstance(hits_list, list):
            return False
        return any(
            isinstance(h, dict)
            and h.get('score', 0.0) >= min_score
            and h.get('num_matches', 0) >= min_frags
            for h in hits_list
        )

    in_feature_df = ms2_df[ms2_df['in_feature'] == True]
    if in_feature_df.empty:
        keep_uids = set()
    else:
        compounds_with_hits_mask = in_feature_df.groupby('mz_rt_uid')['hits'].apply(
            lambda x: any(_scan_has_passing_hit(h, min_score, min_frags) for h in x)
        )
        keep_uids = set(compounds_with_hits_mask[compounds_with_hits_mask].index)
    ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(keep_uids)].reset_index(drop=True)
    ms2_steps.append((f"hits (score>={min_score}, frags>={min_frags})", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))

    ldt.log_filter_table(ms2_steps, starting_scans, starting_uids, entries_label="Scans", title="MS2 hit filtering summary")

    ms2_df = ms2_df.reindex(columns=final_columns)

    # drop compounds with no passing MS2 hits (if we made it this far, there were MS2 filters)
    if not ms1_df.empty and not ms2_df.empty:
        valid_uids = ms2_df['mz_rt_uid'].unique()
        ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(valid_uids)].copy()
        logger.info(f"Synced MS1 and MS2 data: retained {len(ms1_df)} EIC points for {ms1_df['mz_rt_uid'].nunique()} compounds.")

    return ms2_df, ms1_df

def _keep_top_scan_per_compound_file(ms2_df: pd.DataFrame) -> pd.DataFrame:
    """Retain only the highest-scoring scan per ``(mz_rt_uid, filename)`` group.

    For each ``(mz_rt_uid, filename)`` group the scan whose best hit score is
    highest is kept; all other scans in that group are dropped.  The best hit
    score for a scan is defined as the ``score`` field of the first element of
    its ``hits`` list (hits are stored in descending score order), or ``-1``
    when the ``hits`` list is empty.

    This filter is applied **after** :func:`_assign_hits` so that every scan
    already has its ``hits`` list populated.  It runs **before**
    :func:`_filter_out_ms2_data` so that the compound-level gate operates on
    the already-reduced set of scans, saving memory.

    When ``keep_top_scan_per_compound_file=False`` in the workflow params this
    function is not called and all scans are forwarded unchanged.

    Args:
        ms2_df: Wide-format MS2 DataFrame (one row per scan) with a ``hits``
                column populated by :func:`_assign_hits`.

    Returns:
        Filtered MS2 DataFrame with at most one scan row per
        ``(mz_rt_uid, filename)`` group.
    """
    if ms2_df.empty:
        return ms2_df

    starting_scans = len(ms2_df)
    starting_uids = ms2_df['mz_rt_uid'].nunique()

    def _best_score(hits):
        """Return the top hit score for a scan, or -1 if no hits."""
        if isinstance(hits, list) and hits and isinstance(hits[0], dict):
            return hits[0].get('score', -1.0)
        return -1.0

    best_scores = ms2_df['hits'].apply(_best_score)
    # For each (mz_rt_uid, filename) group, find the index of the row with the
    # highest best_score.  idxmax() returns the first occurrence on ties.
    group_best_idx = (
        best_scores
        .groupby([ms2_df['mz_rt_uid'], ms2_df['filename']])
        .idxmax()
    )
    ms2_df = ms2_df.loc[group_best_idx.values].reset_index(drop=True)

    logger.info(
        f"keep_top_scan_per_compound_file: reduced MS2 from {starting_scans} scans "
        f"({starting_uids} compounds) to {len(ms2_df)} scans "
        f"({ms2_df['mz_rt_uid'].nunique()} compounds)."
    )
    return ms2_df


def _assign_hits(ms2_df, results_map):
    """Assign scored hits back to every row of ms2_df in a single O(n) pass.

    results_map[(uid, filename)] is a list of hit lists, one per scan in the
    order they appear in ms2_df for that group (all scans, in_feature or not).
    Compound retention is decided separately by :func:`_filter_out_ms2_data`.

    The ``hits`` column is assigned directly without copying the full DataFrame
    first, avoiding a redundant deep copy of all list-valued columns.
    """
    logger.info("Assigning hits back to MS2 dataframe...")
    hits_col = [[] for _ in range(len(ms2_df))]
    uid_col = ms2_df['mz_rt_uid'].tolist()
    filename_col = ms2_df['filename'].tolist()
    group_counter: dict[tuple, int] = {}
    for row_idx in range(len(ms2_df)):
        key = (uid_col[row_idx], filename_col[row_idx])
        hits_list = results_map.get(key, [])
        scan_idx = group_counter.get(key, 0)
        if scan_idx < len(hits_list):
            hits_col[row_idx] = hits_list[scan_idx]
        group_counter[key] = scan_idx + 1
    # Assign the new column directly — no full-frame copy needed.
    ms2_df = ms2_df.assign(hits=hits_col)
    return ms2_df

def find_ms2_hits(auto_id_obj):
    """Run the full MS2 hit-detection pipeline and attach results to ``auto_id_obj``.

    Loads reference MSMS spectra for every InChIKey present in the MS2 data,
    dispatches per-(compound, file) scoring jobs to a :class:`~concurrent.futures.ProcessPoolExecutor`,
    assigns the scored hits back to the MS2 DataFrame via :func:`_assign_hits`,
    and filters both the MS2 and MS1 DataFrames via :func:`_filter_out_ms2_data`.
    Results are written back to ``auto_id_obj.experimental_data``.

    Args:
        auto_id_obj: An auto-identification result object exposing
            ``experimental_data`` (with ``ms2_df`` and ``ms1_df``),
            ``ta.params``, ``ta.polarity``, ``paths``, and
            ``msms_refs_db_filter``.

    Returns:
        None.  Results are attached to ``auto_id_obj.experimental_data.ms2_df``
        and ``auto_id_obj.experimental_data.ms1_df``.
    """
    dataset = auto_id_obj.experimental_data
    wp = auto_id_obj.ta.params
    polarity = auto_id_obj.ta.polarity

    ms2_df = dataset.ms2_df
    if ms2_df.empty:
        logger.warning("No MS2 data found. Skipping hit detection.")
        return

    unique_ms2_inchi_keys = ms2_df['inchi_key'].dropna().unique()
    groups = ms2_df.groupby(['mz_rt_uid', 'filename'])

    msms_refs_path = getattr(auto_id_obj.config, 'msms_refs_path', None) or None
    main_db_path = auto_id_obj.paths.get('main_db_path')

    if msms_refs_path:
        # Analyst override: load from a specific .jsonl file on disk
        logger.info(f"MSMS refs override: loading from file {msms_refs_path}")
        refs_by_inchi_key = ldt.load_msms_refs_file(
            file_path=Path(msms_refs_path),
            database_filter=auto_id_obj.msms_refs_db_filter,
            polarity=polarity,
            inchi_keys=unique_ms2_inchi_keys,
        )
    else:
        if not main_db_path:
            raise ValueError(
                "No msms_refs_path override and no main_db_path available. "
                "Cannot load MSMS reference spectra."
            )
        refs_by_inchi_key = ldt.load_msms_refs_from_db(
            db_path=main_db_path,
            database_filter=auto_id_obj.msms_refs_db_filter,
            polarity=polarity,
            inchi_keys=unique_ms2_inchi_keys,
        )
    ms2_inchi_keys_without_refs = set(unique_ms2_inchi_keys) - set(refs_by_inchi_key.keys())
    if ms2_inchi_keys_without_refs:
        logger.warning(f"No reference spectra found for {len(ms2_inchi_keys_without_refs)} inchi_keys: {', '.join(list(ms2_inchi_keys_without_refs))}")

    jobs = []
    for (uid, filename), group in groups:
        inchi_key = group['inchi_key'].iloc[0] if 'inchi_key' in group.columns else ""
        ref_subset = refs_by_inchi_key.get(inchi_key, [])
        scans_data = group[['frag_mzs', 'frag_ints', 'precursor_MZ', 'precursor_intensity']].to_dict('records')
        
        jobs.append((
            uid, filename, scans_data, ref_subset,
            wp.get('ms2_frag_mz_tolerance', 0.05),
            wp.get('ms2_min_score', 0),
            wp.get('ms2_min_matching_frags', 0),
            wp.get('ms2_mz_tolerance_ppm', 5.0),
            wp.get('limit_to_n_hits', 20)
        ))

    # Free reference spectra now that all jobs are built
    del refs_by_inchi_key

    logger.info(f"Finding reference hits for {len(jobs)} compound-file groups...")
    results_map = {}
    max_workers = get_max_workers(auto_id_obj.config.max_workers if auto_id_obj.config else None)
    logger.info(f"Using {max_workers} worker processes for MS2 hit detection.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_compound_batch, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Detecting MS2 Hits", disable=should_disable_tqdm()):
            uid, filename, hits_list = fut.result()
            results_map[(uid, filename)] = hits_list

    ms2_df = _assign_hits(ms2_df, results_map)
    # Free the hit-record dict now that jobs are built
    del results_map

    if wp.get('keep_top_scan_per_compound_file', True):
        ms2_df = _keep_top_scan_per_compound_file(ms2_df)

    ms2_df, ms1_df = _filter_out_ms2_data(ms2_df, auto_id_obj.experimental_data.ms1_df, wp.get('ms2_min_score', 0), wp.get('ms2_min_matching_frags', 0))

    logger.info("Attaching MS2 data (and MS1 data, if MS2 data filtered) to AutoID object...")
    dataset.ms2_df = ms2_df
    dataset.ms1_df = ms1_df
    auto_id_obj.experimental_data = dataset

    logger.info("MS2 hit detection complete.")