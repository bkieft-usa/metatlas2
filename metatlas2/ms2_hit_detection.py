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
from typing import Dict

import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf

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

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

def _no_match_alignment(query_mz, query_int, ref_mz, ref_int) -> Dict:
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
) -> Dict:
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
    (uid, filename, scans_data, ref_spectra, 
     frag_mz_tolerance, min_score, min_frags, 
     ms2_mz_tolerance_ppm, limit_to_n_hits) = job
    
    if not ref_spectra:
        return uid, filename, [[] for _ in range(len(scans_data))]

    # 1. Prepare matchms Spectra
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

    # 2. Vectorized PPM Masking
    ref_precursor_mzs = np.array([float(r.get('precursor_mz', 0.0) or 0.0) for r in ref_spectra])
    q_mzs_np = np.array(q_mzs)[:, None] 
    
    if ms2_mz_tolerance_ppm is None:
        candidate_mask = np.ones((len(queries), len(ref_spectra)), dtype=bool)
    else:
        tol_matrix = ref_precursor_mzs * (ms2_mz_tolerance_ppm * 1e-6)
        candidate_mask = np.abs(q_mzs_np - ref_precursor_mzs) <= tol_matrix

    # 3. Scoring
    cosine_hungarian = CosineHungarian(tolerance=frag_mz_tolerance)
    hit_method = "matrix"
    if hit_method == "matrix":
        with _suppress_tqdm():
            # Use matrix method
            score_matrix = cosine_hungarian.matrix(references=ref_spectra, queries=queries)
            scores = score_matrix['score'].T
            matches = score_matrix['matches'].T
    elif hit_method == "pair":
        pair_results = np.array([
            cosine_hungarian.pair(r, q) for q in queries for r in ref_spectra
        ])
        scores = pair_results['score'].reshape(len(queries), len(ref_spectra))
        matches = pair_results['matches'].reshape(len(queries), len(ref_spectra))
    passing = candidate_mask & (scores >= min_score) & (matches >= min_frags)

    # 4. Build Rich Metadata Hits
    all_scan_results = [[] for _ in range(len(scans_data))]
    
    for q_idx in range(len(queries)):
        ref_indices = np.where(passing[q_idx])[0]
        if ref_indices.size == 0:
            continue
            
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
            
            # Perform alignment ONLY for the top N hits
            align_res = _align_spectra_for_plotting(
                q_mz_np, q_int_np, 
                ref.mz, ref.intensities, 
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

    # Collect per-step stats for the summary table: (step_label, scans_after, compounds_after)
    ms2_steps = [("all ms2 scans", starting_scans, starting_uids)]

    # Only in_feature=True scans count toward compound retention.
    in_feature_df = ms2_df[ms2_df['in_feature'] == True]
    if in_feature_df.empty:
        keep_uids = set()
    else:
        compounds_with_hits_mask = in_feature_df.groupby('mz_rt_uid')['hits'].apply(
            lambda x: any(isinstance(h, list) and len(h) > 0 for h in x)
        )
        keep_uids = set(compounds_with_hits_mask[compounds_with_hits_mask].index)
    ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(keep_uids)].reset_index(drop=True)
    ms2_steps.append((f"hits (score>={min_score}, frags>={min_frags})", len(ms2_df), ms2_df['mz_rt_uid'].nunique()))

    ldt.log_filter_table(ms2_steps, starting_scans, starting_uids, entries_label="Scans", title="MS2 hit filtering summary")

    ms2_df = ms2_df.reindex(columns=final_columns)

    # Synchronize MS1: drop compounds with no passing MS2 hits (if we made it this far, there are MS2 filters)
    if not ms1_df.empty and not ms2_df.empty:
        ms1_starting_entries = len(ms1_df)
        ms1_starting_uids = ms1_df['mz_rt_uid'].nunique()
        valid_uids = ms2_df['mz_rt_uid'].unique()
        ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(valid_uids)].copy()
        ms1_steps = [
            ("all ms2 data", ms1_starting_entries, ms1_starting_uids),
            ("pass MS1 & MS2", len(ms1_df), ms1_df['mz_rt_uid'].nunique()),
        ]
        ldt.log_filter_table(ms1_steps, ms1_starting_entries, ms1_starting_uids, title="MS1 sync with MS2 hits summary")

    return ms2_df, ms1_df

def _assign_hits(ms2_df, results_map):
    """Assign scored hits back to every row of ms2_df in a single O(n) pass.
    
    results_map[(uid, filename)] is a list of hit lists, one per scan in the
    order they appear in ms2_df for that group (all scans, in_feature or not).
    Compound retention is decided separately by _filter_out_ms2_data.
    """
    logger.info("Assigning hits back to MS2 dataframe...")
    hits_col = [[] for _ in range(len(ms2_df))]
    uid_col = ms2_df['mz_rt_uid'].tolist()
    filename_col = ms2_df['filename'].tolist()
    group_counter: Dict[tuple, int] = {}
    for row_idx in range(len(ms2_df)):
        key = (uid_col[row_idx], filename_col[row_idx])
        hits_list = results_map.get(key, [])
        scan_idx = group_counter.get(key, 0)
        if scan_idx < len(hits_list):
            hits_col[row_idx] = hits_list[scan_idx]
        group_counter[key] = scan_idx + 1
    ms2_df = ms2_df.copy()
    ms2_df['hits'] = hits_col
    return ms2_df

def find_ms2_hits(auto_id_obj):
    dataset = auto_id_obj.experimental_data
    wp = auto_id_obj.ta.params
    polarity = auto_id_obj.ta.polarity
    
    ms2_df = dataset.ms2_df
    if ms2_df.empty:
        logger.warning("No MS2 data found. Skipping hit detection.")
        return

    unique_ms2_inchi_keys = ms2_df['inchi_key'].dropna().unique()
    groups = ms2_df.groupby(['mz_rt_uid', 'filename'])
    refs_by_inchi_key = ldt.load_msms_refs_file(
        file_path=Path(auto_id_obj.paths['msms_refs_path']),
        database_filter=auto_id_obj.msms_refs_db_filter,
        polarity=polarity,
        inchi_keys=unique_ms2_inchi_keys
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

    logger.info(f"Finding reference hits for {len(jobs)} compound-file groups...")
    results_map = {}
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 10)) as executor:
        futures = [executor.submit(_process_compound_batch, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Detecting MS2 Hits", disable=should_disable_tqdm()):
            uid, filename, hits_list = fut.result()
            results_map[(uid, filename)] = hits_list

    ms2_df = _assign_hits(ms2_df, results_map)

    ms2_df, ms1_df = _filter_out_ms2_data(ms2_df, auto_id_obj.experimental_data.ms1_df, wp.get('ms2_min_score', 0), wp.get('ms2_min_matching_frags', 0))

    logger.info("Attaching MS2 data (and MS1 data, if MS2 data filtered) to AutoID object...")
    dataset.ms2_df = ms2_df
    dataset.ms1_df = ms1_df
    auto_id_obj.experimental_data = dataset

    logger.info("MS2 hit detection complete.")