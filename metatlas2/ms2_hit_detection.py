import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
from contextlib import contextmanager
import tqdm as tqdm_module

from matchms.similarity import CosineHungarian
from matchms import Spectrum
from scipy.optimize import linear_sum_assignment

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def _process_job_global(job):
    (mz_rt_uid, file_path, ms2_df, ref_spectra,
     frag_mz_tolerance, min_score, min_frags, 
     max_ppm_error, limit_to_n_hits) = job
    try:
        hits_df = _find_hits_from_ms2_df(
            ms2_df, mz_rt_uid, ref_spectra,
            frag_mz_tolerance=frag_mz_tolerance,
            min_score=min_score,
            min_frags=min_frags,
            max_ppm_error=max_ppm_error,
            limit_to_n_hits=limit_to_n_hits,
        )
        return (mz_rt_uid, file_path, hits_df)
    except Exception as e:
        logger.exception(f"Error in hit detection for {mz_rt_uid} {file_path}: {e}")
        return (mz_rt_uid, file_path, pd.DataFrame())

def find_ms2_hits(auto_id_obj) -> None:
    """
    Find MS2 hits for all compounds with MS2 data in the experimental dataset using a
    reference spectra database. Uses MatchMS cosine similarity scoring and Hungarian
    algorithm for peak alignment. Parallelized across compounds/files with MS2 data.
    """
    from metatlas2.workflow_objects import MS2Hit

    # load msms refs data
    refs_by_inchi_key = ldt.load_msms_refs_file(
        Path(auto_id_obj.paths['msms_refs_path']),
        database_filter=auto_id_obj.config.get('WORKFLOWS').get('PATHS').get('msms_refs_db_filter', "metatlas"),
    )
    if not refs_by_inchi_key:
        raise FileNotFoundError("No reference database found - skipping hit detection")

    # Warn about ms2_data entries with no matching refs (informational only)
    missing_uids = 0
    missing_inchi_keys = set()
    for ms2 in auto_id_obj.experimental_data.ms2_data:
        if not refs_by_inchi_key.get(ms2.inchi_key, []):
            missing_uids += 1
            missing_inchi_keys.add(ms2.inchi_key)
    logger.info(f"{missing_uids} ms2_data entries have no matching refs.")
    logger.info(f"Unique inchi_keys from ms2_data not in reference database: {missing_inchi_keys}")

    # Define filtering parameters
    wp = auto_id_obj.workflow_params
    frag_mz_tolerance = wp.get('ms2_frag_mz_tolerance', 0.05)
    min_score = wp.get('ms2_min_score', 0)
    min_frags = wp.get('ms2_min_matching_frags', 0)
    max_ppm_error = wp.get('ppm_error', 5.0)
    limit_to_n_hits = wp.get('limit_to_n_hits', None)

    # set up data to pass to parallel jobs
    jobs = [
        (ms2.mz_rt_uid, ms2.filename, ms2.data,
        refs_by_inchi_key.get(getattr(ms2, 'inchi_key', ''), []),
        frag_mz_tolerance, min_score, min_frags, max_ppm_error, limit_to_n_hits)
        for ms2 in auto_id_obj.experimental_data.ms2_data
        if not ms2.data.empty and ms2.mz_rt_uid
    ]
    if not jobs:
        logger.warning("No MS2 data found, skipping hit detection")
        return

    logger.info(f"Searching {len(jobs)} query sets...")
    max_workers = min(mp.cpu_count(), len(jobs))
    results = []
    if max_workers > 1 and len(jobs) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers for hit detection...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_job_global, job) for job in jobs]
            with tqdm(total=len(jobs), desc="Detecting MS2 hits") as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
    else:
        logger.info("Using sequential processing for hit detection...")
        with tqdm(total=len(jobs), desc="Detecting MS2 hits") as pbar:
            for job in jobs:
                results.append(_process_job_global(job))
                pbar.update(1)

    # Assign results to ExperimentalData.ms2_hits as MS2Hit objects
    for mz_rt_uid, filename, ms2_hits_df in results:
        ms2_hit_obj = MS2Hit(
            mz_rt_uid=mz_rt_uid,
            filename=filename,
            data=ms2_hits_df,
        )
        auto_id_obj.experimental_data.ms2_hits.append(ms2_hit_obj)

    # Summary
    hits_list = [
        (h.mz_rt_uid, h.filename, len(h.data))
        for h in auto_id_obj.experimental_data.ms2_hits
        if not h.data.empty
    ]
    total_hits = sum(n for _, _, n in hits_list)
    unique_compounds = len({uid for uid, _, _ in hits_list})
    unique_files = len({fn for _, fn, _ in hits_list})
    limit_msg = ""
    limit_to_n_hits = auto_id_obj.workflow_params.get('limit_to_n_hits', None)
    if limit_to_n_hits is not None:
        limit_msg = f" (capped at {limit_to_n_hits} hits per compound per file)"
    logger.info(
        f"Hit detection complete: {total_hits} total hits across "
        f"{unique_compounds} compounds and {unique_files} files{limit_msg}."
    )

def _find_hits_from_ms2_df(
    ms2_df: pd.DataFrame,
    mz_rt_uid: str,
    ref_spectra: List,
    frag_mz_tolerance: float,
    min_score: float,
    min_frags: int,
    max_ppm_error: float,
    limit_to_n_hits: int,
) -> pd.DataFrame:
    if ms2_df.empty or not ref_spectra:
        return pd.DataFrame()

    # 1. Convert EVERYTHING to numpy immediately. 
    # This eliminates Pandas overhead inside the loop.
    ms2_df = ms2_df.sort_values('rt', kind='stable')
    
    all_rt = ms2_df['rt'].to_numpy()
    all_pmz = ms2_df['precursor_MZ'].to_numpy()
    all_pint = ms2_df['precursor_intensity'].to_numpy()
    all_in_feature = ms2_df['in_feature'].to_numpy()
    all_mz = ms2_df['mz'].to_numpy() # Added this
    all_i = ms2_df['i'].to_numpy()   # Added this
    
    _, indices = np.unique(all_rt, return_index=True)
    
    query_meta = [] 
    queries = []
    
    # 2. NumPy-only loop for Spectrum creation
    for i in range(len(indices)):
        start = indices[i]
        end = indices[i+1] if i+1 < len(indices) else len(all_rt)
        
        # Slice numpy arrays instead of DataFrame .iloc
        s_mz = all_mz[start:end]
        s_i = all_i[start:end]
        
        # NumPy sort is orders of magnitude faster than df.sort_values
        sort_idx = np.argsort(s_mz)
        f_mz = s_mz[sort_idx].astype(np.float32)
        f_int = s_i[sort_idx].astype(np.float32)
        
        qry = Spectrum(mz=f_mz, intensities=f_int, metadata={'precursor_mz': all_pmz[start]})
        queries.append(qry)
        
        query_meta.append({
            'rt': all_rt[start],
            'precursor_mz': all_pmz[start],
            'precursor_int': all_pint[start],
            'in_feature': all_in_feature[start],
            'frag_mz': f_mz,
            'frag_int': f_int,
        })

    if not queries:
        return pd.DataFrame()

    ref_precursor_mz = np.array([float(r.get('precursor_mz', 0.0) or 0.0) for r in ref_spectra], dtype=np.float64)
    
    # Vectorized PPM mask
    n_queries, n_refs = len(queries), len(ref_spectra)
    q_mz = np.array([m['precursor_mz'] for m in query_meta])[:, None]
    
    if max_ppm_error is None:
        candidate_mask = np.ones((n_queries, n_refs), dtype=bool)
    else:
        tol_matrix = ref_precursor_mz * (max_ppm_error * 1e-6)
        candidate_mask = np.abs(q_mz - ref_precursor_mz) <= tol_matrix

    if not candidate_mask.any():
        return pd.DataFrame()

    # Heavy computation
    with _suppress_tqdm():
        score_matrix = CosineHungarian(tolerance=frag_mz_tolerance).matrix(references=ref_spectra, queries=queries)
    
    scores = score_matrix['score'].T
    matches = score_matrix['matches'].T

    passing = candidate_mask & (scores >= min_score) & (matches >= min_frags)
    
    if not passing.any():
        return pd.DataFrame()

    if limit_to_n_hits is not None:
        n_passing = int(passing.sum())
        if n_passing > limit_to_n_hits:
            flat_scores = np.where(passing, scores, -np.inf).ravel()
            top_flat_idx = np.argpartition(flat_scores, -limit_to_n_hits)[-limit_to_n_hits:]
            new_passing = np.zeros(passing.size, dtype=bool)
            new_passing[top_flat_idx] = True
            passing = new_passing.reshape(passing.shape)

    # Final build: Pre-extract ref metadata to avoid dictionary lookups in the loop
    ref_metadata = [
        {
            'db': r.get('database', 'unknown'),
            'id': r.get('id', ''),
            'name': r.get('name') or r.get('compound_name') or 'Unknown',
            'mz': float(ref_precursor_mz[idx])
        } 
        for idx, r in enumerate(ref_spectra)
    ]

    all_hits = []
    passing_qi, passing_ri = np.where(passing)
    
    for qi, ri in zip(passing_qi, passing_ri):
        meta = query_meta[qi]
        ref = ref_spectra[ri]
        r_meta = ref_metadata[ri]
        
        actual_ppm = ((meta['precursor_mz'] - r_meta['mz']) / r_meta['mz']) * 1e6

        alignment_data = _align_spectra_for_plotting(
            meta['frag_mz'], meta['frag_int'], ref.mz, ref.intensities, frag_mz_tolerance
        )

        all_hits.append({
            'mz_rt_uid': mz_rt_uid,
            'in_feature': meta['in_feature'],
            'database': r_meta['db'],
            'ref_id': r_meta['id'],
            'ref_name': r_meta['name'],
            'score': float(scores[qi, ri]),
            'num_matches': int(matches[qi, ri]),
            'mz_theoretical': r_meta['mz'],
            'mz_measured': meta['precursor_mz'],
            'ppm_error': actual_ppm,
            'rt': float(meta['rt']),
            'qry_intensity_peak': meta['precursor_int'],
            'ref_frags': len(ref.mz),
            'data_frags': len(meta['frag_mz']),
            'matched_fragments': alignment_data['matched_fragments'],
            'aligned_fragment_colors': alignment_data['fragment_colors'],
            'qry_spectrum': alignment_data['query_aligned'],
            'ref_spectrum': alignment_data['ref_aligned'],
        })

    return pd.DataFrame(all_hits)

def _align_spectra_for_plotting(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    ref_mz: np.ndarray,
    ref_int: np.ndarray,
    frag_mz_tolerance: float,
) -> Dict:
    """
    Align query and reference spectra for mirror plotting.

    Produces two parallel arrays where matched peaks share an index position,
    and unmatched peaks have NaN on the opposing side. This lets a plotter
    draw connecting lines between matched fragments.
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

        # Build aligned arrays:
        #   [unmatched-ref peaks] + [all query peaks, with matched-ref peaks slotted in]
        n_unmatched_r = int(unmatched_r.sum())
        n_query = len(query_mz)

        # Query side: NaN for unmatched-ref slots, then query peaks as-is
        q_aligned_mz = np.concatenate([np.full(n_unmatched_r, np.nan), query_mz])
        q_aligned_int = np.concatenate([np.full(n_unmatched_r, np.nan), query_int])

        # Ref side: unmatched-ref peaks first, then NaN-filled query-length array
        # with matched ref peaks placed at their query partner's index
        r_slot_mz = np.full(n_query, np.nan)
        r_slot_int = np.full(n_query, np.nan)
        r_slot_mz[matched_q] = ref_mz[matched_r]
        r_slot_int[matched_q] = ref_int[matched_r]
        r_aligned_mz = np.concatenate([ref_mz[unmatched_r], r_slot_mz])
        r_aligned_int = np.concatenate([ref_int[unmatched_r], r_slot_int])

        # Build color + matched-fragment lists
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


def _no_match_alignment(query_mz, query_int, ref_mz, ref_int) -> Dict:
    """Fallback alignment when no peaks match (or on error)."""
    return {
        'matched_fragments': [],
        'fragment_colors': ['red'] * len(query_mz),
        'query_aligned': [query_mz.tolist(), query_int.tolist()],
        'ref_aligned': [ref_mz.tolist(), ref_int.tolist()],
        'num_matched': 0,
    }