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

_worker_refs = None
_worker_params = None

@contextmanager
def _suppress_tqdm():
    """Temporarily disable all tqdm progress bars (e.g. from matchms internals)."""
    original_init = tqdm_module.tqdm.__init__
    def _disabled_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)
    tqdm_module.tqdm.__init__ = _disabled_init
    try:
        yield
    finally:
        tqdm_module.tqdm.__init__ = original_init

def _worker_init(refs_by_inchi_key, workflow_params):
    global _worker_refs, _worker_params
    _worker_refs = refs_by_inchi_key
    _worker_params = workflow_params

def _process_job_global(job):
    inchi_key, adduct, file_path, ms2_df = job
    try:
        hits_df = _find_hits_from_ms2_df(ms2_df, inchi_key, _worker_refs, _worker_params)
        return (inchi_key, adduct, file_path, hits_df)
    except Exception as e:
        logger.error(f"Error in hit detection for {inchi_key} {adduct} {file_path}: {e}")
        return (inchi_key, adduct, file_path, pd.DataFrame())


def find_ms2_hits(auto_id_obj) -> None:
    """
    Find MS2 hits for all compounds with MS2 data in the experimental dataset using a
    reference spectra database. Uses MatchMS cosine similarity scoring and Hungarian
    algorithm for peak alignment. Parallelized across compounds/files with MS2 data.
    """
    from metatlas2.workflow_objects import MS2Hit

    database_filter = auto_id_obj.config.get('WORKFLOWS').get('PATHS').get('msms_refs_db_filter', "metatlas")
    refs_by_inchi_key = ldt.load_msms_refs_file(
        Path(auto_id_obj.paths['msms_refs_path']),
        database_filter=database_filter,
    )
    if not refs_by_inchi_key:
        raise FileNotFoundError("No reference database found - skipping hit detection")

    total_refs = sum(len(v) for v in refs_by_inchi_key.values())
    logger.info(f"Loaded {total_refs} reference spectra across {len(refs_by_inchi_key)} InChI keys.")

    jobs = [
        (ms2.inchi_key, ms2.adduct, ms2.filename, ms2.data)
        for ms2 in auto_id_obj.experimental_data.ms2_data
        if not ms2.data.empty
    ]
    if not jobs:
        logger.warning("No MS2 data found, skipping hit detection")
        return

    logger.info(f"Searching {len(jobs)} query sets...")
    max_workers = min(mp.cpu_count(), len(jobs), 8)

    results = []
    if max_workers > 1 and len(jobs) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers for hit detection...")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(refs_by_inchi_key, auto_id_obj.workflow_params)
        ) as executor:
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
    for inchi_key, adduct, filename, ms2_hits_df in results:
        ms2_hit_obj = MS2Hit(
            inchi_key=inchi_key,
            adduct=adduct,
            filename=filename,
            data=ms2_hits_df
        )
        auto_id_obj.experimental_data.ms2_hits.append(ms2_hit_obj)

    # Limit to top N hits per compound (inchi_key + adduct) by score, across all files
    max_hits = 100
    hits_by_compound: Dict[Tuple[str, str], List] = {}
    logger.info(f"Limiting to top {max_hits} hits per compound (InChI key + adduct) across all files...")
    for hit_obj in auto_id_obj.experimental_data.ms2_hits:
        hits_by_compound.setdefault((hit_obj.inchi_key, hit_obj.adduct), []).append(hit_obj)

    for (inchi_key, adduct), hit_objs in hits_by_compound.items():
        hit_objs_with_data = [h for h in hit_objs if not h.data.empty and 'score' in h.data.columns]
        if not hit_objs_with_data:
            continue
        combined = pd.concat(
            [h.data.assign(_hit_obj_id=id(h)) for h in hit_objs_with_data],
            ignore_index=True
        )
        total_for_compound = len(combined)
        if total_for_compound > max_hits:
            combined = combined.nlargest(max_hits, 'score')
            for hit_obj in hit_objs_with_data:
                mask = combined['_hit_obj_id'] == id(hit_obj)
                hit_obj.data = combined.loc[mask].drop(columns='_hit_obj_id').reset_index(drop=True)
            logger.debug(f"Compound {inchi_key} {adduct}: trimmed {total_for_compound} hits to top {max_hits} by score.")

    # Summary
    hits_list = [
        (h.inchi_key, h.adduct, h.filename, len(h.data))
        for h in auto_id_obj.experimental_data.ms2_hits
        if not h.data.empty
    ]
    total_hits = sum(n for _, _, _, n in hits_list)
    unique_compounds = len({(ik, ad) for ik, ad, _, _ in hits_list})
    unique_files = len({fn for _, _, fn, _ in hits_list})
    logger.info(
        f"Hit detection complete: {total_hits} total hits across "
        f"{unique_compounds} compounds and {unique_files} files."
    )


def _find_hits_from_ms2_df(
    ms2_df: pd.DataFrame,
    inchi_key: str,
    refs_by_inchi_key: Dict[str, list],
    workflow_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Find reference hits for all MS2 scans in a DataFrame using cosine scoring.

    Args:
        ms2_df: MS2 data DataFrame with columns: rt, mz, i, precursor_MZ, precursor_intensity
        inchi_key: InChI key for the compound being searched
        refs_by_inchi_key: Pre-built dict mapping inchi_key -> list of matchms Spectrum objects
        workflow_params: Workflow parameters dict for thresholds etc.

    Returns:
        DataFrame with one row per hit, columns include all hit metadata
    """
    if ms2_df.empty:
        return pd.DataFrame()

    ref_spectra = refs_by_inchi_key.get(inchi_key)
    if not ref_spectra:
        return pd.DataFrame()

    frag_mz_tolerance = workflow_params.get('ms2_frag_mz_tolerance', 0.05)
    min_score = workflow_params.get('ms2_min_score', 0.01)
    min_frags = workflow_params.get('ms2_min_matching_frags', 1)
    max_ppm_error = workflow_params.get('ppm_error')

    cos = CosineHungarian(tolerance=frag_mz_tolerance)

    # Build one query Spectrum per RT group, with the actual measured precursor_mz
    query_meta = []  # (rt, precursor_mz, precursor_intensity, frag_mz, frag_int, Spectrum)

    for rt_val, rt_group in ms2_df.groupby('rt', sort=False):
        frag_mz = rt_group['mz'].values.astype(np.float32)
        frag_int = rt_group['i'].values.astype(np.float32)
        precursor_mz = float(rt_group['precursor_MZ'].iloc[0])
        precursor_int = float(rt_group['precursor_intensity'].iloc[0])

        qry = Spectrum(
            mz=frag_mz,
            intensities=frag_int,
            metadata={'precursor_mz': precursor_mz}
        )
        query_meta.append((rt_val, precursor_mz, precursor_int, frag_mz, frag_int, qry))

    if not query_meta:
        return pd.DataFrame()

    all_hits = []

    for rt_val, precursor_mz, precursor_int, frag_mz, frag_int, qry in query_meta:
        for ref in ref_spectra:
            ref_mz = ref.mz
            ref_int = ref.intensities
            precursor_mz_ref = ref.get('precursor_mz', 0.0) or 0.0

            # Compute PPM error and apply precursor MZ tolerance filter
            if precursor_mz_ref != 0:
                ppm_error = ((precursor_mz - precursor_mz_ref) / precursor_mz_ref) * 1e6
            else:
                ppm_error = np.nan
            if max_ppm_error is not None and (np.isnan(ppm_error) or abs(ppm_error) > max_ppm_error):
                continue

            # Score using pair(ref, query) — returns structured array with 'score' and 'matches'
            with _suppress_tqdm():
                result = cos.pair(ref, qry)
            score = float(result['score'])
            num_matches = int(result['matches'])

            if score < min_score:
                continue
            if num_matches < min_frags:
                continue

            # Alignment for plotting only (not used for score/match counting)
            qry_arr = np.array([frag_mz, frag_int])
            ref_arr = np.array([ref_mz, ref_int])
            alignment_data = _align_spectra_for_plotting(qry_arr, ref_arr, frag_mz_tolerance)

            ref_name = ref.get('name') or ref.get('compound_name') or ref.get('id') or 'Unknown'

            all_hits.append({
                'inchi_key': inchi_key,
                'database': ref.get('database', 'unknown'),
                'ref_id': ref.get('id', ''),
                'ref_name': ref_name,
                'score': score,
                'num_matches': num_matches,
                'mz_theoretical': float(precursor_mz_ref),
                'mz_measured': float(precursor_mz),
                'ppm_error': ppm_error,
                'rt': float(rt_val),
                'qry_intensity_peak': float(precursor_int),
                'ref_frags': len(ref_mz),
                'data_frags': len(frag_mz),
                'matched_fragments': alignment_data['matched_fragments'],
                'aligned_fragment_colors': alignment_data['fragment_colors'],
                'qry_spectrum': alignment_data['query_aligned'],
                'ref_spectrum': alignment_data['ref_aligned'],
            })

    return pd.DataFrame(all_hits) if all_hits else pd.DataFrame()


def _align_spectra_for_plotting(
    query_spectrum: np.ndarray,
    ref_spectrum: np.ndarray,
    frag_mz_tolerance: float
) -> Dict:
    """
    Align spectra using Hungarian assignment algorithm for proper mirror plotting.
    """
    try:
        query_mz = query_spectrum[0]
        ref_mz = ref_spectrum[0]

        matched_peaks = _match_peaks(query_spectrum, ref_spectrum, frag_mz_tolerance)
        matrix_size = max(len(query_mz), len(ref_mz))
        filtered_coords = _hungarian_assignment(matched_peaks, matrix_size)
        query_aligned, ref_aligned = _link_aligned_spectra(query_spectrum, ref_spectrum, filtered_coords)

        min_len = min(len(query_aligned[0]), len(ref_aligned[0]))
        fragment_colors = ["red"] * min_len
        matched_fragments = []
        for idx in range(min_len):
            q_val = query_aligned[0][idx]
            r_val = ref_aligned[0][idx]
            if not np.isnan(q_val) and not np.isnan(r_val):
                matched_fragments.append(float(q_val))
                fragment_colors[idx] = "green"

        return {
            'matched_fragments': matched_fragments,
            'fragment_colors': fragment_colors,
            'query_aligned': query_aligned.tolist() if query_aligned is not None else [[], []],
            'ref_aligned': ref_aligned.tolist() if ref_aligned is not None else [[], []],
            'num_matched': len(matched_fragments)
        }

    except Exception as e:
        logger.error(f"Error in spectrum alignment: {e}")
        return {
            'matched_fragments': [],
            'fragment_colors': ['red'] * len(query_spectrum[0]) if len(query_spectrum) > 0 else [],
            'query_aligned': [query_spectrum[0].tolist(), query_spectrum[1].tolist()] if len(query_spectrum) >= 2 else [[], []],
            'ref_aligned': [ref_spectrum[0].tolist(), ref_spectrum[1].tolist()] if len(ref_spectrum) >= 2 else [[], []],
            'num_matched': 0
        }


def _match_peaks(
    spec1: np.ndarray,
    spec2: np.ndarray,
    frag_mz_tolerance: float
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Match MS2 fragment peaks within m/z tolerance.
    Returns list of ((query_idx, ref_idx), intensity_product) tuples.
    """
    matched_peaks = []
    spec1_mz, spec1_intensity = spec1[0], spec1[1]
    spec2_mz, spec2_intensity = spec2[0], spec2[1]

    for i in range(len(spec1_mz)):
        mz_diffs = np.abs(spec2_mz - spec1_mz[i])
        within_tolerance = mz_diffs <= frag_mz_tolerance
        if np.any(within_tolerance):
            for j in np.where(within_tolerance)[0]:
                matched_peaks.append(((i, j), float(spec1_intensity[i] * spec2_intensity[j])))

    return matched_peaks


def _hungarian_assignment(
    matched_peaks: List[Tuple[Tuple[int, int], float]],
    matrix_size: int
) -> List[Tuple[int, int]]:
    """
    Filter matched peaks by maximizing matched intensity product using Hungarian algorithm.
    """
    if not matched_peaks:
        return []

    cost_matrix = np.zeros((matrix_size, matrix_size))
    for (row, col), value in matched_peaks:
        if row < matrix_size and col < matrix_size:
            cost_matrix[row, col] = value

    row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)

    return [
        (row_idx[i], col_idx[i])
        for i in range(len(row_idx))
        if cost_matrix[row_idx[i], col_idx[i]] > 0
    ]


def _link_aligned_spectra(
    spec1: np.ndarray,
    spec2: np.ndarray,
    filtered_coords: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create linked and aligned MS2 spectra using filtered matching fragment indices.
    Returns aligned spectra suitable for mirror plotting.
    """
    try:
        spec1_mz, spec1_intensity = spec1[0], spec1[1]
        spec2_mz, spec2_intensity = spec2[0], spec2[1]

        if not filtered_coords:
            return spec1, spec2

        shared_spec1_idxs = [coord[0] for coord in filtered_coords]
        shared_spec2_idxs = [coord[1] for coord in filtered_coords]

        shared_spec2_mzs = np.array([spec2_mz[i] for i in shared_spec2_idxs])
        shared_spec2_intensities = np.array([spec2_intensity[i] for i in shared_spec2_idxs])

        unshared_mask = np.ones(len(spec2_mz), dtype=bool)
        unshared_mask[shared_spec2_idxs] = False
        unshared_spec2_mzs = spec2_mz[unshared_mask]
        unshared_spec2_intensities = spec2_intensity[unshared_mask]

        # Aligned spec1: nan placeholders for unshared spec2 peaks, then all spec1 peaks
        spec1_alignment_linker = np.full(len(unshared_spec2_mzs), np.nan)
        spec1_aligned = np.array([
            np.concatenate((spec1_alignment_linker, spec1_mz)),
            np.concatenate((spec1_alignment_linker, spec1_intensity))
        ])

        # Aligned spec2: unshared spec2 peaks, then nan-filled linker with matched peaks inserted
        spec2_alignment_linker_mz = np.full(len(spec1_mz), np.nan)
        spec2_alignment_linker_intensity = np.full(len(spec1_intensity), np.nan)
        for i, spec1_idx in enumerate(shared_spec1_idxs):
            if spec1_idx < len(spec2_alignment_linker_mz):
                spec2_alignment_linker_mz[spec1_idx] = shared_spec2_mzs[i]
                spec2_alignment_linker_intensity[spec1_idx] = shared_spec2_intensities[i]

        spec2_aligned = np.array([
            np.concatenate((unshared_spec2_mzs, spec2_alignment_linker_mz)),
            np.concatenate((unshared_spec2_intensities, spec2_alignment_linker_intensity))
        ])

        return spec1_aligned, spec2_aligned

    except Exception as e:
        logger.error(f"Error in linking aligned spectra: {e}")
        return spec1, spec2