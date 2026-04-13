import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

from matchms.similarity import CosineHungarian
from matchms import Spectrum
from scipy.optimize import linear_sum_assignment

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
logger = lcf.get_logger('ms2_hit_detection')

def process_job(job, reference_df, workflow_params):
    inchi_key, adduct, file_path, ms2_df = job
    try:
        ms2_hits_df = _find_hits_from_ms2_df(ms2_df, inchi_key, reference_df, workflow_params)
        return (inchi_key, adduct, file_path, ms2_hits_df)
    except Exception as e:
        logger.error(f"Error in hit detection for {inchi_key} {adduct} {file_path}: {e}")
        return (inchi_key, adduct, file_path, pd.DataFrame())

def find_ms2_hits(
    auto_id_obj: "AutoID",
) -> "ExperimentalData":
    """
    Find MS2 hits for all compounds with MS2 data in the experimental dataset using a reference spectra database.
    Uses MatchMS cosine similarity scoring and Hungarian algorithm for peak alignment.
    Parallelized across compounds/files with MS2 data for faster processing.
    """

    from metatlas2.workflow_objects import MS2Hit

    reference_df = ldt.load_msms_refs_file(Path(auto_id_obj.paths['msms_refs_path']))
    if reference_df.empty:
        raise FileNotFoundError("No reference database found - skipping hit detection")

    logger.info(f"Finding MS2 hits using reference database with {len(reference_df)} entries...")

    jobs = []
    for ms2 in auto_id_obj.experimental_data.ms2_data:
        if not ms2.data.empty:
            jobs.append((ms2.inchi_key, ms2.adduct, ms2.filename, ms2.data))

    if not jobs:
        logger.warning("No MS2 data found for any compounds, skipping hit detection")
        return
    else:
        logger.info(f"Prepared hit detection input for {len(jobs)} files with MS2 data")

    max_workers = min(mp.cpu_count(), len(jobs), 8)
    use_parallel = max_workers > 1 and len(jobs) > 1

    results = []
    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers for hit detection...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_job, job, reference_df, auto_id_obj.workflow_params) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        logger.info("Using sequential processing for hit detection...")
        results = [process_job(job, reference_df, auto_id_obj.workflow_params) for job in jobs]

    # Assign results to ExperimentalData.ms2_hits as MS2Hit objects
    for inchi_key, adduct, filename, ms2_hits_df in results:
        ms2_hit_obj = MS2Hit(
            inchi_key=inchi_key,
            adduct=adduct,
            filename=filename,
            data=ms2_hits_df
        )
        auto_id_obj.experimental_data.ms2_hits.append(ms2_hit_obj)

    # Limit to top N hits per compound (inchi_key) by score, across all files and references
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
            logger.info(f"Compound {inchi_key} {adduct}: trimmed {total_for_compound} hits to top {max_hits} by score.")

    # Print summary broken down by compound and file
    hits_list = [(h.inchi_key, h.adduct, h.filename, len(h.data))
                 for h in auto_id_obj.experimental_data.ms2_hits if not h.data.empty]
    total_hits = sum(n for _, _, _, n in hits_list)
    unique_compounds = len({(ik, ad) for ik, ad, _, _ in hits_list})
    unique_files = len({fn for _, _, fn, _ in hits_list})
    logger.info(f"Hit detection complete: {total_hits} total hits across {unique_compounds} compounds and {unique_files} files.")

    return

def _find_hits_from_ms2_df(
    ms2_df: pd.DataFrame, 
    inchi_key: str, 
    reference_df: pd.DataFrame,
    workflow_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Find reference hits for all MS2 scans in a DataFrame.
    
    Args:
        ms2_df: MS2 data DataFrame with columns: rt, mz, i, precursor_MZ, precursor_intensity
        inchi_key: InChI key for compound
        reference_df: Reference spectra database
        workflow_params: Dictionary of workflow parameters from config for hit detection thresholds, etc.
        
    Returns:
        DataFrame with one row per hit, columns include all hit metadata
    """
    if ms2_df.empty:
        return pd.DataFrame()
    
    all_hits = []
    
    # Set up MatchMS scoring
    frag_mz_tolerance = workflow_params.get('ms2_frag_mz_tolerance', 0.05)
    cos = CosineHungarian(tolerance=frag_mz_tolerance)
    
    # Find matching reference spectra by inchi_key
    matching_refs = reference_df[reference_df['inchi_key'] == inchi_key]
    if matching_refs.empty:
        return pd.DataFrame()
    
    # Group by RT to get individual MS2 scans
    for rt_val, rt_group in ms2_df.groupby('rt'):
        if rt_group.empty:
            continue
            
        # Get precursor info (should be same for all fragments in scan)
        precursor_mz = rt_group['precursor_MZ'].iloc[0]
        precursor_intensity = rt_group['precursor_intensity'].iloc[0]
        
        # Build spectrum from fragments
        fragment_mz = rt_group['mz'].values
        fragment_intensity = rt_group['i'].values
        
        # Create MatchMS query spectrum
        mms_query = Spectrum(mz=fragment_mz, intensities=fragment_intensity, metadata={'precursor_mz': np.nan})
        
        # Compare against each reference
        for _, ref_row in matching_refs.iterrows():
            try:
                ref_spectrum = ref_row.get('spectrum', None)
                if ref_spectrum is None or len(ref_spectrum) != 2:
                    continue
                
                ref_mz = np.array(ref_spectrum[0], dtype=np.float64)
                ref_intensity = np.array(ref_spectrum[1], dtype=np.float64)
                
                if len(ref_mz) == 0 or len(ref_intensity) == 0:
                    continue
                
                # Get precursor m/z from reference for ppm error calculation
                precursor_mz_ref = ref_row.get('precursor_mz', 0.0)
                
                # Create MatchMS reference spectrum
                mms_ref = Spectrum(mz=ref_mz, intensities=ref_intensity, metadata={'precursor_mz': precursor_mz_ref})
                
                # Calculate MatchMS score
                mms_comparison = cos.pair(mms_query, mms_ref)
                score = mms_comparison['score'] if mms_comparison['score'] is not None else 0.0
                if score < workflow_params.get('ms2_min_score', 0.01):
                    continue

                # Perform custom alignment for plotting
                query_spectrum_array = np.array([fragment_mz, fragment_intensity])
                ref_spectrum_array = np.array([ref_mz, ref_intensity])
                alignment_data = _align_spectra_for_plotting(query_spectrum_array, ref_spectrum_array, frag_mz_tolerance)
                if len(alignment_data.get('matched_fragments', [])) < workflow_params.get('ms2_min_matching_frags', 1):
                    continue
                
                # Create hit data row
                hit_data = {
                    'inchi_key': inchi_key,
                    'database': str(ref_row.get('database', 'unknown')),
                    'ref_id': str(ref_row.get('id', '')),
                    'ref_name': str(ref_row.get('name', 'Unknown')),
                    'score': float(score),
                    'num_matches': len(alignment_data.get('matched_fragments', [])),
                    'mz_theoretical': float(precursor_mz_ref),
                    'mz_measured': float(precursor_mz),
                    'ppm_error': ((precursor_mz - precursor_mz_ref) / precursor_mz_ref) * 1e6 if precursor_mz_ref != 0 else np.nan,
                    'rt': float(rt_val),
                    'qry_intensity_peak': float(precursor_intensity),
                    'ref_frags': len(ref_mz),
                    'data_frags': len(fragment_mz),
                    'matched_fragments': alignment_data.get('matched_fragments', []),
                    'aligned_fragment_colors': alignment_data.get('fragment_colors', []),
                    'qry_spectrum': alignment_data.get('query_aligned', [[], []]),
                    'ref_spectrum': alignment_data.get('ref_aligned', [[], []]),
                }
                
                all_hits.append(hit_data)
                
            except Exception as e:
                logger.error(f"Error processing reference hit: {e}")
                continue
    
    # Convert to DataFrame
    if all_hits:
        return pd.DataFrame(all_hits)
    else:
        return pd.DataFrame()

def _align_spectra_for_plotting(query_spectrum: np.ndarray, ref_spectrum: np.ndarray, 
                               frag_mz_tolerance: float) -> Dict:
    """
    Align spectra using Hungarian assignment algorithm for proper mirror plotting.
    Based on the provided MatchMS workflow.
    """
    try:
        query_mz = query_spectrum[0]
        ref_mz = ref_spectrum[0]
        
        matched_peaks = _match_peaks(query_spectrum, ref_spectrum, frag_mz_tolerance)
        matrix_size = max(len(query_mz), len(ref_mz))
        filtered_coords = _hungarian_assignment(matched_peaks, matrix_size)
        query_aligned, ref_aligned = _link_aligned_spectra(query_spectrum, ref_spectrum, filtered_coords)
        
        # Generate fragment colors and matched fragment list based on aligned spectra
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

def _match_peaks(spec1: np.ndarray, spec2: np.ndarray, frag_mz_tolerance: float) -> List[Tuple[Tuple[int, int], float]]:
    """
    Match MS2 fragment peaks within m/z tolerance.
    Returns list of ((query_idx, ref_idx), intensity_product) tuples.
    """
    matched_peaks = []
    spec1_mz, spec1_intensity = spec1[0], spec1[1]
    spec2_mz, spec2_intensity = spec2[0], spec2[1]
    
    for i in range(len(spec1_mz)):
        # Find peaks in spec2 within tolerance of current spec1 peak
        mz_diffs = np.abs(spec2_mz - spec1_mz[i])
        within_tolerance = mz_diffs <= frag_mz_tolerance
        
        if np.any(within_tolerance):
            matching_indices = np.where(within_tolerance)[0]
            for j in matching_indices:
                match_coords = (i, j)
                match_value = float(spec1_intensity[i] * spec2_intensity[j])
                matched_peaks.append((match_coords, match_value))
    
    return matched_peaks

def _hungarian_assignment(matched_peaks: List[Tuple[Tuple[int, int], float]], 
                         matrix_size: int) -> List[Tuple[int, int]]:
    """
    Filter matched peaks by maximizing the matched intensity product using Hungarian algorithm.
    """
    if not matched_peaks:
        return []
    
    # Create cost matrix
    cost_matrix = np.zeros((matrix_size, matrix_size))
    for match in matched_peaks:
        coords, value = match
        row, col = coords
        if row < matrix_size and col < matrix_size:
            cost_matrix[row, col] = value
    
    # Solve assignment problem (maximize)
    row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=True)
    
    # Filter to only include assignments with non-zero costs
    filtered_coords = []
    for i in range(len(row_idx)):
        row, col = row_idx[i], col_idx[i]
        if cost_matrix[row, col] > 0:
            filtered_coords.append((row, col))
    
    return filtered_coords

def _link_aligned_spectra(spec1: np.ndarray, spec2: np.ndarray, 
                         filtered_coords: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create linked and aligned MS2 spectra using filtered matching fragment indices.
    Returns aligned spectra suitable for mirror plotting.
    """
    try:
        spec1_mz, spec1_intensity = spec1[0], spec1[1]
        spec2_mz, spec2_intensity = spec2[0], spec2[1]
        
        if not filtered_coords:
            # No matches, return original spectra
            return spec1, spec2
        
        shared_spec1_idxs = [coord[0] for coord in filtered_coords]
        shared_spec2_idxs = [coord[1] for coord in filtered_coords]
        
        # Get shared and unshared peaks from spec2
        shared_spec2_mzs = np.array([spec2_mz[i] for i in shared_spec2_idxs])
        shared_spec2_intensities = np.array([spec2_intensity[i] for i in shared_spec2_idxs])
        
        unshared_spec2_mzs = np.array([spec2_mz[i] for i in range(len(spec2_mz)) if i not in shared_spec2_idxs])
        unshared_spec2_intensities = np.array([spec2_intensity[i] for i in range(len(spec2_intensity)) if i not in shared_spec2_idxs])
        
        # Create aligned spec1: [nan placeholders for unshared spec2 peaks] + [all spec1 peaks]
        spec1_alignment_linker = np.full(len(unshared_spec2_mzs), np.nan)
        aligned_spec1_mz = np.concatenate((spec1_alignment_linker, spec1_mz))
        aligned_spec1_intensity = np.concatenate((spec1_alignment_linker, spec1_intensity))
        spec1_aligned = np.array([aligned_spec1_mz, aligned_spec1_intensity])
        
        # Create aligned spec2: [unshared spec2 peaks] + [matched spec2 peaks + nan placeholders]
        spec2_alignment_linker_mz = np.full(len(spec1_mz), np.nan)
        spec2_alignment_linker_intensity = np.full(len(spec1_intensity), np.nan)
        
        # Fill in the matched peaks at their corresponding positions
        for i, spec1_idx in enumerate(shared_spec1_idxs):
            if spec1_idx < len(spec2_alignment_linker_mz):
                spec2_alignment_linker_mz[spec1_idx] = shared_spec2_mzs[i]
                spec2_alignment_linker_intensity[spec1_idx] = shared_spec2_intensities[i]
        
        aligned_spec2_mz = np.concatenate((unshared_spec2_mzs, spec2_alignment_linker_mz))
        aligned_spec2_intensity = np.concatenate((unshared_spec2_intensities, spec2_alignment_linker_intensity))
        spec2_aligned = np.array([aligned_spec2_mz, aligned_spec2_intensity])
        
        return spec1_aligned, spec2_aligned
        
    except Exception as e:
        logger.error(f"Error in linking aligned spectra: {e}")
        return spec1, spec2