import sys
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from IPython.display import display

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import ms2_hit_detection as mhd
import logging_config as lcf
import extract_data_from_parquet as pdx
import ms1_ms2_summarizer as mss

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_analysis')
current_time = datetime.now().isoformat()


def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str,
                                    rt_alignment_number: int,
                                    analysis_number: int,
                                    config: Dict,
                                    analysis_params: Dict) -> Tuple[pd.DataFrame, Dict]: 
    """
    Execute the complete targeted analysis workflow independently of workflow objects.
    Returns atlas DataFrame and analysis results dictionary that can be used to 
    create workflow objects in the calling module.
    
    Returns:
        Tuple of (atlas_dataframe, analysis_results_dict)
        
    analysis_results_dict contains:
        - compounds: Dict[str, Dict] - compound analysis data by inchi_key
        - atlas_info: Dict - atlas metadata
        - summary_stats: Dict - analysis summary statistics
    """
    logger.info("Running new targeted analysis...")
    
    logger.info("Loading target atlas...")
    main_db_path = config["ENV"]["PATHS"]["main_database"]
    atlas_dataframe = dbi.get_atlas_compounds_table(database_path=main_db_path, 
                                                    atlas_uid=target_atlas_uid)
    if len(atlas_dataframe) == 0:
        raise ValueError(f"No compounds found in RT-aligned atlas")
    else:
        logger.info(f"Created Atlas dataframe with {len(atlas_dataframe)} compounds")

    logger.info("Loading experimental files from project database...")
    project_files_df = dbi.get_files_by_type_from_db(project_db_path=project_db_path, 
                                                     file_types=['experimental', 'istd', 'exctrl'])
    project_files = project_files_df['file_path'].tolist()
    if len(project_files) == 0:
        raise ValueError("No experimental files found in project database")
    else:
        logger.info(f"Found {len(project_files)} experimental files")

    logger.info("Extracting EIC and MS2 data from parquet files...")
    experimental_data_no_hits = pdx.extract_eic_and_ms2_from_parquet(atlas_df=atlas_dataframe,
                                                                     parquet_files=project_files,
                                                                     ppm_tolerance=analysis_params["default_ppm_error"],
                                                                     extra_time=analysis_params["extra_time"],
                                                                     use_parallel=True if len(project_files) > 1 else False)
    
    logger.info("Finding MS2 reference hits...")
    experimental_data_with_hits = mhd.find_ms2_hits(experimental_data=experimental_data_no_hits, 
                                                    config=config)
    
    logger.info("Calculating MS1 and MS2 summary statistics for each compound and file...")
    experimental_data_with_hits_and_summaries = mss.create_ms_summaries(experimental_data_with_hits)

    logger.info("Saving experimental data to project database...")
    _save_experimental_data_to_db(exp_data=experimental_data_with_hits_and_summaries,
                                  rt_alignment_number=rt_alignment_number,
                                  analysis_number=analysis_number,
                                  project_db_path=project_db_path)

    logger.info("Setting up targeted analysis results structure...")
    analysis_results = _create_analysis_results_dict(exp_data=experimental_data_with_hits_and_summaries,
                                                      main_db_path=main_db_path,
                                                      target_atlas_uid=target_atlas_uid,
                                                      project_db_path=project_db_path,
                                                      atlas_dataframe=atlas_dataframe)

    logger.info("Calculating summary statistics for analysis results...")
    _run_analysis_summary(analysis_results)

    return atlas_dataframe, analysis_results

def _save_experimental_data_to_db(exp_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], 
                                  rt_alignment_number: int,
                                  analysis_number: int,
                                  project_db_path: str) -> None:
    """
    Save experimental data from analysis results structure to project database.
    Independent of workflow objects.
    """
    logger.info("Saving all MS1 and MS2 data")
    
    for inchi_key, file_results in exp_data.items():
        for filename, ms_level_results in file_results.items():
            for ms_level, data in ms_level_results.items():
                if data.empty:
                    logger.debug(f"Skipping saving empty dataframe for {ms_level} for {inchi_key} in file {filename}")
                    continue
                if ms_level == "ms1_data":
                    try:
                        dbi.save_ms1_data_to_db(
                            project_db_path=project_db_path,
                            inchi_key=inchi_key,
                            rt_alignment_number=rt_alignment_number,
                            analysis_number=analysis_number,
                            file_path=filename,
                            ms1_df=data
                        )
                    except Exception as e:
                        raise ValueError(f"Error saving MS1 data for {inchi_key} in file {filename}: {e}")
                if ms_level == "ms2_data":
                    try:
                        dbi.save_ms2_data_to_db(
                            project_db_path=project_db_path,
                            inchi_key=inchi_key,
                            rt_alignment_number=rt_alignment_number,
                            analysis_number=analysis_number,
                            file_path=filename,
                            ms2_df=data
                        )
                    except Exception as e:
                        raise ValueError(f"Error saving MS2 data for {inchi_key} in file {filename}: {e}")
                elif ms_level == "ms2_hits":
                    try:
                        dbi.save_ms2_hits_to_db(
                            project_db_path=project_db_path,
                            inchi_key=inchi_key,
                            rt_alignment_number=rt_alignment_number,
                            analysis_number=analysis_number,
                            file_path=filename,
                            ms2_hits_df=data
                        )
                    except Exception as e:
                        raise ValueError(f"Error saving MS2 hits for {inchi_key} in file {filename}: {e}")
                elif ms_level == "ms1_summary":
                    try:
                        dbi.save_ms1_summary_to_db(
                            project_db_path=project_db_path,
                            inchi_key=inchi_key,
                            rt_alignment_number=rt_alignment_number,
                            analysis_number=analysis_number,
                            file_path=filename,
                            ms1_summary_df=data
                        )
                    except Exception as e:
                        raise ValueError(f"Error saving MS1 summary for {inchi_key} in file {filename}: {e}")
                elif ms_level == "ms2_summary":
                    try:
                        dbi.save_ms2_summary_to_db(
                            project_db_path=project_db_path,
                            inchi_key=inchi_key,
                            rt_alignment_number=rt_alignment_number,
                            analysis_number=analysis_number,
                            file_path=filename,
                            ms2_summary_df=data
                        )
                    except Exception as e:
                        raise ValueError(f"Error saving MS2 summary for {inchi_key} in file {filename}: {e}")
    
    logger.info("Experimental data saved to database")
    return

def _create_analysis_results_dict(exp_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
                                  main_db_path: str,
                                  target_atlas_uid: str,
                                  project_db_path: str,
                                  atlas_dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Set up the database structure for storing analysis results.
    """

    logger.info("Creating analysis results structure with compound data from atlas...")
    analysis_results = _create_analysis_results_structure(target_atlas_uid = target_atlas_uid,
                                                          main_db_path = main_db_path,
                                                          project_db_path = project_db_path,
                                                          atlas_dataframe = atlas_dataframe)
    
    logger.info("Add experimental data to analysis results...")
    _apply_experimental_data_to_results(analysis_results, exp_data)
    _apply_isomer_detection_to_results(analysis_results, atlas_dataframe)
    _apply_rt_bounds_suggestions_to_results(analysis_results)

    return analysis_results

def _create_analysis_results_structure(target_atlas_uid: str,
                                       main_db_path: str,
                                       project_db_path: str,
                                       atlas_dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Create the initial analysis results structure with compound data from the atlas.
    """

    analysis_results = {
        'compounds': {},
        'atlas_info': {
            'atlas_uid': target_atlas_uid,
            'project_db_path': project_db_path,
            'main_db_path': main_db_path
        }
    }

    # Initialize compound data from atlas
    for _, row in atlas_dataframe.iterrows():
        inchi_key = row['inchi_key']
        analysis_results['compounds'][inchi_key] = {
            # Core identifiers
            'compound_uid': row['compound_uid'],
            'inchi_key': inchi_key,
            'compound_name': row.get('compound_name', row.get('label', '')),
            'formula': row.get('formula', ''),
            'mz': row.get('mz', 0.0),
            'adduct': row.get('adduct', ''),
            'polarity': row.get('polarity', ''),
            'chromatography': row.get('chromatography', ''),
            'mz_tolerance': row.get('mz_tolerance', 5.0),
            
            # Original RT data from atlas
            'original_rt_peak': row.get('rt_peak', 0.0),
            'original_rt_min': row.get('rt_min', 0.0),
            'original_rt_max': row.get('rt_max', 0.0),
            
            # Current RT data (modifiable)
            'rt_peak': row.get('rt_peak', 0.0),
            'rt_min': row.get('rt_min', 0.0),
            'rt_max': row.get('rt_max', 0.0),
            
            # Analysis annotations
            'ms1_notes': 'keep',
            'ms2_notes': 'no selection',
            'analyst_notes': '',
            'identification_notes': '',
            
            # Experimental data containers
            'eic_data_files': {},
            'ms2_data_files': {},
            'isomers': [],
            'suggested_rt_bounds': None,
            
            # Best results
            'best_eic_file': '',
            'best_eic_rt': 0.0,
            'best_eic_mz': 0.0,
            'best_eic_intensity': 0.0,
            'best_eic_ppm_error': 0.0,
            'best_eic_rt_error': 0.0,
            
            'best_ms2_file': '',
            'best_ms2_database': '',
            'best_ms2_score': 0.0,
            'best_ms2_num_matches': 0,
            'best_ms2_matched_fragments': [],
            
            # Summary stats
            'total_files_detected': 0,
            'ms2_files_with_data': 0,
            
            # Modification tracking
            'is_rt_modified': False,
            'is_annotation_modified': False
        }

    if not analysis_results['compounds']:
        raise ValueError("No compounds initialized in analysis results structure")
    else:
        logger.info(f"Initialized analysis results structure for {len(analysis_results['compounds'])} compounds")

    return analysis_results

def _build_isomer_dict(atlas_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Return a dict: inchi_key → list of isomer dicts (empty list if none).
    Isomers are defined as:
      - mz or mono_isotopic_molecular_weight within 0.005
      - OR inchi_key prefix (before '-') identical
    """
    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in atlas_df.iterrows():
        mz = row["mz"]
        mono_isotopic_molecular_weight = row.get("mono_isotopic_molecular_weight", None)
        inchi_prefix = row["inchi_key"].split("-")[0]
        def is_isomer(r):
            if r["inchi_key"] == row["inchi_key"]:
                return False
            mz_close = abs(r["mz"] - mz) <= 0.005
            mass_close = (
                mono_isotopic_molecular_weight is not None and r.get("mono_isotopic_molecular_weight", None) is not None and
                abs(r["mono_isotopic_molecular_weight"] - mono_isotopic_molecular_weight) <= 0.005
            )
            prefix_match = r["inchi_key"].split("-")[0] == inchi_prefix
            return mz_close or mass_close or prefix_match
        isomers = atlas_df[atlas_df.apply(is_isomer, axis=1)]
        isomer_dict[row["inchi_key"]] = [
            {
                "inchi_key": r["inchi_key"],
                "compound_name": r["label"],
                "rt": r["rt_peak"],
                "mz": r["mz"],
                "mz_tolerance": r.get("mz_tolerance_ppm", 10.0),
            }
            for _, r in isomers.iterrows()
        ]
    return isomer_dict

def _suggest_rt_bounds_from_eic(
    eic_data: Dict[str, Dict[str, Any]],
    atlas_rt_peak: float,
    atlas_rt_min: float,
    atlas_rt_max: float
) -> Optional[Dict[str, float]]:
    """
    Compute RT bounds from the *average* extracted‑ion chromatogram (EIC)
    of many LC‑MS/MS files.
    """
    if not eic_data:
        return None

    # Don't look through every file, too cumbersome
    sorted_files = sorted(
        eic_data.items(),
        key=lambda kv: float(kv[1].get("intensity_peak", 0)),
        reverse=True,
    )
    selected = sorted_files[:50]

    # Check EICs for bad data
    rt_lists: List[np.ndarray] = []
    int_lists: List[np.ndarray] = []
    weights: List[float] = []

    for fname, trace in selected:
        rt_raw = trace.get("rt_vals", [])
        i_raw  = trace.get("i_vals", [])

        rt_arr = np.asarray(rt_raw, dtype=np.float64)
        i_arr  = np.asarray(i_raw,  dtype=np.float64)

        valid = (~np.isnan(rt_arr)) & (~np.isnan(i_arr)) & (i_arr >= 0)
        rt_arr = rt_arr[valid]
        i_arr  = i_arr[valid]

        if rt_arr.size < 5:
            continue

        rt_lists.append(rt_arr)
        int_lists.append(i_arr)

        w = float(trace.get("intensity_peak", 1.0))
        if np.isnan(w) or w <= 0:
            w = 1.0
        weights.append(w)

    if not rt_lists:
        return None

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    # Combine samples
    global_min = min(rt.min() for rt in rt_lists)
    global_max = max(rt.max() for rt in rt_lists)

    # Choose a step size
    all_spacings = np.concatenate(
        [np.diff(rt) for rt in rt_lists if rt.size > 1]
    )
    step = np.median(all_spacings) if all_spacings.size else 0.01

    # Guard against a zero step (unlikely but possible)
    if step <= 0:
        step = 0.01

    common_rt = np.arange(global_min, global_max + step, step)

    # Make a commond grid by interpolating (not all rts the same)
    interpolated = []
    for rt, intensity in zip(rt_lists, int_lists):
        if np.array_equal(rt, common_rt):
            interp_i = intensity
        else:
            f = interp1d(rt, intensity, kind="linear",
                         bounds_error=False, fill_value=0.0)
            interp_i = f(common_rt)
        interpolated.append(interp_i)

    intensity_matrix = np.vstack(interpolated)

    # Average EICs
    weighted_avg = np.average(intensity_matrix, axis=0, weights=weights)
    ma_window = 5
    smoothed = _moving_average(weighted_avg, window=ma_window)
    if ma_window > 1:
        pad = (ma_window - 1) // 2
        smoothed = np.pad(smoothed, (pad, pad), mode="edge")
        smoothed = smoothed[:common_rt.size]

    # Peak detection
    if np.max(smoothed) <= 0:
        return None

    # Use a modest height / prominence threshold (10 % / 5 % of max)
    max_int = np.max(smoothed)
    min_height = max_int * 0.10
    min_prom   = max_int * 0.05

    peaks, _ = find_peaks(
        smoothed,
        height=min_height,
        prominence=min_prom,
        distance=5
    )

    if peaks.size == 0:
        peaks = np.array([np.argmax(smoothed)])

    # Choose best peak
    best_idx = np.argmax(smoothed[peaks])
    best_peak = peaks[best_idx]
    best_rt   = common_rt[best_peak]
    best_int  = smoothed[best_peak]

    # Calculate peak bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominences = peak_prominences(smoothed, [best_peak])[0]
    if len(prominences) == 0 or prominences[0] == 0.0:
        left_ips = [0]
        right_ips = [len(smoothed) - 1]
        widths = [right_ips[0] - left_ips[0]]
    else:
        widths, _, left_ips, right_ips = peak_widths(
            smoothed,
            [best_peak],
            rel_height=0.5
        )
    # Convert from index space to RT space
    left_idx  = int(np.floor(left_ips[0]))
    right_idx = int(np.ceil(right_ips[0]))
    left_idx  = max(0, left_idx)
    right_idx = min(len(common_rt) - 1, right_idx)

    rt_left  = common_rt[left_idx]
    rt_right = common_rt[right_idx]

    # Add a small padding (5 % of the width or 0.05 min, whichever is larger)
    width_rt = rt_right - rt_left
    pad = max(0.05, width_rt * 0.05)
    rt_min = rt_left - pad
    rt_max = rt_right + pad

    # Conf score
    # intensity score
    intensity_score = min(1.0, best_int / max(max_int, 1e3))

    # prominence score
    if len(prominences) == 0 or prominences[0] == 0.0:
        prominence_score = 0.0
    else:
        prominence_score = min(1.0, prominences[0] / max(best_int * 0.5, 1e3))

    # width score (optimal width ≈ 20 points for typical chromatograms)
    optimal_width_pts = 20
    width_score = 1.0 - min(1.0, abs(widths[0] - optimal_width_pts) / optimal_width_pts)

    # RT proximity score
    max_expected_dev = max(abs(atlas_rt_max - atlas_rt_min), 1.0)
    rt_dev = abs(best_rt - atlas_rt_peak)
    rt_score = max(0.0, 1.0 - (rt_dev / max_expected_dev))

    # shape symmetry score
    left_tail  = best_peak - left_idx
    right_tail = right_idx - best_peak
    asym = abs(left_tail - right_tail) / max(left_tail + right_tail, 1)
    shape_score = max(0.0, 1.0 - asym)

    # weight scores
    confidence = (
        0.30 * intensity_score +
        0.20 * prominence_score +
        0.15 * width_score +
        0.20 * rt_score +
        0.15 * shape_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < 0.1:
        return None
    if width_rt > (atlas_rt_max - atlas_rt_min) * 2:
        confidence *= 0.5
    if rt_dev > max_expected_dev * 2:
        confidence *= 0.3

    return {
        "rt_min": float(rt_min),
        "rt_max": float(rt_max),
        "rt_peak": float(best_rt),
        "confidence": float(confidence),
    }

def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving‑average.  window=1 returns the original array."""
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def _apply_experimental_data_to_results(analysis_results: Dict, experimental_data: Dict) -> None:
    """
    Add experimental data to raw analysis results structure (analysis_results is an empty object for filling).
    
    Args:
        analysis_results: Analysis results dictionary to populate
        experimental_data: Data from extract_eic_and_ms2_from_parquet() and find_ms2_hits()
            Format: {inchi_key: {filename: {'ms1_data': df, 'ms1_summary': df, 'ms2_data': df, 'ms2_hits': df, 'ms2_summary': df}}}
    """
    logger.info("Filling in analysis results template with experimental data...")

    for inchi_key, file_results in experimental_data.items():
        if inchi_key not in analysis_results['compounds']:
            logger.debug(f"  Compound {inchi_key} not found in analysis results, skipping experimental data addition")
            continue
            
        logger.debug(f"  Found compound {inchi_key} in analysis results template, adding data...")
        compound = analysis_results['compounds'][inchi_key]
        
        # Process each file for this compound
        for filename, data_dict in file_results.items():

            # Extract MS1 (EIC) data
            ms1_data = data_dict.get('ms1_data', pd.DataFrame())
            ms1_summary = data_dict.get('ms1_summary', pd.DataFrame())
            
            if not ms1_data.empty and not ms1_summary.empty:
                # Convert ms1_summary row to dict for storage
                summary_dict = ms1_summary.iloc[0].to_dict() if len(ms1_summary) > 0 else {}
                
                eic_data = {
                    'rt_vals': ms1_data['rt'].tolist(),
                    'mz_vals': ms1_data['mz'].tolist(),
                    'i_vals': ms1_data['i'].tolist(),
                    'rt_peak': float(summary_dict.get('rt_peak', 0.0)),
                    'mz_peak': float(summary_dict.get('mz_peak', 0.0)),
                    'intensity_peak': float(summary_dict.get('intensity_peak', 0.0)),
                    'ppm_diff': float(summary_dict.get('ppm_diff', 0.0)),
                    'rt_diff': float(summary_dict.get('rt_diff', 0.0)),
                    'num_datapoints': int(summary_dict.get('num_datapoints', 0))
                }
                compound['eic_data_files'][filename] = eic_data
                logger.debug(f"    Added MS1 data for file {filename}")
            
            # Extract MS2 data
            ms2_data = data_dict.get('ms2_data', pd.DataFrame())
            ms2_hits = data_dict.get('ms2_hits', pd.DataFrame())
            ms2_summary = data_dict.get('ms2_summary', pd.DataFrame())
            
            if not ms2_data.empty:
                # Store MS2 data in a structured format
                ms2_file_data = {
                    'ms2_data': ms2_data,
                    'ms2_hits': ms2_hits,
                    'ms2_summary': ms2_summary,
                }
                
                # Extract summary statistics if available
                if not ms2_summary.empty:
                    summary_dict = ms2_summary.iloc[0].to_dict()
                    ms2_file_data['num_scans'] = int(summary_dict.get('num_scans', 0))
                    ms2_file_data['num_fragments'] = int(summary_dict.get('num_fragments', 0))
                    ms2_file_data['best_ms2_rt'] = float(summary_dict.get('best_ms2_rt', 0.0))
                    ms2_file_data['best_ms2_mz'] = float(summary_dict.get('best_ms2_mz', 0.0))
                    ms2_file_data['best_ms2_intensity'] = float(summary_dict.get('best_ms2_intensity', 0.0))
                    ms2_file_data['num_hits'] = int(summary_dict.get('num_hits', 0))
                    
                    # Extract best hit info
                    if not ms2_hits.empty and summary_dict.get('num_hits', 0) > 0:
                        best_hit_idx = ms2_hits['score'].idxmax()
                        best_hit = ms2_hits.loc[best_hit_idx]
                        
                        ms2_file_data['best_hit'] = {
                            'database': str(best_hit['database']),
                            'ref_id': str(best_hit['ref_id']),
                            'ref_name': str(best_hit['ref_name']),
                            'score': float(best_hit['score']),
                            'num_matches': int(best_hit['num_matches']),
                            'mz_theoretical': float(best_hit['mz_theoretical']),
                            'mz_measured': float(best_hit['mz_measured']),
                            'rt_measured': float(best_hit['rt_measured']),
                            'matched_fragments': best_hit['matched_fragments'][0] if isinstance(best_hit['matched_fragments'], list) else best_hit['matched_fragments']
                        }
                else:
                    ms2_file_data['num_scans'] = 0
                    ms2_file_data['num_hits'] = 0
                    ms2_file_data['best_hit'] = {}
                
                compound['ms2_data_files'][filename] = ms2_file_data
                logger.debug(f"    Added MS2 data for file {filename} ({ms2_file_data.get('num_scans', 0)} scans, {ms2_file_data.get('num_hits', 0)} hits)")
        
        # Update summary statistics
        compound['total_files_detected'] = len(compound['eic_data_files'])
        if compound['ms2_data_files']:
            compound['ms2_files_with_data'] = len([
                f for f, data in compound['ms2_data_files'].items() 
                if data.get('num_scans', 0) > 0
            ])
        
        # Update best EIC and MS2 results
        logger.debug(f"    Calculating best EIC and MS2 results for compound {inchi_key}...")
        _calculate_best_eic_results(compound)
        _calculate_best_ms2_results(compound)

def _calculate_best_eic_results(compound: Dict) -> None:
    """Calculate best EIC statistics from current data."""
    if not compound['eic_data_files']:
        logger.debug("      No EIC data files to calculate best EIC results")
        return
    
    best_intensity = 0.0    
    for filename, eic_data in compound['eic_data_files'].items():
        intensity = eic_data.get('intensity_peak', 0.0)
        if intensity > best_intensity:
            logger.debug(f"      New best EIC found in file {filename} with intensity {intensity}")
            best_intensity = intensity
            compound['best_eic_file'] = filename
            compound['best_eic_rt'] = eic_data.get('rt_peak', 0.0)
            compound['best_eic_mz'] = eic_data.get('mz_peak', 0.0)
            compound['best_eic_intensity'] = intensity
            compound['best_eic_ppm_error'] = eic_data.get('ppm_diff', 0.0)
            compound['best_eic_rt_error'] = eic_data.get('rt_diff', 0.0)

def _calculate_best_ms2_results(compound: Dict) -> None:
    """Calculate best MS2 statistics from current data."""
    if not compound['ms2_data_files']:
        logger.debug("      No MS2 data files to calculate best MS2 results")
        return
    
    best_score = 0.0
    best_hit_data = None
    for filename, ms2_data in compound['ms2_data_files'].items():
        if isinstance(ms2_data, dict):
            best_hit = ms2_data.get('best_hit', {})
            if best_hit and best_hit.get('score', 0.0) > best_score:
                best_score = best_hit.get('score', 0.0)
                best_hit_data = best_hit
                compound['best_ms2_file'] = filename
    
    if best_hit_data:
        logger.debug(f"      New best MS2 found in file {compound['best_ms2_file']} with score {best_score}")
        compound['best_ms2_database'] = best_hit_data.get('database', '')
        compound['best_ms2_score'] = best_hit_data.get('score', 0.0)
        compound['best_ms2_num_matches'] = best_hit_data.get('num_matches', 0)
        compound['best_ms2_matched_fragments'] = best_hit_data.get('matched_fragments', [])

def _apply_isomer_detection_to_results(analysis_results: Dict, atlas_dataframe: pd.DataFrame) -> None:
    """
    Apply isomer detection to raw analysis results structure.
    Independent of workflow objects.
    """
    logger.info("Applying isomer detection...")
    
    # Generate isomers dictionary
    isomer_dict = _build_isomer_dict(atlas_dataframe)
        
    # Debug: Show some examples
    compounds_with_isomers = 0
    for inchi_key, isomers in isomer_dict.items():
        if isomers:
            compounds_with_isomers += 1
    
    logger.info(f"Found {compounds_with_isomers} compounds with isomers")
    
    # Apply to compounds in analysis results
    for inchi_key, compound in analysis_results['compounds'].items():
        compound['isomers'] = isomer_dict.get(inchi_key, [])

    return

def _apply_rt_bounds_suggestions_to_results(analysis_results: Dict) -> None:
    """
    Apply RT bounds suggestions to raw analysis results structure.
    Independent of workflow objects.
    """
    logger.info("Applying RT bounds suggestions to EIC data...")
    
    compounds_processed = 0
    compounds_with_suggestions = 0
    
    for inchi_key, compound in analysis_results['compounds'].items():
        logger.debug(f"Processing compound {inchi_key} for RT bounds suggestion...")
        if compound['eic_data_files']:
            compounds_processed += 1
            
            # Prepare EIC data for RT bounds calculation
            eic_dict = {}
            for filename, eic_data in compound['eic_data_files'].items():
                eic_dict[filename] = {
                    "rt_vals": eic_data.get("rt_vals", []),
                    "i_vals": eic_data.get("i_vals", []),
                    "intensity_peak": eic_data.get("intensity_peak", 0.0)
                }
            
            # Calculate suggested bounds
            suggested_bounds = _suggest_rt_bounds_from_eic(
                eic_dict,
                compound['original_rt_peak'],
                compound['original_rt_min'],
                compound['original_rt_max']
            )
            
            if suggested_bounds:
                compounds_with_suggestions += 1
                compound['suggested_rt_bounds'] = suggested_bounds
        else:
            logger.debug(f"No EIC data for compound {inchi_key}, skipping RT bounds suggestion")
    
    logger.info(f"Processed {compounds_processed} compounds with EIC data")
    logger.info(f"Generated RT suggestions for {compounds_with_suggestions} compounds")

    return

def _run_analysis_summary(analysis_results: Dict) -> None:
    """
    Calculate analysis summary statistics from raw results structure.
    Independent of workflow objects.
    """

    compounds_with_eic = sum(
        1 for compound in analysis_results['compounds'].values() 
        if compound['eic_data_files']
    )
    compounds_with_ms2 = sum(
        1 for compound in analysis_results['compounds'].values() 
        if compound['ms2_data_files']
    )
    modified_compounds = sum(
        1 for compound in analysis_results['compounds'].values() 
        if compound['is_rt_modified'] or compound['is_annotation_modified']
    )
    
    total_summary = {
        'total_compounds': len(analysis_results['compounds']),
        'compounds_with_eic': compounds_with_eic,
        'compounds_with_ms2': compounds_with_ms2,
        'modified_compounds': modified_compounds,
        'atlas_uid': analysis_results['atlas_info']['atlas_uid'],
        'project_db_path': analysis_results['atlas_info']['project_db_path']
    }

    logger.info(f"Analysis complete:")
    logger.info(f"  Total compounds: {total_summary['total_compounds']}")
    logger.info(f"  Compounds with EIC data: {total_summary['compounds_with_eic']}")
    logger.info(f"  Compounds with MS2 data: {total_summary['compounds_with_ms2']}")

    return