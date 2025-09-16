import sys
import copy
from datetime import datetime
from pathlib import Path
import getpass
import copy
import pandas as pd
import numpy as np
import json
import pickle
import re
import os
import glob
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm.notebook import tqdm
import time
import uuid
import glob
import warnings
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Output
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter1d

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import ms1_ms2_analysis as msa
import load_tools as ldt
import logging_config as lcf
import json

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_analysis')
current_time = datetime.now().isoformat()

def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str, 
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
    logger.info("Setting up targeted analysis database...")

    # Get project directory for caching
    project_dir = str(Path(project_db_path).parent)

    # Run fresh analysis
    logger.info("Running fresh targeted analysis...")
    main_db_path = config["ENV"]["PATHS"]["main_database"]

    logger.info("Loading target atlas...")
    atlas_dataframe = dbi.get_atlas_compounds_table(main_db_path, atlas_uid=target_atlas_uid)

    if len(atlas_dataframe) == 0:
        raise ValueError(f"No compounds found in RT-corrected atlas")

    logger.info(f"Created Atlas dataframe with {len(atlas_dataframe)} compounds")

    # Initialize analysis data structure (no workflow objects dependency)
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
            
            # Best results (to be populated)
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

    logger.info("Loading experimental files from project database...")
    project_files = dbi.get_lcmsruns_from_db(project_db_path, file_types=['experimental', 'istd', 'exctrl'])

    if len(project_files) == 0:
        raise ValueError("No experimental files found in project database")

    logger.info(f"Found {len(project_files)} experimental files")

    logger.info("Preparing inputs for feature extraction...")
    input_data_list = msa.prepare_feature_tools_inputs(
        atlas_df=atlas_dataframe,
        h5_files=project_files,
        ppm_tolerance=analysis_params["default_ppm_error"],
        extra_time=analysis_params["extra_time"]
    )
    logger.info(f"Created {len(input_data_list)} input dictionaries for feature extraction")

    logger.info("Extracting EIC and MS2 data...")
    experimental_data_no_hits = msa.extract_eic_and_ms2_data(input_data_list, atlas_dataframe, config)
    
    logger.info("Finding MS2 reference hits...")
    experimental_data_with_hits = msa.find_ms2_hits(experimental_data_no_hits, config)

    # Add experimental data to analysis results
    add_experimental_data_to_results(analysis_results, experimental_data_with_hits)

    # Apply post-processing features independently
    apply_isomer_detection_to_results(analysis_results, atlas_dataframe)
    apply_rt_bounds_suggestions_to_results(analysis_results)
    
    # Calculate summary statistics
    analysis_results['summary_stats'] = calculate_analysis_summary(analysis_results)

    logger.info(f"Analysis complete:")
    logger.info(f"  Total compounds: {len(analysis_results['compounds'])}")
    logger.info(f"  Compounds with EIC data: {analysis_results['summary_stats']['compounds_with_eic']}")
    logger.info(f"  Compounds with MS2 data: {analysis_results['summary_stats']['compounds_with_ms2']}")

    return atlas_dataframe, analysis_results

def build_isomer_dict(atlas_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
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

def suggest_rt_bounds_from_eic(
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

def add_experimental_data_to_results(analysis_results: Dict, experimental_data: Dict) -> None:
    """
    Add experimental data to raw analysis results structure.
    Independent of workflow objects.
    """
    for inchi_key, compound_experimental_data in experimental_data.items():
        if inchi_key in analysis_results['compounds']:
            compound = analysis_results['compounds'][inchi_key]
            
            # Add EIC data directly from simplified format
            eic_files = compound_experimental_data.get('eic_files', {})
            for filename, eic_data in eic_files.items():
                compound['eic_data_files'][filename] = eic_data
            
            # Add MS2 data using consistent per-file structure
            ms2_files = compound_experimental_data.get('ms2_files', {})
            for filename, file_data in ms2_files.items():
                compound['ms2_data_files'][filename] = file_data
            
            # Update summary statistics
            compound['total_files_detected'] = len(compound['eic_data_files'])
            if compound['ms2_data_files']:
                compound['ms2_files_with_data'] = len([
                    f for f, data in compound['ms2_data_files'].items() 
                    if data.get('ms2_entries')
                ])
            
            # Update best EIC and MS2 results
            _update_best_eic_results(compound)
            _update_best_ms2_results(compound)

def _update_best_eic_results(compound: Dict) -> None:
    """Update best EIC statistics from current data."""
    if not compound['eic_data_files']:
        return
    
    best_intensity = 0.0
    best_file = ""
    
    for filename, eic_data in compound['eic_data_files'].items():
        intensity = eic_data.get('intensity_peak', 0.0)
        if intensity > best_intensity:
            best_intensity = intensity
            best_file = filename
            compound['best_eic_file'] = filename
            compound['best_eic_rt'] = eic_data.get('rt_peak', 0.0)
            compound['best_eic_mz'] = eic_data.get('mz_peak', 0.0)
            compound['best_eic_intensity'] = intensity
            compound['best_eic_ppm_error'] = eic_data.get('ppm_diff', 0.0)
            compound['best_eic_rt_error'] = eic_data.get('rt_diff', 0.0)

def _update_best_ms2_results(compound: Dict) -> None:
    """Update best MS2 statistics from current data."""
    if not compound['ms2_data_files']:
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
        compound['best_ms2_database'] = best_hit_data.get('database', '')
        compound['best_ms2_score'] = best_hit_data.get('score', 0.0)
        compound['best_ms2_num_matches'] = best_hit_data.get('num_matches', 0)
        compound['best_ms2_matched_fragments'] = best_hit_data.get('matched_fragments', [])

def apply_isomer_detection_to_results(analysis_results: Dict, atlas_dataframe: pd.DataFrame) -> None:
    """
    Apply isomer detection to raw analysis results structure.
    Independent of workflow objects.
    """
    logger.info("Applying isomer detection...")
    
    # Generate isomers dictionary
    isomer_dict = build_isomer_dict(atlas_dataframe)
        
    # Debug: Show some examples
    compounds_with_isomers = 0
    for inchi_key, isomers in isomer_dict.items():
        if isomers:
            compounds_with_isomers += 1
    
    logger.info(f"Found {compounds_with_isomers} compounds with isomers")
    
    # Apply to compounds in analysis results
    for inchi_key, compound in analysis_results['compounds'].items():
        compound['isomers'] = isomer_dict.get(inchi_key, [])

def apply_rt_bounds_suggestions_to_results(analysis_results: Dict) -> None:
    """
    Apply RT bounds suggestions to raw analysis results structure.
    Independent of workflow objects.
    """
    logger.info("Applying RT bounds suggestions...")
    
    compounds_processed = 0
    compounds_with_suggestions = 0
    
    for inchi_key, compound in analysis_results['compounds'].items():
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
            suggested_bounds = suggest_rt_bounds_from_eic(
                eic_dict,
                compound['original_rt_peak'],
                compound['original_rt_min'],
                compound['original_rt_max']
            )
            
            if suggested_bounds:
                compounds_with_suggestions += 1
                compound['suggested_rt_bounds'] = suggested_bounds
    
    logger.info(f"Processed {compounds_processed} compounds with EIC data")
    logger.info(f"Generated RT suggestions for {compounds_with_suggestions} compounds")

def calculate_analysis_summary(analysis_results: Dict) -> Dict[str, Any]:
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
    
    return {
        'total_compounds': len(analysis_results['compounds']),
        'compounds_with_eic': compounds_with_eic,
        'compounds_with_ms2': compounds_with_ms2,
        'modified_compounds': modified_compounds,
        'atlas_uid': analysis_results['atlas_info']['atlas_uid'],
        'project_db_path': analysis_results['atlas_info']['project_db_path']
    }