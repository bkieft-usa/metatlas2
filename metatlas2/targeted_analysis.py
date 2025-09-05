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
import data_classes as dcl
import logging_config as lcf
import simple_cache as scache
#import spectrum_handlers as sph

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_analysis')
current_time = datetime.now().isoformat()

def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str, 
                                    config: Dict,) -> Tuple[pd.DataFrame, dcl.ProjectAnalysis]: 
    """
    Execute the complete targeted analysis workflow with comprehensive caching support.
    Returns atlas DataFrame and ProjectAnalysis object - no more plot_data dictionary.
    """
    logger.info("Setting up targeted analysis database...")

    # Get project directory for caching
    project_dir = str(Path(project_db_path).parent)
    
    # Handle data cache parameter
    use_data_cache = config['analysis_settings'].get("use_data_cache", False)
    
    # First check for analysis cache
    if use_data_cache is not False:
        logger.info("Checking for analysis cache...")
        
        cached_analysis = scache.load_analysis_cache(project_dir, use_data_cache, target_atlas_uid)
        if cached_analysis is not None:
            logger.info(f"Loaded analysis cache with {len(cached_analysis.compounds)} compounds")
            
            # Also need atlas_df for compatibility
            atlas_df = dbi.get_atlas_compounds_with_metadata(
                project_db_path=project_db_path,
                main_db_path=config["paths"]["main_database"],
                atlas_uid=target_atlas_uid
            )
            
            logger.info("Resuming analysis from cache")
            return atlas_df, cached_analysis

    # Run fresh analysis if no cache or cache failed
    logger.info("Running fresh targeted analysis...")
    main_db_path = config["paths"]["main_database"]
    analysis_settings = config["analysis_settings"]

    logger.info("Loading target atlas...")
    atlas_dataframe = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path, 
        main_db_path=main_db_path, 
        atlas_uid=target_atlas_uid
    )

    if len(atlas_dataframe) == 0:
        raise ValueError(f"No compounds found in RT-corrected atlas")

    logger.info(f"Created Atlas dataframe with {len(atlas_dataframe)} compounds")

    # Initialize project analysis with flat classes
    project_analysis = dcl.ProjectAnalysis(
        project_db_path=project_db_path,
        atlas_uid=target_atlas_uid
    )
    project_analysis.load_from_atlas(atlas_dataframe)

    # Save initialization checkpoint
    scache.save_progress_checkpoint(project_analysis, project_dir, target_atlas_uid, "initialized")

    logger.info("Loading experimental files from project database...")
    project_files = dbi.get_experimental_files_from_db(project_db_path)

    if len(project_files) == 0:
        raise ValueError("No experimental files found in project database")

    logger.info(f"Found {len(project_files)} experimental files")

    logger.info("Preparing inputs for feature extraction...")
    input_data_list = msa.prepare_feature_tools_inputs(
        atlas_df=atlas_dataframe,
        h5_files=project_files,
        ppm_tolerance=analysis_settings["default_ppm_error"],
        extra_time=analysis_settings["extra_time"]
    )
    logger.info(f"Created {len(input_data_list)} input dictionaries for feature extraction")

    logger.info("Extracting EIC and MS2 data with hits...")
    experimental_data = msa.extract_eic_and_ms2_data(input_data_list, config)

    # Add experimental data to project analysis using simplified format
    project_analysis.add_experimental_data_simple(experimental_data)

    # Save data extraction checkpoint
    scache.save_progress_checkpoint(project_analysis, project_dir, target_atlas_uid, "data_extracted")

    # Generate isomers and suggested RT bounds for each compound
    isomer_dict = build_isomer_dict(atlas_dataframe)
    for inchi_key, compound in project_analysis.compounds.items():
        # Add isomers
        compound.isomers = isomer_dict.get(inchi_key, [])
        
        # Generate suggested RT bounds if EIC data exists
        if compound.eic_data_files:
            eic_dict = {}
            for filename, eic_data in compound.eic_data_files.items():
                eic_dict[filename] = {
                    "rt_vals": eic_data.get("rt_vals", []),
                    "i_vals": eic_data.get("i_vals", []),
                    "intensity_peak": eic_data.get("intensity_peak", 0.0)
                }
            
            suggested_bounds = suggest_rt_bounds_from_eic(
                eic_dict,
                compound.original_rt_peak,
                compound.original_rt_min,
                compound.original_rt_max
            )
            compound.suggested_rt_bounds = suggested_bounds

    # Save final analysis cache (ready for GUI)
    data_timestamp = scache.save_analysis_cache(
        project_analysis, 
        project_dir,
        target_atlas_uid
    )
    logger.info(f"Saved complete analysis cache with timestamp: {data_timestamp}")

    return atlas_dataframe, project_analysis

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

def create_post_analysis_atlas_v2(project_analysis: dcl.ProjectAnalysis, 
                                  config: Dict, 
                                  output_dir: str = None) -> str:
    """
    Create post-analysis atlas using class-based approach.
    """
    # Find modified compounds
    modified_compounds = {
        inchi_key: compound for inchi_key, compound in project_analysis.compounds.items()
        if compound.is_rt_modified or compound.is_annotation_modified
    }
    
    if not modified_compounds:
        logger.info("No compounds were modified during analysis")
        return project_analysis.atlas_uid
    
    logger.info(f"Creating post-analysis atlas with {len(modified_compounds)} modified compounds...")
    
    # Prepare compound updates for database
    compound_updates = {}
    main_db_path = config["paths"]["main_database"]
    
    # Get original atlas information for the new atlas
    atlas_df = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_analysis.project_db_path,
        main_db_path=main_db_path,
        atlas_uid=project_analysis.atlas_uid
    )
    
    for compound in modified_compounds.values():
        # Find compound rows in atlas
        compound_rows = atlas_df[atlas_df['inchi_key'] == compound.inchi_key]
        for _, row in compound_rows.iterrows():
            compound_uid = row['compound_uid']
            
            update_dict = {}
            
            if compound.is_rt_modified:
                update_dict.update({
                    'rt_peak': compound.rt_peak,
                    'rt_min': compound.rt_min,
                    'rt_max': compound.rt_max
                })
            
            if compound.is_annotation_modified:
                update_dict.update({
                    'ms1_notes': compound.ms1_notes,
                    'ms2_notes': compound.ms2_notes,
                    'analyst_notes': compound.analyst_notes,
                    'identification_notes': compound.identification_notes
                })
            
            compound_updates[compound_uid] = update_dict
    
    # Create new atlas using existing database function
    new_atlas_uid = dbi.clone_and_modify_atlas(
        project_analysis.project_db_path,
        project_analysis.project_db_path,
        project_analysis.atlas_uid,
        config,
        compound_updates,
        use_experimental_table=False,
        new_atlas_description="Targeted Analysis Completed"
    )
    
    # Save atlas data to file if output directory specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        new_atlas_table = dbi.get_atlas_compounds_with_metadata(
            project_db_path=project_analysis.project_db_path, 
            main_db_path=main_db_path, 
            atlas_uid=new_atlas_uid
        )
        
        output_path = Path(output_dir) / f"atlas_post_targeted_analysis_{new_atlas_uid}.tsv"
        new_atlas_table.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved new atlas table to: {output_path}")
    
    logger.info(f"Created post-analysis atlas: {new_atlas_uid}")
    
    return new_atlas_uid

def run_post_analysis_workflow_v2(project_db_path: str,
                                  analysis_atlas_uid: str,
                                  project_analysis: dcl.ProjectAnalysis,
                                  atlas_dataframe: pd.DataFrame,
                                  project_name: str,
                                  config: Dict,
                                  analysis_output_path: str) -> Tuple[str, str, Dict]:
    """
    Execute complete post-analysis workflow using class-based approach.
    No plot_data dictionary needed - works directly with ProjectAnalysis.
    """
    logger.info("Starting class-based post-analysis workflow...")
    
    # Step 1: Create post-analysis atlas with modifications
    os.makedirs(analysis_output_path, exist_ok=True)
    logger.info("Creating post-analysis atlas...")
    post_analysis_atlas_uid = create_post_analysis_atlas_v2(
        project_analysis, 
        config, 
        analysis_output_path
    )
    logger.info(f"Created post-analysis atlas: {post_analysis_atlas_uid}")
    
    # Step 2: Save targeted analysis results using simplified approach
    logger.info("Saving targeted analysis results...")
    targeted_analysis_uid = dbi.save_targeted_analysis_from_project_analysis(
        project_analysis, 
        project_name, 
        post_analysis_atlas_uid
    )
    logger.info(f"Saved targeted analysis with UID: {targeted_analysis_uid}")
    
    # Step 3: Generate comprehensive report
    logger.info("Generating comprehensive analysis report...")
    comprehensive_report = dbi.generate_comprehensive_targeted_analysis_report(
        project_db_path, 
        config, 
        targeted_analysis_uid, 
        atlas_dataframe, 
        post_analysis_atlas_uid, 
        analysis_output_path
    )
    logger.info(f"Generated comprehensive report and saved to {analysis_output_path}")
    
    logger.info("Class-based post-analysis workflow completed successfully")
    
    return post_analysis_atlas_uid, targeted_analysis_uid, comprehensive_report

# def run_complete_targeted_analysis_v3(project_db_path: str,
#                                       target_atlas_uid: str,
#                                       project_name: str,
#                                       config: Dict,
#                                       analysis_output_path: str,
#                                       resume_from_cache: bool = True) -> Tuple[str, str, str]:
#     """
#     Complete targeted analysis workflow with comprehensive caching and resumption support.
#     """
#     logger.info("Starting complete targeted analysis workflow v3 with caching...")
    
#     project_dir = str(Path(project_db_path).parent)
    
#     # Check cache status
#     cache_status = scache.get_cache_status(project_dir, target_atlas_uid)
#     logger.info(f"Cache status: {cache_status['total_caches']} caches, {cache_status['total_checkpoints']} checkpoints")
    
#     # Step 1: Run targeted analysis (with caching support)
#     if resume_from_cache and cache_status['has_latest_cache']:
#         logger.info("Attempting to resume from latest cache...")
#         config['analysis_settings']['use_data_cache'] = True
#     else:
#         logger.info("Running fresh analysis...")
#         config['analysis_settings']['use_data_cache'] = False
    
#     atlas_df, project_analysis, plot_data = run_targeted_analysis_workflow(
#         project_db_path, target_atlas_uid, config
#     )
    
#     # Step 2: Create post-analysis atlas
#     logger.info("Creating post-analysis atlas...")
#     os.makedirs(analysis_output_path, exist_ok=True)
#     post_analysis_atlas_uid = create_post_analysis_atlas_v2(
#         project_analysis, config, analysis_output_path
#     )
    
#     # Save post-atlas checkpoint
#     scache.save_progress_checkpoint(project_analysis, project_dir, target_atlas_uid, "post_atlas_created")
    
#     # Step 3: Save analysis to database
#     logger.info("Saving analysis results to database...")
#     analysis_uid = project_analysis.save_to_database(project_name, post_analysis_atlas_uid)
    
#     # Save database checkpoint
#     scache.save_progress_checkpoint(project_analysis, project_dir, target_atlas_uid, "database_saved")
    
#     # Step 4: Generate report
#     logger.info("Generating comprehensive report...")
#     comprehensive_report = dbi.generate_comprehensive_targeted_analysis_report(
#         project_db_path, config, analysis_uid, atlas_df,
#         post_analysis_atlas_uid, analysis_output_path
#     )
    
#     report_path = Path(analysis_output_path) / f"targeted_analysis_report_{analysis_uid}.xlsx"
    
#     # Save final completion checkpoint
#     scache.save_progress_checkpoint(project_analysis, project_dir, target_atlas_uid, "completed")
    
#     # Cleanup old caches to save space
#     scache.cleanup_old_caches(project_dir, target_atlas_uid, keep_last_n=3)
    
#     logger.info("Complete targeted analysis workflow v3 finished successfully")
    
#     return post_analysis_atlas_uid, analysis_uid, str(report_path)

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