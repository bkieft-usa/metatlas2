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
from typing import Dict, List, Optional, Any, Tuple
from tqdm.notebook import tqdm
import time
import duckdb
import uuid
import glob
import warnings

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
from scipy.interpolate import interp1d

sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.database_interact as dbi
import metatlas2.ms1_ms2_analysis as msa
import metatlas2.load_tools as ldt

def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str, 
                                    config: Dict) -> Tuple[pd.DataFrame, Dict, Dict]: 
    """ Execute the complete targeted analysis workflow.
    Returns:
        Tuple of (atlas_df, eics, ms2_data_with_hits)
    """
    print("Setting up targeted analysis database...")

    main_db_path = config["paths"]["main_database"]
    analysis_settings = config["analysis_settings"]

    print("Loading target atlas...")
    atlas_df_ft = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path, 
        main_db_path=main_db_path, 
        atlas_uid=target_atlas_uid
    )

    if len(atlas_df_ft) == 0:
        raise ValueError(f"No compounds found in RT-corrected atlas")

    print(f"Created Atlas dataframe with {len(atlas_df_ft)} compounds")

    print("Loading experimental files from project database...")
    project_files = dbi.get_experimental_files_from_db(project_db_path)

    if len(project_files) == 0:
        raise ValueError("No experimental files found in project database")

    print(f"Found {len(project_files)} experimental files")

    print("Preparing inputs for feature extraction...")
    input_data_list = msa.prepare_feature_tools_inputs(
        atlas_df=atlas_df_ft,
        h5_files=project_files,
        ppm_tolerance=analysis_settings["default_ppm_error"],
        extra_time=analysis_settings["extra_time"]
    )
    print(f"Created {len(input_data_list)} input dictionaries")

    print("Extracting EIC and MS2 data with hits...")
    eics, ms2_data_with_hits = msa.extract_eic_and_ms2_data_with_hits(
        input_data_list, atlas_df_ft, config
    )

    return atlas_df_ft, eics, ms2_data_with_hits

def set_up_gui_data(eics, atlas_df_ft, ms2_data_with_hits):
    """Return a dict keyed by InChI‑key with all EIC & MS2 info ready for plotting."""
    isomer_dict = build_isomer_dict(atlas_df_ft)
    metadata = {}
    for _, row in atlas_df_ft.iterrows():
        compound_inchi = row["inchi_key"]
        atlas_entry = make_atlas_entry(row, isomer_dict)
        eic_rows = collect_eic_rows(compound_inchi, eics)
        eic_dict, best_eic, avg_eic, suggested_rt_bounds = summarize_eic(eic_rows, atlas_entry)
        ms2_data = collect_ms2(compound_inchi, ms2_data_with_hits, atlas_entry)
        best_ms2, avg_ms2 = summarize_ms2(ms2_data["all_hits"])
        metadata[compound_inchi] = assemble_compound_block(
            atlas_entry,
            eic_dict,
            best_eic,
            avg_eic,
            suggested_rt_bounds,
            ms2_data,
            best_ms2,
            avg_ms2,
        )
    return metadata

def build_isomer_dict(atlas_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Return a dict: inchi_key → list of isomer dicts (empty list if none)."""
    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in atlas_df.iterrows():
        mz = row["mz"]
        tol = row.get("mz_tolerance_ppm", 10.0) * 1e-3
        mask = np.isclose(atlas_df["mz"], mz, atol=tol)
        isomers = atlas_df[mask & (atlas_df["inchi_key"] != row["inchi_key"])]
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

def make_atlas_entry(row: pd.Series,
                     isomer_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Create the “original_atlas_data” block for a single compound."""
    return {
        "rt_min": row["rt_min"],
        "rt_max": row["rt_max"],
        "rt_peak": row["rt_peak"],
        "mz": row["mz"],
        "mz_tolerance": row.get("mz_tolerance_ppm", 10.0),
        "adduct": row.get("adduct", "[M+H]+"),
        "polarity": row.get("polarity", "positive"),
        "compound_name": row["label"],
        "inchi_key": row["inchi_key"],
        "formula": row.get("formula", ""),
        "exact_mass": row.get("exact_mass", None),
        "isomers": isomer_dict.get(row["inchi_key"], []),
        "ms2_notes": "no selection",
        "ms1_notes": "keep",
        "identification_notes": row.get("identification_notes", ""),
        "analyst_notes": row.get("analyst_notes", "")
    }

def collect_eic_rows(compound_inchi: str,
                     eics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the rows that match `compound_inchi` from every file."""
    frames = []
    for file_path, eic_df in eics.items():
        sub = eic_df[eic_df["inchi_key"] == compound_inchi]
        if not sub.empty:
            sub = sub.copy()
            sub["file_name"] = Path(file_path).name
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def make_eic_dict(eic_df: pd.DataFrame,
                  atlas_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {file_name: trace_dict}."""
    eic_dict: Dict[str, Dict[str, Any]] = {}
    for _, row in eic_df.iterrows():
        rt_vals = np.array(row["rt"])
        i_vals = np.array(row["i"])
        if rt_vals.size > 1:
            order = np.argsort(rt_vals)
            rt_vals = rt_vals[order]
            i_vals = i_vals[order]

        eic_dict[row["file_name"]] = {
            "rt_vals": rt_vals,
            "i_vals": i_vals,
            "mz_vals": row.get("mz", []),
            "intensity_peak": row.get("intensity_peak"),
            "rt_peak": row.get("rt_peak"),
            "mz_peak": row.get("mz_peak"),
            "ppm_diff": (abs(row.get("mz_peak", 0) - atlas_data["mz"]) / atlas_data["mz"] * 1e6),
            "rt_diff": row.get("rt_peak", 0) - atlas_data["rt_peak"],
        }
    return eic_dict

def summarize_eic(
    eic_df: pd.DataFrame,
    atlas_data: Dict[str, Any]
) -> Tuple[
    Dict[str, Dict[str, Any]],  # eic_dict
    Dict[str, Any],             # best_eic
    Dict[str, Any],             # avg_eic
    Dict[str, Any] | None,      # suggested_rt_bounds (or None if no EIC)
]:
    """Compute the three EIC related structures."""
    if eic_df.empty:
        return {}, {}, {}, None

    eic_dict = make_eic_dict(eic_df, atlas_data)
    best_row = eic_df.loc[eic_df["intensity_peak"].idxmax()]
    best_eic = {
        "file_peak": best_row["file_name"],
        "rt_peak": best_row["rt_peak"],
        "intensity_peak": best_row["intensity_peak"],
        "mz_peak": best_row["mz_peak"],
        "ppm_diff": (
            abs(best_row["mz_peak"] - atlas_data["mz"]) / atlas_data["mz"] * 1e6
        ),
        "rt_diff": best_row["rt_peak"] - atlas_data["rt_peak"],
    }

    avg_eic = {
        "rt_peak": eic_df["rt_peak"].mean(),
        "intensity_peak": eic_df["intensity_peak"].mean(),
        "mz_peak": eic_df["mz_peak"].mean(),
    }

    suggested_rt_bounds = suggest_rt_bounds_from_eic(
        eic_dict,
        atlas_data["rt_peak"],
        atlas_data["rt_min"],
        atlas_data["rt_max"],
    )
    return eic_dict, best_eic, avg_eic, suggested_rt_bounds

def collect_ms2(
    compound_inchi: str,
    ms2_data_with_hits: Dict[str, List[Dict[str, Any]]],
    atlas_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a dict with:
      - 'files': {file_name: {summary info for MS2 datapoints and hits}}
      - 'all_hits': flat list of reference hits for summary
      - 'all_ms2_entries': flat list of all MS2 entries for fallback best selection
    """
    files: Dict[str, Dict[str, Any]] = {}
    all_hits: List[Dict[str, Any]] = []
    all_ms2_entries: List[Dict[str, Any]] = []  # Add this to track all MS2 entries
    
    for file_path, ms2_datapoints in ms2_data_with_hits.items():
        file_name = Path(file_path).name
        file_hits: List[Dict[str, Any]] = []
        file_ms2_entries = []
        
        # Collect ALL MS2 datapoints for this compound, regardless of hits
        for datum in ms2_datapoints:
            if datum.get("inchi_key") == compound_inchi:
                # Make sure intensity_peak is calculated properly
                spectrum = datum.get("spectrum", None)
                if spectrum is not None:
                    datum['intensity_peak'] = max(spectrum[1])
                else:
                    datum['intensity_peak'] = 0.0
                datum['filename'] = file_name
                file_ms2_entries.append(datum)
                all_ms2_entries.append(datum)  # Add to global list
                
                # Process hits if they exist
                for hit in datum.get("hits", []):
                    ref = hit.get("msv_ref_aligned")
                    qry = hit.get("msv_query_aligned")
                    if ref is None or qry is None or len(ref) != 2 or len(qry) != 2:
                        continue
                    ref_mz, ref_int = np.array(ref[0]), np.array(ref[1])
                    qry_mz, qry_int = np.array(qry[0]), np.array(qry[1])
                    frag_matches_ref, frag_matches_ref_colors = _frag_match_colors(
                        ref_mz, ref_int, qry_mz, qry_int
                    )
                    ms2_information = {
                        "filename": file_name,
                        "score": hit.get("score", 0.0),
                        "database": hit.get("database", None),
                        "ref_id": hit.get("id", None),
                        "rt_theoretical": atlas_data.get("rt_peak", 0.0),
                        "rt_measured": hit.get("msms_scan", 0.0),
                        "num_matches": hit.get("num_matches", 0),
                        "ref_frags": len(hit.get("msv_ref_unaligned", [[], []])[0]),
                        "data_frags": len(hit.get("msv_query_unaligned", [[], []])[0]),
                        "mz_theoretical": hit.get("precursor_mz", 0.0),
                        "mz_measured": hit.get("measured_precursor_mz", 0.0),
                        "ppm_diff": (
                            abs(
                                hit.get("precursor_mz", 0.0)
                                - hit.get("measured_precursor_mz", 0.0)
                            )
                            / hit.get("measured_precursor_mz", 1.0)
                            * 1e6
                        ),
                        "qry_intensity_peak": qry_int.max() if qry_int.size else 0,
                        "qry_mz_peak": qry_mz[qry_int.argmax()] if qry_int.size else 0,
                        "qry_frag_matches": frag_matches_ref,
                        "qry_frag_colors": frag_matches_ref_colors,
                        "qry_spectrum": qry,
                        "ref_spectrum": ref
                    }
                    file_hits.append(ms2_information)
                    all_hits.append(ms2_information)
        
        # Include file information if there are ANY MS2 entries (with or without hits)
        if file_ms2_entries:
            file_info = {
                "num_ms2_entries": len(file_ms2_entries),
                "num_hits": len(file_hits),
                "ms2_entries": file_ms2_entries  # Include all MS2 datapoints
            }
            
            # Add best hit info if hits exist
            if file_hits:
                best_hit = max(file_hits, key=lambda h: h.get("score", 0.0))
                file_info["best_hit"] = best_hit
            else:
                file_info["best_hit"] = {}

            file_info["best_ms2"] = max(file_ms2_entries, key=lambda d: d.get("intensity_peak", 0.0))

            files[file_name] = file_info
    
    # Make sure we're returning all_ms2_entries in the dictionary
    return {"files": files, "all_hits": all_hits, "all_ms2_entries": all_ms2_entries}

def _frag_match_colors(
    ref_mz: np.ndarray,
    ref_int: np.ndarray,
    qry_mz: np.ndarray,
    qry_int: np.ndarray,
) -> Tuple[List[float], List[str]]:
    """
    Colour-coding logic: for each fragment position, if both ref_int and qry_int are non-zero,
    color green (match), else red (no match). Assumes arrays are same length and aligned.
    Returns:
        - list of colour strings for each fragment
    """
    colors: List[str] = []
    frag_matches: List[float] = []

    if len(ref_int) != len(qry_int):
        raise ValueError("Input arrays must have the same length")
    for i in range(len(ref_int)):
        if ref_int[i] > 0 and qry_int[i] > 0:
            colors.append("green")
            frag_matches.append(ref_mz[i])
        else:
            colors.append("red")
    return frag_matches, colors

def summarize_ms2(all_hits: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return best_ms2 dict and avg_ms2 dict (empty if no hits)."""
    if not all_hits:
        return {}, {"avg_score": 0.0}

    best = max(all_hits, key=lambda h: h.get("score", 0.0))
    best_ms2 = {
        "file_peak": best.get("filename"),
        "database": best.get("database"),
        "ref_id": best.get("ref_id"),
        "rt_peak": best.get("rt_measured", 0.0),
        "intensity_peak": best.get("qry_intensity_peak", 0.0),
        "mz_peak": best.get("mz_measured", 0.0),
        "score": best.get("score", 0.0),
        "num_matches": best.get("num_matches", 0),
        "ref_frags": best.get("ref_frags", 0),
        "data_frags": best.get("data_frags", 0),
        "frags_matching": best.get("qry_frag_matches", []),
        "qry_spectrum": best.get("qry_spectrum", []),
        "ref_spectrum": best.get("ref_spectrum", [])
    }
    avg_ms2 = {"avg_score": float(np.mean([h.get("score", 0.0) for h in all_hits]))}
    return best_ms2, avg_ms2

def assemble_compound_block(
    atlas_entry: Dict[str, Any],
    eic_dict: Dict[str, Dict[str, Any]],
    best_eic: Dict[str, Any],
    avg_eic: Dict[str, Any],
    suggested_rt_bounds: Dict[str, Any] | None,
    ms2_data: Dict[str, Any],  # Now receives the full ms2_data dict
    best_ms2: Dict[str, Any],
    avg_ms2: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the nested dict that lives under a single InChI‑key."""
    
    # Enhanced best MS2 selection logic - now accessing the correct data
    all_hits = ms2_data.get("all_hits", [])
    all_ms2_entries = ms2_data.get("all_ms2_entries", [])
    
    enhanced_best_ms2 = {}
    
    if all_hits:
        # Use existing best_ms2 from hits (highest score)
        enhanced_best_ms2 = best_ms2.copy()
        enhanced_best_ms2["selection_method"] = "reference_hit"
    elif all_ms2_entries:
        # No hits, select best by intensity from all MS2 entries
        best_entry = max(all_ms2_entries, key=lambda d: d.get("intensity_peak", 0.0))
        enhanced_best_ms2 = {
            "file_peak": best_entry.get("filename", ""),
            "database": None,
            "ref_id": None,
            "rt_peak": best_entry.get("rt", 0.0),
            "intensity_peak": best_entry.get("intensity_peak", 0.0),
            "mz_peak": best_entry.get("precursor_mz", 0.0),
            "score": None,
            "num_matches": None,
            "ref_frags": None,
            "data_frags": len(best_entry.get("spectrum", [[], []])[0]),
            "frags_matching": [],
            "qry_spectrum": best_entry.get("spectrum", []),
            "ref_spectrum": [],
            "selection_method": "highest_intensity"
        }
    else:
        # No MS2 data at all
        enhanced_best_ms2 = {
            "file_peak": None,
            "database": None,
            "ref_id": None,
            "rt_peak": None,
            "intensity_peak": None,
            "mz_peak": None,
            "score": None,
            "num_matches": None,
            "ref_frags": None,
            "data_frags": None,
            "frags_matching": [],
            "qry_spectrum": [],
            "ref_spectrum": [],
            "selection_method": "none"
        }
    
    return {
        "original_atlas_data": atlas_entry.copy(),
        "new_atlas_data": atlas_entry.copy(),
        "suggested_rt_bounds_data": suggested_rt_bounds,
        "eic_data": eic_dict,
        "best_eic": best_eic,
        "avg_eic": avg_eic,
        "best_ms2": enhanced_best_ms2,  # Use enhanced version
        "avg_ms2": avg_ms2,
        "ms2_data": ms2_data["files"],  # Store only the files part in ms2_data for compatibility
    }

def create_post_analysis_atlas(project_db_path, analysis_atlas_uid, plot_data, config):
    """
    Clone the ANALYSIS_ATLAS_UID atlas and amend it based on the RT bounds and annotations in plot_data.

    Args:
        project_db_path (str): Path to the project database.
        analysis_atlas_uid (str): UID of the atlas to clone.
        plot_data (dict): Dictionary keyed by InChI key with updated RT bounds and annotations.
        config (dict): Metatlas config.

    Returns:
        str: UID of the new amended atlas.
    """
    # Prepare compound_updates dict keyed by compound_uid
    main_db_path = config["paths"]["main_database"]
    atlas_df = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path,
        main_db_path=main_db_path,
        atlas_uid=analysis_atlas_uid
    )
    compound_updates = {}
    for inchi_key, meta in plot_data.items():
        new_data = meta.get('new_atlas_data', {})
        ms2_notes = new_data.get('ms2_notes', None)
        ms1_notes = new_data.get('ms1_notes', None)
        rt_min = new_data.get('rt_min', None)
        rt_max = new_data.get('rt_max', None)
        rt_peak = new_data.get('rt_peak', None)
        # Find the compound row in the atlas
        compound_rows = atlas_df[atlas_df['inchi_key'] == inchi_key]
        for _, row in compound_rows.iterrows():
            compound_uid = row['compound_uid']
            compound_updates[compound_uid] = {
                'rt_min': rt_min,
                'rt_max': rt_max,
                'rt_peak': rt_peak,
                'ms2_notes': ms2_notes,
                'ms1_notes': ms1_notes
            }
    # Use consolidated function to clone and amend atlas
    new_atlas_uid = dbi.clone_and_modify_atlas(
        project_db_path,
        project_db_path,
        analysis_atlas_uid,
        config,
        compound_updates,
        use_experimental_table=False
    )
    return new_atlas_uid

def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving‑average.  window=1 returns the original array."""
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def suggest_rt_bounds_from_eic(
    eic_data: Dict[str, Dict[str, Any]],
    atlas_rt_peak: float,
    atlas_rt_min: float,
    atlas_rt_max: float
) -> Optional[Dict[str, float]]:
    """
    Compute RT bounds from the *average* extracted‑ion chromatogram (EIC)
    of many LC‑MS/MS files.

    Parameters
    ----------
    eic_data : dict
        Mapping ``file_name → {'rt_vals': [...], 'i_vals': [...],
        'intensity_peak': float}``.
    atlas_rt_peak, atlas_rt_min, atlas_rt_max : float
        Expected RT window from the atlas (used for confidence scoring).

    Returns
    -------
    dict or None
        ``{'rt_min':…, 'rt_max':…, 'rt_peak':…, 'confidence':…}``
        or ``None`` if a suitable peak cannot be found.
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

    # Use a modest height / prominence threshold (10 % / 5 % of max)
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

    # Add a small padding (5 % of the width or 0.05 min, whichever is larger)
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