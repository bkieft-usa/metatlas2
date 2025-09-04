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
import metatlas2.data_classes as dcl
import metatlas2.logging_config as lcf
import metatlas2.checkpoint_manager as chk

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_analysis')

def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str, 
                                    config: Dict,
                                    use_cache: Union[bool, str] = True) -> Tuple[pd.DataFrame, dcl.ProjectDataCollection, Optional[chk.CheckpointManager]]: 
    """
    Execute the complete targeted analysis workflow with checkpoint support.
    
    Args:
        project_db_path: Path to project database
        target_atlas_uid: UID of the target atlas
        config: Configuration dictionary
        use_cache: Cache behavior control:
                  - True: Use most recent cache if available, otherwise run fresh
                  - False: Always run fresh analysis
                  - str (datetime): Use cache matching this timestamp
    
    Returns:
        Tuple of (atlas_df, experiment_data, checkpoint_manager)
    """
    logger.info("Setting up targeted analysis database...")

    # Create checkpoint manager
    checkpoint_manager = chk.CheckpointManager(project_db_path, target_atlas_uid)
    
    # Handle cache parameter
    if use_cache is False:
        logger.info("Cache disabled, running fresh analysis...")
    elif use_cache is True:
        if checkpoint_manager.has_checkpoint():
            try:
                logger.info("Found existing checkpoint, attempting to load most recent...")
                project_data, atlas_df, plot_data, modifications, metadata = checkpoint_manager.load_session()
                
                # Verify the checkpoint is compatible
                if metadata.get('analysis_atlas_uid') == target_atlas_uid:
                    logger.info(f"Loaded session from {metadata['timestamp']}")
                    logger.info(f"Session has {metadata['total_compounds']} compounds, {metadata['modified_compounds']} modified")
                    return atlas_df, project_data, checkpoint_manager
                else:
                    logger.warning("Checkpoint atlas UID doesn't match, running fresh analysis")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}, running fresh analysis")
        else:
            logger.info("No checkpoint found, running fresh analysis...")
    elif isinstance(use_cache, str):
        # Try to load specific timestamp
        try:
            checkpoint_info = checkpoint_manager.get_checkpoint_info()
            if checkpoint_info["exists"]:
                saved_timestamp = checkpoint_info["metadata"]["timestamp"]
                if saved_timestamp == use_cache:
                    logger.info(f"Loading checkpoint from specific timestamp: {use_cache}")
                    project_data, atlas_df, plot_data, modifications, metadata = checkpoint_manager.load_session()
                    return atlas_df, project_data, checkpoint_manager
                else:
                    logger.warning(f"Requested timestamp {use_cache} doesn't match saved timestamp {saved_timestamp}")
                    logger.info("Running fresh analysis...")
            else:
                logger.warning(f"No checkpoint found for timestamp {use_cache}")
                logger.info("Running fresh analysis...")
        except Exception as e:
            logger.error(f"Failed to load checkpoint for timestamp {use_cache}: {e}")
            logger.info("Running fresh analysis...")
    else:
        raise ValueError(f"Invalid use_cache parameter: {use_cache}. Must be True, False, or datetime string.")

    # Run fresh analysis
    main_db_path = config["paths"]["main_database"]
    analysis_settings = config["analysis_settings"]

    logger.info("Loading target atlas...")
    atlas_df_ft = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path, 
        main_db_path=main_db_path, 
        atlas_uid=target_atlas_uid
    )

    if len(atlas_df_ft) == 0:
        raise ValueError(f"No compounds found in RT-corrected atlas")

    logger.info(f"Created Atlas dataframe with {len(atlas_df_ft)} compounds")

    logger.info("Loading experimental files from project database...")
    project_files = dbi.get_experimental_files_from_db(project_db_path)

    if len(project_files) == 0:
        raise ValueError("No experimental files found in project database")

    logger.info(f"Found {len(project_files)} experimental files")

    logger.info("Preparing inputs for feature extraction...")
    input_data_list = msa.prepare_feature_tools_inputs(
        atlas_df=atlas_df_ft,
        h5_files=project_files,
        ppm_tolerance=analysis_settings["default_ppm_error"],
        extra_time=analysis_settings["extra_time"]
    )
    logger.info(f"Created {len(input_data_list)} input dictionaries")

    logger.info("Extracting EIC and MS2 data with hits...")
    experiment_data = msa.extract_eic_and_ms2_data(
        input_data_list, atlas_df_ft, config
    )

    # Save initial checkpoint for fresh analysis
    plot_data = set_up_gui_data(experiment_data, atlas_df_ft)
    empty_modifications = dcl.AnalystModifications()
    
    checkpoint_manager.save_session(
        experiment_data, 
        atlas_df_ft, 
        plot_data, 
        empty_modifications,
        timestamp=datetime.now().isoformat()
    )

    return atlas_df_ft, experiment_data, checkpoint_manager

def set_up_gui_data(experiment_data: dcl.ProjectDataCollection, atlas_df_ft: pd.DataFrame):
    """Return a dict keyed by InChI‑key with all EIC & MS2 info ready for plotting."""
    isomer_dict = build_isomer_dict(atlas_df_ft)
    metadata = {}
    
    for _, row in atlas_df_ft.iterrows():
        compound_inchi = row["inchi_key"]
        atlas_entry = make_atlas_entry(row, isomer_dict)
        
        # Get compound data from experiment_data
        compound_data = experiment_data.get_compound(compound_inchi)
        if compound_data is None:
            # Create empty data structures for compounds with no data
            eic_dict = {}
            best_eic = {}
            avg_eic = {}
            suggested_rt_bounds = None
            ms2_data = {"files": {}, "all_hits": [], "all_ms2_entries": []}
            best_ms2 = {}
            avg_ms2 = {"avg_score": 0.0}
        else:
            # Convert class-based data to GUI format
            eic_dict, best_eic, avg_eic, suggested_rt_bounds = convert_eic_data_to_gui_format(
                compound_data, atlas_entry
            )
            ms2_data = convert_ms2_data_to_gui_format(compound_data, atlas_entry)
            best_ms2, avg_ms2 = summarize_ms2_from_classes(compound_data)
        
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

def convert_eic_data_to_gui_format(compound_data: dcl.CompoundDataCollection, 
                                   atlas_data: Dict[str, Any]) -> Tuple[
    Dict[str, Dict[str, Any]],  # eic_dict
    Dict[str, Any],             # best_eic
    Dict[str, Any],             # avg_eic
    Dict[str, Any] | None,      # suggested_rt_bounds
]:
    """Convert class-based EIC data to GUI format."""
    if not compound_data.eic_data:
        return {}, {}, {}, None
    
    # Build eic_dict from EICData objects
    eic_dict = {}
    for eic in compound_data.eic_data:
        eic_dict[eic.filename] = {
            "rt_vals": eic.rt_values,
            "i_vals": eic.intensity_values,
            "mz_vals": eic.mz_values,
            "intensity_peak": eic.intensity_peak,
            "rt_peak": eic.rt_peak,
            "mz_peak": eic.mz_peak,
            "ppm_diff": eic.ppm_error,
            "rt_diff": eic.rt_error,
        }
    
    # Get best EIC (highest intensity)
    best_eic_obj = compound_data.best_eic_by_intensity
    if best_eic_obj:
        best_eic = {
            "file_peak": best_eic_obj.filename,
            "rt_peak": best_eic_obj.rt_peak,
            "intensity_peak": best_eic_obj.intensity_peak,
            "mz_peak": best_eic_obj.mz_peak,
            "ppm_diff": best_eic_obj.ppm_error,
            "rt_diff": best_eic_obj.rt_error,
        }
    else:
        best_eic = {}
    
    # Calculate average EIC
    if compound_data.eic_data:
        avg_eic = {
            "rt_peak": np.mean([eic.rt_peak for eic in compound_data.eic_data]),
            "intensity_peak": np.mean([eic.intensity_peak for eic in compound_data.eic_data]),
            "mz_peak": np.mean([eic.mz_peak for eic in compound_data.eic_data]),
        }
    else:
        avg_eic = {}
    
    # Generate suggested RT bounds
    suggested_rt_bounds = suggest_rt_bounds_from_eic_objects(
        compound_data.eic_data,
        atlas_data["rt_peak"],
        atlas_data["rt_min"],
        atlas_data["rt_max"],
    )
    
    return eic_dict, best_eic, avg_eic, suggested_rt_bounds

def convert_ms2_data_to_gui_format(compound_data: dcl.CompoundDataCollection, 
                                   atlas_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert class-based MS2 data to GUI format for compatibility."""
    files_data = {}
    all_hits = []
    all_ms2_entries = []
    
    # Group spectra by file
    spectra_by_file = {}
    for spectrum in compound_data.ms2_spectra:
        if spectrum.filename not in spectra_by_file:
            spectra_by_file[spectrum.filename] = []
        spectra_by_file[spectrum.filename].append(spectrum)
    
    for filename, spectra in spectra_by_file.items():
        file_hits = []
        file_ms2_entries = []
        
        for spectrum in spectra:
            # Convert spectrum to entry format
            ms2_entry = {
                "inchi_key": spectrum.inchi_key,
                "spectrum": [spectrum.mz_values.tolist(), spectrum.intensity_values.tolist()],
                "intensity_peak": spectrum.max_intensity,
                "rt": spectrum.rt,
                "precursor_mz": spectrum.precursor_mz,
                "filename": spectrum.filename
            }
            file_ms2_entries.append(ms2_entry)
            all_ms2_entries.append(ms2_entry)
            
            # Process hits
            for hit in spectrum.hits:
                hit_info = {
                    "filename": filename,
                    "score": hit.score,
                    "database": hit.database,
                    "ref_id": hit.ref_id,
                    "rt_theoretical": atlas_data.get("rt_peak", 0.0),
                    "rt_measured": spectrum.rt,
                    "num_matches": hit.num_matches,
                    "ref_frags": len(hit.ref_mz_values),
                    "data_frags": len(spectrum.mz_values),
                    "mz_theoretical": hit.ref_precursor_mz,
                    "mz_measured": spectrum.precursor_mz,
                    "ppm_diff": abs(hit.ref_precursor_mz - spectrum.precursor_mz) / spectrum.precursor_mz * 1e6 if spectrum.precursor_mz > 0 else 0,
                    "qry_intensity_peak": spectrum.max_intensity,
                    "qry_mz_peak": spectrum.mz_values[np.argmax(spectrum.intensity_values)] if len(spectrum.intensity_values) > 0 else 0,
                    "qry_frag_matches": hit.matched_fragments,
                    "qry_frag_colors": hit.fragment_colors,
                    "qry_spectrum": [hit.query_mz_aligned.tolist(), hit.query_intensity_aligned.tolist()],
                    "ref_spectrum": [hit.ref_mz_aligned.tolist(), hit.ref_intensity_aligned.tolist()]
                }
                file_hits.append(hit_info)
                all_hits.append(hit_info)
        
        if file_ms2_entries:
            file_info = {
                "num_ms2_entries": len(file_ms2_entries),
                "num_hits": len(file_hits),
                "ms2_entries": file_ms2_entries
            }
            
            # Add best hit info if hits exist
            if file_hits:
                best_hit = max(file_hits, key=lambda h: h.get("score", 0.0))
                file_info["best_hit"] = best_hit
            else:
                file_info["best_hit"] = {}
            
            # Add best MS2 by intensity
            best_ms2 = max(file_ms2_entries, key=lambda d: d.get("intensity_peak", 0.0))
            file_info["best_ms2"] = best_ms2
            
            files_data[filename] = file_info
    
    return {"files": files_data, "all_hits": all_hits, "all_ms2_entries": all_ms2_entries}

def summarize_ms2_from_classes(compound_data: dcl.CompoundDataCollection) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate best_ms2 and avg_ms2 summaries from class-based data."""
    # Get all hits from all spectra
    all_hits = []
    for spectrum in compound_data.ms2_spectra:
        for hit in spectrum.hits:
            all_hits.append(hit)
    
    if not all_hits:
        # No hits, try to find best by intensity
        best_ms2_spectrum = compound_data.best_ms2_by_intensity
        if best_ms2_spectrum:
            best_ms2 = {
                "file_peak": best_ms2_spectrum.filename,
                "database": None,
                "ref_id": None,
                "rt_peak": best_ms2_spectrum.rt,
                "intensity_peak": best_ms2_spectrum.max_intensity,
                "mz_peak": best_ms2_spectrum.precursor_mz,
                "score": None,
                "num_matches": None,
                "ref_frags": None,
                "data_frags": len(best_ms2_spectrum.mz_values),
                "frags_matching": [],
                "qry_spectrum": [best_ms2_spectrum.mz_values.tolist(), best_ms2_spectrum.intensity_values.tolist()],
                "ref_spectrum": [],
                "selection_method": "highest_intensity"
            }
        else:
            best_ms2 = {"selection_method": "none"}
        
        avg_ms2 = {"avg_score": 0.0}
        return best_ms2, avg_ms2
    
    # Find best hit by score
    best_hit = max(all_hits, key=lambda h: h.score)
    best_spectrum = None
    for spectrum in compound_data.ms2_spectra:
        if best_hit in spectrum.hits:
            best_spectrum = spectrum
            break
    
    if best_spectrum:
        best_ms2 = {
            "file_peak": best_spectrum.filename,
            "database": best_hit.database,
            "ref_id": best_hit.ref_id,
            "rt_peak": best_spectrum.rt,
            "intensity_peak": best_spectrum.max_intensity,
            "mz_peak": best_spectrum.precursor_mz,
            "score": best_hit.score,
            "num_matches": best_hit.num_matches,
            "ref_frags": len(best_hit.ref_mz_values),
            "data_frags": len(best_spectrum.mz_values),
            "frags_matching": best_hit.matched_fragments,
            "qry_spectrum": [best_hit.query_mz_aligned.tolist(), best_hit.query_intensity_aligned.tolist()],
            "ref_spectrum": [best_hit.ref_mz_aligned.tolist(), best_hit.ref_intensity_aligned.tolist()],
            "selection_method": "reference_hit"
        }
    else:
        best_ms2 = {"selection_method": "none"}
    
    # Calculate average score
    avg_score = np.mean([hit.score for hit in all_hits])
    avg_ms2 = {"avg_score": float(avg_score)}
    
    return best_ms2, avg_ms2

def suggest_rt_bounds_from_eic_objects(
    eic_data: List[dcl.EICData],
    atlas_rt_peak: float,
    atlas_rt_min: float,
    atlas_rt_max: float
) -> Optional[Dict[str, float]]:
    """
    Compute RT bounds from EICData objects instead of dict format.
    """
    if not eic_data:
        return None
    
    # Convert to the format expected by the original function
    eic_dict = {}
    for eic in eic_data:
        eic_dict[eic.filename] = {
            "rt_vals": eic.rt_values,
            "i_vals": eic.intensity_values,
            "intensity_peak": eic.intensity_peak
        }
    
    # Use the existing function
    return suggest_rt_bounds_from_eic(eic_dict, atlas_rt_peak, atlas_rt_min, atlas_rt_max)

def build_isomer_dict(atlas_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Return a dict: inchi_key → list of isomer dicts (empty list if none).
    Isomers are defined as:
      - mz or exact_mass within 0.005
      - OR inchi_key prefix (before '-') identical
    """
    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in atlas_df.iterrows():
        mz = row["mz"]
        exact_mass = row.get("exact_mass", None)
        inchi_prefix = row["inchi_key"].split("-")[0]
        def is_isomer(r):
            if r["inchi_key"] == row["inchi_key"]:
                return False
            mz_close = abs(r["mz"] - mz) <= 0.005
            mass_close = (
                exact_mass is not None and r.get("exact_mass", None) is not None and
                abs(r["exact_mass"] - exact_mass) <= 0.005
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

def create_post_analysis_atlas(project_db_path, analysis_atlas_uid, gui_container, config):
    """
    Clone the ANALYSIS_ATLAS_UID atlas and amend it based on analyst modifications.
    Updated to work with AnalystModifications class.

    Args:
        project_db_path (str): Path to the project database.
        analysis_atlas_uid (str): UID of the atlas to clone.
        gui_container: GUI container with get_modifications() method
        config (dict): Metatlas config.

    Returns:
        str: UID of the new amended atlas.
    """
    # Get the AnalystModifications object directly
    analyst_modifications = gui_container.get_modifications()
    
    # Get original compound metadata for reference
    original_metadata = gui_container.metadata
    
    # Prepare compound updates directly from AnalystModifications
    main_db_path = config["paths"]["main_database"]
    atlas_df = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path,
        main_db_path=main_db_path,
        atlas_uid=analysis_atlas_uid
    )
    
    compound_updates = {}
    for inchi_key in analyst_modifications.get_modified_compounds():
        # Get RT modifications
        rt_mods = analyst_modifications.get_rt_bounds(inchi_key)
        
        # Get annotation modifications
        annotation_mods = analyst_modifications.get_annotations(inchi_key)
        
        # Find the compound row(s) in the atlas
        compound_rows = atlas_df[atlas_df['inchi_key'] == inchi_key]
        for _, row in compound_rows.iterrows():
            compound_uid = row['compound_uid']
            
            # Prepare update dict
            update_dict = {}
            
            # Add RT updates if they exist
            if rt_mods:
                update_dict.update({
                    'rt_min': rt_mods['rt_min'],
                    'rt_max': rt_mods['rt_max'],
                    'rt_peak': rt_mods['rt_peak']
                })
            
            # Add annotation updates if they exist
            if annotation_mods:
                if 'ms2_notes' in annotation_mods:
                    update_dict['ms2_notes'] = annotation_mods['ms2_notes']
                if 'ms1_notes' in annotation_mods:
                    update_dict['ms1_notes'] = annotation_mods['ms1_notes']
                if 'analyst_notes' in annotation_mods:
                    update_dict['analyst_notes'] = annotation_mods['analyst_notes']
                if 'identification_notes' in annotation_mods:
                    update_dict['identification_notes'] = annotation_mods['identification_notes']
            
            compound_updates[compound_uid] = update_dict
    
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