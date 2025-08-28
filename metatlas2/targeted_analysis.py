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

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Output

sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.database_interact as dbi
import metatlas2.ms1_ms2_analysis as msa

def load_msms_refs_file(file_path):
    """
    Load the msms_refs.tab file and convert it to a DataFrame format suitable for MS2 matching.
    
    Args:
        file_path: Path to the msms_refs.tab file
        
    Returns:
        DataFrame with columns: ['database', 'id', 'name', 'spectrum', 'collision_energy', 
                                'precursor_mz', 'polarity', 'adduct', 'formula', 'exact_mass', 
                                'inchi_key', 'inchi', 'smiles']
    """
    import ast

    # Read the tab-separated file
    df = pd.read_csv(file_path, sep='\t', header=None, names=[
        'id', 'database', 'compound_id', 'name', 'spectrum', 'collision_energy', 
        'precursor_mz', 'polarity', 'adduct', 'fragmentation_method', 'other_id', 
        'experiment', 'instrument', 'formula', 'exact_mass', 'inchi_key', 'inchi', 'smiles'
    ])

    # Convert spectrum strings to numpy arrays
    def parse_spectrum(spec_str):
        try:
            # Parse the string representation of the spectrum
            spectrum = ast.literal_eval(spec_str)
            if len(spectrum) == 2 and len(spectrum[0]) == len(spectrum[1]):
                return np.array(spectrum)
            else:
                return None
        except:
            return None

    df['spectrum_parsed'] = df['spectrum'].apply(parse_spectrum)

    # Remove rows with unparseable spectra
    df = df.dropna(subset=['spectrum_parsed'])
    df['spectrum'] = df['spectrum_parsed']
    df = df.drop('spectrum_parsed', axis=1)

    # Clean up data types
    df['precursor_mz'] = pd.to_numeric(df['precursor_mz'], errors='coerce')
    df['collision_energy'] = pd.to_numeric(df['collision_energy'], errors='coerce') 
    df['exact_mass'] = pd.to_numeric(df['exact_mass'], errors='coerce')

    return df

def process_ms2_hits(ms2_data: Dict, reference_df: pd.DataFrame, min_score: float = 0.4, min_matches: int = 2) -> Tuple[Dict, pd.DataFrame]:
    """
    Process MS2 data against reference database and return enhanced data with hits.
    
    Args:
        ms2_data: Raw MS2 data dictionary
        reference_df: Reference spectra DataFrame
        min_score: Minimum similarity score for high-quality matches
        min_matches: Minimum number of fragment matches
        
    Returns:
        Tuple of (enhanced_ms2_data, all_hits_df)
    """
    print("Running MS2 hits analysis...")
    all_hits = msa.get_ms2_hits_from_data(
        ms2_data_dict=ms2_data,
        reference_df=reference_df,
        frag_mz_tolerance=0.01,
        keep_nonmatches=True,
        precursor_mz_tolerance_ppm=20.0
    )
    
    if all_hits.empty:
        print("No MS2 hits found")
        return ms2_data, pd.DataFrame()
    
    print(f"MS2 hits identified: {all_hits.shape[0]}")
    
    # Filter high-quality matches
    high_quality_hits = all_hits[
        (all_hits.get('score', 0) >= min_score) & 
        (all_hits.get('num_matches', 0) >= min_matches)
    ].copy()
    
    if high_quality_hits.empty:
        print("No high-quality matches found")
        return ms2_data, all_hits
    
    print(f"Found {len(high_quality_hits)} high-quality matches")
    
    # Enhance MS2 data with hits
    enhanced_ms2_data = {}
    
    for file_path, ms2_entries in ms2_data.items():
        file_name = Path(file_path).name
        enhanced_ms2_data[file_path] = []
        
        if isinstance(ms2_entries, pd.DataFrame):
            file_hits = high_quality_hits[high_quality_hits['file_name'] == file_name]

            for _, entry in ms2_entries.iterrows():
                enhanced_entry = entry.to_dict()
                inchi_key = enhanced_entry.get('inchi_key', 'unknown')
                matching_hits = file_hits[file_hits['inchi_key'] == inchi_key]
                enhanced_entry['hits'] = matching_hits.to_dict('records') if not matching_hits.empty else []
                enhanced_ms2_data[file_path].append(enhanced_entry)
        else:
            print(f"Warning: ms2_entries for {file_path} is not a DataFrame: {type(ms2_entries)}")
            enhanced_ms2_data[file_path] = []
    
    return enhanced_ms2_data, all_hits

def run_targeted_analysis_workflow(project_db_path: str, 
                                    target_atlas_uid: str, 
                                    config: Dict) -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]: 
    """ Execute the complete targeted analysis workflow.
    Returns:
        Tuple of (atlas_df, eics, ms2_data_with_hits, ms2_hits)
    """
    print("Setting up targeted analysis database...")
    dbi.create_targeted_analysis_table(project_db_path)

    main_db_path = config["paths"]["main_database"]
    msms_refs_path = Path(config["paths"]["msms_refs"])
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

    print("Extracting EIC and MS2 data...")
    eics, ms2_data = msa.extract_eic_and_ms2_data(input_data_list, atlas_df_ft)

    # Process MS2 reference data if available
    ms2_hits = pd.DataFrame()
    ms2_data_with_hits = ms2_data

    if Path(msms_refs_path).exists():
        print("Loading reference spectra from msms_refs.tab...")
        reference_df = load_msms_refs_file(msms_refs_path)
        
        if not reference_df.empty:
            print(f"Reference DataFrame shape: {reference_df.shape}")
            print(f"Number of unique InChI keys: {reference_df['inchi_key'].nunique()}")
            
            ms2_data_with_hits, ms2_hits = process_ms2_hits(ms2_data, reference_df)
        else:
            print("Reference DataFrame is empty")
    else:
        print(f"MSMS reference file not found at {msms_refs_path}")
        print("Continuing without MS2 reference analysis")

    return atlas_df_ft, eics, ms2_data_with_hits, ms2_hits

def analyze_ms2_hits_results(ms2_hits: pd.DataFrame, min_score: float = 0.3, min_matches: int = 2) -> Dict:
    """Analyze MS2 hits results and return summary statistics."""
    if ms2_hits.empty:
        return {
            'total_spectra': 0,
            'spectra_with_hits': 0,
            'high_quality_matches': 0,
            'mean_score': 0.0,
            'median_score': 0.0
        }
    
    # Basic statistics
    total_spectra = len(ms2_hits)
    spectra_with_hits = (ms2_hits.get('score', pd.Series([0])) > 0).sum()
    high_quality_matches = ((ms2_hits.get('score', pd.Series([0])) >= min_score) & 
                           (ms2_hits.get('num_matches', pd.Series([0])) >= min_matches)).sum()
    
    scores = ms2_hits.get('score', pd.Series([0])).dropna()
    mean_score = scores.mean() if len(scores) > 0 else 0.0
    median_score = scores.median() if len(scores) > 0 else 0.0
    
    return {
        'total_spectra': total_spectra,
        'spectra_with_hits': spectra_with_hits,
        'high_quality_matches': high_quality_matches,
        'mean_score': mean_score,
        'median_score': median_score
    }

def plot_ms2_score_distribution(hits_df, title="MS2 Similarity Score Distribution"):
    """
    Plot the distribution of MS2 similarity scores
    
    Args:
        hits_df: DataFrame from get_msms_hits_from_data()
        title: Plot title
    """
    if hits_df.empty:
        print("No hits to plot")
        return
    
    scores = hits_df['score'].dropna()
    if len(scores) == 0:
        print("No valid scores to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(scores, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Box plot by database
    hits_reset = hits_df.reset_index()
    if 'database' in hits_reset.columns:
        sns.boxplot(data=hits_reset, x='database', y='score', ax=ax2)
        ax2.set_xlabel('Database')
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Scores by Database')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Median: {scores.median():.3f}")
    print(f"  Min: {scores.min():.3f}")
    print(f"  Max: {scores.max():.3f}")
    print(f"  Total hits: {len(scores)}")

def filter_ms2_hits(ms2_hits: pd.DataFrame, min_score: float = 0.4, min_matches: int = 2) -> pd.DataFrame:
    """Filter MS2 hits based on quality criteria."""
    if ms2_hits.empty:
        return pd.DataFrame()
    
    filtered = ms2_hits[
        (ms2_hits.get('score', 0) >= min_score) & 
        (ms2_hits.get('num_matches', 0) >= min_matches)
    ].copy()
    
    return filtered

def suggest_rt_bounds_from_eic(eic_data, atlas_rt_peak, atlas_rt_min, atlas_rt_max):
    """
    Suggest new RT bounds based on sophisticated peak detection in EIC data.
    Uses scipy peak finding to identify peak boundaries and calculate confidence.
    
    Args:
        eic_data: Dictionary of file_name -> EIC trace data
        atlas_rt_peak: Original atlas RT peak
        atlas_rt_min: Original atlas RT min
        atlas_rt_max: Original atlas RT max
    
    Returns:
        dict: {'rt_min': float, 'rt_max': float, 'rt_peak': float, 'confidence': float}
        Returns None if no good suggestions can be made
    """
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    from scipy.ndimage import gaussian_filter1d
    
    if not eic_data:
        return None
    
    # Find the EIC with highest peak intensity
    best_file = None
    best_intensity = 0
    
    for file_name, trace_data in eic_data.items():
        peak_intensity = trace_data.get('intensity_peak', 0)
        if peak_intensity and peak_intensity > best_intensity:
            best_intensity = peak_intensity
            best_file = file_name
    
    if best_file is None:
        return None
    
    trace_data = eic_data[best_file]
    rt_vals_raw = trace_data.get('rt_vals', [])
    i_vals_raw = trace_data.get('i_vals', [])
    
    # Clean and convert data to proper numeric arrays
    try:
        # Filter out None values and convert to numeric
        valid_indices = []
        rt_vals_clean = []
        i_vals_clean = []
        
        for i, (rt, intensity) in enumerate(zip(rt_vals_raw, i_vals_raw)):
            if rt is not None and intensity is not None:
                try:
                    rt_float = float(rt)
                    i_float = float(intensity)
                    if not np.isnan(rt_float) and not np.isnan(i_float) and i_float >= 0:
                        valid_indices.append(i)
                        rt_vals_clean.append(rt_float)
                        i_vals_clean.append(i_float)
                except (ValueError, TypeError):
                    continue
        
        if len(rt_vals_clean) < 5:
            return None
            
        rt_vals = np.array(rt_vals_clean, dtype=np.float64)
        i_vals = np.array(i_vals_clean, dtype=np.float64)
        
    except Exception as e:
        # If data cleaning fails, fall back to simple approach
        try:
            rt_vals = np.array([x for x in rt_vals_raw if x is not None], dtype=np.float64)
            i_vals = np.array([x for x in i_vals_raw if x is not None], dtype=np.float64)
            if len(rt_vals) != len(i_vals) or len(rt_vals) < 5:
                return None
        except:
            return None
    
    # Ensure we have enough data points and valid data
    if len(rt_vals) == 0 or len(i_vals) == 0 or len(rt_vals) < 5:
        return None
    
    if np.max(i_vals) <= 0:
        return None
    
    # Sort by RT to ensure proper order
    sort_idx = np.argsort(rt_vals)
    rt_vals = rt_vals[sort_idx]
    i_vals = i_vals[sort_idx]
    
    # Smooth the data slightly to reduce noise for peak detection
    try:
        i_vals_smooth = gaussian_filter1d(i_vals.astype(np.float64), sigma=1.0)
    except Exception:
        # If smoothing fails, use original data
        i_vals_smooth = i_vals.astype(np.float64)
    
    # Find all peaks with minimum height and prominence
    max_intensity = np.max(i_vals_smooth)
    min_height = max_intensity * 0.1  # At least 10% of max intensity
    min_prominence = max_intensity * 0.05  # At least 5% prominence
    
    try:
        peaks, peak_properties = find_peaks(
            i_vals_smooth, 
            height=min_height,
            prominence=min_prominence,
            distance=5  # Minimum 5 data points between peaks
        )
    except Exception:
        # If peak finding fails, find maximum manually
        max_idx = np.argmax(i_vals_smooth)
        peaks = np.array([max_idx])
    
    if len(peaks) == 0:
        return None
    
    # Find the peak closest to the atlas RT peak
    peak_rts = rt_vals[peaks]
    peak_intensities = i_vals_smooth[peaks]
    
    # Calculate score for each peak (combination of intensity and proximity to atlas)
    intensity_scores = peak_intensities / np.max(peak_intensities)
    rt_deviations = np.abs(peak_rts - atlas_rt_peak)
    max_expected_deviation = max(abs(atlas_rt_max - atlas_rt_min), 1.0)
    rt_scores = np.maximum(0.0, 1.0 - (rt_deviations / max_expected_deviation))
    
    # Combined score (weight intensity more heavily)
    combined_scores = 0.7 * intensity_scores + 0.3 * rt_scores
    best_peak_idx = np.argmax(combined_scores)
    selected_peak = peaks[best_peak_idx]
    
    # Get peak properties for the selected peak
    try:
        prominences = peak_prominences(i_vals_smooth, [selected_peak])[0]
        widths_result = peak_widths(i_vals_smooth, [selected_peak], rel_height=0.5)
        
        # Extract width information
        width_indices = widths_result[0][0]  # Width in indices
        left_idx = int(widths_result[2][0])  # Left boundary index
        right_idx = int(widths_result[3][0])  # Right boundary index
        
    except Exception:
        # If width calculation fails, use simple approach
        prominences = np.array([i_vals_smooth[selected_peak] * 0.5])
        width_indices = 10.0  # Default width
        left_idx = max(0, selected_peak - 10)
        right_idx = min(len(rt_vals) - 1, selected_peak + 10)
    
    # Ensure boundaries are within array bounds
    left_idx = max(0, left_idx)
    right_idx = min(len(rt_vals) - 1, right_idx)
    
    # Advanced peak boundary detection using derivative analysis
    def find_peak_boundaries(rt_vals, i_vals, peak_idx, initial_left, initial_right):
        """Find more precise peak boundaries using slope analysis"""
        
        try:
            # Calculate first derivative (gradient)
            gradient = np.gradient(i_vals)
            
            # Find where slope becomes very small (< 5% of max gradient)
            max_grad = np.max(np.abs(gradient))
            slope_threshold = max_grad * 0.05 if max_grad > 0 else 0.1
            
            # Search left from peak
            final_left = initial_left
            for i in range(peak_idx, max(0, peak_idx - 20), -1):
                if i < len(gradient) and abs(gradient[i]) < slope_threshold:
                    # Check if intensity has dropped significantly
                    if i_vals[i] < i_vals[peak_idx] * 0.1:  # Below 10% of peak
                        final_left = i
                        break
                    # Or if we're in a valley (derivative changes sign)
                    if i > 0 and gradient[i] * gradient[i-1] < 0 and gradient[i] > 0:
                        final_left = i
                        break
            
            # Search right from peak
            final_right = initial_right
            for i in range(peak_idx, min(len(gradient), peak_idx + 20)):
                if i < len(gradient) and abs(gradient[i]) < slope_threshold:
                    # Check if intensity has dropped significantly
                    if i_vals[i] < i_vals[peak_idx] * 0.1:  # Below 10% of peak
                        final_right = i
                        break
                    # Or if we're in a valley (derivative changes sign)
                    if i < len(gradient) - 1 and gradient[i] * gradient[i+1] < 0 and gradient[i] < 0:
                        final_right = i
                        break
            
            return final_left, final_right
            
        except Exception:
            # If derivative analysis fails, return initial boundaries
            return initial_left, initial_right
    
    # Apply advanced boundary detection
    refined_left, refined_right = find_peak_boundaries(
        rt_vals, i_vals_smooth, selected_peak, left_idx, right_idx
    )
    
    # Calculate suggested RT bounds with small padding
    base_rt_min = rt_vals[refined_left]
    base_rt_max = rt_vals[refined_right]
    
    # Add minimal padding (5% of peak width or 0.05 min, whichever is larger)
    peak_width_rt = base_rt_max - base_rt_min
    padding = max(0.05, peak_width_rt * 0.05)
    
    suggested_rt_peak = rt_vals[selected_peak]
    suggested_rt_min = base_rt_min - padding
    suggested_rt_max = base_rt_max + padding
    
    # Calculate comprehensive confidence score
    def calculate_confidence(peak_intensity, prominence, width_indices, rt_deviation, max_expected_dev):
        """Calculate confidence based on multiple peak quality metrics"""
        
        try:
            # Intensity score (normalized to 0-1)
            max_possible_intensity = np.max(i_vals_smooth)
            intensity_score = min(1.0, peak_intensity / max(max_possible_intensity, 1e3))
            
            # Prominence score (how well separated the peak is)
            prominence_score = min(1.0, prominence / max(peak_intensity * 0.5, 1e3))
            
            # Width score (prefer peaks that are not too narrow or too wide)
            # Optimal width is around 10-30 data points for chromatographic peaks
            optimal_width = 20
            width_score = 1.0 - min(1.0, abs(width_indices - optimal_width) / optimal_width)
            
            # RT proximity score
            rt_score = max(0.0, 1.0 - (rt_deviation / max(max_expected_dev, 1.0)))
            
            # Peak shape score (symmetry and sharpness)
            left_tail = selected_peak - refined_left
            right_tail = refined_right - selected_peak
            asymmetry = abs(left_tail - right_tail) / max(left_tail + right_tail, 1)
            shape_score = max(0.0, 1.0 - asymmetry)  # More symmetric = higher score
            
            # Weighted combination of all scores
            weights = {
                'intensity': 0.3,
                'prominence': 0.2,
                'width': 0.15,
                'rt_proximity': 0.2,
                'shape': 0.15
            }
            
            final_confidence = (
                weights['intensity'] * intensity_score +
                weights['prominence'] * prominence_score +
                weights['width'] * width_score +
                weights['rt_proximity'] * rt_score +
                weights['shape'] * shape_score
            )
            
            return min(1.0, max(0.0, final_confidence))
            
        except Exception:
            # If confidence calculation fails, return basic score
            return min(1.0, peak_intensity / max(np.max(i_vals_smooth), 1e3))
    
    # Calculate final confidence
    peak_intensity = i_vals_smooth[selected_peak]
    prominence = prominences[0] if len(prominences) > 0 else peak_intensity * 0.5
    rt_deviation = abs(suggested_rt_peak - atlas_rt_peak)
    
    confidence = calculate_confidence(
        peak_intensity, prominence, width_indices, rt_deviation, max_expected_deviation
    )
    
    # Additional quality checks
    if confidence < 0.1:  # Very low confidence
        return None
    
    if peak_width_rt > (atlas_rt_max - atlas_rt_min) * 2:  # Peak too wide
        confidence *= 0.5
    
    if rt_deviation > max_expected_deviation * 2:  # Peak too far from expected
        confidence *= 0.3
    
    return {
        'rt_min': float(suggested_rt_min),
        'rt_max': float(suggested_rt_max),
        'rt_peak': float(suggested_rt_peak),
        'confidence': float(confidence),
        'source_file': best_file,
        'peak_intensity': float(peak_intensity),
        'prominence': float(prominence),
        'peak_width_rt': float(peak_width_rt),
        'num_peaks_found': len(peaks)
    }

def set_up_plot_data(eics, atlas_df_ft, ms2_data_with_hits):
    """
    Build a complete compound metadata dictionary from EICs and atlas DataFrame.
    Returns a dictionary keyed by InChI key, with all EIC and MS2 info for plotting.
    """
    print("Building complete compound metadata dictionary...")

    metadata = {}

    # Set up isomer dictionary
    isomer_dict = {}
    for _, row in atlas_df_ft.iterrows():
        isomer_mz = row['mz']
        isomer_tolerance = row.get('mz_tolerance_ppm', 10.0)
        isomers = atlas_df_ft[np.isclose(atlas_df_ft['mz'], isomer_mz, atol=isomer_tolerance*1e-3)]
        isomers = isomers[isomers['inchi_key'] != row['inchi_key']]
        if not isomers.empty:
            isomer_list = []
            for _, isomer_row in isomers.iterrows():
                isomer_list.append({
                    'inchi_key': isomer_row['inchi_key'],
                    'compound_name': isomer_row['label'],
                    'rt': isomer_row['rt_peak'],
                    'mz': isomer_row['mz'],
                    'tolerance': isomer_row.get('mz_tolerance_ppm', 10.0),
                })
            isomer_dict[row['inchi_key']] = isomer_list
        else:
            isomer_dict[row['inchi_key']] = []

    for _, row in tqdm(atlas_df_ft.iterrows(), total=len(atlas_df_ft), desc="Processing compounds"):
        compound_name = row['label']
        compound_inchi = row['inchi_key']

        # Set up atlas data
        atlas_data = {
            'rt_min': row['rt_min'],
            'rt_max': row['rt_max'],
            'rt_peak': row['rt_peak'],
            'mz': row['mz'],
            'mz_tolerance': row.get('mz_tolerance_ppm', 10.0),
            'adduct': row.get('adduct', '[M+H]+'),
            'polarity': row.get('polarity', 'positive'),
            'compound_name': compound_name,
            'inchi_key': compound_inchi,
            'formula': row.get('formula', ''),
            'exact_mass': row.get('exact_mass', None),
            'isomers': isomer_dict.get(compound_inchi, []),
            #'ms2_notes': 'no selection',
            #'ms1_notes': 'keep'
        }

        # Initialize metadata structure
        metadata[compound_inchi] = {
            'original_atlas_data': atlas_data.copy(),
            'is_modified': False,
            'new_atlas_data': atlas_data.copy(),
            'suggested_atlas_data': None,  # Will be populated below
            'eic_data': {},
            'number_of_files': 0,
            'best_eic': {
                'file_peak': None,
                'rt_peak': None,
                'intensity_peak': None,
                'mz_peak': None,
                'ppm_diff': None,
                'rt_diff': None
            },
            'ms2_data': {}
        }

        # EIC extraction
        compound_eic_data = []
        for file_path, eic_df in eics.items():
            compound_eics = eic_df[eic_df['inchi_key'] == compound_inchi]
            if not compound_eics.empty:
                compound_eics = compound_eics.copy()
                compound_eics.loc[:, 'file_name'] = Path(file_path).name
                compound_eic_data.append(compound_eics)
        all_compound_eic_data = pd.concat(compound_eic_data, ignore_index=True) if compound_eic_data else pd.DataFrame()
        if not all_compound_eic_data.empty:
            all_compound_eic_data.sort_values(by=['intensity_peak'], ascending=False, inplace=True)
            # Store EIC traces as a dict of file_name: {rt_vals, i_vals}
            eic_dict = {}
            for _, eic_row in all_compound_eic_data.iterrows():
                # Sort RT and intensity values by RT ascending
                rt_vals = np.array(eic_row['rt'])
                i_vals = np.array(eic_row['i'])
                if len(rt_vals) > 1:
                    sort_idx = np.argsort(rt_vals)
                    rt_vals = rt_vals[sort_idx]
                    i_vals = i_vals[sort_idx]
                eic_dict[eic_row['file_name']] = {
                    'rt_vals': rt_vals,
                    'i_vals': i_vals,
                    'mz_vals': eic_row.get('mz', []),
                    'intensity_peak': eic_row.get('intensity_peak', None),
                    'rt_peak': eic_row.get('rt_peak', None),
                    'mz_peak': eic_row.get('mz_peak', None),
                    'rt_ms1_measured': eic_row.get('rt_peak', None),
                    'rt_theoretical': atlas_data['rt_peak'],
                    'ppm_diff': abs(eic_row.get('mz_peak') - atlas_data['mz']) / atlas_data['mz'] * 1e6,
                    'rt_diff': eic_row.get('rt_peak') - atlas_data['rt_peak']
                }

            # Best EIC (highest intensity)
            best_idx = all_compound_eic_data['intensity_peak'].idxmax()
            best_row = all_compound_eic_data.loc[best_idx]

            metadata[compound_inchi]['best_eic'] = {
                'file_peak': best_row['file_name'],
                'rt_peak': best_row['rt_peak'],
                'intensity_peak': best_row['intensity_peak'],
                'mz_peak': best_row['mz_peak'],
                'ppm_diff': abs(best_row['mz_peak'] - atlas_data['mz']) / atlas_data['mz'] * 1e6,
                'rt_diff': best_row['rt_peak'] - atlas_data['rt_peak']
            }
            metadata[compound_inchi]['eic_data'] = eic_dict
            metadata[compound_inchi]['number_of_files'] = len(eic_dict)
            
            # Calculate suggested RT bounds from EIC data
            suggested_bounds = suggest_rt_bounds_from_eic(
                eic_dict, 
                atlas_data['rt_peak'], 
                atlas_data['rt_min'], 
                atlas_data['rt_max']
            )
            metadata[compound_inchi]['suggested_atlas_data'] = suggested_bounds

        # MS2 experimental spectra and reference spectra (hits) - SIMPLIFIED STRUCTURE
        ms2_files_data = {}  # Changed to organize by file_name instead of file_path
        
        for file_path, ms2_datapoints in ms2_data_with_hits.items():
            file_name = Path(file_path).name
            extracted_spectra = []
            hit_pairs = []  # List of (query_spectrum, reference_spectrum) pairs
            
            for ms2_datum in ms2_datapoints:
                # Add experimental MS2 spectrum if inchi_key matches
                if ms2_datum.get('inchi_key', None) == compound_inchi:
                    spectrum = ms2_datum.get('spectrum', None)
                    if spectrum is not None and len(spectrum) == 2:
                        # Create experimental spectrum entry
                        exp_spectrum = {
                            'mz': np.array(spectrum[0]),
                            'intensity': np.array(spectrum[1]),
                            'precursor_mz': ms2_datum.get('precursor_mz', 0.0),
                            'precursor_intensity': ms2_datum.get('precursor_intensity', 0.0),
                            'rt': ms2_datum.get('rt', None),
                            'adduct': ms2_datum.get('adduct', ''),
                            'ppm_diff': abs(ms2_datum.get('precursor_mz', 0.0) - atlas_data['mz']) / atlas_data['mz'] * 1e6,
                        }
                        extracted_spectra.append(exp_spectrum)
                        
                        # Check for hits and create query/reference pairs
                        hits = ms2_datum.get('hits', [])
                        for hit in hits:
                            ref_spec = hit.get('msv_ref_aligned', None)
                            query_spec = hit.get('msv_query_aligned', None)
                            if ref_spec is not None and query_spec is not None and len(ref_spec) == 2 and len(query_spec) == 2:
                                # Compute frag_matches for query and reference
                                query_mz = np.array(query_spec[0])
                                query_intensity = np.array(query_spec[1])
                                ref_mz = np.array(ref_spec[0])
                                ref_intensity = np.array(ref_spec[1])

                                # For each query mz, check if there is a ref intensity at the same index that is nonzero and mz matches within 0.001
                                frag_matches_query = []
                                for n in range(len(query_mz)):
                                    if n < len(ref_intensity):
                                        if abs(query_mz[n] - ref_mz[n]) <= 0.001 and ref_intensity[n] > 0:
                                            frag_matches_query.append("green")
                                        else:
                                            frag_matches_query.append("red")
                                    else:
                                        frag_matches_query.append("red")

                                # For each ref mz, check if there is a query intensity at the same index that is nonzero and mz matches within 0.001
                                frag_matches_ref = []
                                for n in range(len(ref_mz)):
                                    if n < len(query_intensity):
                                        if abs(ref_mz[n] - query_mz[n]) <= 0.001 and query_intensity[n] > 0:
                                            frag_matches_ref.append("green")
                                        else:
                                            frag_matches_ref.append("red")
                                    else:
                                        frag_matches_ref.append("red")

                                query_spectrum = {
                                    'mz': query_mz,
                                    'intensity': query_intensity,
                                    'precursor_mz': hit.get('precursor_mz', 0.0),
                                    'rt': hit.get('msms_scan', 0.0),
                                    'data_frags': hit.get('data_frags', 0),
                                    'frag_matches': frag_matches_query,
                                }
                                
                                reference_spectrum = {
                                    'mz': ref_mz,
                                    'intensity': ref_intensity,
                                    'precursor_mz': hit.get('precursor_mz', 0.0),
                                    'adduct': hit.get('adduct', ms2_datum.get('adduct', '')),
                                    'score': hit.get('score', 0.0),
                                    'database': hit.get('database', 'unknown'),
                                    'rt': hit.get('msms_scan', 0.0),
                                    'num_matches': hit.get('num_matches', 0),
                                    'ref_frags': hit.get('ref_frags', 0),
                                    'hit_info': {
                                        'rt_ms2_measured': hit.get('msms_scan', 0.0),
                                        'precursor_mz': ms2_datum.get('precursor_mz', 0.0),
                                        'mz_theoretical': hit.get('precursor_mz', 0.0),
                                        'mz_measured': hit.get('measured_precursor_mz', 0.0),
                                        'ppm_diff': abs(hit.get('precursor_mz', 0.0) - hit.get('measured_precursor_mz', 0.0)) / hit.get('measured_precursor_mz', 1.0) * 1e6,
                                    },
                                    'frag_matches': frag_matches_ref,
                                }
                                hit_pairs.append((query_spectrum, reference_spectrum))
            
            # Only add file data if there are spectra
            if extracted_spectra or hit_pairs:
                ms2_files_data[file_name] = {
                    'extracted_spectra': extracted_spectra,
                    'hit_pairs': hit_pairs,
                    'has_extracted': len(extracted_spectra) > 0,
                    'has_hits': len(hit_pairs) > 0
                }

        metadata[compound_inchi]['ms2_data'] = ms2_files_data

        metadata[compound_inchi]['ms2_data']['total_files'] = len(ms2_files_data)
        metadata[compound_inchi]['ms2_data']['has_any_data'] = len(ms2_files_data) > 0

    return metadata

def filter_ms2_experimental_spectra_by_rt(experimental_spectra, rt_min, rt_max):
    """
    Filter experimental MS2 spectra to only include those within the specified RT bounds.
    
    Parameters:
    experimental_spectra: List of experimental spectra dictionaries
    rt_min: Minimum retention time
    rt_max: Maximum retention time

    Returns:
    List of filtered experimental spectra
    """
    filtered_spectra = []

    for spectrum in experimental_spectra:
        spectrum_rt = spectrum.get('rt', spectrum.get('scan_time', None))
        
        # Only include spectra within RT bounds
        if spectrum_rt is not None and rt_min <= spectrum_rt <= rt_max:
            filtered_spectra.append(spectrum)

    return filtered_spectra

def get_current_rt_bounds(compound_metadata):
    """
    Get the current RT bounds for a compound (either atlas defaults or user-modified values).
    
    Parameters:
    compound_metadata: The complete compound metadata dictionary

    Returns:
    tuple: (rt_min, rt_max, rt_peak)
    """
    
    current_rt_min = compound_metadata['new_atlas_data']['rt_min']
    current_rt_max = compound_metadata['new_atlas_data']['rt_max']
    current_rt_peak = compound_metadata['new_atlas_data']['rt_peak']
    
    return (current_rt_min, current_rt_max, current_rt_peak)

def write_ms2_title(compound_metadata, ms2_info, ms2_data_type, ms2_file):
    """Create MS2 title and subheadings"""
    
    # Create MS2 title (exact same format as original)
    name = compound_metadata['original_atlas_data']['compound_name']
    index = 1
    heading = f"{name} {index}"

    if ms2_data_type == 'hits' and ms2_info is not None:
        hit_info = ms2_info.get('hit_info', {})
        if hit_info:
            subheading1 = (
                f"Measured m/z: {hit_info['mz_measured']:.4f}  |  Reference m/z: {hit_info['mz_theoretical']:.4f}  |  "
                f"Reference DB: {ms2_info['database']}  |  Ref Matches: {ms2_info['num_matches']}  |  Score: {ms2_info['score']:.4f}"
            )
        else:
            subheading1 = (
                f'Measured m/z: N/A  |  Reference DB: N/A  |  Ref Matches: N/A  |  Score: N/A'
            )
    elif ms2_data_type == 'extracted' and ms2_info is not None:
        subheading1 = (
                    f"Measured m/z: {ms2_info.get('precursor_mz', 0.0)}  |  Theoretical m/z: {compound_metadata['original_atlas_data']['mz']:.4f}  |  "
                    f"Reference DB: N/A  |  Ref Matches: N/A  |  Score: N/A"
                )
    else:
        # Handle case when there's no MS2 data at all
        subheading1 = 'No MS2 data available'

    if ms2_file:
        subheading2 = f'Experimental: {ms2_file}'
    else:
        subheading2 = 'No file data'

    ms2_title = f'{heading}<br><sub>{subheading1}</sub><br><sub>{subheading2}</sub>'

    return ms2_title

def write_eic_title(compound_metadata):
    # Create EIC title (exact same format as original)
    eic_title_parts = []
    
    # Basic info
    if compound_metadata['original_atlas_data']['adduct']:
        eic_title_parts.append(f'Adduct: {compound_metadata["original_atlas_data"]["adduct"]}')
    if compound_metadata['original_atlas_data']['inchi_key']:
        eic_title_parts.append(f'InChI Key: {compound_metadata["original_atlas_data"]["inchi_key"]}')
    eic_title_parts.append(f'LCMS files: {compound_metadata["number_of_files"]}')
    subheading1 = ' | '.join(eic_title_parts)

    # RT and m/z comparison using CURRENT RT values
    rt_parts = [f'Atlas RT: {compound_metadata["original_atlas_data"]["rt_peak"]:.2f}']

    best_rt = compound_metadata['best_eic']['rt_peak']
    if best_rt is not None:
        # Recalculate RT diff using current RT peak
        current_rt_diff = abs(compound_metadata["original_atlas_data"]["rt_peak"] - best_rt)
        rt_parts.extend([
            f'Measured RT: {best_rt:.2f} min',
            f'RT diff: {current_rt_diff:.2f} min'
        ])
    else:
        rt_parts.extend(['Measured RT: N/A', 'RT diff: N/A'])
    
    rt_parts.append(f'Atlas m/z: {compound_metadata["original_atlas_data"]["mz"]:.4f}')
    
    best_mz = compound_metadata['best_eic']['mz_peak']
    if best_mz is not None:
        ppm_diff = compound_metadata['best_eic']['ppm_diff']
        rt_parts.extend([
            f'Measured m/z: {best_mz:.4f}',
            f'ppm diff: {ppm_diff:.2f}' if ppm_diff is not None else 'ppm diff: N/A'
        ])
    else:
        rt_parts.extend(['Measured m/z: N/A', 'ppm diff: N/A'])
    
    subheading2 = '   |   '.join(rt_parts)

    # Isomers
    isomers = compound_metadata['original_atlas_data']['isomers']
    isomer_strings = []
    for isomer in isomers:
        name = isomer.get('compound_name', '')
        isomer_inchi_key = isomer.get('inchi_key', '')
        rt_pk = isomer.get('rt', '')
        isomer_strings.append(f"{name} {rt_pk:.2f}")
    
    subheading3 = f'Isomers: {", ".join(isomer_strings)}' if isomer_strings else 'No isomers'

    eic_title = f'<sub>{subheading1}</sub><br><sub>{subheading2}</sub><br><sub>{subheading3}</sub>'

    return eic_title

def create_targeted_analysis_plot(compound_metadata, 
                                  height=600, file_color_dict=None, 
                                  ms2_data_type=None, ms2_plot_data=None):
    """
    Create the plot using pre-calculated metadata and selected MS2 spectra.
    Args:
        ms2_plot_data: tuple (has_hits, has_data, selected_query, selected_ref)
    """

    current_rt_min, current_rt_max, current_rt_peak = get_current_rt_bounds(compound_metadata)
    x_min = current_rt_min - 1.0
    x_max = current_rt_max + 1.0

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.35,  # Increased from 0.3 to add more space between MS2 and EIC plots
        subplot_titles=('', ''),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    eic_row = 2
    ms2_row = 1

    # MS2 plot logic
    selected_query, selected_ref, selected_file = ms2_plot_data
    exp_max_intensity = 0
    ref_max_intensity = 0
    ms2_title = ""

    # Handle case when there's no MS2 data at all
    if ms2_data_type is None or (selected_query is None and selected_ref is None):
        # Create empty MS2 plot with proper structure
        ms2_title = write_ms2_title(compound_metadata, None, None, None)
        pmz = 0.0
        max_mz = 500.0
        # Add an invisible trace to properly initialize the subplot
        fig.add_trace(go.Scatter(
            x=[0, max_mz],
            y=[0, 0],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),  # Completely transparent
            showlegend=False,
            hoverinfo='skip',
            name='Empty'
        ), row=ms2_row, col=1)
        
    elif ms2_data_type == 'hits':
        # Plot query (top) with hover info
        mz_vals = selected_query.get('mz', [])
        pmz = selected_query.get('precursor_mz', 0.0)
        mz_colors = selected_query.get('frag_matches', [])
        intensity_vals = selected_query.get('intensity', [])
        if len(mz_vals) > 0 and len(intensity_vals) > 0:
            max_mz = np.max(mz_vals)
            exp_max_intensity = max(exp_max_intensity, np.max(intensity_vals))
            
            # Create stick plot for experimental spectrum with per-stick color
            for mz, intensity, color in zip(mz_vals, intensity_vals, mz_colors):
                fig.add_trace(go.Scatter(
                    x=[mz, mz],
                    y=[0, intensity],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip', name='Experimental'
                ), row=ms2_row, col=1)
            
            # Add invisible markers at peak tops for hover interaction
            fig.add_trace(go.Scatter(
                x=list(mz_vals), y=list(intensity_vals),
                mode='markers',
                marker=dict(size=1, color="black", opacity=0.01),
                showlegend=False, name='Experimental Peaks',
                hovertemplate='m/z: %{x:.4f}<br>Intensity: %{y:.0f}<extra>Experimental</extra>'
            ), row=ms2_row, col=1)
        
        # Plot reference (bottom) with hover info
        mz_vals = selected_ref.get('mz', [])
        mz_colors = selected_ref.get('frag_matches', [])
        intensity_vals = selected_ref.get('intensity', [])
        if len(mz_vals) > 0 and len(intensity_vals) > 0:
            max_mz = np.max(mz_vals)
            ref_max_intensity = max(ref_max_intensity, np.max(intensity_vals))
            # Scale reference intensities to match experimental scale
            if exp_max_intensity > 0 and ref_max_intensity > 0:
                scaled_intensities = [i * (exp_max_intensity / ref_max_intensity) for i in intensity_vals]
            else:
                scaled_intensities = intensity_vals

            # Create stick plot for reference spectrum with per-stick color
            for mz, scaled_int, color in zip(mz_vals, scaled_intensities, mz_colors):
                fig.add_trace(go.Scatter(
                    x=[mz, mz],
                    y=[0, -scaled_int],  # Negative for downward
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip', name='Reference'
                ), row=ms2_row, col=1)

            # Add invisible markers at peak tops for hover interaction
            fig.add_trace(go.Scatter(
                x=list(mz_vals), y=[-s for s in scaled_intensities],
                mode='markers',
                marker=dict(size=1, color='red', opacity=0.01),
                showlegend=False, name='Reference Peaks',
                hovertemplate='m/z: %{x:.4f}<br>Intensity: %{customdata:.0f}<extra>Reference</extra>',
                customdata=list(intensity_vals)  # Use original intensities for hover
            ), row=ms2_row, col=1)
        
        ms2_title = write_ms2_title(compound_metadata, selected_ref, ms2_data_type, selected_file)
        
    elif ms2_data_type == 'extracted':
        # Plot only experimental spectrum (top) with hover info
        mz_vals = selected_query.get('mz', [])
        pmz = selected_query.get('precursor_mz', 0.0)
        intensity_vals = selected_query.get('intensity', [])
        if len(mz_vals) > 0 and len(intensity_vals) > 0:
            max_mz = np.max(mz_vals)
            exp_max_intensity = max(exp_max_intensity, np.max(intensity_vals))
            
            # Create stick plot for experimental spectrum
            x_coords = []
            y_coords = []
            for mz, intensity in zip(mz_vals, intensity_vals):
                x_coords.extend([mz, mz, None])
                y_coords.extend([0, intensity, None])
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords, mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False, hoverinfo='skip', name='Experimental'
            ), row=ms2_row, col=1)
            
            # Add invisible markers at peak tops for hover interaction
            fig.add_trace(go.Scatter(
                x=list(mz_vals), y=list(intensity_vals),
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.01),
                showlegend=False, name='Experimental Peaks',
                hovertemplate='m/z: %{x:.4f}<br>Intensity: %{y:.0f}<extra>Experimental</extra>'
            ), row=ms2_row, col=1)
        
        ms2_title = write_ms2_title(compound_metadata, selected_query, ms2_data_type, selected_file)

    # Configure MS2 subplot
    fig.update_xaxes(
        title_text="m/z",
        showgrid=False,
        range=[0, max_mz*1.1] if max_mz > 0 else [0, 500],
        row=ms2_row, col=1
    )
    
    # Set y-range for MS2 plot - handle empty plot case
    if exp_max_intensity > 0:
        y_range = [-exp_max_intensity * 1.1, exp_max_intensity * 1.1]
    else:
        y_range = [-1000, 1000]  # Default range for empty plot
        
    fig.update_yaxes(
        title_text="Intensity",
        showgrid=False,
        range=y_range,
        row=ms2_row, col=1,
        exponentformat="e",  # Use scientific notation
        showexponent="all"
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=ms2_row, col=1)
    fig.add_vline(x=pmz, line_dash="dash", line_color="black",
                   line_width=1, row=ms2_row, col=1)

    # Add EIC traces
    def get_file_color(file_name, color_dict, default_color="gray"):
        """Get color for file based on color dictionary or default"""
        if not color_dict:
            return default_color
        for key, color in color_dict.items():
            if key.lower() in file_name.lower():
                return color
        return default_color

    for file_name, trace_data in compound_metadata['eic_data'].items():
        color = get_file_color(file_name, file_color_dict)
        
        fig.add_trace(go.Scatter(
            x=trace_data['rt_vals'],
            y=trace_data['i_vals'],
            mode='lines',
            name=file_name,
            line=dict(color=color, width=2),
            hovertemplate=f'{file_name}<extra></extra>',
            showlegend=False
        ), row=eic_row, col=1)

    # Add RT reference lines using CURRENT working values
    fig.add_vline(x=current_rt_peak, line_dash="dash", line_color="black", line_width=3, row=eic_row, col=1)
    fig.add_vline(x=current_rt_min, line_dash="dot", line_color="green", line_width=2, row=eic_row, col=1)
    fig.add_vline(x=current_rt_max, line_dash="dot", line_color="green", line_width=2, row=eic_row, col=1)

    # Add suggested RT bounds as two light green vlines if available
    suggested_data = compound_metadata.get('suggested_atlas_data', None)
    if suggested_data is not None:
        rt_min = suggested_data['rt_min']
        rt_max = suggested_data['rt_max']
        # Add light green vlines for suggested RT min and max
        fig.add_vline(
            x=rt_min,
            line_dash="dot",
            line_color="lightgreen",
            line_width=3,
            row=eic_row,
            col=1
        )
        fig.add_vline(
            x=rt_max,
            line_dash="dot",
            line_color="lightgreen",
            line_width=3,
            row=eic_row,
            col=1
        )

    # Configure EIC subplot
    fig.update_xaxes(title_text='RT', 
                    range=[x_min, x_max],
                    showgrid=False, zeroline=False, showline=True,
                    linewidth=1, linecolor='black', ticks='outside',
                    tickwidth=1, tickcolor='black', row=eic_row, col=1)
    fig.update_yaxes(title_text='Intensity',
                    range=[0, None],
                    showgrid=False, zeroline=False, showline=True,
                    linewidth=1, linecolor='black', ticks='outside',
                    tickwidth=1, tickcolor='black', row=eic_row, col=1,
                    exponentformat="e",  # Use scientific notation
                    showexponent="all"
    )
    
    eic_title = write_eic_title(compound_metadata)

    # Adjusted annotation positions
    fig.add_annotation(
        text=eic_title,
        x=0.5, y=0.53,  # Reduced from 0.58 to move EIC title closer to plot
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12),
        xanchor="center"
    )

    fig.add_annotation(
        text=ms2_title,
        x=0.5, y=1.13,  # Increased from 1.1 to add more space above MS2 plot
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12),
        xanchor="center"
    )

    # Final layout configuration
    fig.update_layout(
        margin=dict(l=40, r=40, t=80, b=40),  # Reduced margins
        hovermode='closest',
        height=height,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        xaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
        yaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True)
    )

    # Add scale toggle buttons for both plots
    updatemenus = [
        # EIC plot buttons (bottom left)
        dict(
            type="buttons", direction="right",
            buttons=[
                dict(args=[{f"yaxis{eic_row}.type": "linear", f"yaxis{eic_row}.autorange": True}], 
                        label="Lin", method="relayout"),
                dict(args=[{f"yaxis{eic_row}.type": "log", f"yaxis{eic_row}.autorange": True}], 
                        label="Log", method="relayout")
            ],
            x=0.02, xanchor="left", y=0.02, yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        ),
        # MS2 plot buttons (top left)
        dict(
            type="buttons", direction="right",
            buttons=[
                dict(args=[{f"yaxis{ms2_row}.type": "linear", f"yaxis{ms2_row}.autorange": True}], 
                        label="Lin", method="relayout"),
                dict(args=[{f"yaxis{ms2_row}.type": "log", f"yaxis{ms2_row}.autorange": True}], 
                        label="Log", method="relayout")
            ],
            x=0.02, xanchor="left", y=0.65, yanchor="bottom",  # Positioned for MS2 plot
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
    ]
    
    fig.update_layout(updatemenus=updatemenus)

    return fig

def create_gui(compound_metadata, atlas_df, config):
    """
    Create an enhanced RT editor that includes MS2 file browsing capability.
    This combines the RT editing functionality with the ability to scroll through MS2 files.
    
    Args:
        compound_metadata: Pre-calculated metadata dictionary
        atlas_df: Original atlas DataFrame
    
    Returns:
        ipywidgets container with enhanced RT editor and MS2 browser
    """
    
    # Set starting status
    file_color_dict = config['plot_settings']['file_color_mapping'] if 'file_color_mapping' in config['plot_settings'] else None
    compound_list = list(compound_metadata.keys())
    all_compound_names = [meta['original_atlas_data']['compound_name'] for meta in compound_metadata.values()]
    current_compound_index = [0]
    current_ms2_file_index = [0]
    updating = [False]  # Prevent recursive updates
    rt_slider = [None]  # Store slider reference
    
    if not compound_list:
        print("No compounds found in metadata")
        return None

    # Define annotation options
    ms2_options = ['no selection',
                   '-1.0, poor match, should remove',
                   '0.0, no match or no MSMS collected',
                   '0.5, partial or putative match of fragments',
                   '1.0, good match',
                   '0.5, co-isolated precursor, partial match',
                   '1.0, co-isolated precursor, good match',
                   '0.5, single ion match, no evidence',
                   '1.0, single ion match, ISTD/ref evidence']
    
    ms1_options = ['keep', 'remove', 'unresolvable isomers', 'poor peak shape']

    # Create dropdown options: format as "index: compound_name" for clarity
    dropdown_options = []
    for i, compound_name in enumerate(all_compound_names):
        dropdown_options.append(f"{i+1}: {compound_name}")

    # Create main widgets with better height management
    plot_output = widgets.Output(layout=widgets.Layout(width='70%', height='600px'))
    prev_button = widgets.Button(description="◀ Prev", button_style='primary', layout=widgets.Layout(width='80px'))
    next_button = widgets.Button(description="Next ▶", button_style='primary', layout=widgets.Layout(width='80px'))
    counter_label = widgets.HTML(value=f"Compound 1 of {len(compound_list)}", layout=widgets.Layout(width='150px'))
    
    # Add compound dropdown for direct navigation
    compound_dropdown = widgets.Dropdown(
        options=dropdown_options,
        value=dropdown_options[0],
        description='Jump to:',
        layout=widgets.Layout(width='400px'),
        style={'description_width': '60px'}
    )

    ms2_prev_button = widgets.Button(description="◀ Prev MS2", button_style='info', layout=widgets.Layout(width='100px'))
    ms2_next_button = widgets.Button(description="Next MS2 ▶", button_style='info', layout=widgets.Layout(width='100px'))
    ms2_counter_label = widgets.HTML(value="", layout=widgets.Layout(width='150px'))
    
    reset_button = widgets.Button(description="Reset to Original", button_style='warning', layout=widgets.Layout(width='200px'))
    accept_suggestions_button = widgets.Button(description="Accept Suggestions", button_style='success', layout=widgets.Layout(width='200px'))

    # Create annotation radio buttons with compact layouts
    ms2_radio = widgets.RadioButtons(
        options=ms2_options,
        value=ms2_options[0],
        description='MS2 Notes:',
        layout=widgets.Layout(width='100%', height='240px'),
        style={'description_width': '80px'}
    )
    
    ms1_radio = widgets.RadioButtons(
        options=ms1_options,
        value=ms1_options[0],
        description='MS1 Notes:',
        layout=widgets.Layout(width='100%', height='360px'),
        style={'description_width': '80px'}
    )

    # Create placeholder for slider that will be replaced dynamically
    slider_container = widgets.HBox([], layout=widgets.Layout(justify_content='center', width='73%', margin='-110px 0 0 0'))

    def on_rt_slider_change(change):
        """Callback that fires whenever the user drags the slider."""
        if updating[0]:
            return
        
        # Additional check: only respond to user-initiated changes
        # by checking if the new value is actually different from what's in metadata
        compound_inchi, compound_name, meta = get_current_compound()
        if meta is None:
            return

        new_min, new_max = change['new']
        current_rt_min = meta['new_atlas_data']['rt_min']
        current_rt_max = meta['new_atlas_data']['rt_max']
        
        # Only update if the slider value is actually different from metadata
        # (with small tolerance for floating point comparison)
        tolerance = 1e-6
        if (abs(new_min - current_rt_min) < tolerance and 
            abs(new_max - current_rt_max) < tolerance):
            return

        # Write the new bounds to the metadata
        meta['new_atlas_data']['rt_min'] = float(new_min)
        meta['new_atlas_data']['rt_max'] = float(new_max)
        meta['new_atlas_data']['rt_peak'] = float(new_min + (new_max - new_min) / 2.0)
        meta['is_modified'] = True

        create_plot_only()

    def get_current_compound():
        """Get current compound name and metadata"""
        idx = current_compound_index[0]
        if 0 <= idx < len(compound_list):
            compound_inchi = compound_list[idx]
            compound_name = compound_metadata[compound_inchi]['original_atlas_data']['compound_name']
            return compound_inchi, compound_name, compound_metadata[compound_inchi]
        return None, None, None
    
    def reset_ms2_file_index():
        """Reset MS2 file index to 0 when switching compounds"""
        current_ms2_file_index[0] = 0

    def get_selected_ms2_spectra(meta, current_ms2_file_index, rt_min, rt_max):
        """
        Returns the selected MS2 spectra for plotting and info for the title.
        Files are sorted by best score (highest to lowest) within RT bounds.
        Returns:
            ms2_data_type: str ('hits', 'extracted', or None)
            selected_query: dict or None
            selected_ref: dict or None  
            selected_file: str or None
        """
        # Get MS2 files data
        ms2_data = meta.get('ms2_data', {})
        if not ms2_data or ms2_data.get('total_files', 0) == 0:
            return None, None, None, None
        
        # Get file names (excluding metadata keys)
        file_names = [k for k in ms2_data.keys() if k not in ['total_files', 'has_any_data']]
        if not file_names:
            return None, None, None, None
        
        # Filter spectra by RT bounds
        def filter_by_rt(spectra_list):
            return [s for s in spectra_list if rt_min <= s.get('rt', 0) <= rt_max]
        
        # Calculate best score for each file within RT bounds for sorting
        file_scores = []
        for file_name in file_names:
            file_data = ms2_data[file_name]
            
            filtered_hit_pairs = [(q, r) for q, r in file_data.get('hit_pairs', []) 
                                 if rt_min <= q.get('rt', 0) <= rt_max]
            filtered_extracted = filter_by_rt(file_data.get('extracted_spectra', []))
            
            # Get best score for this file
            best_score = 0.0
            if filtered_hit_pairs:
                best_score = max(pair[1].get('score', 0) for pair in filtered_hit_pairs)
            elif filtered_extracted:
                # For extracted spectra without hits, use intensity as a proxy score
                # Normalize to 0-1 range to be comparable with similarity scores
                max_intensity = max(np.max(s.get('intensity', [0])) for s in filtered_extracted)
                best_score = min(max_intensity / 1e6, 1.0) if max_intensity > 0 else 0.0
            
            file_scores.append((file_name, best_score))
        
        # Sort files by score (highest first)
        file_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_file_names = [name for name, score in file_scores]
        
        # Clamp file index and get selected file
        idx = min(current_ms2_file_index, len(sorted_file_names) - 1)
        selected_file = sorted_file_names[idx]
        file_data = ms2_data[selected_file]
        
        # Check what type of data we have and select best spectrum
        filtered_hit_pairs = [(q, r) for q, r in file_data.get('hit_pairs', []) 
                             if rt_min <= q.get('rt', 0) <= rt_max]
        filtered_extracted = filter_by_rt(file_data.get('extracted_spectra', []))
        
        if filtered_hit_pairs:
            # Sort hit pairs by reference spectrum score (highest first)
            filtered_hit_pairs.sort(key=lambda pair: pair[1].get('score', 0), reverse=True)
            query_spec, ref_spec = filtered_hit_pairs[0]
            return 'hits', query_spec, ref_spec, selected_file
        elif filtered_extracted:
            # Sort extracted spectra by intensity (highest first) 
            filtered_extracted.sort(key=lambda s: np.max(s.get('intensity', [0])), reverse=True)
            return 'extracted', filtered_extracted[0], None, selected_file
        else:
            return None, None, None, selected_file

    def update_annotation_widgets():
        """Update annotation radio buttons to reflect current compound's values"""
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        # Set radio button values from metadata WITHOUT removing observers
        current_ms2_notes = meta['new_atlas_data'].get('ms2_notes', ms2_options[0])
        current_ms1_notes = meta['new_atlas_data'].get('ms1_notes', ms1_options[0])
        
        # Temporarily disable updating flag to prevent recursion
        temp_updating = updating[0]
        updating[0] = True
        
        try:
            ms2_radio.value = current_ms2_notes
            ms1_radio.value = current_ms1_notes
        finally:
            updating[0] = temp_updating

    def on_ms2_annotation_change(change):
        """Handle MS2 annotation changes"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
                
        # Update metadata
        meta['new_atlas_data']['ms2_notes'] = change['new']

    def on_ms1_annotation_change(change):
        """Handle MS1 annotation changes"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        # Update metadata
        meta['new_atlas_data']['ms1_notes'] = change['new']

    def create_plot_only():
        """
        ONLY create and display the plot - no other side effects.
        """
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return

        plot_output.clear_output(wait=True)

        # Get RT bounds for MS2 filtering
        rt_min, rt_max, rt_peak = get_current_rt_bounds(meta)

        # Get MS2 selection info (now sorted by score)
        ms2_data_type, query_spec, ref_spec, ms2_file = get_selected_ms2_spectra(meta, 
                                                                                current_ms2_file_index[0],
                                                                                rt_min,
                                                                                rt_max)

        # Update MS2 counter label
        ms2_data = meta.get('ms2_data', {})
        total_files = ms2_data.get('total_files', 0)
        if total_files > 0:
            file_names = [k for k in ms2_data.keys() if k not in ['total_files', 'has_any_data']]
            current_file_idx = min(current_ms2_file_index[0], len(file_names) - 1)
            ms2_counter_label.value = f"MS2 File {current_file_idx + 1} of {total_files}"
        else:
            ms2_counter_label.value = "No MS2 data"

        # Create and show plot with fixed height
        fig = create_targeted_analysis_plot(
            compound_metadata=meta,
            file_color_dict=file_color_dict,
            ms2_data_type=ms2_data_type,
            ms2_plot_data=(query_spec, ref_spec, ms2_file)
        )

        if fig is not None:
            with plot_output:
                fig.show()

    def full_update():
        """
        Perform a complete update of all UI elements.
        This is the main orchestrator function.
        """
        if updating[0]:
            return
            
        updating[0] = True
        try:
            compound_inchi, compound_name, meta = get_current_compound()
            if compound_inchi is None or meta is None:
                return
            
            # Ensure annotation defaults are set in new_atlas_data
            if 'ms2_notes' not in meta['new_atlas_data']:
                meta['new_atlas_data']['ms2_notes'] = ms2_options[0]
            if 'ms1_notes' not in meta['new_atlas_data']:
                meta['new_atlas_data']['ms1_notes'] = ms1_options[0]
            
            # Get current RT bounds - this is the SINGLE SOURCE OF TRUTH
            rt_min, rt_max, rt_peak = get_current_rt_bounds(meta)
            
            # Calculate dynamic slider bounds based on compound's RT range
            rt_range = rt_max - rt_min
            #slider_min = max(0, rt_min - rt_range * 2)  # Allow 2x range below
            slider_min = rt_min - 1
            #slider_max = rt_max + rt_range * 2  # Allow 2x range above
            slider_max = rt_max + 1
            slider_step = max(0.01, rt_range / 100)  # 1% of range or 0.01 min minimum
            
            # Create new RT slider with dynamic bounds
            new_slider = widgets.FloatRangeSlider(
                value=(rt_min, rt_max),
                min=slider_min,
                max=slider_max,
                step=slider_step,
                description='RT window',
                continuous_update=True,
                layout=widgets.Layout(width='95%')
            )
            
            # Attach observer to new slider
            new_slider.observe(on_rt_slider_change, names='value')
            
            # Replace slider in container
            slider_container.children = [new_slider]
            rt_slider[0] = new_slider  # Store reference
            
            # Update all UI elements WITHOUT any observers active
            update_annotation_widgets()
            update_dropdown_value()  # Update dropdown to match current compound
            create_plot_only()
            
        finally:
            updating[0] = False

    def on_navigation(direction):
        """Handle compound navigation"""
        if updating[0]:
            return

        idx = current_compound_index[0] + direction
        if 0 <= idx < len(compound_list):
            current_compound_index[0] = idx
            counter_label.value = f"Compound {idx+1} of {len(compound_list)}"

            # Reset MS2 file index when switching compounds
            reset_ms2_file_index()

            # Do a full update
            full_update()

    def on_ms2_navigation(direction):
        """Handle MS2 file navigation (now sorted by score)"""
        if updating[0]:
            return
            
        compound_inchi, _, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return

        ms2_data = meta.get('ms2_data', {})
        total_files = ms2_data.get('total_files', 0)
        if total_files == 0:
            return

        new_idx = current_ms2_file_index[0] + direction
        # Proper bounds checking - files are now sorted by score
        if 0 <= new_idx < total_files:
            current_ms2_file_index[0] = new_idx
            create_plot_only()

    def on_reset(button):
        """Reset current compound to original RT bounds"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None:
            return
        
        updating[0] = True
        try:
            # Get original values
            original_rt_min = meta['original_atlas_data']['rt_min']
            original_rt_max = meta['original_atlas_data']['rt_max']
            original_rt_peak = meta['original_atlas_data']['rt_peak']
            
            # Reset metadata
            meta['new_atlas_data']['rt_min'] = original_rt_min
            meta['new_atlas_data']['rt_max'] = original_rt_max
            meta['new_atlas_data']['rt_peak'] = original_rt_peak
            meta['is_modified'] = False
            
            # Update all UI elements (this will recreate the slider with correct bounds)
            create_plot_only()
            
            # Update slider to match reset values if it exists
            if rt_slider[0] is not None and rt_slider[0].value != (original_rt_min, original_rt_max):
                rt_slider[0].value = (original_rt_min, original_rt_max)
            
        finally:
            updating[0] = False

    def update_dropdown_value():
        """Update dropdown to reflect current compound without triggering observer"""
        idx = current_compound_index[0]
        if 0 <= idx < len(dropdown_options):
            # Temporarily disable updating to prevent recursion
            temp_updating = updating[0]
            updating[0] = True
            try:
                compound_dropdown.value = dropdown_options[idx]
            finally:
                updating[0] = temp_updating

    def on_dropdown_change(change):
        """Handle dropdown selection changes"""
        if updating[0]:
            return
        
        # Extract index from dropdown value (format: "index: compound_name")
        try:
            selected_value = change['new']
            idx = int(selected_value.split(':')[0]) - 1  # Convert from 1-based to 0-based index
            
            if 0 <= idx < len(compound_list) and idx != current_compound_index[0]:
                current_compound_index[0] = idx
                counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
                
                # Reset MS2 file index when switching compounds
                reset_ms2_file_index()
                
                # Do a full update
                full_update()
                
        except (ValueError, IndexError):
            # If parsing fails, ignore the change
            pass

    def on_accept_suggestions(button):
        """Accept suggested RT bounds for current compound"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        suggested_data = meta.get('suggested_atlas_data', None)
        if suggested_data is None:
            return
        
        updating[0] = True
        try:
            # Reset MS2 file index since RT bounds will change
            reset_ms2_file_index()
            
            # Update the compound metadata with suggested values
            suggested_rt_min = suggested_data['rt_min']
            suggested_rt_max = suggested_data['rt_max']
            suggested_rt_peak = suggested_data['rt_peak']
            
            # Update metadata directly
            meta['new_atlas_data']['rt_min'] = suggested_rt_min
            meta['new_atlas_data']['rt_max'] = suggested_rt_max
            meta['new_atlas_data']['rt_peak'] = suggested_rt_peak
            meta['is_modified'] = True

            # Update all UI elements (this will recreate the slider with correct bounds)
            create_plot_only()
            
            # Update slider to match accepted suggestions if it exists
            if rt_slider[0] is not None and rt_slider[0].value != (suggested_rt_min, suggested_rt_max):
                rt_slider[0].value = (suggested_rt_min, suggested_rt_max)
            
        finally:
            updating[0] = False

    # Connect event handlers - ATTACH OBSERVERS ONLY ONCE
    prev_button.on_click(lambda b: on_navigation(-1))
    next_button.on_click(lambda b: on_navigation(1))
    ms2_prev_button.on_click(lambda b: on_ms2_navigation(-1))
    ms2_next_button.on_click(lambda b: on_ms2_navigation(1))
    reset_button.on_click(on_reset)
    accept_suggestions_button.on_click(on_accept_suggestions)
    
    # Add dropdown observer
    compound_dropdown.observe(on_dropdown_change, names='value')
    
    # Add annotation observers only once and never remove them
    ms2_radio.observe(on_ms2_annotation_change, names='value')
    ms1_radio.observe(on_ms1_annotation_change, names='value')

    # Create layout structure
    nav_row = widgets.HBox([prev_button, counter_label, next_button], layout=widgets.Layout(justify_content='center'))
    
    # Add dropdown to navigation - put it in its own row for better layout
    dropdown_row = widgets.HBox([compound_dropdown], layout=widgets.Layout(justify_content='center', margin='5px 0'))
    
    # MS2 browser row
    ms2_nav_row = widgets.HBox([ms2_prev_button, ms2_counter_label, ms2_next_button], layout=widgets.Layout(justify_content='center'))

    # Combine all navigation elements horizontally
    nav_box = widgets.HBox([
        nav_row,
        dropdown_row,
        ms2_nav_row
    ], layout=widgets.Layout(width='100%', align_items='center'))
        
    # Create the main plot and radio button container
    plot_and_radios = widgets.HBox([
        plot_output,
        widgets.VBox([
            widgets.Box([ms2_radio], layout=widgets.Layout(margin='50px 0 50px 0')),
            ms1_radio
        ], layout=widgets.Layout(width='30%', justify_content='flex-start'))
    ], layout=widgets.Layout(width='100%'))
    
    # Create metadata rows
    button_row = widgets.HBox([reset_button, accept_suggestions_button], 
                             layout=widgets.Layout(justify_content='center', margin='0 0 0 0'))
    
    container = widgets.VBox([
        nav_box, 
        plot_and_radios,
        slider_container,
        button_row,
    ], layout=widgets.Layout(width='100%', align_items='flex-start', height='fit-content'))
    
    # Add helper methods
    container.metadata = compound_metadata

    # Initialize with the first compound
    full_update()
    
    return container