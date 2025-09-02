import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
import duckdb
from tqdm.notebook import tqdm

from matchms import Spectrum
from matchms.similarity import CosineHungarian

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

from typing import Dict, List, Optional, Any, Tuple

sys.path.append('/Users/BKieft/Metabolomics/metatlas')
sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
from metatlas.io import feature_tools as ft
import metatlas2.lcmsruns_tools as lrt
import metatlas2.database_interact as dbi
import metatlas2.targeted_analysis as tga

def extract_and_match_qc_compounds(project_db_path: str, qc_atlas_uid: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract MS1 data from QC files and match with QC Atlas compounds.
    Simplified approach that processes all files and matches compounds in a straightforward manner.
    
    Args:
        project_db_path: Path to project database containing lcmsruns
        qc_atlas_uid: UID of QC atlas to use for matching
        config: Dictionary containing extraction parameters

    Returns:
        Tuple of (qc_compound_data, matching_stats)
    """
    print("Loading QC files and atlas compounds from databases...")

    # Define thresholds
    database_path = config['paths']['main_database']
    metadata = config['rt_alignment']['tolerances']

    # Get QC files from project database
    qc_files_df = dbi.get_files_by_type_from_db(project_db_path, 'qc')
    
    # Load QC compounds from atlas
    qc_compounds = dbi.get_atlas_compounds_table(database_path, qc_atlas_uid)
    if qc_compounds.empty or qc_compounds is None:
        raise ValueError(f"No compounds found for QC atlas UID: {qc_atlas_uid}")
    
    print(f"Found {len(qc_files_df)} QC files and {len(qc_compounds)} QC compounds")
    
    # Extract all MS1 data from all QC files
    print("Extracting MS1 data from QC files...")
    all_ms1_data = []
    
    for _, file_row in tqdm(qc_files_df.iterrows(), total=len(qc_files_df), desc="Extracting MS1 data"):
        file_path = file_row['file_path']
        chromatography = file_row['chromatography']
        polarity = file_row['polarity']
        
        try:
            # Determine MS1 key - handle FPS files by trying both polarities
            ms1_keys_to_try = []
            if polarity.upper() == "FPS":
                ms1_keys_to_try = ["ms1_pos", "ms1_neg"]
            elif polarity.lower() == "positive":
                ms1_keys_to_try = ["ms1_pos"]
            elif polarity.lower() == "negative":
                ms1_keys_to_try = ["ms1_neg"]
            
            for ms1_key in ms1_keys_to_try:
                try:
                    ms1_data = lrt.read_hdf_file(file_path, desired_key=ms1_key)
                    
                    if ms1_data is not None and len(ms1_data) > 0:
                        # Add metadata
                        ms1_data['file_path'] = file_path
                        ms1_data['filename'] = os.path.basename(file_path)
                        ms1_data['chromatography'] = chromatography
                        ms1_data['file_polarity'] = polarity  # Original file polarity
                        ms1_data['data_polarity'] = "positive" if ms1_key == "ms1_pos" else "negative"  # Actual data polarity
                        
                        # Filter by minimum intensity
                        if 'i' in ms1_data.columns:
                            ms1_data = ms1_data[ms1_data['i'] >= metadata['i']]
                        
                        # Filter by RT range
                        if 'rt' in ms1_data.columns and len(ms1_data) > 0:
                            rt_max = ms1_data['rt'].max()
                            ms1_data = ms1_data[(ms1_data['rt'] >= 0.0) & (ms1_data['rt'] <= rt_max)]
                        
                        if len(ms1_data) > 0:
                            all_ms1_data.append(ms1_data)
                            print(f"  Extracted {len(ms1_data)} peaks from {os.path.basename(file_path)} ({ms1_key})")
                
                except Exception as e:
                    print(f"  Warning: Could not extract {ms1_key} from {file_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
            continue
    
    if not all_ms1_data:
        raise ValueError("No MS1 data could be extracted from any QC files")
    
    # Combine all MS1 data
    combined_ms1_data = pd.concat(all_ms1_data, ignore_index=True)
    print(f"Total MS1 peaks extracted: {len(combined_ms1_data):,}")
    
    # Match QC compounds to extracted peaks
    print("\nMatching QC compounds to extracted peaks...")
    compound_matches = []
    
    for _, compound in tqdm(qc_compounds.iterrows(), total=len(qc_compounds), desc="Matching compounds"):
        compound_name = compound['compound_name']
        target_mz = compound['mz']
        atlas_rt_peak = compound['rt_peak']
        atlas_rt_min = compound['rt_min'] 
        atlas_rt_max = compound['rt_max']
        compound_chromatography = compound['chromatography']
        compound_polarity = compound['polarity']
        
        # Calculate tolerances
        mz_tolerance = metadata['mz']
        mz_tolerance_da = target_mz * mz_tolerance / 1e6
        
        # RT window with expansion
        rt_min_search = atlas_rt_min - metadata['rt']
        rt_max_search = atlas_rt_max + metadata['rt']
        
        # Filter MS1 data for this compound
        # Match chromatography exactly
        ms1_subset = combined_ms1_data[combined_ms1_data['chromatography'] == compound_chromatography]
        
        # Handle polarity matching (FPS files can match any compound polarity)
        if compound_polarity.upper() == "FPS":
            # Compound is FPS, matches any data polarity
            pass  # Don't filter by polarity
        else:
            # Compound has specific polarity, match with corresponding data polarity or FPS files
            ms1_subset = ms1_subset[
                (ms1_subset['data_polarity'] == compound_polarity.lower()) |
                (ms1_subset['file_polarity'].str.upper() == "FPS")
            ]
        
        # Apply m/z and RT filters
        matching_peaks = ms1_subset[
            (ms1_subset['mz'] >= target_mz - mz_tolerance_da) &
            (ms1_subset['mz'] <= target_mz + mz_tolerance_da) &
            (ms1_subset['rt'] >= rt_min_search) &
            (ms1_subset['rt'] <= rt_max_search)
        ].copy()
        
        if len(matching_peaks) > 0:
            # Calculate errors
            matching_peaks['mz_error_ppm'] = (matching_peaks['mz'] - target_mz) / target_mz * 1e6
            matching_peaks['rt_difference'] = matching_peaks['rt'] - atlas_rt_peak
            
            # Group by file and take best peak per file
            for filename, file_peaks in matching_peaks.groupby('filename'):
                best_peak = file_peaks.loc[file_peaks['i'].idxmax()]
                
                compound_matches.append({
                    'compound_uid': compound['compound_uid'],
                    'compound_name': compound_name,
                    'inchi_key': compound.get('inchi_key', 'unknown'),
                    'atlas_rt_peak': atlas_rt_peak,
                    'atlas_rt_min': atlas_rt_min,
                    'atlas_rt_max': atlas_rt_max,
                    'atlas_mz': target_mz,
                    'observed_rt': best_peak['rt'],
                    'observed_mz': best_peak['mz'],
                    'observed_intensity': best_peak['i'],
                    'mz_error_ppm': best_peak['mz_error_ppm'],
                    'rt_difference': best_peak['rt_difference'],
                    'filename': filename,
                    'file_path': best_peak['file_path'],
                    'chromatography': compound_chromatography,
                    'polarity': compound_polarity,
                    'mz_tolerance_used': mz_tolerance
                })
    
    # Create results
    if compound_matches:
        qc_compound_data = pd.DataFrame(compound_matches)
        
        # Calculate statistics
        matching_stats = {
            'total_compounds': len(qc_compounds),
            'compounds_with_matches': qc_compound_data['compound_uid'].nunique(),
            'compounds_without_matches': len(qc_compounds) - qc_compound_data['compound_uid'].nunique(),
            'total_matches': len(qc_compound_data),
            'total_files': len(qc_files_df),
            'total_peaks_extracted': len(combined_ms1_data)
        }
        
        print(f"\nMatching completed:")
        print(f"  Compounds with matches: {matching_stats['compounds_with_matches']}/{matching_stats['total_compounds']}")
        print(f"  Total compound-file matches: {matching_stats['total_matches']}")
        print(f"  Mean m/z error: {qc_compound_data['mz_error_ppm'].mean():.2f} ± {qc_compound_data['mz_error_ppm'].std():.2f} ppm")
        print(f"  Mean RT difference: {qc_compound_data['rt_difference'].mean():.3f} ± {qc_compound_data['rt_difference'].std():.3f} min")
        
        return qc_compound_data, matching_stats
    else:
        print("No compound matches found. Check Atlas compound definitions, m/z tolerance, RT window settings, and QC file data quality")
        raise ValueError("No compound matches found")

def calculate_mz_tolerance_range(mz: float, tolerance_ppm: float) -> Tuple[float, float]:
    """Calculate m/z tolerance range in Daltons."""
    tolerance_da = mz * tolerance_ppm / 1e6
    return mz - tolerance_da, mz + tolerance_da

def find_peaks_in_rt_window(ms1_data: pd.DataFrame, target_mz: float, 
                           mz_tolerance_ppm: float, rt_data: Dict[str, float], 
                           rt_window: float = 0.5) -> pd.DataFrame:
    """
    Find peaks within m/z tolerance and RT window.
    
    Args:
        ms1_data: MS1 data DataFrame
        target_mz: Target m/z value
        mz_tolerance_ppm: m/z tolerance in ppm
        rt_data: RT data dictionary with 'center', 'min', and 'max' keys
        rt_window: RT window around center (minutes)
    
    Returns:
        DataFrame of matching peaks
    """
    # Calculate m/z range
    mz_min, mz_max = calculate_mz_tolerance_range(target_mz, mz_tolerance_ppm)
    
    # Calculate RT range
    rt_min = rt_data['min'] - rt_window
    rt_max = rt_data['max'] + rt_window
    print(mz_min, mz_max, rt_min, rt_max)
    display(ms1_data)
    # Filter peaks
    matching_peaks = ms1_data[
        (ms1_data['mz'] >= mz_min) & 
        (ms1_data['mz'] <= mz_max) & 
        (ms1_data['rt'] >= rt_min) & 
        (ms1_data['rt'] <= rt_max)
    ].copy()
    display(matching_peaks)
    if len(matching_peaks) > 0:
        # Calculate m/z error
        matching_peaks['mz_error_ppm'] = (
            (matching_peaks['mz'] - target_mz) / target_mz * 1e6
        )
        # Calculate RT difference
        rt_diff = matching_peaks['rt'] - rt_center
        if abs(rt_diff).any() > 1:
            print(f"RT difference exceeds threshold for {len(matching_peaks)} peaks")
        matching_peaks['rt_difference'] = rt_diff

    return matching_peaks

def align_ms_arrays(query_mz, query_intensity, ref_mz, ref_intensity, mz_tolerance=0.005, intensity_tolerance=1000):
    """
    Align MS2 vectors using the metatlas approach for consistent fragment matching.
    Produces aligned arrays for query and reference spectra, matching fragments within mz_tolerance
    and intensity above intensity_tolerance. Non-matching fragments get intensity 0.

    Args:
        query_mz: Query spectrum m/z values (array-like)
        query_intensity: Query spectrum intensities (array-like)
        ref_mz: Reference spectrum m/z values (array-like)
        ref_intensity: Reference spectrum intensities (array-like)
        mz_tolerance: m/z tolerance for matching (default: 0.005)
        intensity_tolerance: minimum intensity for counting a match (default: 1000)

    Returns:
        tuple: (aligned_query_mz, aligned_query_intensity, aligned_ref_mz, aligned_ref_intensity, num_matches)
    """
    # Convert to numpy arrays
    query_mz = np.array(query_mz, dtype=np.float64)
    query_intensity = np.array(query_intensity, dtype=np.float64)
    ref_mz = np.array(ref_mz, dtype=np.float64)
    ref_intensity = np.array(ref_intensity, dtype=np.float64)

    # Check lengths
    if len(query_mz) != len(query_intensity):
        raise ValueError("query_mz and query_intensity must have the same length")
    if len(ref_mz) != len(ref_intensity):
        raise ValueError("ref_mz and ref_intensity must have the same length")

    # Combine all m/z values and sort
    all_mz = np.concatenate([query_mz, ref_mz])
    all_mz_unique = np.unique(np.round(all_mz, 10))  # rounding to avoid floating point issues

    aligned_query_intensity = []
    aligned_ref_intensity = []
    aligned_mz = []

    num_matches = 0

    for mz in all_mz_unique:
        # Find closest query peak within tolerance
        query_idx = np.where(np.abs(query_mz - mz) <= mz_tolerance)[0]
        query_val = np.max(query_intensity[query_idx]) if len(query_idx) > 0 else 0.0

        # Find closest ref peak within tolerance
        ref_idx = np.where(np.abs(ref_mz - mz) <= mz_tolerance)[0]
        ref_val = np.max(ref_intensity[ref_idx]) if len(ref_idx) > 0 else 0.0

        aligned_mz.append(mz)
        aligned_query_intensity.append(query_val)
        aligned_ref_intensity.append(ref_val)

        # Count as match if both intensities above threshold
        if query_val > intensity_tolerance and ref_val > intensity_tolerance:
            num_matches += 1

    return (np.array(aligned_mz), np.array(aligned_query_intensity),
            np.array(aligned_mz), np.array(aligned_ref_intensity), num_matches)

def get_ms2_hits_from_data(ms2_data_dict, reference_df, 
                          frag_mz_tolerance=0.005,
                          intensity_tolerance=1000,
                          keep_nonmatches=True,
                          precursor_mz_tolerance_ppm=10.0):
    """
    Get MS2 hits from your ms2_data object using a reference DataFrame.
    
    Updated to use metatlas-style alignment with matchms Spectrum objects and cosine similarity
    for consistency with existing metatlas workflows while leveraging matchms capabilities.
    
    Args:
        ms2_data_dict: Dictionary with file paths as keys and DataFrames as values
                      Each DataFrame should have columns from ft.calculate_ms2_summary():
                      ['label', 'spectrum', 'rt', 'precursor_mz', 'precursor_intensity', 
                       'inchi_key', 'compound_uid', 'adduct']
        reference_df: DataFrame from load_msms_refs_file() with columns:
                     ['database', 'id', 'name', 'spectrum', 'precursor_mz', 'inchi_key', ...]
        frag_mz_tolerance: m/z tolerance for fragment matching (default: 0.005)
        keep_nonmatches: Whether to keep entries without reference matches (default: True)
        precursor_mz_tolerance_ppm: Precursor m/z tolerance in ppm (default: 10.0)
        
    Returns:
        DataFrame with MS2 hits in the same format as get_msms_hits()
        Columns: ['database', 'id', 'file_name', 'msms_scan', 'score',
                 'msv_query_aligned', 'msv_ref_aligned', 
                 'name', 'adduct', 'inchi_key', 'precursor_mz', 'measured_precursor_mz',
                 'measured_precursor_intensity']
    """
    
    # Initialize the matchms similarity scorer
    cos = CosineHungarian(tolerance=frag_mz_tolerance)
    
    # Columns for the output DataFrame (matching get_msms_hits format)
    msms_hits_cols = ['database', 'id', 'file_name', 'msms_scan', 'score', 'num_matches',
                     'msv_query_unaligned', 'msv_ref_unaligned',
                     'msv_query_aligned', 'msv_ref_aligned', 'name', 'adduct', 'inchi_key',
                     'precursor_mz', 'measured_precursor_mz',
                     'measured_precursor_intensity']
    
    all_hits = []
    
    # Process each file
    for file_path, ms2_df in tqdm(ms2_data_dict.items(), desc="Processing MS2 files"):
        file_name = Path(file_path).name
        
        # Process each spectrum in the file
        for idx, row in ms2_df.iterrows():
            # Extract spectrum data - handle different possible column names
            spectrum_data = None
            for col in ['spectrum', 'spectra', 'ms2_spectrum']:
                if col in row and row[col] is not None:
                    spectrum_data = row[col]
                    break
            
            if spectrum_data is None:
                continue
                
            # Parse spectrum data format from ft.calculate_ms2_summary
            try:
                if isinstance(spectrum_data, (list, tuple)) and len(spectrum_data) == 2:
                    mz_values = np.array(spectrum_data[0], dtype=np.float64)
                    intensity_values = np.array(spectrum_data[1], dtype=np.float64)
                elif isinstance(spectrum_data, np.ndarray) and spectrum_data.shape[0] == 2:
                    mz_values = np.array(spectrum_data[0], dtype=np.float64)
                    intensity_values = np.array(spectrum_data[1], dtype=np.float64)
                else:
                    continue
            except:
                continue
            
            if len(mz_values) == 0 or len(intensity_values) == 0:
                continue
            
            # Get precursor m/z - handle different possible column names
            precursor_mz = None
            for col in ['precursor_mz', 'mz', 'parent_mz']:
                if col in row and not pd.isna(row[col]):
                    precursor_mz = float(row[col])
                    break
            
            if precursor_mz is None:
                continue
            
            # Get precursor intensity - handle different possible column names
            precursor_intensity = 1.0  # Default
            for col in ['precursor_intensity', 'intensity', 'precursor_peak_height', 'base_peak_intensity']:
                if col in row and not pd.isna(row[col]):
                    precursor_intensity = float(row[col])
                    break
            
            # Get RT value - handle different possible column names
            rt_value = 0.0  # Default
            for col in ['rt', 'retention_time', 'scan_time']:
                if col in row and not pd.isna(row[col]):
                    rt_value = float(row[col])
                    break
            
            # Create matchms Spectrum object for the query
            query_spectrum = Spectrum(mz=mz_values, 
                                    intensities=intensity_values,
                                    metadata={'precursor_mz': precursor_mz})
            
            # Get inchi_key for matching
            inchi_key = row.get('inchi_key', '')
            if pd.isna(inchi_key):
                inchi_key = ''
            
            # Find matching reference spectra
            if inchi_key == '':
                if keep_nonmatches:
                    # Create a non-match entry
                    hit = {
                        'database': None,
                        'id': None,
                        'file_name': file_name,
                        'msms_scan': rt_value,
                        'score': 0.0,
                        'msv_query_aligned': np.array([mz_values, intensity_values]),
                        'msv_ref_aligned': np.array([[], []]),
                        'name': row.get('label', 'Unknown'),
                        'adduct': row.get('adduct', ''),
                        'inchi_key': inchi_key,
                        'precursor_mz': np.nan,
                        'measured_precursor_mz': precursor_mz,
                        'measured_precursor_intensity': precursor_intensity
                    }
                    all_hits.append(hit)
                continue
            
            # Find reference spectra with matching inchi_key
            matching_refs = reference_df[reference_df['inchi_key'] == inchi_key]
            
            if matching_refs.empty:
                if keep_nonmatches:
                    # Create a non-match entry
                    hit = {
                        'database': 'no_match',
                        'id': f'no_match_{inchi_key}',
                        'file_name': file_name,
                        'msms_scan': rt_value,
                        'score': 0.0,
                        'msv_query_aligned': np.array([mz_values, intensity_values]),
                        'msv_ref_aligned': np.array([[], []]),
                        'name': row.get('label', 'Unknown'),
                        'adduct': row.get('adduct', ''),
                        'inchi_key': inchi_key,
                        'precursor_mz': np.nan,
                        'measured_precursor_mz': precursor_mz,
                        'measured_precursor_intensity': precursor_intensity
                    }
                    all_hits.append(hit)
                continue
            
            # Compare against each matching reference spectrum
            for _, ref_row in matching_refs.iterrows():
                # Extract reference spectrum data
                ref_spectrum_data = ref_row.get('spectrum', None)
                if ref_spectrum_data is None:
                    continue
                
                # Handle reference spectrum format
                try:
                    if isinstance(ref_spectrum_data, np.ndarray) and ref_spectrum_data.shape[0] == 2:
                        ref_mz = np.array(ref_spectrum_data[0], dtype=np.float64)
                        ref_intensity = np.array(ref_spectrum_data[1], dtype=np.float64)
                    elif isinstance(ref_spectrum_data, (list, tuple)) and len(ref_spectrum_data) == 2:
                        ref_mz = np.array(ref_spectrum_data[0], dtype=np.float64)
                        ref_intensity = np.array(ref_spectrum_data[1], dtype=np.float64)
                    else:
                        continue
                except:
                    continue
                
                if len(ref_mz) == 0 or len(ref_intensity) == 0:
                    continue
                
                # Check precursor m/z tolerance if available
                ref_precursor_mz = ref_row.get('precursor_mz', np.nan)
                if not pd.isna(ref_precursor_mz):
                    ppm_error = abs(precursor_mz - ref_precursor_mz) / ref_precursor_mz * 1e6
                    if ppm_error > precursor_mz_tolerance_ppm:
                        continue
                
                # Create matchms Spectrum object for the reference
                ref_spectrum = Spectrum(mz=ref_mz, 
                                      intensities=ref_intensity,
                                      metadata={'precursor_mz': ref_precursor_mz})
                
                # Calculate similarity score using matchms
                try:
                    similarity_result = cos.pair(query_spectrum, ref_spectrum)
                    score = float(similarity_result['score'])
                    matchms_matches = int(similarity_result['matches'])
                except Exception as e:
                    print(f"Error calculating similarity with matchms: {e}")
                    score = 0.0
                    matchms_matches = 0
                
                # Align MS vectors using metatlas approach for consistent output format
                try:
                    (aligned_query_mz, aligned_query_intensity,
                     aligned_ref_mz, aligned_ref_intensity, num_matches) = align_ms_arrays(
                        mz_values, intensity_values, ref_mz, ref_intensity, frag_mz_tolerance
                    )
                    
                except Exception as e:
                    print(f"Error aligning spectra: {e}")
                    aligned_query_mz = mz_values
                    aligned_query_intensity = intensity_values
                    aligned_ref_mz = ref_mz
                    aligned_ref_intensity = ref_intensity
                
                # Create hit entry
                hit = {
                    'database': ref_row.get('database', 'unknown'),
                    'id': str(ref_row.get('id', '')),
                    'file_name': file_name,
                    'msms_scan': rt_value,
                    'score': float(score),
                    'num_matches': num_matches,
                    'msv_query_unaligned': np.array([mz_values, intensity_values]),
                    'msv_ref_unaligned': np.array([ref_mz, ref_intensity]),
                    'msv_query_aligned': np.array([aligned_query_mz, aligned_query_intensity]),
                    'msv_ref_aligned': np.array([aligned_ref_mz, aligned_ref_intensity]),
                    'name': ref_row.get('name', row.get('label', 'Unknown')),
                    'adduct': ref_row.get('adduct', row.get('adduct', '')),
                    'inchi_key': inchi_key,
                    'precursor_mz': float(ref_precursor_mz) if not pd.isna(ref_precursor_mz) else np.nan,
                    'measured_precursor_mz': precursor_mz,
                    'measured_precursor_intensity': precursor_intensity
                }
                all_hits.append(hit)
    
    # Convert to DataFrame
    if not all_hits:
        return pd.DataFrame(columns=msms_hits_cols)
    
    hits_df = pd.DataFrame(all_hits)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['score', 'msms_scan', 'num_matches',
                   'precursor_mz', 'measured_precursor_mz', 'measured_precursor_intensity']
    for col in numeric_cols:
        if col in hits_df.columns:
            hits_df[col] = pd.to_numeric(hits_df[col], errors='coerce')
    
    # Filter out non-matches if requested
    if not keep_nonmatches:
        hits_df = hits_df.dropna(subset=['id'], how='all')
    
    if hits_df.empty:
        hits_df = pd.DataFrame(columns=msms_hits_cols)

    return hits_df

def prepare_feature_tools_inputs(atlas_df: pd.DataFrame, h5_files: List[str], 
                                ppm_tolerance: float = 20, extra_time: float = 0.1) -> List[Dict]:
    """
    Prepare input parameters for feature_tools.get_data() function using setup_file_slicing_parameters
    
    Args:
        atlas_df: Atlas DataFrame with required columns
        h5_files: List of H5 file paths
        ppm_tolerance: m/z tolerance in ppm (default: 20)
        extra_time: Additional time window in minutes (default: 0.1)
    
    Returns:
        List of input dictionaries for feature_tools.get_data()
    """
    # Auto-detect polarity from atlas
    polarity = 'positive'
    if 'polarity' in atlas_df.columns:
        polarity = atlas_df['polarity'].iloc[0] if not atlas_df['polarity'].empty else 'positive'

    # Use setup_file_slicing_parameters to prepare inputs
    input_data_list = ft.setup_file_slicing_parameters(
        atlas=atlas_df,
        filenames=h5_files,
        extra_time=extra_time,
        ppm_tolerance=ppm_tolerance,
        polarity=polarity,
        project_dir=False,  # Don't save intermediate files
        overwrite=True
    )
    
    return input_data_list

def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """    
    Args:
        input_data_list: List of input dictionaries from setup_file_slicing_parameters
        atlas_df: Atlas DataFrame containing compound metadata including inchi_key
    
    Returns:
        Tuple: (eics_dict, ms2_data_dict)
            - eics_dict: Dictionary mapping file paths to grouped EIC data with inchi_key
            - ms2_data_dict: Dictionary mapping file paths to MS2 summary data with inchi_key and sorted by intensity
    """
    eics = {}
    ms2_data = {}
    
    # Create a mapping from compound label to metadata
    compound_metadata = {}
    for _, row in atlas_df.iterrows():
        compound_metadata[row['label']] = {
            'inchi_key': row.get('inchi_key', ''),
            'compound_uid': row.get('compound_uid', ''),
            'adduct': row.get('adduct', ''),
            'mz': row.get('mz', 0.0),
            'rt_min': row.get('rt_min', 0.0),
            'rt_max': row.get('rt_max', 0.0),
            'rt_peak': row.get('rt_peak', 0.0)
        }
    
    print(f"Extracting enhanced EIC and MS2 data from {len(input_data_list)} files...")
    print(f"Using metadata for {len(compound_metadata)} compounds from atlas...")
    
    for i, file_input in enumerate(tqdm(input_data_list, desc="Processing enhanced data")):
        file_path = file_input['lcmsrun']
        file_name = Path(file_path).name
        
        try:
            # Extract data with ms1_feature_filter=False to get all data
            data = ft.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
            
            # Group duplicates to create adduct EICs
            if not data['ms1_data'].empty:
                adduct_eics = ft.group_duplicates(data['ms1_data'], 'label', make_string=False)
                
                # Add inchi_key column to EIC data
                if not adduct_eics.empty and 'label' in adduct_eics.columns:
                    # Map compound labels to inchi_keys
                    adduct_eics['inchi_key'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                    )
                    
                    # Add other useful metadata
                    adduct_eics['compound_uid'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                    )
                    adduct_eics['adduct'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('adduct', '')
                    )

                    # Calculate rt_peak for each row: the RT at which intensity is maximal
                    def calc_best_rt_and_i(row):
                        intensities = row.get('i', None)
                        rts = row.get('rt', None)
                        mzs = row.get('mz', None)
                        if isinstance(intensities, (np.ndarray, list)) and isinstance(rts, (np.ndarray, list)):
                            if len(intensities) > 0 and len(rts) > 0:
                                max_idx = np.argmax(intensities)
                                return pd.Series({'rt_peak': rts[max_idx], 'intensity_peak': intensities[max_idx], 'mz_peak': mzs[max_idx]})
                        return pd.Series({'rt_peak': np.nan, 'intensity_peak': np.nan, 'mz_peak': np.nan})

                    best_rt_i = adduct_eics.apply(calc_best_rt_and_i, axis=1)
                    adduct_eics['rt_peak'] = best_rt_i['rt_peak']
                    adduct_eics['intensity_peak'] = best_rt_i['intensity_peak']
                    adduct_eics['mz_peak'] = best_rt_i['mz_peak']
                    adduct_eics.sort_values(by='rt_peak', inplace=True)

                eics[file_path] = adduct_eics
                print(f"  {i+1}/{len(input_data_list)}: {file_name} -> EIC data extracted ({len(adduct_eics)} compounds)")
            else:
                print(f"  {i+1}/{len(input_data_list)}: {file_name} -> No MS1 data found")
            
            # Calculate MS2 summary
            if not data['ms2_data'].empty:
                ms2_summary = ft.calculate_ms2_summary(data['ms2_data'])
                if not ms2_summary.empty:
                    # Add inchi_key column to MS2 data
                    if 'label' in ms2_summary.columns:
                        ms2_summary['inchi_key'] = ms2_summary['label'].map(
                            lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                        )
                        
                        # Add other useful metadata
                        ms2_summary['compound_uid'] = ms2_summary['label'].map(
                            lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                        )
                        ms2_summary['adduct'] = ms2_summary['label'].map(
                            lambda x: compound_metadata.get(x, {}).get('adduct', '')
                        )
                    
                    # Ensure MS2 data has intensity column and sort by it
                    # Check if intensity column exists, if not try to find one
                    intensity_col = None
                    potential_intensity_cols = ['intensity', 'i', 'peak_height', 'precursor_intensity', 'base_peak_intensity']
                    
                    for col in potential_intensity_cols:
                        if col in ms2_summary.columns:
                            intensity_col = col
                            break
                    
                    if intensity_col is None:
                        # If no standard intensity column found, check for numeric columns that might be intensity
                        numeric_cols = ms2_summary.select_dtypes(include=[np.number]).columns
                        intensity_candidates = [col for col in numeric_cols if 'int' in col.lower() or 'height' in col.lower() or 'area' in col.lower()]
                        if intensity_candidates:
                            intensity_col = intensity_candidates[0]
                    
                    if intensity_col is not None:
                        # Sort by intensity in descending order
                        ms2_summary = ms2_summary.sort_values(by=intensity_col, ascending=False)
                        print(f"    MS2 summary: {len(ms2_summary)} spectra")
                    else:
                        # If no intensity column found, create a placeholder and add a warning
                        ms2_summary['intensity'] = 1.0  # Placeholder intensity
                        print(f"    MS2 summary: {len(ms2_summary)} spectra (no intensity column found, added placeholder)")
                    
                    ms2_data[file_path] = ms2_summary
                else:
                    print(f"    MS2 summary: No valid spectra")
            else:
                print(f"    MS2: No data found")
                
        except Exception as e:
            print(f"  {i+1}/{len(input_data_list)}: {file_name} -> ERROR: {e}")
            continue

    print(f"\nExtraction complete:")
    print(f"  EIC data: {len(eics)} files")
    print(f"  MS2 data: {len(ms2_data)} files")
    
    return eics, ms2_data

def process_ms2_hits(ms2_data: Dict, reference_df: pd.DataFrame, config: Dict) -> Tuple[Dict, pd.DataFrame]:
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
    all_hits = get_ms2_hits_from_data(
        ms2_data_dict=ms2_data,
        reference_df=reference_df,
        keep_nonmatches=True,
    )
    
    if all_hits.empty:
        print("No MS2 hits found")
        return ms2_data, pd.DataFrame()

    print(f"MS2 hits identified: {len(all_hits)}")

    # Filter high-quality matches
    min_score = config["analysis_settings"].get("ms2_min_score", 0.1)
    min_matches = config["analysis_settings"].get("ms2_min_matches", 2)
    high_quality_hits = all_hits[
        (all_hits.get('score', 0) >= min_score) & 
        (all_hits.get('num_matches', 0) >= min_matches)
    ].copy()
    
    if high_quality_hits.empty:
        print("No high-quality matches found")
        return ms2_data, all_hits

    print(f"MS2 hits with >={min_score} score and >={min_matches} matches: {len(high_quality_hits)}")

    ms2_data_and_hits = {}
    
    for file_path, ms2_entries in ms2_data.items():
        file_name = Path(file_path).name
        ms2_data_and_hits[file_path] = []

        if isinstance(ms2_entries, pd.DataFrame) and not ms2_entries.empty:
            file_hits = high_quality_hits[high_quality_hits['file_name'] == file_name]

            for _, entry in ms2_entries.iterrows():
                enhanced_entry = entry.to_dict()
                inchi_key = enhanced_entry.get('inchi_key', 'unknown')
                matching_hits = file_hits[file_hits['inchi_key'] == inchi_key]
                enhanced_entry['hits'] = matching_hits.to_dict('records') if not matching_hits.empty else []
                ms2_data_and_hits[file_path].append(enhanced_entry)
        else:
            print(f"Warning: ms2_entries for {file_path} is not a DataFrame: {type(ms2_entries)}")
            ms2_data_and_hits[file_path] = []

    return ms2_data_and_hits


def filter_ms2_hits(ms2_hits: pd.DataFrame, min_score: float, min_matches: int) -> pd.DataFrame:
    """Filter MS2 hits based on quality criteria."""
    if ms2_hits.empty:
        return pd.DataFrame()
    
    filtered = ms2_hits[
        (ms2_hits.get('score', 0) >= min_score) & 
        (ms2_hits.get('num_matches', 0) >= min_matches)
    ].copy()
    
    return filtered


def analyze_ms2_hits_results(ms2_hits: pd.DataFrame, min_score: float, min_matches: int) -> Dict:
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

def summarize_ms2_hits(ms2_data_and_hits: Dict, config: Dict):
    """
    Summarize MS2 hits analysis results using enhanced MS2 data with hits.

    Args:
        ms2_data_and_hits: Output dictionary from process_ms2_hits (enhanced MS2 data with hits)
        config: Configuration dictionary containing analysis settings

    Returns:
        None
    """
    analysis_settings = config["analysis_settings"]

    # Collect all hits from ms2_data_and_hits into a single DataFrame
    all_hits = []
    for _, file_hits in ms2_data_and_hits.items():
        for entry in file_hits:
            if 'hits' in entry and entry['hits']:
                all_hits.extend(entry['hits'])

    ms2_hits_df = pd.DataFrame(all_hits)

    if not ms2_hits_df.empty:
        print("=== MS2 Hits Analysis Results ===")

        # Analyze the results using config values
        min_score = analysis_settings["ms2_min_score"]
        min_matches = analysis_settings["ms2_min_matches"]

        analysis = analyze_ms2_hits_results(ms2_hits_df, min_score=min_score, min_matches=min_matches)

        print(f"Analysis Summary:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        # Plot score distribution
        print(f"\n=== Score Distribution ===")
        plot_ms2_score_distribution(ms2_hits_df, "MS2 Similarity Scores vs Reference Database")

        # Show high-quality matches summary
        print(f"\n=== High-Quality Matches (Score >= {min_score}, Matches >= {min_matches}) ===")
        high_quality_hits = filter_ms2_hits(ms2_hits_df, min_score=min_score, min_matches=min_matches)

        if not high_quality_hits.empty:
            print(f"Found {len(high_quality_hits)} high-quality matches")
            print(f"Unique compounds with high-quality matches: {high_quality_hits['inchi_key'].nunique()}")
        else:
            print("No high-quality matches found")
    else:
        print("No MS2 hits data available for analysis")

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