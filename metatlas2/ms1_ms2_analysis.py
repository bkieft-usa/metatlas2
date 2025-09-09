import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
import duckdb
from tqdm.notebook import tqdm

from matchms.similarity import CosineHungarian
from matchms import Spectrum
from scipy.optimize import linear_sum_assignment

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import feature_tools as ftt
import lcmsruns_tools as lrt
import database_interact as dbi
import targeted_analysis as tga
import load_tools as ldt
import data_classes as dcl
import logging_config as lcf
#import spectrum_handlers as sph

# Initialize logger properly at module level
logger = lcf.get_logger('ms1_ms2_analysis')

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
    logger.info("Loading QC files and atlas compounds from databases...")

    # Define thresholds
    database_path = config['paths']['main_database']
    metadata = config['rt_alignment']['tolerances']

    # Get QC files from project database
    qc_files_df = dbi.get_files_by_type_from_db(project_db_path, 'qc')
    
    # Load QC compounds from atlas - use main database
    qc_compounds = dbi.get_atlas_compounds_table(database_path, qc_atlas_uid)
    if qc_compounds.empty or qc_compounds is None:
        raise ValueError(f"No compounds found for QC atlas UID: {qc_atlas_uid}")
    
    logger.info(f"Found {len(qc_files_df)} QC files and {len(qc_compounds)} QC compounds")
    
    # Extract all MS1 data from all QC files
    logger.info("Extracting MS1 data from QC files...")
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
            else:
                logger.warning(f"  Warning: Unknown polarity '{polarity}' for file {Path(file_path).name}")
                continue
            
            file_has_data = False
            for ms1_key in ms1_keys_to_try:
                try:
                    ms1_data = lrt.read_hdf_file(file_path, desired_key=ms1_key)
                    
                    if ms1_data is not None and len(ms1_data) > 0:
                        ms1_data['file_path'] = file_path
                        ms1_data['filename'] = os.path.basename(file_path)
                        ms1_data['chromatography'] = chromatography
                        ms1_data['file_polarity'] = polarity  # Original file polarity
                        ms1_data['data_polarity'] = "positive" if ms1_key == "ms1_pos" else "negative"  # Actual data polarity
                        
                        # Filter by minimum intensity
                        if 'i' in ms1_data.columns:
                            initial_count = len(ms1_data)
                            ms1_data = ms1_data[ms1_data['i'] >= metadata['i']]
                        
                        # Filter by RT range
                        if 'rt' in ms1_data.columns and len(ms1_data) > 0:
                            rt_max = ms1_data['rt'].max()
                            ms1_data = ms1_data[(ms1_data['rt'] >= 0.0) & (ms1_data['rt'] <= rt_max)]
                        
                        if len(ms1_data) > 0:
                            all_ms1_data.append(ms1_data)
                            logger.info(f"  Extracted {len(ms1_data)} peaks from {os.path.basename(file_path)} ({ms1_key})")
                            file_has_data = True
                
                except Exception as e:
                    logger.warning(f"  Warning: Could not extract {ms1_key} from {Path(file_path).name}: {e}")
                    continue
            
            if not file_has_data:
                logger.warning(f"  Warning: No MS1 data extracted from {Path(file_path).name}")
                    
        except Exception as e:
            logger.warning(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue
    
    if not all_ms1_data:
        raise ValueError("No MS1 data could be extracted from any QC files")
    
    # Combine all MS1 data
    combined_ms1_data = pd.concat(all_ms1_data, ignore_index=True)
    logger.info(f"Total MS1 peaks extracted: {len(combined_ms1_data):,}")

    # Match QC compounds to extracted peaks
    logger.info("Matching QC compounds to extracted peaks...")
    compound_matches = []
    
    # Log counters
    compounds_processed = 0
    compounds_with_matches = 0
    
    for _, compound in tqdm(qc_compounds.iterrows(), total=len(qc_compounds), desc="Matching compounds"):
        compounds_processed += 1
        compound_name = compound['compound_name']
        target_mz = compound['mz']
        atlas_rt_peak = compound['rt_peak']
        atlas_rt_min = compound['rt_min'] 
        atlas_rt_max = compound['rt_max']
        compound_chromatography = compound['chromatography']
        if "HILIC" in compound_chromatography:
            compound_chromatography = "HILIC"
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
        
        # Handle polarity matching - fix the logic here
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
            compounds_with_matches += 1
            
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
    
    logger.info(f"Processed {compounds_processed} compounds, {compounds_with_matches} had matches")
    
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
        
        logger.info(f"Matching completed:")
        logger.info(f"  Compounds with matches: {matching_stats['compounds_with_matches']}/{matching_stats['total_compounds']}")
        logger.info(f"  Total compound-file matches: {matching_stats['total_matches']}")
        logger.info(f"  Mean m/z error: {qc_compound_data['mz_error_ppm'].mean():.2f} ± {qc_compound_data['mz_error_ppm'].std():.2f} ppm")
        logger.info(f"  Mean RT difference: {qc_compound_data['rt_difference'].mean():.3f} ± {qc_compound_data['rt_difference'].std():.3f} min")
        
        return qc_compound_data, matching_stats
    else:
        logger.warning("No compound matches found. Check Atlas compound definitions, m/z tolerance, RT window settings, and QC file data quality")
        logger.warning(f"Debugging information:")
        logger.warning(f"  Total compounds to match: {len(qc_compounds)}")
        logger.warning(f"  Total MS1 peaks available: {len(combined_ms1_data):,}")
        logger.warning(f"  m/z tolerance: {metadata['mz']} ppm")
        logger.warning(f"  RT tolerance: {metadata['rt']} min")
        logger.warning(f"  Intensity threshold: {metadata['i']}")
        
        # Show some sample data ranges
        if len(combined_ms1_data) > 0:
            logger.warning(f"  MS1 data m/z range: {combined_ms1_data['mz'].min():.4f} - {combined_ms1_data['mz'].max():.4f}")
            logger.warning(f"  MS1 data RT range: {combined_ms1_data['rt'].min():.2f} - {combined_ms1_data['rt'].max():.2f}")
            logger.warning(f"  MS1 data intensity range: {combined_ms1_data['i'].min():.0f} - {combined_ms1_data['i'].max():.0f}")
        
        # Show compound target ranges
        logger.warning(f"  Compound m/z range: {qc_compounds['mz'].min():.4f} - {qc_compounds['mz'].max():.4f}")
        logger.warning(f"  Compound RT range: {qc_compounds['rt_peak'].min():.2f} - {qc_compounds['rt_peak'].max():.2f}")
        
        raise ValueError("No compound matches found")

def calculate_mz_tolerance_range(mz: float, tolerance_ppm: float) -> Tuple[float, float]:
    """Calculate m/z tolerance range in Daltons."""
    tolerance_da = mz * tolerance_ppm / 1e6
    return mz - tolerance_da, mz + tolerance_da

def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_dataframe: pd.DataFrame, config: Dict) -> Dict[str, Dict]:
    """
    Extract EIC and MS2 data using simplified approach - returns raw data only.
    ProjectAnalysis will handle object creation and management.
    
    Returns:
        Dict keyed by inchi_key containing simple experimental data structures
    """
    # Load reference database for MS2 matching
    msms_refs_path = Path(config["paths"]["msms_refs"])
    reference_df = ldt.load_msms_refs_file(msms_refs_path)
    
    # Create a mapping from compound label to metadata
    compound_metadata = {}
    for _, row in atlas_dataframe.iterrows():
        compound_metadata[row['label']] = {
            'inchi_key': row.get('inchi_key', ''),
            'compound_uid': row.get('compound_uid', ''),
            'adduct': row.get('adduct', ''),
            'mz': row.get('mz', 0.0),
            'rt_min': row.get('rt_min', 0.0),
            'rt_max': row.get('rt_max', 0.0),
            'rt_peak': row.get('rt_peak', 0.0)
        }

    # Extract experimental data using simplified approach
    experimental_data = _extract_experimental_data_simple(input_data_list, compound_metadata, reference_df, config)
    
    # Print summary
    logger.info(f"Extraction complete:")
    logger.info(f"  Total compounds with data: {len(experimental_data)}")
    compounds_with_eic = sum(1 for data in experimental_data.values() if data.get('eic_files'))
    compounds_with_ms2 = sum(1 for data in experimental_data.values() if data.get('ms2_files'))
    logger.info(f"  Compounds with EIC data: {compounds_with_eic}")
    logger.info(f"  Compounds with MS2 data: {compounds_with_ms2}")
    
    return experimental_data

def _extract_experimental_data_simple(input_data_list: List[Dict], compound_metadata: Dict[str, Dict], 
                                     reference_df: Optional[pd.DataFrame], config: Dict) -> Dict[str, Dict]:
    """
    Extract experimental data in simplified format - pure data extraction only.
    
    Returns:
        Dict keyed by inchi_key containing experimental data
    """
    experimental_data = {}
    
    # Determine processing approach
    max_workers = min(mp.cpu_count(), len(input_data_list), 8)
    
    if max_workers > 1 and len(input_data_list) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        experimental_data = _extract_data_parallel_simple(input_data_list, compound_metadata, reference_df, config, max_workers)
    else:
        logger.info("Using sequential processing...")
        experimental_data = _extract_data_sequential_simple(input_data_list, compound_metadata, reference_df, config)

    return experimental_data

def _extract_data_parallel_simple(input_data_list: List[Dict], compound_metadata: Dict[str, Dict], 
                                 reference_df: Optional[pd.DataFrame], config: Dict, max_workers: int) -> Dict[str, Dict]:
    """Extract data in parallel - returns simple data structures only."""
    
    logger.info(f"Setting up {max_workers} workers for parallel processing...")
    
    # Process files in parallel
    experimental_data = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, file_input in enumerate(input_data_list):
            # Pass compound_metadata along with other parameters
            future = executor.submit(_process_single_file_simple, i, file_input, compound_metadata, reference_df, config)
            futures.append(future)
        
        # Collect results with progress bar
        for i, future in enumerate(tqdm(futures, desc="Processing files in parallel")):
            try:
                file_path, file_experimental_data = future.result()
                
                # Merge file data into experimental_data - simple merging
                for inchi_key, compound_data in file_experimental_data.items():
                    if inchi_key not in experimental_data:
                        experimental_data[inchi_key] = {
                            'eic_files': {},
                            'ms2_files': {}
                        }
                    
                    # Merge EIC files
                    if 'eic_files' in compound_data:
                        experimental_data[inchi_key]['eic_files'].update(compound_data['eic_files'])
                    
                    # Merge MS2 files
                    if 'ms2_files' in compound_data:
                        experimental_data[inchi_key]['ms2_files'].update(compound_data['ms2_files'])
                
                # Log progress
                if len(file_experimental_data) > 0:
                    logger.info(f"  Completed {i+1}/{len(input_data_list)}: {Path(file_path).name} - {len(file_experimental_data)} compounds")
                else:
                    logger.warning(f"  Completed {i+1}/{len(input_data_list)}: {Path(file_path).name} - No data extracted")
                    
            except Exception as e:
                logger.error(f"  Error in parallel processing for file {i+1}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    # Post-process to add summary statistics
    for inchi_key, compound_data in experimental_data.items():
        _add_summary_statistics(compound_data)
    
    logger.info(f"Parallel processing complete: {len(experimental_data)} compounds with data")
    return experimental_data

def _extract_data_sequential_simple(input_data_list: List[Dict], compound_metadata: Dict[str, Dict], 
                                   reference_df: Optional[pd.DataFrame], config: Dict) -> Dict[str, Dict]:
    """Extract data sequentially - returns only raw data dictionaries."""
    experimental_data = {}
    
    for i, file_input in enumerate(tqdm(input_data_list, desc="Processing files")):
        file_path = file_input['lcmsrun']
        filename = Path(file_path).name
        
        try:
            data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
            
            # Process EIC data - simple data only
            if not data['ms1_data'].empty:
                adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
                if not adduct_eics.empty and 'label' in adduct_eics.columns:
                    adduct_eics['inchi_key'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                    )
                    adduct_eics['compound_uid'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                    )
                    adduct_eics['adduct'] = adduct_eics['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('adduct', '')
                    )
                    for _, eic_row in adduct_eics.iterrows():
                        inchi_key = eic_row.get('inchi_key', '')
                        if not inchi_key:
                            continue
                        
                        # Initialize simple data structure
                        if inchi_key not in experimental_data:
                            experimental_data[inchi_key] = {
                                'eic_files': {},
                                'ms2_files': {}
                            }
                        
                        # Extract simple EIC values
                        eic_data = _extract_simple_eic_data(eic_row, filename)
                        if eic_data:
                            experimental_data[inchi_key]['eic_files'][filename] = eic_data
            
            # Process MS2 data - simple data only
            if not data['ms2_data'].empty:
                logger.info(f"Processing MS2 data for {filename} with {len(data['ms2_data'])} points...")
                ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
                if not ms2_summary.empty and 'label' in ms2_summary.columns:
                    ms2_summary['inchi_key'] = ms2_summary['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                    )
                    ms2_summary['compound_uid'] = ms2_summary['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                    )
                    ms2_summary['adduct'] = ms2_summary['label'].map(
                        lambda x: compound_metadata.get(x, {}).get('adduct', '')
                    )
                if not ms2_summary.empty:
                    for _, ms2_row in ms2_summary.iterrows():

                        inchi_key = ms2_row.get('inchi_key', '')
                        if not inchi_key:
                            continue
                        
                        # Initialize simple data structure
                        if inchi_key not in experimental_data:
                            experimental_data[inchi_key] = {
                                'eic_files': {},
                                'ms2_files': {}
                            }
                        
                        # Extract simple MS2 data
                        ms2_data = _extract_simple_ms2_data(ms2_row, filename, reference_df, config)
                        if ms2_data:
                            if filename not in experimental_data[inchi_key]['ms2_files']:
                                experimental_data[inchi_key]['ms2_files'][filename] = {
                                    "ms2_entries": [],
                                    "all_hits": []
                                }
                            
                            experimental_data[inchi_key]['ms2_files'][filename]["ms2_entries"].append(ms2_data['entry'])
                            experimental_data[inchi_key]['ms2_files'][filename]["all_hits"].extend(ms2_data['hits'])
            
            logger.info(f"  File {filename} processed successfully: {len([k for k in experimental_data.keys()])} compounds found")
            
        except Exception as e:
            logger.error(f"  Error processing {filename}: {e}")
            continue
    
    # Post-process to add summary statistics
    for inchi_key, compound_data in experimental_data.items():
        _add_summary_statistics(compound_data)
    
    return experimental_data

def _extract_simple_eic_data(eic_row: pd.Series, filename: str) -> Optional[Dict]:
    """Extract simple EIC data from a row - pure data extraction."""
    try:
        intensities = eic_row.get('i', np.array([]))
        rts = eic_row.get('rt', np.array([]))
        mzs = eic_row.get('mz', np.array([]))

        if (isinstance(intensities, np.ndarray) and intensities.size == 0) or \
           (isinstance(rts, np.ndarray) and rts.size == 0) or \
           (not isinstance(intensities, np.ndarray) and len(intensities) == 0) or \
           (not isinstance(rts, np.ndarray) and len(rts) == 0):
            return None
        
        # Find peak values
        max_idx = np.argmax(intensities)
        rt_peak = rts[max_idx]
        intensity_peak = intensities[max_idx]
        mz_peak = mzs[max_idx] if (isinstance(mzs, (np.ndarray, list)) and len(mzs) > 0) else 0.0
        
        # Calculate simple errors
        atlas_rt_peak = eic_row.get('rt_peak', 0.0)
        atlas_mz = eic_row.get('mz', 0.0)
        # Ensure atlas_mz is a scalar
        if isinstance(atlas_mz, (np.ndarray, list, pd.Series)):
            if len(atlas_mz) > 0:
                atlas_mz_val = float(atlas_mz[0])
            else:
                atlas_mz_val = 0.0
        else:
            atlas_mz_val = float(atlas_mz)
        ppm_error = abs(mz_peak - atlas_mz_val) / atlas_mz_val * 1e6 if atlas_mz_val > 0 else 0.0
        rt_error = rt_peak - atlas_rt_peak
        
        result = {
            "rt_vals": rts.tolist(),
            "i_vals": intensities.tolist(),
            "mz_vals": mzs.tolist(),
            "intensity_peak": float(intensity_peak),
            "rt_peak": float(rt_peak),
            "mz_peak": float(mz_peak),
            "ppm_diff": float(ppm_error),
            "rt_diff": float(rt_error),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting EIC data for {filename}: {e}")
        return None

def _extract_simple_ms2_data(ms2_row: pd.Series, filename: str, reference_df: Optional[pd.DataFrame], 
                            config: Dict) -> Optional[Dict]:
    """Extract simple MS2 data from a row - pure data extraction."""
    try:
        inchi_key = ms2_row.get('inchi_key', '')

        # Parse spectrum data
        spectrum_data = ms2_row.get('spectrum', [[], []])
        mz_values = np.array(spectrum_data[0]) if len(spectrum_data) == 2 else np.array([])
        intensity_values = np.array(spectrum_data[1]) if len(spectrum_data) == 2 else np.array([])

        # Check for empty arrays
        if (isinstance(mz_values, np.ndarray) and mz_values.size == 0) or \
           (isinstance(intensity_values, np.ndarray) and intensity_values.size == 0) or \
           (not isinstance(mz_values, np.ndarray) and len(mz_values) == 0) or \
           (not isinstance(intensity_values, np.ndarray) and len(intensity_values) == 0):
            return None

        # Find peak values
        max_idx = np.argmax(intensity_values)
        intensity_peak = intensity_values[max_idx]
        mz_peak = mz_values[max_idx]
        rt_peak = float(ms2_row.get('rt', 0.0))
        precursor_mz = ms2_row.get('precursor_mz', 0.0)
        
        logger.info(ms2_row.keys())
        # Extract measured values from the MS2 scan
        mz_measured = float(ms2_row.get('precursor_mz', precursor_mz))  # Use precursor_mz as measured m/z
        rt_measured = float(ms2_row.get('rt', rt_peak))  # Use rt as measured RT

        # Calculate ppm error for precursor
        ppm_error = abs(mz_peak - precursor_mz) / precursor_mz * 1e6 if precursor_mz > 0 else 0.0

        ms2_entry = {
            "inchi_key": inchi_key,
            "spectrum": [mz_values.tolist(), intensity_values.tolist()],
            "intensity_peak": float(intensity_peak),
            "rt_peak": float(rt_peak),
            "rt_measured": rt_measured,  # Add measured RT
            "mz_peak": float(mz_peak),
            "mz_measured": mz_measured,  # Add measured m/z
            "precursor_mz": float(precursor_mz),
            "ppm_diff": float(ppm_error),
        }

        # Find hits if reference database available
        hits = []
        if reference_df is not None:
            hits = _find_simple_hits(ms2_entry, reference_df, config)

        return {
            'entry': ms2_entry,
            'hits': hits
        }

    except Exception as e:
        logger.error(f"Error extracting MS2 data for {filename}: {e}")
        return None

def _find_simple_hits(ms2_entry: Dict, reference_df: pd.DataFrame, config: Dict) -> List[Dict]:
    """Find reference hits using MatchMS CosineHungarian scoring and proper alignment."""
    inchi_key = ms2_entry.get("inchi_key", "")
    mz_values = np.array(ms2_entry.get("spectrum", [[], []])[0])
    intensity_values = np.array(ms2_entry.get("spectrum", [[], []])[1])
    mz_measured = ms2_entry.get("mz_measured", 0.0)
    rt_measured = ms2_entry.get("rt_measured", 0.0)

    if not inchi_key:
        return []
    
    # Find matching reference spectra by inchi_key
    matching_refs = reference_df[reference_df['inchi_key'] == inchi_key]
    if matching_refs.empty:
        return []
    
    hits = []
    
    # Set up MatchMS scoring
    frag_mz_tolerance = config.get('ms2_matching', {}).get('mz_tolerance_da', 0.05)  # Default 0.05 Da
    cos = CosineHungarian(tolerance=frag_mz_tolerance)
    
    # Create MatchMS query spectrum
    mms_query = Spectrum(mz=mz_values, intensities=intensity_values, metadata={'precursor_mz': np.nan})
    
    for _, ref_row in matching_refs.iterrows():
        try:
            ref_spectrum = ref_row.get('spectrum', None)
            if ref_spectrum is None or len(ref_spectrum) != 2:
                continue
            
            ref_mz = np.array(ref_spectrum[0], dtype=np.float64)
            ref_intensity = np.array(ref_spectrum[1], dtype=np.float64)
            
            if len(ref_mz) == 0 or len(ref_intensity) == 0:
                continue
            
            # Filter reference spectrum to remove peaks above precursor + 2.5 Da
            precursor_mz = ref_row.get('precursor_mz', 0.0)
            if precursor_mz > 0:
                valid_peaks = ref_mz < (precursor_mz + 2.5)
                ref_mz = ref_mz[valid_peaks]
                ref_intensity = ref_intensity[valid_peaks]
            
            if len(ref_mz) == 0:
                continue
            
            # Create MatchMS reference spectrum
            mms_ref = Spectrum(mz=ref_mz, intensities=ref_intensity, metadata={'precursor_mz': np.nan})
            
            # Calculate MatchMS score
            mms_comparison = cos.pair(mms_query, mms_ref)
            score = mms_comparison['score'] if mms_comparison['score'] is not None else 0.0
            num_matches = mms_comparison['matches'] if mms_comparison['matches'] is not None else 0
            
            # Perform custom alignment for plotting
            query_spectrum_array = np.array([mz_values, intensity_values])
            ref_spectrum_array = np.array([ref_mz, ref_intensity])
            alignment_data = _align_spectra_for_plotting(query_spectrum_array, ref_spectrum_array, frag_mz_tolerance)
            
            # Create hit data with MatchMS scoring and alignment
            hit_data = {
                'database': str(ref_row.get('database', 'unknown')),
                'ref_id': str(ref_row.get('id', '')),
                'score': float(score),
                'num_matches': int(num_matches),
                'ref_name': str(ref_row.get('name', 'Unknown')),
                'mz_theoretical': float(precursor_mz),
                'mz_measured': mz_measured,  # Add measured m/z from MS2 scan
                'rt_measured': rt_measured,  # Add measured RT from MS2 scan
                'qry_intensity_peak': float(np.max(intensity_values)) if len(intensity_values) > 0 else 0.0,
                'ref_frags': len(ref_mz),
                'data_frags': len(mz_values),
                'matched_fragments': alignment_data.get('matched_fragments', []),
                'qry_frag_colors': alignment_data.get('fragment_colors', []),
                'qry_spectrum': alignment_data.get('query_aligned', []),
                'ref_spectrum': alignment_data.get('ref_aligned', []),
                'qry_spectrum_original': [mz_values.tolist(), intensity_values.tolist()],
                'ref_spectrum_original': [ref_mz.tolist(), ref_intensity.tolist()]
            }

            hits.append(hit_data)
            
        except Exception as e:
            logger.error(f"Error processing reference hit: {e}")
            continue
    
    return hits

def _align_spectra_for_plotting(query_spectrum: np.ndarray, ref_spectrum: np.ndarray, 
                               frag_mz_tolerance: float) -> Dict:
    """
    Align spectra using Hungarian assignment algorithm for proper mirror plotting.
    Based on the provided MatchMS workflow.
    """
    try:
        query_mz = query_spectrum[0]
        query_intensity = query_spectrum[1]
        ref_mz = ref_spectrum[0]
        ref_intensity = ref_spectrum[1]
        
        matched_peaks = _match_peaks(query_spectrum, ref_spectrum, frag_mz_tolerance)
        matrix_size = max(len(query_mz), len(ref_mz))
        filtered_coords = _hungarian_assignment(matched_peaks, matrix_size)
        query_aligned, ref_aligned = _link_aligned_spectra(query_spectrum, ref_spectrum, filtered_coords)
        
        # Step: Generate fragment colors and matched fragment list based on aligned spectra
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

def _add_summary_statistics(compound_data: Dict):
    """Add summary statistics to compound data - ensures consistent per-file structure."""
    try:
        # Add EIC summary statistics
        eic_files = compound_data.get('eic_files', {})
        if eic_files:
            compound_data['total_files_detected'] = len(eic_files)
        
        # Add MS2 summary statistics and ensure per-file structure
        ms2_files = compound_data.get('ms2_files', {})
        if ms2_files:
            # Count files with MS2 data
            files_with_data = len([f for f in ms2_files.values() if f.get('ms2_entries')])
            compound_data['ms2_files_with_data'] = files_with_data
            
            # Ensure each file has the complete per-file structure with best_hit and best_ms2
            for filename, file_data in ms2_files.items():
                hits = file_data.get('all_hits', [])
                entries = file_data.get('ms2_entries', [])
                
                # Best hit by score for this file
                if hits:
                    best_hit = max(hits, key=lambda h: h.get('score', 0.0))
                    file_data['best_hit'] = best_hit
                    file_data['num_hits'] = len(hits)
                else:
                    file_data['best_hit'] = {}
                    file_data['num_hits'] = 0
                
                # Best MS2 by intensity for this file
                if entries:
                    best_ms2 = max(entries, key=lambda e: e.get('intensity_peak', 0.0))
                    file_data['best_ms2'] = best_ms2
                    file_data['num_ms2_entries'] = len(entries)
                else:
                    file_data['best_ms2'] = {}
                    file_data['num_ms2_entries'] = 0
        
    except Exception as e:
        logger.error(f"Error adding summary statistics: {e}")

def _process_single_file_simple(file_index: int, file_input: Dict, compound_metadata: Dict[str, Dict], 
                               reference_df: Optional[pd.DataFrame], config: Dict) -> Tuple[str, Dict[str, Dict]]:
    """Process a single file - returns simple data structures only."""
    file_path = file_input['lcmsrun']
    filename = Path(file_path).name
    
    try:
        # Extract raw data
        data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
        
        file_experimental_data = {}
        
        # Process EIC data - simple extraction only
        if not data['ms1_data'].empty:
            adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
            
            if not adduct_eics.empty and 'label' in adduct_eics.columns:
                # Add metadata mapping like in sequential version
                adduct_eics['inchi_key'] = adduct_eics['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                )
                adduct_eics['compound_uid'] = adduct_eics['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                )
                adduct_eics['adduct'] = adduct_eics['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('adduct', '')
                )
                
                for idx, eic_row in adduct_eics.iterrows():
                    inchi_key = eic_row.get('inchi_key', '')
                    if not inchi_key:
                        continue
                    
                    # Initialize simple data structure
                    if inchi_key not in file_experimental_data:
                        file_experimental_data[inchi_key] = {
                            'eic_files': {},
                            'ms2_files': {}
                        }
                    
                    # Extract simple EIC data
                    eic_data = _extract_simple_eic_data(eic_row, filename)
                    if eic_data:
                        file_experimental_data[inchi_key]['eic_files'][filename] = eic_data
        
        # Process MS2 data - simple extraction only
        if not data['ms2_data'].empty:
            ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
            
            if not ms2_summary.empty and 'label' in ms2_summary.columns:
                logger.info(f"Colnames in MS2 summary: {ms2_summary.columns.tolist()}")
                # Add metadata mapping like in sequential version
                ms2_summary['inchi_key'] = ms2_summary['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
                )
                ms2_summary['compound_uid'] = ms2_summary['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
                )
                ms2_summary['adduct'] = ms2_summary['label'].map(
                    lambda x: compound_metadata.get(x, {}).get('adduct', '')
                )
                
                for idx, ms2_row in ms2_summary.iterrows():
                    inchi_key = ms2_row.get('inchi_key', '')
                    if not inchi_key:
                        continue
                    
                    # Initialize simple data structure
                    if inchi_key not in file_experimental_data:
                        file_experimental_data[inchi_key] = {
                            'eic_files': {},
                            'ms2_files': {}
                        }
                    
                    # Extract simple MS2 data
                    ms2_data = _extract_simple_ms2_data(ms2_row, filename, reference_df, config)
                    if ms2_data:
                        if filename not in file_experimental_data[inchi_key]['ms2_files']:
                            file_experimental_data[inchi_key]['ms2_files'][filename] = {
                                "ms2_entries": [],
                                "all_hits": []
                            }
                        
                        file_experimental_data[inchi_key]['ms2_files'][filename]["ms2_entries"].append(ms2_data['entry'])
                        file_experimental_data[inchi_key]['ms2_files'][filename]["all_hits"].extend(ms2_data['hits'])
        
        logger.debug(f"Worker {file_index}: Extracted data for {len(file_experimental_data)} compounds")
        return file_path, file_experimental_data
        
    except Exception as e:
        # Log the error but return empty data structure
        logger.error(f"Worker {file_index}: Error processing {filename}: {e}")
        import traceback
        logger.error(f"Worker {file_index}: Traceback: {traceback.format_exc()}")
        return file_path, {}

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
    input_data_list = ftt.setup_file_slicing_parameters(
        atlas=atlas_df,
        filenames=h5_files,
        extra_time=extra_time,
        ppm_tolerance=ppm_tolerance,
        polarity=polarity,
        project_dir=False,  # Don't save intermediate files
        overwrite=True
    )
    
    return input_data_list