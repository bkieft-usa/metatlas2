import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
import duckdb
from tqdm.notebook import tqdm

from matchms import Spectrum
from matchms.similarity import CosineHungarian

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


def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_df: pd.DataFrame, config: Dict) -> Dict[str, Dict]:
    """
    Extract EIC and MS2 data using simplified approach - returns raw data only.
    ProjectAnalysis will handle object creation and management.
    
    Returns:
        Dict keyed by inchi_key containing simple experimental data structures
    """
    # Load reference database for MS2 matching
    msms_refs_path = Path(config["paths"]["msms_refs"])
    reference_df = None
    if msms_refs_path.exists():
        reference_df = ldt.load_msms_refs_file(msms_refs_path)
        if reference_df is not None:
            logger.info(f"Loaded {len(reference_df)} reference spectra for MS2 matching")
        else:
            logger.info("MS2 reference file found but could not be loaded")
    else:
        logger.info(f"MS2 reference file not found at {msms_refs_path}")
        logger.info("All MS2 datapoints will be preserved but without reference hits")
    
    logger.info(f"Extracting data from {len(input_data_list)} files...")
    
    # Extract experimental data using simplified approach
    experimental_data = _extract_experimental_data_simple(input_data_list, reference_df, config)
    
    # Print summary
    logger.info(f"Extraction complete:")
    logger.info(f"  Total compounds with data: {len(experimental_data)}")
    compounds_with_eic = sum(1 for data in experimental_data.values() if data.get('eic_files'))
    compounds_with_ms2 = sum(1 for data in experimental_data.values() if data.get('ms2_files'))
    logger.info(f"  Compounds with EIC data: {compounds_with_eic}")
    logger.info(f"  Compounds with MS2 data: {compounds_with_ms2}")
    
    return experimental_data

def _extract_experimental_data_simple(input_data_list: List[Dict], reference_df: Optional[pd.DataFrame], 
                                     config: Dict) -> Dict[str, Dict]:
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
        experimental_data = _extract_data_parallel_simple(input_data_list, reference_df, config, max_workers)
    else:
        logger.info("Using sequential processing...")
        experimental_data = _extract_data_sequential_simple(input_data_list, reference_df, config)
    
    return experimental_data

def _extract_data_sequential_simple(input_data_list: List[Dict], reference_df: Optional[pd.DataFrame], 
                                   config: Dict) -> Dict[str, Dict]:
    """Extract data sequentially - returns only raw data dictionaries."""
    experimental_data = {}
    
    for i, file_input in enumerate(tqdm(input_data_list, desc="Processing files")):
        file_path = file_input['lcmsrun']
        filename = Path(file_path).name
        
        try:
            # Extract raw data using feature_tools
            logger.info(f"Processing file {i+1}/{len(input_data_list)}: {filename}")
            logger.info(f"File input keys: {list(file_input.keys())}")
            logger.info(f"Atlas DF columns: {list(file_input['atlas'].columns)}")  # Changed from 'atlas_df'
            
            data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
            
            # Process EIC data - simple data only
            if not data['ms1_data'].empty:
                adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
                
                if not adduct_eics.empty and 'label' in adduct_eics.columns:
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
                ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
                
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
        
        if len(intensities) == 0 or len(rts) == 0:
            return None
        
        # Find peak values
        max_idx = np.argmax(intensities)
        rt_peak = rts[max_idx]
        intensity_peak = intensities[max_idx]
        mz_peak = mzs[max_idx] if len(mzs) > 0 else 0.0
        
        # Calculate simple errors
        atlas_rt_peak = eic_row.get('rt_peak', 0.0)
        atlas_mz = eic_row.get('mz', 0.0)
        ppm_error = abs(mz_peak - atlas_mz) / atlas_mz * 1e6 if atlas_mz > 0 else 0.0
        rt_error = rt_peak - atlas_rt_peak
        
        result = {
            "rt_vals": rts.tolist() if hasattr(rts, 'tolist') else list(rts),
            "i_vals": intensities.tolist() if hasattr(intensities, 'tolist') else list(intensities),
            "mz_vals": mzs.tolist() if hasattr(mzs, 'tolist') else list(mzs),
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
        if len(spectrum_data) != 2 or len(spectrum_data[0]) == 0:
            return None
        
        mz_values = np.array(spectrum_data[0])
        intensity_values = np.array(spectrum_data[1])
        
        # Create simple MS2 entry
        ms2_entry = {
            "inchi_key": inchi_key,
            "spectrum": [mz_values.tolist(), intensity_values.tolist()],
            "intensity_peak": float(np.max(intensity_values)) if len(intensity_values) > 0 else 0.0,
            "rt": float(ms2_row.get('rt', 0.0)),
            "precursor_mz": float(ms2_row.get('precursor_mz', 0.0)),
            "filename": filename
        }
        
        # Find hits if reference database available
        hits = []
        if reference_df is not None:
            hits = _find_simple_hits(inchi_key, mz_values, intensity_values, reference_df, config)
        
        return {
            'entry': ms2_entry,
            'hits': hits
        }
        
    except Exception as e:
        logger.error(f"Error extracting MS2 data for {filename}: {e}")
        return None

def _find_simple_hits(inchi_key: str, mz_values: np.ndarray, intensity_values: np.ndarray, 
                     reference_df: pd.DataFrame, config: Dict) -> List[Dict]:
    """Find reference hits - returns simple hit dictionaries."""
    if not inchi_key:
        return []
    
    # Find matching reference spectra
    matching_refs = reference_df[reference_df['inchi_key'] == inchi_key]
    if matching_refs.empty:
        return []
    
    hits = []
    
    for _, ref_row in matching_refs.iterrows():
        try:
            ref_spectrum_data = ref_row.get('spectrum', None)
            if ref_spectrum_data is None or len(ref_spectrum_data) != 2:
                continue
            
            # Simple spectrum comparison
            match_data = _simple_spectrum_comparison(
                mz_values, intensity_values,
                np.array(ref_spectrum_data[0]), np.array(ref_spectrum_data[1])
            )

            # Create simple hit data
            hit_data = {
                'database': str(ref_row.get('database', 'unknown')),
                'ref_id': str(ref_row.get('id', '')),
                'score': float(match_data.get('similarity_score', 0.0)),
                'num_matches': int(match_data.get('num_matched_fragments', 0)),
                'ref_name': str(ref_row.get('name', 'Unknown')),
                'ref_precursor_mz': float(ref_row.get('precursor_mz', 0.0)),
                'ref_mz_values': match_data.get('ref_mz', []),
                'ref_intensity_values': match_data.get('ref_intensity', []),
                'matched_fragments': match_data.get('matching_fragments', []),
                'fragment_colors': match_data.get('fragment_colors', [])
            }

            hits.append(hit_data)
            
        except Exception as e:
            logger.error(f"Error processing reference hit: {e}")
            continue
    
    return hits

def _simple_spectrum_comparison(query_mz: np.ndarray, query_intensity: np.ndarray,
                               ref_mz: np.ndarray, ref_intensity: np.ndarray) -> Dict:
    """Simple spectrum comparison - returns basic similarity metrics."""
    try:
        # Basic fragment matching with simple tolerance
        mz_tolerance = 0.02  # 20 ppm at m/z 1000
        
        matched_fragments = []
        fragment_colors = []
        
        for i, qmz in enumerate(query_mz):
            # Find matching reference peaks
            mz_diffs = np.abs(ref_mz - qmz)
            match_idx = np.where(mz_diffs <= mz_tolerance)[0]
            
            if len(match_idx) > 0:
                # Take closest match
                best_match = match_idx[np.argmin(mz_diffs[match_idx])]
                matched_fragments.append(float(qmz))
                fragment_colors.append('red')  # Simple coloring
        
        # Calculate simple similarity score
        num_matches = len(matched_fragments)
        total_possible = min(len(query_mz), len(ref_mz))
        similarity_score = num_matches / max(total_possible, 1) if total_possible > 0 else 0.0
        
        return {
            'similarity_score': similarity_score,
            'num_matched_fragments': num_matches,
            'matching_fragments': matched_fragments,
            'fragment_colors': fragment_colors,
            'ref_mz': ref_mz.tolist(),
            'ref_intensity': ref_intensity.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in spectrum comparison: {e}")
        return {
            'similarity_score': 0.0,
            'num_matched_fragments': 0,
            'matching_fragments': [],
            'fragment_colors': [],
            'ref_mz': [],
            'ref_intensity': []
        }

def _add_summary_statistics(compound_data: Dict):
    """Add summary statistics to compound data - simple calculations only."""
    try:
        # Add EIC summary statistics
        eic_files = compound_data.get('eic_files', {})
        if eic_files:
            # Simple file counting
            compound_data['total_files_detected'] = len(eic_files)
        
        # Add MS2 summary statistics
        ms2_files = compound_data.get('ms2_files', {})
        if ms2_files:
            # Count files with MS2 data
            files_with_data = len([f for f in ms2_files.values() if f.get('ms2_entries')])
            compound_data['ms2_files_with_data'] = files_with_data
            
            # Find best hit and best MS2 across all files
            all_hits = []
            all_entries = []
            
            for file_data in ms2_files.values():
                all_hits.extend(file_data.get('all_hits', []))
                all_entries.extend(file_data.get('ms2_entries', []))
            
            # Add file-level best hit and best MS2
            for filename, file_data in ms2_files.items():
                hits = file_data.get('all_hits', [])
                entries = file_data.get('ms2_entries', [])
                
                # Best hit by score
                if hits:
                    best_hit = max(hits, key=lambda h: h.get('score', 0.0))
                    file_data['best_hit'] = best_hit
                    file_data['num_hits'] = len(hits)
                else:
                    file_data['best_hit'] = {}
                    file_data['num_hits'] = 0
                
                # Best MS2 by intensity
                if entries:
                    best_ms2 = max(entries, key=lambda e: e.get('intensity_peak', 0.0))
                    file_data['best_ms2'] = best_ms2
                    file_data['num_ms2_entries'] = len(entries)
                else:
                    file_data['best_ms2'] = {}
                    file_data['num_ms2_entries'] = 0
        
    except Exception as e:
        logger.error(f"Error adding summary statistics: {e}")

def _extract_data_parallel_simple(input_data_list: List[Dict], reference_df: Optional[pd.DataFrame], 
                                 config: Dict, max_workers: int) -> Dict[str, Dict]:
    """Extract data in parallel - returns simple data structures only."""
    
    logger.info(f"Setting up {max_workers} workers for parallel processing...")
    
    # Process files in parallel
    experimental_data = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, file_input in enumerate(input_data_list):
            # Pass file_input directly rather than as part of a tuple
            future = executor.submit(_process_single_file_simple, i, file_input, reference_df, config)
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

def _process_single_file_simple(file_index: int, file_input: Dict, reference_df: Optional[pd.DataFrame], 
                               config: Dict) -> Tuple[str, Dict[str, Dict]]:
    """Process a single file - returns simple data structures only."""
    file_path = file_input['lcmsrun']
    filename = Path(file_path).name
    
    try:
        # Log what we received to debug
        logger.debug(f"Worker {file_index}: Processing {filename}")
        logger.debug(f"Worker {file_index}: file_input keys: {list(file_input.keys())}")
        logger.debug(f"Worker {file_index}: atlas shape: {file_input['atlas'].shape}")  # Changed from 'atlas_df'
        logger.debug(f"Worker {file_index}: atlas columns: {list(file_input['atlas'].columns)}")  # Changed from 'atlas_df'
        
        # Extract raw data
        data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
        
        file_experimental_data = {}
        
        # Process EIC data - simple extraction only
        if not data['ms1_data'].empty:
            adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
            
            if not adduct_eics.empty and 'label' in adduct_eics.columns:
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
            
            if not ms2_summary.empty:
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

def prepare_feature_tools_inputs(
    atlas_df: pd.DataFrame,
    h5_files: List[str],
    ppm_tolerance: float = 5.0,
    extra_time: float = 1.0
) -> List[Dict]:
    """
    Prepare input dictionaries for feature extraction.
    Simplified to focus on data preparation only.
    
    Args:
        atlas_df: DataFrame containing atlas compounds
        h5_files: List of HDF5 file paths
        ppm_tolerance: m/z tolerance in ppm
        extra_time: Extra RT window in minutes
    
    Returns:
        List of input dictionaries for feature_tools
    """
    logger.info(f"Preparing feature extraction inputs for {len(h5_files)} files...")
    
    # Ensure atlas_df has all required columns for feature_tools
    required_columns = ['inchi_key', 'compound_name', 'label', 'mz', 'rt_peak', 'rt_min', 'rt_max', 
                       'adduct', 'polarity', 'chromatography', 'mz_tolerance']
    
    # Check for missing columns and add defaults if needed
    atlas_df_copy = atlas_df.copy()
    
    for col in required_columns:
        if col not in atlas_df_copy.columns:
            if col == 'label':
                atlas_df_copy['label'] = atlas_df_copy.get('compound_name', 'Unknown')
            elif col == 'polarity':
                # Try to infer from existing columns or use a default
                atlas_df_copy['polarity'] = atlas_df_copy.get('polarity', 'positive')
            elif col == 'chromatography':
                atlas_df_copy['chromatography'] = atlas_df_copy.get('chromatography', 'HILIC')
            elif col == 'mz_tolerance':
                atlas_df_copy['mz_tolerance'] = atlas_df_copy.get('mz_tolerance', ppm_tolerance)
            else:
                logger.warning(f"Missing required column '{col}' in atlas DataFrame")
                atlas_df_copy[col] = ''
    
    # Validate that we have the essential data
    essential_columns = ['inchi_key', 'mz', 'rt_peak']
    missing_essential = [col for col in essential_columns if col not in atlas_df_copy.columns or atlas_df_copy[col].isna().all()]
    
    if missing_essential:
        raise ValueError(f"Atlas DataFrame is missing essential columns: {missing_essential}")
    
    # Extract atlas-level metadata for feature_tools
    atlas_polarity = atlas_df_copy['polarity'].iloc[0] if not atlas_df_copy.empty else 'positive'
    atlas_chromatography = atlas_df_copy['chromatography'].iloc[0] if not atlas_df_copy.empty else 'HILIC'
    
    logger.info(f"Atlas DataFrame prepared with {len(atlas_df_copy)} compounds and columns: {list(atlas_df_copy.columns)}")
    logger.info(f"Atlas polarity: {atlas_polarity}, chromatography: {atlas_chromatography}")
    
    # Create input list for feature_tools
    input_data_list = []
    
    for file_path in h5_files:
        try:
            # Create input dictionary with correct key names for feature_tools
            file_input = {
                'lcmsrun': file_path,
                'atlas': atlas_df_copy,
                'polarity': atlas_polarity,
                'chromatography': atlas_chromatography,
                'ppm_tolerance': ppm_tolerance,
                'extra_time': extra_time
            }
            input_data_list.append(file_input)
            
        except Exception as e:
            logger.error(f"Error preparing input for {Path(file_path).name}: {e}")
            continue
    
    logger.info(f"Prepared {len(input_data_list)} input dictionaries")
    return input_data_list