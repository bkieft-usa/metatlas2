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


def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_df: pd.DataFrame, config: Dict) -> dcl.ProjectDataCollection:
    """
    Extract EIC and MS2 data using class-based structure.
    
    Returns:
        ProjectDataCollection containing all compound data
    """
    # Load reference database
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
    
    # Create compound metadata mapping
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
    
    # Initialize experiment data collection
    experiment_data = dcl.ProjectDataCollection()
    
    logger.info(f"Extracting data from {len(input_data_list)} files...")
    
    # Determine number of workers for parallel processing
    max_workers = min(mp.cpu_count(), len(input_data_list), 8)
    
    if max_workers > 1 and len(input_data_list) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        experiment_data = _extract_data_parallel(
            input_data_list, compound_metadata, reference_df, config, max_workers
        )
    else:
        logger.info("Using sequential processing...")
        experiment_data = _extract_data_sequential(
            input_data_list, compound_metadata, reference_df, config
        )
    
    # Print summary
    summary = experiment_data.compounds_summary
    logger.info(f"Extraction complete:")
    logger.info(f"  Total compounds: {summary['total_compounds']}")
    logger.info(f"  Compounds with EIC data: {summary['compounds_with_eic']}")
    logger.info(f"  Compounds with MS2 data: {summary['compounds_with_ms2']}")
    logger.info(f"  Compounds with MS2 hits: {summary['compounds_with_hits']}")
    logger.info(f"  Total EIC traces: {summary['total_eic_traces']}")
    logger.info(f"  Total MS2 spectra: {summary['total_ms2_spectra']}")
    
    return experiment_data

def _extract_data_sequential(input_data_list: List[Dict], compound_metadata: Dict, 
                           reference_df: Optional[pd.DataFrame], config: Dict) -> dcl.ProjectDataCollection:
    """Extract data using sequential processing."""
    experiment_data = dcl.ProjectDataCollection()
    
    for i, file_input in enumerate(tqdm(input_data_list, desc="Processing files")):
        file_path = file_input['lcmsrun']
        filename = Path(file_path).name
        
        try:
            # Extract data
            data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
            
            # Process EIC data
            if not data['ms1_data'].empty:
                adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
                
                if not adduct_eics.empty and 'label' in adduct_eics.columns:
                    for _, eic_row in adduct_eics.iterrows():
                        label = eic_row['label']
                        metadata = compound_metadata.get(label, {})
                        
                        if not metadata.get('inchi_key'):
                            continue
                        
                        # Calculate peak values
                        intensities = eic_row.get('i', np.array([]))
                        rts = eic_row.get('rt', np.array([]))
                        mzs = eic_row.get('mz', np.array([]))
                        
                        if len(intensities) > 0 and len(rts) > 0:
                            max_idx = np.argmax(intensities)
                            rt_peak = rts[max_idx]
                            intensity_peak = intensities[max_idx]
                            mz_peak = mzs[max_idx] if len(mzs) > 0 else 0.0
                        else:
                            rt_peak = intensity_peak = mz_peak = 0.0
                        
                        # Create EIC object
                        eic = dcl.EICData(
                            inchi_key=metadata['inchi_key'],
                            compound_uid=metadata['compound_uid'],
                            label=label,
                            adduct=metadata['adduct'],
                            filename=filename,
                            file_path=file_path,
                            rt_values=rts,
                            mz_values=mzs,
                            intensity_values=intensities,
                            rt_peak=rt_peak,
                            mz_peak=mz_peak,
                            intensity_peak=intensity_peak,
                            atlas_rt_min=metadata['rt_min'],
                            atlas_rt_max=metadata['rt_max'],
                            atlas_rt_peak=metadata['rt_peak'],
                            atlas_mz=metadata['mz']
                        )
                        experiment_data.add_eic_data(eic)
            
            # Process MS2 data
            if not data['ms2_data'].empty:
                ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
                
                if not ms2_summary.empty:
                    for _, ms2_row in ms2_summary.iterrows():
                        label = ms2_row.get('label', '')
                        metadata = compound_metadata.get(label, {})
                        
                        if not metadata.get('inchi_key'):
                            continue
                        
                        # Parse spectrum data
                        spectrum_data = ms2_row.get('spectrum', [[], []])
                        if len(spectrum_data) != 2 or len(spectrum_data[0]) == 0:
                            continue
                        
                        mz_values = np.array(spectrum_data[0])
                        intensity_values = np.array(spectrum_data[1])
                        
                        # Create MS2 spectrum object
                        spectrum = dcl.MS2Spectrum(
                            inchi_key=metadata['inchi_key'],
                            compound_uid=metadata['compound_uid'],
                            label=label,
                            adduct=metadata['adduct'],
                            filename=filename,
                            file_path=file_path,
                            precursor_mz=ms2_row.get('precursor_mz', 0.0),
                            precursor_intensity=ms2_row.get('precursor_intensity', 0.0),
                            rt=ms2_row.get('rt', 0.0),
                            mz_values=mz_values,
                            intensity_values=intensity_values,
                            atlas_rt_min=metadata['rt_min'],
                            atlas_rt_max=metadata['rt_max'],
                            atlas_rt_peak=metadata['rt_peak'],
                            atlas_mz=metadata['mz']
                        )
                        
                        # Find hits if reference database available
                        if reference_df is not None:
                            hits = _find_hits_for_spectrum(spectrum, reference_df, config)
                            spectrum.hits = hits
                        
                        experiment_data.add_ms2_spectrum(spectrum)
            
        except Exception as e:
            logger.error(f"  Error processing {filename}: {e}")
            continue
    
    return experiment_data

def _extract_data_parallel(input_data_list: List[Dict], compound_metadata: Dict, 
                         reference_df: Optional[pd.DataFrame], config: Dict, max_workers: int) -> dcl.ProjectDataCollection:
    """Extract data using parallel processing."""
    
    # Prepare arguments for each worker
    worker_args = []
    for i, file_input in enumerate(input_data_list):
        worker_args.append((i, file_input, compound_metadata, reference_df, config))
    
    # Process files in parallel
    experiment_data = dcl.ProjectDataCollection()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in worker_args:
            future = executor.submit(_process_single_file, *args)
            futures.append(future)
        
        # Collect results with progress bar
        for i, future in enumerate(tqdm(futures, desc="Processing files in parallel")):
            try:
                file_path, eic_objects, ms2_objects = future.result()
                
                # Add EIC objects to experiment data
                for eic in eic_objects:
                    experiment_data.add_eic_data(eic)
                
                # Add MS2 objects to experiment data
                for spectrum in ms2_objects:
                    experiment_data.add_ms2_spectrum(spectrum)
                
                if (i + 1) % 10 == 0:  # Print every 10 files
                    file_name = Path(file_path).name
                    logger.info(f"  Completed {i+1}/{len(input_data_list)}: {file_name}")
                    
            except Exception as e:
                logger.error(f"  Error in parallel processing: {e}")

    return experiment_data

def _process_single_file(file_index: int, file_input: Dict, compound_metadata: Dict, 
                       reference_df: Optional[pd.DataFrame], config: Dict) -> Tuple[str, List[dcl.EICData], List[dcl.MS2Spectrum]]:
    """Process a single file for EIC and MS2 data extraction."""
    file_path = file_input['lcmsrun']
    filename = Path(file_path).name
    
    # Extract data
    data = ftt.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
    
    eic_objects = []
    ms2_objects = []
    
    # Process EIC data
    if not data['ms1_data'].empty:
        adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
        
        if not adduct_eics.empty and 'label' in adduct_eics.columns:
            for _, eic_row in adduct_eics.iterrows():
                label = eic_row['label']
                metadata = compound_metadata.get(label, {})
                
                if not metadata.get('inchi_key'):
                    continue
                
                # Calculate peak values
                intensities = eic_row.get('i', np.array([]))
                rts = eic_row.get('rt', np.array([]))
                mzs = eic_row.get('mz', np.array([]))
                
                if len(intensities) > 0 and len(rts) > 0:
                    max_idx = np.argmax(intensities)
                    rt_peak = rts[max_idx]
                    intensity_peak = intensities[max_idx]
                    mz_peak = mzs[max_idx] if len(mzs) > 0 else 0.0
                else:
                    rt_peak = intensity_peak = mz_peak = 0.0
                
                # Create EIC object
                eic = dcl.EICData(
                    inchi_key=metadata['inchi_key'],
                    compound_uid=metadata['compound_uid'],
                    label=label,
                    adduct=metadata['adduct'],
                    filename=filename,
                    file_path=file_path,
                    rt_values=rts,
                    mz_values=mzs,
                    intensity_values=intensities,
                    rt_peak=rt_peak,
                    mz_peak=mz_peak,
                    intensity_peak=intensity_peak,
                    atlas_rt_min=metadata['rt_min'],
                    atlas_rt_max=metadata['rt_max'],
                    atlas_rt_peak=metadata['rt_peak'],
                    atlas_mz=metadata['mz']
                )
                eic_objects.append(eic)
    
    # Process MS2 data
    if not data['ms2_data'].empty:
        ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
        
        if not ms2_summary.empty:
            for _, ms2_row in ms2_summary.iterrows():
                label = ms2_row.get('label', '')
                metadata = compound_metadata.get(label, {})
                
                if not metadata.get('inchi_key'):
                    continue
                
                # Parse spectrum data
                spectrum_data = ms2_row.get('spectrum', [[], []])
                if len(spectrum_data) != 2 or len(spectrum_data[0]) == 0:
                    continue
                
                mz_values = np.array(spectrum_data[0])
                intensity_values = np.array(spectrum_data[1])
                
                # Create MS2 spectrum object
                spectrum = dcl.MS2Spectrum(
                    inchi_key=metadata['inchi_key'],
                    compound_uid=metadata['compound_uid'],
                    label=label,
                    adduct=metadata['adduct'],
                    filename=filename,
                    file_path=file_path,
                    precursor_mz=ms2_row.get('precursor_mz', 0.0),
                    precursor_intensity=ms2_row.get('precursor_intensity', 0.0),
                    rt=ms2_row.get('rt', 0.0),
                    mz_values=mz_values,
                    intensity_values=intensity_values,
                    atlas_rt_min=metadata['rt_min'],
                    atlas_rt_max=metadata['rt_max'],
                    atlas_rt_peak=metadata['rt_peak'],
                    atlas_mz=metadata['mz']
                )
                
                # Find hits if reference database available
                if reference_df is not None:
                    hits = _find_hits_for_spectrum(spectrum, reference_df, config)
                    spectrum.hits = hits
                
                ms2_objects.append(spectrum)
    
    return file_path, eic_objects, ms2_objects

def _align_spectra_for_comparison(query_mz: np.ndarray, query_intensity: np.ndarray,
                                ref_mz: np.ndarray, ref_intensity: np.ndarray,
                                mz_tolerance: float = 0.005, 
                                intensity_threshold: float = 100) -> Dict:
    """
    Align query and reference spectra for comparison and scoring.
    
    Args:
        query_mz: Query spectrum m/z values
        query_intensity: Query spectrum intensities
        ref_mz: Reference spectrum m/z values  
        ref_intensity: Reference spectrum intensities
        mz_tolerance: m/z tolerance for matching peaks
        intensity_threshold: Minimum intensity for counting matches
    
    Returns:
        Dict with all comparison results
    """
    # Convert to numpy arrays and validate
    query_mz = np.asarray(query_mz, dtype=np.float64)
    query_intensity = np.asarray(query_intensity, dtype=np.float64)
    ref_mz = np.asarray(ref_mz, dtype=np.float64)
    ref_intensity = np.asarray(ref_intensity, dtype=np.float64)
    
    if len(query_mz) != len(query_intensity):
        raise ValueError("query_mz and query_intensity must have same length")
    if len(ref_mz) != len(ref_intensity):
        raise ValueError("ref_mz and ref_intensity must have same length")
    
    # Get all unique m/z values for alignment grid
    all_mz = np.concatenate([query_mz, ref_mz])
    unique_mz = np.unique(np.round(all_mz, 10))  # Round to avoid floating point issues
    
    # Initialize aligned arrays
    aligned_query_intensity = []
    aligned_ref_intensity = []
    matched_mz_values = []
    matched_colors = []
    num_matched_fragments = 0
    
    for mz in unique_mz:
        # Find matching peaks within tolerance
        query_matches = np.where(np.abs(query_mz - mz) <= mz_tolerance)[0]
        ref_matches = np.where(np.abs(ref_mz - mz) <= mz_tolerance)[0]
        
        # Take max intensity if multiple matches within tolerance
        query_intensity_at_mz = np.max(query_intensity[query_matches]) if len(query_matches) > 0 else 0.0
        ref_intensity_at_mz = np.max(ref_intensity[ref_matches]) if len(ref_matches) > 0 else 0.0
        
        aligned_query_intensity.append(query_intensity_at_mz)
        aligned_ref_intensity.append(ref_intensity_at_mz)
        
        # Determine if this is a match and assign color
        if query_intensity_at_mz > intensity_threshold and ref_intensity_at_mz > intensity_threshold:
            matched_colors.append('green')
            matched_mz_values.append(mz)
            num_matched_fragments += 1
        else:
            matched_colors.append('red')
    
    # Calculate similarity score using cosine similarity
    similarity_score = _calculate_cosine_similarity(query_mz, query_intensity, ref_mz, ref_intensity)

    aligned_and_scored = {
        "ref_mz": ref_mz,
        "ref_intensity": ref_intensity,
        "query_mz": query_mz,
        "query_intensity": query_intensity,
        "aligned_query_mz": unique_mz,
        "aligned_query_intensity": np.array(aligned_query_intensity),
        "aligned_ref_mz": unique_mz,
        "aligned_ref_intensity": np.array(aligned_ref_intensity),
        "similarity_score": similarity_score,
        "matching_fragments": matched_mz_values,
        "num_matched_fragments": num_matched_fragments,
        "fragment_colors": matched_colors
    }

    return aligned_and_scored


def _calculate_cosine_similarity(query_mz: np.ndarray, query_intensity: np.ndarray,
                               ref_mz: np.ndarray, ref_intensity: np.ndarray,
                               tolerance: float = 0.005) -> float:
    """
    Calculate cosine similarity between two spectra using matchms.
    
    Args:
        query_mz: Query spectrum m/z values
        query_intensity: Query spectrum intensities  
        ref_mz: Reference spectrum m/z values
        ref_intensity: Reference spectrum intensities
        tolerance: m/z tolerance for matching
    
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        # Filter out zero intensities and ensure arrays are positive
        query_mask = query_intensity > 0
        ref_mask = ref_intensity > 0
        
        if not np.any(query_mask) or not np.any(ref_mask):
            return 0.0
        
        query_mz_filtered = query_mz[query_mask]
        query_intensity_filtered = query_intensity[query_mask]
        ref_mz_filtered = ref_mz[ref_mask]
        ref_intensity_filtered = ref_intensity[ref_mask]
        
        # Create matchms Spectrum objects with proper metadata
        query_spectrum = Spectrum(
            mz=query_mz_filtered, 
            intensities=query_intensity_filtered,
            metadata={'precursor_mz': float(np.median(query_mz_filtered))}
        )
        ref_spectrum = Spectrum(
            mz=ref_mz_filtered, 
            intensities=ref_intensity_filtered,
            metadata={'precursor_mz': float(np.median(ref_mz_filtered))}
        )
        
        # Calculate cosine similarity - correct calling syntax
        cosine_hungarian = CosineHungarian(tolerance=tolerance)
        score = cosine_hungarian(query_spectrum, ref_spectrum)
        logger.info(f"Cosine similarity score between query and reference spectra: {score}")

        return float(score)
    
    except Exception as e:
        # If similarity calculation fails, return 0.0
        return 0.0

def _find_hits_for_spectrum(spectrum: dcl.MS2Spectrum, reference_df: pd.DataFrame, config: Dict) -> List[dcl.MS2Hit]:
    """Find reference hits for an MS2Spectrum object using standardized spectrum handlers."""
    if not spectrum.inchi_key:
        return []
    
    # Find matching reference spectra
    matching_refs = reference_df[reference_df['inchi_key'] == spectrum.inchi_key]
    if matching_refs.empty:
        return []
    
    hits = []
    
    for _, ref_row in matching_refs.iterrows():
        ref_spectrum_data = ref_row.get('spectrum', None)
        if ref_spectrum_data is None:
            continue
        
        # Use standardized spectrum comparison for alignment and fragment matching
        match_data = _align_spectra_for_comparison(
            spectrum.mz_values, spectrum.intensity_values,
            ref_spectrum_data[0], ref_spectrum_data[1]
        )

        hit_data = {
            'database': ref_row.get('database', 'unknown'),
            'ref_id': str(ref_row.get('id', '')),
            'score': match_data.get('similarity_score', 0.0),
            'num_matches': match_data.get('num_matched_fragments', 0),
            'ref_name': ref_row.get('name', 'Unknown'),
            'ref_precursor_mz': ref_row.get('precursor_mz', 0.0),
            'ref_mz_values': match_data.get('ref_mz', []),
            'ref_intensity_values': match_data.get('ref_intensity', []),
            'query_mz_aligned': match_data.get('aligned_query_mz', []),
            'query_intensity_aligned': match_data.get('aligned_query_intensity', []),
            'ref_mz_aligned': match_data.get('aligned_ref_mz', []),
            'ref_intensity_aligned': match_data.get('aligned_ref_intensity', []),
            'matched_fragments': match_data.get('matching_fragments', []),
            'fragment_colors': match_data.get('fragment_colors', [])
        }

        # Create hit object
        hit = dcl.MS2Hit(**hit_data)
        hits.append(hit)
    
    return hits

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