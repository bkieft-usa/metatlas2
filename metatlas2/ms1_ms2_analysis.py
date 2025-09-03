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

sys.path.append('/Users/BKieft/Metabolomics/metatlas')
sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
from metatlas.io import feature_tools as ft
import metatlas2.lcmsruns_tools as lrt
import metatlas2.database_interact as dbi
import metatlas2.targeted_analysis as tga
import metatlas2.load_tools as ldt
import metatlas2.data_classes as dc

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
    
    # Load QC compounds from atlas - use main database
    qc_compounds = dbi.get_atlas_compounds_table(database_path, qc_atlas_uid)
    if qc_compounds.empty or qc_compounds is None:
        raise ValueError(f"No compounds found for QC atlas UID: {qc_atlas_uid}")
    
    print(f"Found {len(qc_files_df)} QC files and {len(qc_compounds)} QC compounds")
    
    # Debug: Print some compound information
    print(f"Sample compounds from atlas:")
    for i, (_, row) in enumerate(qc_compounds.head(3).iterrows()):
        print(f"  {i+1}: {row.get('compound_name', 'No name')} - {row.get('chromatography', 'No chrom')}/{row.get('polarity', 'No pol')} - RT: {row.get('rt_peak', 'No RT')} - m/z: {row.get('mz', 'No mz')}")
    
    # Debug: Print some file information
    print(f"Sample QC files:")
    for i, (_, row) in enumerate(qc_files_df.head(3).iterrows()):
        print(f"  {i+1}: {Path(row['file_path']).name} - {row.get('chromatography', 'No chrom')}/{row.get('polarity', 'No pol')}")
    
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
            else:
                print(f"  Warning: Unknown polarity '{polarity}' for file {Path(file_path).name}")
                continue
            
            file_has_data = False
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
                            initial_count = len(ms1_data)
                            ms1_data = ms1_data[ms1_data['i'] >= metadata['i']]
                            print(f"  Intensity filter: {initial_count} -> {len(ms1_data)} peaks (threshold: {metadata['i']})")
                        
                        # Filter by RT range
                        if 'rt' in ms1_data.columns and len(ms1_data) > 0:
                            rt_max = ms1_data['rt'].max()
                            ms1_data = ms1_data[(ms1_data['rt'] >= 0.0) & (ms1_data['rt'] <= rt_max)]
                        
                        if len(ms1_data) > 0:
                            all_ms1_data.append(ms1_data)
                            print(f"  Extracted {len(ms1_data)} peaks from {os.path.basename(file_path)} ({ms1_key})")
                            file_has_data = True
                
                except Exception as e:
                    print(f"  Warning: Could not extract {ms1_key} from {Path(file_path).name}: {e}")
                    continue
            
            if not file_has_data:
                print(f"  Warning: No MS1 data extracted from {Path(file_path).name}")
                    
        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue
    
    if not all_ms1_data:
        raise ValueError("No MS1 data could be extracted from any QC files")
    
    # Combine all MS1 data
    combined_ms1_data = pd.concat(all_ms1_data, ignore_index=True)
    print(f"Total MS1 peaks extracted: {len(combined_ms1_data):,}")
    
    # Debug: Show data polarity distribution
    if 'data_polarity' in combined_ms1_data.columns:
        polarity_counts = combined_ms1_data['data_polarity'].value_counts()
        print(f"MS1 data polarity distribution: {dict(polarity_counts)}")

    if 'chromatography' in combined_ms1_data.columns:
        chromatography_counts = combined_ms1_data['chromatography'].value_counts()
        print(f"MS1 data chromatography distribution: {dict(chromatography_counts)}")

    # Match QC compounds to extracted peaks
    print("\nMatching QC compounds to extracted peaks...")
    compound_matches = []
    
    # Debug counters
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
        
        # Debug: Print compound being processed
        if compounds_processed <= 3:
            print(f"  Processing compound {compounds_processed}: {compound_name}")
            print(f"    Target: {compound_chromatography}/{compound_polarity}, m/z={target_mz:.4f}, RT={atlas_rt_peak:.2f}")
        
        # Calculate tolerances
        mz_tolerance = metadata['mz']
        mz_tolerance_da = target_mz * mz_tolerance / 1e6
        
        # RT window with expansion
        rt_min_search = atlas_rt_min - metadata['rt']
        rt_max_search = atlas_rt_max + metadata['rt']
        
        # Filter MS1 data for this compound
        # Match chromatography exactly
        ms1_subset = combined_ms1_data[combined_ms1_data['chromatography'] == compound_chromatography]
        print(f"    After chromatography filter ({compound_chromatography}): {len(ms1_subset)} peaks")
        
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
        print(f"    After polarity filter ({compound_polarity}): {len(ms1_subset)} peaks")
        
        # Apply m/z and RT filters
        matching_peaks = ms1_subset[
            (ms1_subset['mz'] >= target_mz - mz_tolerance_da) &
            (ms1_subset['mz'] <= target_mz + mz_tolerance_da) &
            (ms1_subset['rt'] >= rt_min_search) &
            (ms1_subset['rt'] <= rt_max_search)
        ].copy()
        
        print(f"    After m/z and RT filters: {len(matching_peaks)} peaks")
        
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
    
    print(f"\nProcessed {compounds_processed} compounds, {compounds_with_matches} had matches")
    
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
        print(f"\nDebugging information:")
        print(f"  Total compounds to match: {len(qc_compounds)}")
        print(f"  Total MS1 peaks available: {len(combined_ms1_data):,}")
        print(f"  m/z tolerance: {metadata['mz']} ppm")
        print(f"  RT tolerance: {metadata['rt']} min")
        print(f"  Intensity threshold: {metadata['i']}")
        
        # Show some sample data ranges
        if len(combined_ms1_data) > 0:
            print(f"  MS1 data m/z range: {combined_ms1_data['mz'].min():.4f} - {combined_ms1_data['mz'].max():.4f}")
            print(f"  MS1 data RT range: {combined_ms1_data['rt'].min():.2f} - {combined_ms1_data['rt'].max():.2f}")
            print(f"  MS1 data intensity range: {combined_ms1_data['i'].min():.0f} - {combined_ms1_data['i'].max():.0f}")
        
        # Show compound target ranges
        print(f"  Compound m/z range: {qc_compounds['mz'].min():.4f} - {qc_compounds['mz'].max():.4f}")
        print(f"  Compound RT range: {qc_compounds['rt_peak'].min():.2f} - {qc_compounds['rt_peak'].max():.2f}")
        
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
        rt_data: RT data dictionary with 'center', 'min' and 'max' keys
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

def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_df: pd.DataFrame, config: Dict) -> dc.ExperimentDataCollection:
    """
    Extract EIC and MS2 data using class-based structure.
    
    Returns:
        ExperimentDataCollection containing all compound data
    """
    # Load reference database
    msms_refs_path = Path(config["paths"]["msms_refs"])
    reference_df = None
    if msms_refs_path.exists():
        reference_df = ldt.load_msms_refs_file(msms_refs_path)
        if reference_df is not None:
            print(f"Loaded {len(reference_df)} reference spectra for MS2 matching")
        else:
            print("MS2 reference file found but could not be loaded")
    else:
        print(f"MS2 reference file not found at {msms_refs_path}")
        print("All MS2 datapoints will be preserved but without reference hits")
    
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
    experiment_data = dc.ExperimentDataCollection()
    
    print(f"Extracting data from {len(input_data_list)} files...")
    
    # Determine number of workers for parallel processing
    max_workers = min(mp.cpu_count(), len(input_data_list), 8)
    
    if max_workers > 1 and len(input_data_list) > 1:
        print(f"Using parallel processing with {max_workers} workers...")
        experiment_data = _extract_data_parallel(
            input_data_list, compound_metadata, reference_df, config, max_workers
        )
    else:
        print("Using sequential processing...")
        experiment_data = _extract_data_sequential(
            input_data_list, compound_metadata, reference_df, config
        )
    
    # Print summary
    summary = experiment_data.compounds_summary
    print(f"\nExtraction complete:")
    print(f"  Total compounds: {summary['total_compounds']}")
    print(f"  Compounds with EIC data: {summary['compounds_with_eic']}")
    print(f"  Compounds with MS2 data: {summary['compounds_with_ms2']}")
    print(f"  Compounds with MS2 hits: {summary['compounds_with_hits']}")
    print(f"  Total EIC traces: {summary['total_eic_traces']}")
    print(f"  Total MS2 spectra: {summary['total_ms2_spectra']}")
    
    return experiment_data

def _extract_data_sequential(input_data_list: List[Dict], compound_metadata: Dict, 
                           reference_df: Optional[pd.DataFrame], config: Dict) -> dc.ExperimentDataCollection:
    """Extract data using sequential processing."""
    experiment_data = dc.ExperimentDataCollection()
    
    for i, file_input in enumerate(tqdm(input_data_list, desc="Processing files")):
        file_path = file_input['lcmsrun']
        filename = Path(file_path).name
        
        try:
            # Extract data
            data = ft.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
            
            # Process EIC data
            if not data['ms1_data'].empty:
                adduct_eics = ft.group_duplicates(data['ms1_data'], 'label', make_string=False)
                
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
                        eic = dc.EICData(
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
                ms2_summary = ft.calculate_ms2_summary(data['ms2_data'])
                
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
                        spectrum = dc.MS2Spectrum(
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
            
            # Progress info
            if (i + 1) % 10 == 0:  # Print every 10 files
                eic_count = sum(len(c.eic_data) for c in experiment_data.compounds.values())
                ms2_count = sum(len(c.ms2_spectra) for c in experiment_data.compounds.values())
                print(f"  Processed {i+1}/{len(input_data_list)} files - EIC: {eic_count}, MS2: {ms2_count}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    return experiment_data

def _extract_data_parallel(input_data_list: List[Dict], compound_metadata: Dict, 
                         reference_df: Optional[pd.DataFrame], config: Dict, max_workers: int) -> dc.ExperimentDataCollection:
    """Extract data using parallel processing."""
    
    # Prepare arguments for each worker
    worker_args = []
    for i, file_input in enumerate(input_data_list):
        worker_args.append((i, file_input, compound_metadata, reference_df, config))
    
    # Process files in parallel
    experiment_data = dc.ExperimentDataCollection()
    
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
                    print(f"  Completed {i+1}/{len(input_data_list)}: {file_name}")
                    
            except Exception as e:
                print(f"  Error in parallel processing: {e}")
    
    return experiment_data

def _process_single_file(file_index: int, file_input: Dict, compound_metadata: Dict, 
                       reference_df: Optional[pd.DataFrame], config: Dict) -> Tuple[str, List[dc.EICData], List[dc.MS2Spectrum]]:
    """Process a single file for EIC and MS2 data extraction."""
    file_path = file_input['lcmsrun']
    filename = Path(file_path).name
    
    # Extract data
    data = ft.get_data(file_input, save_file=False, return_data=True, ms1_feature_filter=False)
    
    eic_objects = []
    ms2_objects = []
    
    # Process EIC data
    if not data['ms1_data'].empty:
        adduct_eics = ft.group_duplicates(data['ms1_data'], 'label', make_string=False)
        
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
                eic = dc.EICData(
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
        ms2_summary = ft.calculate_ms2_summary(data['ms2_data'])
        
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
                spectrum = dc.MS2Spectrum(
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

def _find_hits_for_spectrum(spectrum: dc.MS2Spectrum, reference_df: pd.DataFrame, config: Dict) -> List[dc.MS2Hit]:
    """Find reference hits for an MS2Spectrum object."""
    if not spectrum.inchi_key:
        return []
    
    # Find matching reference spectra
    matching_refs = reference_df[reference_df['inchi_key'] == spectrum.inchi_key]
    if matching_refs.empty:
        return []
    
    cos = CosineHungarian(tolerance=0.005)
    query_spectrum = Spectrum(mz=spectrum.mz_values, 
                            intensities=spectrum.intensity_values,
                            metadata={'precursor_mz': spectrum.precursor_mz})
    
    hits = []
    
    for _, ref_row in matching_refs.iterrows():
        ref_spectrum_data = ref_row.get('spectrum', None)
        if ref_spectrum_data is None:
            continue
        
        # Parse reference spectrum
        try:
            if isinstance(ref_spectrum_data, np.ndarray) and ref_spectrum_data.shape[0] == 2:
                ref_mz = np.array(ref_spectrum_data[0])
                ref_intensity = np.array(ref_spectrum_data[1])
            elif isinstance(ref_spectrum_data, (list, tuple)) and len(ref_spectrum_data) == 2:
                ref_mz = np.array(ref_spectrum_data[0])
                ref_intensity = np.array(ref_spectrum_data[1])
            else:
                continue
        except:
            continue
        
        if len(ref_mz) == 0:
            continue
        
        # Calculate similarity
        ref_spectrum = Spectrum(mz=ref_mz, 
                              intensities=ref_intensity,
                              metadata={'precursor_mz': ref_row.get('precursor_mz', 0.0)})
        
        try:
            similarity_result = cos.pair(query_spectrum, ref_spectrum)
            score = float(similarity_result['score'])
            num_matches = int(similarity_result['matches'])
        except:
            score = 0.0
            num_matches = 0
        
        # Align spectra
        try:
            (aligned_query_mz, aligned_query_intensity,
             aligned_ref_mz, aligned_ref_intensity, _) = align_ms_arrays(
                spectrum.mz_values, spectrum.intensity_values, ref_mz, ref_intensity, 0.005
            )
            
            # Calculate fragment colors for visualization
            fragment_colors = []
            matched_fragments = []
            for i in range(len(aligned_query_mz)):
                if aligned_query_intensity[i] > 0 and aligned_ref_intensity[i] > 0:
                    fragment_colors.append('green')
                    matched_fragments.append(aligned_query_mz[i])
                else:
                    fragment_colors.append('red')
                    
        except:
            aligned_query_mz = spectrum.mz_values
            aligned_query_intensity = spectrum.intensity_values
            aligned_ref_mz = ref_mz
            aligned_ref_intensity = ref_intensity
            fragment_colors = ['red'] * len(spectrum.mz_values)
            matched_fragments = []
        
        # Create hit object
        hit = dc.MS2Hit(
            database=ref_row.get('database', 'unknown'),
            ref_id=str(ref_row.get('id', '')),
            score=score,
            num_matches=num_matches,
            ref_name=ref_row.get('name', 'Unknown'),
            ref_precursor_mz=ref_row.get('precursor_mz', 0.0),
            ref_mz_values=ref_mz,
            ref_intensity_values=ref_intensity,
            query_mz_aligned=aligned_query_mz,
            query_intensity_aligned=aligned_query_intensity,
            ref_mz_aligned=aligned_ref_mz,
            ref_intensity_aligned=aligned_ref_intensity,
            matched_fragments=matched_fragments,
            fragment_colors=fragment_colors
        )
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