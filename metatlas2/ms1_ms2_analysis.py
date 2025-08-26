import sys
import pandas as pd
import os
from pathlib import Path
import duckdb
from tqdm.notebook import tqdm
sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.lcmsruns_tools as lrt
import metatlas2.database_interact as dbi
from typing import Dict, List, Optional, Any, Tuple

def extract_and_match_qc_compounds(project_db_path: str, database_path: str, qc_atlas_uid: str, metadata: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract MS1 data from QC files and match with QC Atlas compounds.
    Simplified approach that processes all files and matches compounds in a straightforward manner.
    
    Args:
        project_db_path: Path to project database containing lcmsruns
        database_path: Path to main compounds database
        qc_atlas_uid: UID of QC atlas to use for matching
        metadata: Dictionary containing extraction parameters
    
    Returns:
        Tuple of (matches_df, matching_stats)
    """
    print("Loading QC files and atlas compounds from databases...")
    
    # Get QC files from project database
    qc_files_df = lrt.get_files_by_type_from_db(project_db_path, 'qc')
    
    # Load QC compounds from atlas
    qc_compounds = dbi.get_atlas_compounds_from_db(database_path, qc_atlas_uid)
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
                            ms1_data = ms1_data[ms1_data['i'] >= metadata['min_peak_intensity']]
                        
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
        mz_tolerance = compound['mz_tolerance'] if pd.notna(compound['mz_tolerance']) else metadata['default_ppm_tolerance']
        mz_tolerance_da = target_mz * mz_tolerance / 1e6
        
        # RT window with expansion
        rt_min_search = atlas_rt_min - metadata['window_expansion']
        rt_max_search = atlas_rt_max + metadata['window_expansion']
        
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
        matches_df = pd.DataFrame(compound_matches)
        
        # Calculate statistics
        matching_stats = {
            'total_compounds': len(qc_compounds),
            'compounds_with_matches': matches_df['compound_uid'].nunique(),
            'compounds_without_matches': len(qc_compounds) - matches_df['compound_uid'].nunique(),
            'total_matches': len(matches_df),
            'total_files': len(qc_files_df),
            'total_peaks_extracted': len(combined_ms1_data)
        }
        
        print(f"\nMatching completed:")
        print(f"  Compounds with matches: {matching_stats['compounds_with_matches']}/{matching_stats['total_compounds']}")
        print(f"  Total compound-file matches: {matching_stats['total_matches']}")
        print(f"  Mean m/z error: {matches_df['mz_error_ppm'].mean():.2f} ± {matches_df['mz_error_ppm'].std():.2f} ppm")
        print(f"  Mean RT difference: {matches_df['rt_difference'].mean():.3f} ± {matches_df['rt_difference'].std():.3f} min")
        
        return matches_df, matching_stats
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