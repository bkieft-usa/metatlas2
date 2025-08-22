import sys
import pandas as pd
import os
sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.lcmsruns_tools as lrt
from typing import Dict, List, Optional, Any, Tuple

def extract_ms1_data_from_file(file_path: str, polarity: str = 'positive') -> pd.DataFrame:
    """
    Extract MS1 data from a single QC file.
    
    Args:
        file_path: Path to the h5 file
        polarity: MS polarity ('positive' or 'negative')
    
    Returns:
        DataFrame with columns: mz, rt, intensity, file_path
    """
    try:
        # Determine the correct key based on polarity
        ms1_key = f"ms1_{'pos' if polarity == 'positive' else 'neg'}"
        
        # Extract MS1 data using feature_tools
        ms1_data = lrt.read_hdf_file(file_path, desired_key=ms1_key)

        if ms1_data is None or len(ms1_data) == 0:
            print(f"   No MS1 data found in {os.path.basename(file_path)}")
            return pd.DataFrame()
        
        # Add file information
        ms1_data['file_path'] = file_path
        ms1_data['filename'] = os.path.basename(file_path)
        ms1_data['polarity'] = polarity
        
        # Filter by minimum intensity
        if 'i' in ms1_data.columns:
            ms1_data = ms1_data[ms1_data['i'] >= MIN_PEAK_INTENSITY]

        rt_range_min, rt_range_max = 0.0, round(ms1_data.rt.max(), 2)

        if 'rt' in ms1_data.columns:
            ms1_data = ms1_data[ms1_data['rt'] >= rt_range_min]
            ms1_data = ms1_data[ms1_data['rt'] <= rt_range_max]

        print(f"Extracted {len(ms1_data)} MS1 peaks from {os.path.basename(file_path)}")
        return ms1_data
        
    except Exception as e:
        print(f"    Error extracting from {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

def calculate_mz_tolerance_range(mz: float, tolerance_ppm: float) -> Tuple[float, float]:
    """Calculate m/z tolerance range in Daltons."""
    tolerance_da = mz * tolerance_ppm / 1e6
    return mz - tolerance_da, mz + tolerance_da

def find_peaks_in_rt_window(ms1_data: pd.DataFrame, target_mz: float, 
                           mz_tolerance_ppm: float, rt_center: float, 
                           rt_window: float = 0.5) -> pd.DataFrame:
    """
    Find peaks within m/z tolerance and RT window.
    
    Args:
        ms1_data: MS1 data DataFrame
        target_mz: Target m/z value
        mz_tolerance_ppm: m/z tolerance in ppm
        rt_center: Center RT value (minutes)
        rt_window: RT window around center (minutes)
    
    Returns:
        DataFrame of matching peaks
    """
    # Calculate m/z range
    mz_min, mz_max = calculate_mz_tolerance_range(target_mz, mz_tolerance_ppm)
    
    # Calculate RT range
    rt_min = rt_center - rt_window
    rt_max = rt_center + rt_window
    
    # Filter peaks
    matching_peaks = ms1_data[
        (ms1_data['mz'] >= mz_min) & 
        (ms1_data['mz'] <= mz_max) & 
        (ms1_data['rt'] >= rt_min) & 
        (ms1_data['rt'] <= rt_max)
    ].copy()
    
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