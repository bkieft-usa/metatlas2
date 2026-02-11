"""
Efficient feature extraction from parquet files using sorted m/z indices.
Direct extraction without intermediate preparation steps.
"""
import pandas as pd
import sys
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm.notebook import tqdm

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('extract_data_from_parquet')

def view_parquet_file_contents(parquet_file: str, num_rows: int = 5, rt_slice: float = None) -> pd.DataFrame:
    """Utility function to view contents of a parquet file."""
    try:
        df = pq.read_table(parquet_file).to_pandas()
        logger.info(f"Successfully read {parquet_file} with {len(df)} rows.")
        if rt_slice:
            df = df[df['rt'] == rt_slice]
            logger.info(f"Filtered to {len(df)} rows with rt == {rt_slice}.")
        return df.head(num_rows)
    except Exception as e:
        logger.error(f"Error reading {parquet_file}: {e}")
        return pd.DataFrame()

def extract_eic_and_ms2_from_parquet(
    atlas_df: pd.DataFrame,
    parquet_files: List[str],
    ppm_tolerance: float = 20.0,
    extra_time: float = 0.1,
    use_parallel: bool = True,
    max_workers: int = None
) -> Dict[str, Dict]:
    """
    Orchestator function that calls helper functions to extract EIC and MS2 data directly from parquet files.
    Extract EIC and MS2 data directly from parquet files with optional parallel processing.
    
    Args:
        atlas_df: Atlas DataFrame with columns [label, inchi_key, mz, rt_min, rt_max, rt_peak]
        parquet_files: List of parquet file paths (e.g., *_ms1_pos.parquet, *_ms2_neg.parquet)
        ppm_tolerance: m/z tolerance in ppm
        extra_time: Extra RT time to extract beyond feature bounds
        use_parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of parallel workers (default: min(cpu_count, len(files), 8))
    
    Returns:
        Dict mapping inchi_key to dict of file data
        Dict for each file is in the format:
        {
            'ms1_data': DataFrame with columns [label, rt, mz, i, in_feature],
            'ms1_summary': DataFrame with summary stats for MS1 data,
            'ms2_data': DataFrame with columns [label, rt, mz, i, precursor_MZ, precursor_intensity, collision_energy, in_feature]
        }
    """

    logger.info(f"Starting data extraction for {len(atlas_df)} compounds from {len(parquet_files)} parquet files...")

    # Initialize results structure
    results = {}
    for _, row in atlas_df.iterrows():
        inchi_key = row.get('inchi_key', '')
        if not inchi_key:
            logger.warning(f"Missing inchi_key for row with label {row.get('label', 'unknown')}")
            continue
        results[inchi_key] = {}
    
    # Determine parallelization strategy
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(parquet_files), 8)
    use_parallel = use_parallel and max_workers > 1 and len(parquet_files) > 1
    
    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for parquet_file in parquet_files:
                future = executor.submit(_process_single_parquet_file, 
                                        parquet_file, atlas_df, ppm_tolerance, extra_time)
                futures.append((future, parquet_file))
            
            # Collect results
            for future, parquet_file in tqdm(futures, desc="Extracting data from parquet files"):
                try:
                    file_results = future.result()
                    
                    # Merge file results into main results
                    for label, file_data in file_results.items():
                        results[label].update(file_data)
                                        
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
    else:
        logger.info("Using sequential processing...")
        
        for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
            try:
                file_results = _process_single_parquet_file(
                    parquet_file, atlas_df, ppm_tolerance, extra_time
                )
                
                # Merge file results into main results
                for label, file_data in file_results.items():
                    results[label].update(file_data)
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue
    
    logger.debug("Data extraction results:")
    for label, file_data in results.items():
        logger.debug(f"  {label}: {len(file_data)}/{len(parquet_files)} files with data")
    return results


def _process_single_parquet_file(parquet_file: str, atlas_df: pd.DataFrame, 
                                  ppm_tolerance: float, extra_time: float) -> Dict[str, Dict]:
    """Process a single parquet file - worker function for parallel processing."""
    results = {}
    
    # Determine if this is MS1 or MS2 from filename
    filename = Path(parquet_file).name
    is_ms1 = filename.endswith('_ms1_pos.parquet') or filename.endswith('_ms1_neg.parquet')
    is_ms2 = filename.endswith('_ms2_pos.parquet') or filename.endswith('_ms2_neg.parquet')
    
    if not (is_ms1 or is_ms2):
        raise ValueError(f"Cannot determine MS level from filename: {filename}")
    if (is_ms1 and is_ms2) or (not is_ms1 and not is_ms2):
        raise ValueError(f"Filename does not clearly indicate MS level: {filename}")
    if not Path(parquet_file).exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    # Extract features for each compound
    for _, row in atlas_df.iterrows():
        label = row.get('label', '')
        inchi_key = row.get('inchi_key', '')
        
        if not inchi_key:
            logger.warning(f"Missing inchi_key for label {label}")
            continue
        
        compound_data = {
            'ms1_data': pd.DataFrame(),
            'ms1_summary': pd.DataFrame(),
            'ms2_data': pd.DataFrame()
        }
        
        # Extract MS1 data
        if is_ms1:
            ms1_data = extract_ms1_from_parquet(
                parquet_file,
                label=label,
                mz=row['mz'],
                rt_min=row['rt_min'],
                rt_max=row['rt_max'],
                ppm_tolerance=ppm_tolerance,
                extra_time=extra_time
            )
            ms1_data = ms1_data.sort_values(by=['rt', 'i'], ascending=[True, False]).reset_index(drop=True)
            compound_data['ms1_data'] = ms1_data
            
            # Calculate summary
            if not ms1_data.empty:
                ms1_summary = calculate_ms1_summary(ms1_data, feature_filter=True).reset_index(drop=True)
                if ms1_summary.shape[0] == 0:
                    for c in ['num_datapoints', 'peak_area', 'peak_height', 'mz_centroid', 'rt_peak']:
                        ms1_summary[c] = 0
                compound_data['ms1_summary'] = ms1_summary
        
        # Extract MS2 data
        elif is_ms2:
            ms2_data = extract_ms2_from_parquet(
                parquet_file,
                label=label,
                mz=row['mz'],
                rt_min=row['rt_min'],
                rt_max=row['rt_max'],
                ppm_tolerance=ppm_tolerance,
                extra_time=extra_time
            )
            # sort by rt, then mz within rt
            ms2_data = ms2_data.sort_values(by=['rt', 'mz'], ascending=[True, True]).reset_index(drop=True)
            compound_data['ms2_data'] = ms2_data
            
        
        # Store results for this inchi_key
        if inchi_key not in results:
            results[inchi_key] = {}
        results[inchi_key][parquet_file] = compound_data
    
    return results


def extract_ms1_from_parquet(
    parquet_file: str,
    label: str,
    mz: float,
    rt_min: float,
    rt_max: float,
    ppm_tolerance: float,
    extra_time: float = 0.1
) -> pd.DataFrame:
    """
    Extract a single feature from a parquet file.
    
    Args:
        parquet_file: Path to parquet file (e.g., *_ms1_pos.parquet)
        label: Feature label from atlas
        mz: Target m/z value
        rt_min: Minimum retention time
        rt_max: Maximum retention time
        ppm_tolerance: m/z tolerance in ppm
        extra_time: Extra time to extract beyond rt_min/rt_max
    
    Returns:
        DataFrame with columns: [label, rt, mz, i, in_feature]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, ppm_tolerance)
    
    # Read parquet with m/z filter (uses sorted index efficiently)
    try:
        df = pq.read_table(
            parquet_file,
            filters=[
                ('mz', '>=', mz_min),
                ('mz', '<=', mz_max)
            ]
        ).to_pandas()
    except Exception as e:
        logger.warning(f"Error reading {parquet_file}: {e}")
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'in_feature'])
    
    if df.empty:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'in_feature'])
    
    # Apply RT filter (with extra time)
    rt_filter = (df['rt'] >= rt_min - extra_time) & (df['rt'] <= rt_max + extra_time)
    df = df[rt_filter].copy()
    
    if df.empty:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'in_feature'])
    
    # Mark points within vs outside feature bounds
    df['in_feature'] = (df['rt'] >= rt_min) & (df['rt'] <= rt_max)
    df['label'] = label
    
    return df[['label', 'rt', 'mz', 'i', 'in_feature']]

def calculate_mz_bounds(mz: float, ppm_tolerance: float) -> tuple:
    """Calculate m/z bounds given ppm tolerance."""
    delta = mz * ppm_tolerance / 1e6
    return (mz - delta, mz + delta)

def extract_ms2_from_parquet(
    parquet_file: str,
    label: str,
    mz: float,
    rt_min: float,
    rt_max: float,
    ppm_tolerance: float,
    extra_time: float = 0.1
) -> pd.DataFrame:
    """
    Extract MS2 feature from parquet file.
    
    Returns:
        DataFrame with columns: [label, rt, mz, i, precursor_MZ, 
                                 precursor_intensity, collision_energy, in_feature]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, ppm_tolerance)
    
    # For MS2, filter by precursor m/z
    try:
        df = pq.read_table(
            parquet_file,
            filters=[
                ('precursor_MZ', '>=', mz_min),
                ('precursor_MZ', '<=', mz_max)
            ]
        ).to_pandas()
    except Exception as e:
        logger.warning(f"Error reading {parquet_file}: {e}")
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'precursor_MZ', 
                                     'precursor_intensity', 'collision_energy', 'in_feature'])
    
    if df.empty:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'precursor_MZ', 
                                     'precursor_intensity', 'collision_energy', 'in_feature'])
    
    # Apply RT filter
    rt_filter = (df['rt'] >= rt_min - extra_time) & (df['rt'] <= rt_max + extra_time)
    df = df[rt_filter].copy()
    
    if df.empty:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'precursor_MZ', 
                                     'precursor_intensity', 'collision_energy', 'in_feature'])
    
    df['in_feature'] = (df['rt'] >= rt_min) & (df['rt'] <= rt_max)
    df['label'] = label
    
    return df[['label', 'rt', 'mz', 'i', 'precursor_MZ', 
               'precursor_intensity', 'collision_energy', 'in_feature']]


def calculate_ms1_summary(df: pd.DataFrame, feature_filter: bool = True) -> pd.DataFrame:
    """
    Calculate summary properties for features from MS1 data.
    Use feature_filter=False to keep unmatched data
    """
    summary = {
        'label': [],
        'num_datapoints': [], 
        'peak_area': [], 
        'peak_height': [], 
        'mz_centroid': [],
        'rt_peak': []
    }
        
    if feature_filter:
        df = df[df['in_feature'] == True]

    for label_group, label_data in df.groupby('label'):
        summary['label'].append(label_group)
        summary['num_datapoints'].append(label_data['i'].count())
        sum_intensity = label_data['i'].sum()
        summary['peak_area'].append(sum_intensity)
        
        if sum_intensity > 0:
            idx = label_data['i'].idxmax()
            summary['peak_height'].append(label_data.loc[idx, 'i'])
            summary['mz_centroid'].append(sum(label_data['i'] * label_data['mz']) / sum_intensity)
            summary['rt_peak'].append(label_data.loc[idx, 'rt'])
        else:
            summary['peak_height'].append(0.0)
            summary['mz_centroid'].append(0.0)
            summary['rt_peak'].append(0.0)

    return pd.DataFrame(summary)


