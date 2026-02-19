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
    project_files: dict,
    ppm_tolerance: float = 20.0,
    extra_time: float = 0.1,
    use_parallel: bool = True,
    only_ms_level: int = None,
    max_workers: int = None
) -> Dict[str, Dict]:
    """
    Orchestator function that calls helper functions to extract EIC and MS2 data directly from parquet files.
    Extract EIC and MS2 data directly from parquet files with optional parallel processing.
    
    Args:
        atlas_df: Atlas DataFrame with columns [label, inchi_key, 'adduct', mz, rt_min, rt_max, rt_peak]
        project_files_df: DataFrame with columns ['file_path', 'ms_level', 'polarity']
        ppm_tolerance: m/z tolerance in ppm
        extra_time: Extra RT time to extract beyond feature bounds
        use_parallel: Whether to use parallel processing (default: True)
        only_ms_level: If specified, only extract data for this MS level (1 or 2)
        max_workers: Maximum number of parallel workers (default: min(cpu_count, len(files), 8))
    
    Returns:
        Dict mapping inchi_key and adduct to extracted data for each file, structured as:
        {
            inchi_key: {
                adduct: {
                    file_path: {
                        'ms1_data': DataFrame with columns [label, rt, mz, i],
                        'ms2_data': DataFrame with columns [label, rt, mz, i, precursor_MZ, precursor_intensity, collision_energy]
                    },
                    ...
                },
                ...
            },
            ...
        }
    """

    project_files_df = pd.concat(project_files.values(), ignore_index=True)
    logger.info(f"Starting data extraction for {len(atlas_df)} compounds from {len(project_files_df)} project files...")

    # Initialize results structure
    results = {}
    for _, row in atlas_df.iterrows():
        inchi_key = row.get('inchi_key', '')
        if not inchi_key:
            logger.warning(f"Missing inchi_key for row with label {row.get('label', 'unknown')}")
            continue
        results[inchi_key] = {}
    
    # Determine parallelization strategy
    project_files = project_files_df['file_path'].tolist()
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(project_files), 8)
    use_parallel = use_parallel and max_workers > 1 and len(project_files) > 1
    
    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for parquet_file in project_files:
                future = executor.submit(_process_single_parquet_file, 
                                        parquet_file, atlas_df, ppm_tolerance, extra_time, only_ms_level)
                futures.append((future, parquet_file))
            
            # Collect results
            for future, parquet_file in tqdm(futures, desc="Extracting data from parquet files"):
                try:
                    file_results = future.result()
                    
                    # Merge file results into main results
                    for inchi_key, adduct_data in file_results.items():
                        for adduct, file_data in adduct_data.items():
                            if inchi_key not in results:
                                results[inchi_key] = {}
                            if adduct not in results[inchi_key]:
                                results[inchi_key][adduct] = {}
                            results[inchi_key][adduct][parquet_file] = file_data
                                        
                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
    else:
        logger.info("Using sequential processing...")
        
        for parquet_file in tqdm(project_files, desc="Processing parquet files"):
            try:
                file_results = _process_single_parquet_file(
                    parquet_file, atlas_df, ppm_tolerance, extra_time, only_ms_level
                )
                
                # Merge file results into main results
                for inchi_key, adduct_data in file_results.items():
                    for adduct, file_data in adduct_data.items():
                        if inchi_key not in results:
                            results[inchi_key] = {}
                        if adduct not in results[inchi_key]:
                            results[inchi_key][adduct] = {}
                        results[inchi_key][adduct][parquet_file] = file_data
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue

    return results


def _process_single_parquet_file(parquet_file: str, atlas_df: pd.DataFrame, 
                                  ppm_tolerance: float, extra_time: float, only_ms_level: int = None) -> Dict[str, Dict]:
    """Process a single parquet file - worker function for parallel processing."""    
    filename = Path(parquet_file).name
    is_ms1 = filename.endswith('_ms1_pos.parquet') or filename.endswith('_ms1_neg.parquet')
    is_ms2 = filename.endswith('_ms2_pos.parquet') or filename.endswith('_ms2_neg.parquet')
    
    if not (is_ms1 or is_ms2):
        raise ValueError(f"Cannot determine MS level from filename: {filename}")
    if (is_ms1 and is_ms2) or (not is_ms1 and not is_ms2):
        raise ValueError(f"Filename does not clearly indicate MS level: {filename}")
    if not Path(parquet_file).exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    results = {}
    for _, row in atlas_df.iterrows():
        inchi_key = row.get('inchi_key', '')
        adduct = row.get('adduct', '')
        compound_data = {
            'ms1_data': pd.DataFrame(),
            'ms2_data': pd.DataFrame()
        }
        if is_ms1 and (only_ms_level is None or only_ms_level == 1):
            ms1_data = extract_ms1_from_parquet(
                parquet_file,
                label=row['label'],
                mz=row['mz'],
                rt_min=row['rt_min'],
                rt_max=row['rt_max'],
                ppm_tolerance=ppm_tolerance,
                extra_time=extra_time
            )
            ms1_data = ms1_data.sort_values(by=['rt', 'i'], ascending=[True, False]).reset_index(drop=True)
            compound_data['ms1_data'] = ms1_data
        elif is_ms2 and (only_ms_level is None or only_ms_level == 2):
            ms2_data = extract_ms2_from_parquet(
                parquet_file,
                label=row['label'],
                mz=row['mz'],
                rt_min=row['rt_min'],
                rt_max=row['rt_max'],
                ppm_tolerance=ppm_tolerance,
                extra_time=extra_time
            )
            ms2_data = ms2_data.sort_values(by=['rt', 'mz'], ascending=[True, True]).reset_index(drop=True)
            compound_data['ms2_data'] = ms2_data
        # Store results for this inchi_key and adduct
        if inchi_key not in results:
            results[inchi_key] = {}
        results[inchi_key][adduct] = compound_data
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
        DataFrame with columns: [label, rt, mz, i]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, ppm_tolerance)
    rt_min, rt_max = calculate_rt_bounds(rt_min, rt_max, extra_time)
    
    # Read parquet with m/z filter (uses sorted index efficiently)
    df = pq.read_table(
        parquet_file,
        filters=[
            ('mz', '>=', mz_min),
            ('mz', '<=', mz_max),
            ('rt', '>=', rt_min - extra_time),
            ('rt', '<=', rt_max + extra_time)
        ]
    ).to_pandas()
    
    if not df.empty:
        df['label'] = label
        return df[['label', 'rt', 'mz', 'i']]
    else:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i'])

def calculate_mz_bounds(mz: float, ppm_tolerance: float) -> tuple:
    """Calculate m/z bounds given ppm tolerance."""
    delta = mz * ppm_tolerance / 1e6
    return (mz - delta, mz + delta)

def calculate_rt_bounds(rt_min: float, rt_max: float, extra_time: float) -> tuple:
    """Calculate RT bounds with extra time."""
    return (rt_min - extra_time, rt_max + extra_time)

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
                                 precursor_intensity, collision_energy]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, ppm_tolerance)
    rt_min, rt_max = calculate_rt_bounds(rt_min, rt_max, extra_time)
    
    # For MS2, filter by precursor m/z
    df = pq.read_table(
        parquet_file,
        filters=[
            ('precursor_MZ', '>=', mz_min),
            ('precursor_MZ', '<=', mz_max),
            ('rt', '>=', rt_min - extra_time),
            ('rt', '<=', rt_max + extra_time)
        ]
    ).to_pandas()
    
    if not df.empty:
        df['label'] = label
        return df[['label', 'rt', 'mz', 'i', 'precursor_MZ',
                'precursor_intensity', 'collision_energy']]
    else:
        return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'precursor_MZ',
                                'precursor_intensity', 'collision_energy'])