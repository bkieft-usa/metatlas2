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
from concurrent.futures import as_completed

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('extract_data_from_parquet')

def extract_eic_and_ms2_from_parquet(
    atlas: "Atlas",
    stage: str,
    lcmsruns: List["LCMSRun"],
    workflow_params: dict,
    use_parallel: bool = True,
    only_ms_level: int = None,
    max_workers: int = None
) -> "ExperimentalData":
    """
    Orchestator function that calls helper functions to extract EIC and MS2 data directly from parquet files.
    Extract EIC and MS2 data directly from parquet files with optional parallel processing.
    
    Args:
        atlas: Atlas object with attributes [atlas_uid, atlas_name, ...]
        lcmsruns: List of LCMSRun objects
        ppm_tolerance: m/z tolerance in ppm
        extra_time: Extra RT time to extract beyond feature bounds
        use_parallel: Whether to use parallel processing (default: True)
        only_ms_level: If specified, only extract data for this MS level (1 or 2)
        max_workers: Maximum number of parallel workers (default: min(cpu_count, len(files), 8))

    Returns:
        experimental_data_obj: ExperimentalData object with extracted MS1 and MS2 data

    """
    from workflow_objects import ExperimentalData, MS1Data, MS2Data

    logger.info(f"Starting data extraction based on {atlas.atlas_uid} ({atlas.atlas_name}) for stage '{stage}' from {len(lcmsruns)} LCMS runs...")

    logger.info("Initiating an experimental data object to hold results during analysis...")
    experimental_data_obj = ExperimentalData()

    project_files_list = [run.file_path for run in lcmsruns]
    logger.info(f"Starting data extraction for {len(atlas.compound_mzrts)} compounds from {len(project_files_list)} project files...")

    ppm_tolerance = workflow_params.get("ppm_error", 20.0)
    extra_time = workflow_params.get("extra_time", 1)
    logger.info(f"Using ppm_tolerance={ppm_tolerance} and extra_time={extra_time} for data extraction.")

    if max_workers is None:
        import multiprocessing as mp
        max_workers = min(mp.cpu_count(), len(project_files_list), 8)
    use_parallel = use_parallel and max_workers > 1 and len(project_files_list) > 1

    compound_mzrts = list(atlas.compound_mzrts.values())

    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm.notebook import tqdm
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_process_single_parquet_file, parquet_file, compound_mzrts, ppm_tolerance, extra_time, only_ms_level): parquet_file
                for parquet_file in project_files_list
            }

            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Extracting data from parquet files"):
                parquet_file = future_to_file[future]

                try:
                    file_results = future.result()
                    for inchi_key, adduct_data in file_results.items():
                        for adduct, file_data in adduct_data.items():
                            ms1_obj = MS1Data(
                                inchi_key=inchi_key,
                                adduct=adduct,
                                filename=parquet_file,
                                data=file_data['ms1_data']
                            )
                            experimental_data_obj.ms1_data.append(ms1_obj)
                            ms2_obj = MS2Data(
                                inchi_key=inchi_key,
                                adduct=adduct,
                                filename=parquet_file,
                                data=file_data['ms2_data']
                            )
                            experimental_data_obj.ms2_data.append(ms2_obj)

                except Exception as e:
                    logger.error(f"Error processing {parquet_file}: {e}")
                    continue
    else:
        logger.info("Using sequential processing...")
        from tqdm.notebook import tqdm
        for parquet_file in tqdm(project_files_list, desc="Processing parquet files"):
            try:
                file_results = _process_single_parquet_file(
                    parquet_file, compound_mzrts, ppm_tolerance, extra_time, only_ms_level
                )
                for inchi_key, adduct_data in file_results.items():
                    for adduct, file_data in adduct_data.items():
                        ms1_obj = MS1Data(
                            inchi_key=inchi_key,
                            adduct=adduct,
                            filename=parquet_file,
                            data=file_data['ms1_data']
                        )
                        experimental_data_obj.ms1_data.append(ms1_obj)
                        ms2_obj = MS2Data(
                            inchi_key=inchi_key,
                            adduct=adduct,
                            filename=parquet_file,
                            data=file_data['ms2_data']
                        )
                        experimental_data_obj.ms2_data.append(ms2_obj)
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue

    return experimental_data_obj


def _process_single_parquet_file(
    parquet_file: str, 
    compound_mzrts: list, 
    ppm_tolerance: float, 
    extra_time: float, 
    only_ms_level: int = None
) -> Dict[str, Dict]:
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
    for compound_mzrt in compound_mzrts:
        inchi_key = compound_mzrt.inchi_key
        adduct = compound_mzrt.adduct
        compound_data = {'ms1_data': pd.DataFrame(), 'ms2_data': pd.DataFrame()}
        
        if is_ms1 and (only_ms_level is None or only_ms_level == 1):
            ms1_data = _extract_ms1_from_parquet(
                parquet_file,
                mz=compound_mzrt.mz,
                rt_min=compound_mzrt.rt_min,
                rt_max=compound_mzrt.rt_max,
                ppm_tolerance=ppm_tolerance,
                extra_time=extra_time
            )
            ms1_data = ms1_data.sort_values(by=['rt', 'i'], ascending=[True, False]).reset_index(drop=True)
            compound_data['ms1_data'] = ms1_data
        elif is_ms2 and (only_ms_level is None or only_ms_level == 2):
            ms2_data = _extract_ms2_from_parquet(
                parquet_file,
                mz=compound_mzrt.mz,
                rt_min=compound_mzrt.rt_min,
                rt_max=compound_mzrt.rt_max,
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


def calculate_mz_bounds(mz: float, ppm_tolerance: float) -> tuple:
    """Calculate m/z bounds given ppm tolerance."""
    delta = mz * ppm_tolerance / 1e6
    return (mz - delta, mz + delta)

def calculate_rt_bounds(rt_min: float, rt_max: float, extra_time: float) -> tuple:
    """Calculate RT bounds with extra time."""
    return (rt_min - extra_time, rt_max + extra_time)


def _extract_ms1_from_parquet(
    parquet_file: str,
    mz: float,
    rt_min: float,
    rt_max: float,
    ppm_tolerance: float,
    extra_time: float = 1
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
            ('rt', '>=', rt_min),
            ('rt', '<=', rt_max)
        ]
    ).to_pandas()
    
    if not df.empty:
        return df[['rt', 'mz', 'i']]
    else:
        return pd.DataFrame(columns=['rt', 'mz', 'i'])

def _extract_ms2_from_parquet(
    parquet_file: str,
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
            ('rt', '>=', rt_min),
            ('rt', '<=', rt_max)
        ]
    ).to_pandas()
    
    if not df.empty:
        return df[['rt', 'mz', 'i', 'precursor_MZ',
                'precursor_intensity', 'collision_energy']]
    else:
        return pd.DataFrame(columns=['rt', 'mz', 'i', 'precursor_MZ',
                                'precursor_intensity', 'collision_energy'])