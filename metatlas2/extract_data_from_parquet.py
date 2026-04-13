import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import metatlas2.logging_config as lcf
logger = lcf.get_logger('extract_data_from_parquet')

def extract_eic_and_ms2_from_parquet(
    obj: "RTAlignment" or "AutoIdentification",
    stage: str,
    use_parallel: bool = True,
    max_workers: int = None
) -> "ExperimentalData":
    """
    Orchestator function that calls helper functions to extract EIC and MS2 data directly from parquet files.
    Extract EIC and MS2 data directly from parquet files with optional parallel processing.
    
    Args:
        obj: RTAlignment or AutoIdentification object
        lcmsruns: List of LCMSRun objects
        workflow_params: Dictionary containing workflow parameters
        use_parallel: Whether to use parallel processing (default: True)
        only_ms_level: If specified, only extract data for this MS level (1 or 2)
        max_workers: Maximum number of parallel workers (default: min(cpu_count, len(files), 8))

    Returns:
        experimental_data_obj: ExperimentalData object with extracted MS1 and MS2 data

    """
    from metatlas2.workflow_objects import ExperimentalData, MS1Data, MS2Data

    atlas = obj.align_atlas_obj if stage == "rt_alignment" else obj.pre_autoid_atlas_obj
    lcmsruns = obj.aligner_lcmsruns if stage == "rt_alignment" else obj.autoid_lcmsruns
    workflow_params = obj.rt_alignment_params if stage == "rt_alignment" else obj.workflow_params
    only_ms_level = 1 if stage == "rt_alignment" else None

    logger.info(f"Starting data extraction based on {atlas.atlas_uid} ({atlas.atlas_name}) for stage '{stage}' from {len(lcmsruns)} LCMS runs...")

    logger.info("Initiating an experimental data object to hold results during analysis...")
    experimental_data_obj = ExperimentalData()

    project_files_list = [run.file_path for run in lcmsruns]
    logger.info(f"Starting data extraction for {len(atlas.compound_mzrts)} compounds from {len(project_files_list)} project files...")

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(project_files_list), 8)
    use_parallel = use_parallel and max_workers > 1 and len(project_files_list) > 1

    compound_mzrts = list(atlas.compound_mzrts.values())
    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_process_single_parquet_file, parquet_file, compound_mzrts, workflow_params, only_ms_level): parquet_file
                for parquet_file in project_files_list
            }

            for future in as_completed(future_to_file):
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
        for parquet_file in project_files_list:
            try:
                file_results = _process_single_parquet_file(
                    parquet_file, compound_mzrts, workflow_params, only_ms_level
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
    workflow_params: Dict[str, Any], 
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
                workflow_params=workflow_params
            )
            ms1_data = ms1_data.sort_values(by=['rt', 'i'], ascending=[True, False]).reset_index(drop=True)
            compound_data['ms1_data'] = ms1_data
        elif is_ms2 and (only_ms_level is None or only_ms_level == 2):
            ms2_data = _extract_ms2_from_parquet(
                parquet_file,
                mz=compound_mzrt.mz,
                rt_min=compound_mzrt.rt_min,
                rt_max=compound_mzrt.rt_max,
                workflow_params=workflow_params
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
    workflow_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extract a single feature from a parquet file.
    
    Args:
        parquet_file: Path to parquet file (e.g., *_ms1_pos.parquet)
        compound_name: Feature compound_name from atlas
        mz: Target m/z value
        rt_min: Minimum retention time
        rt_max: Maximum retention time
        workflow_params: Dictionary containing workflow parameters (e.g., ppm_tolerance, extra_time)
    
    Returns:
        DataFrame with columns: [compound_name, rt, mz, i]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, workflow_params.get("ppm_error", 20.0))
    rt_min, rt_max = calculate_rt_bounds(rt_min, rt_max, workflow_params.get("extra_time", 1.0))
    minimum_intensity = workflow_params.get("ms1_min_peak_intensity", 0)
    min_points = workflow_params.get("ms1_min_num_points", 1)
    
    # Read parquet with m/z filter (uses sorted index efficiently)
    df = pq.read_table(
        parquet_file,
        filters=[
            ('mz', '>=', mz_min),
            ('mz', '<=', mz_max),
            ('rt', '>=', rt_min),
            ('rt', '<=', rt_max),
            ('i', '>=', minimum_intensity)
        ]
    ).to_pandas()
    
    if len(df) < min_points:
        return pd.DataFrame(columns=['rt', 'mz', 'i'])

    if not df.empty:
        return df[['rt', 'mz', 'i']]
    else:
        return pd.DataFrame(columns=['rt', 'mz', 'i'])

def _extract_ms2_from_parquet(
    parquet_file: str,
    mz: float,
    rt_min: float,
    rt_max: float,
    workflow_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extract MS2 feature from parquet file.
    
    Returns:
        DataFrame with columns: [compound_name, rt, mz, i, precursor_MZ,
                                 precursor_intensity, collision_energy]
    """
    mz_min, mz_max = calculate_mz_bounds(mz, workflow_params.get("ppm_error", 5.0))
    rt_min, rt_max = calculate_rt_bounds(rt_min, rt_max, workflow_params.get("extra_time", 1.0))

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