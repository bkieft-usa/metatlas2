#!/usr/bin/env python3
"""
Consolidated file conversion script for Metatlas.
Converts .raw files to .mzML, then to .parquet format.
Intended to run as a cronjob.
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymzml
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logger = logging.getLogger(__name__)

# Vars
RAW_IMAGE = "quay.io/biocontainers/thermorawfileparser@sha256:3b930ef774b3d4e0d559f38903da2390f9b24b96a016a1761805b88ae78c2b40"

# Parquet schemas
MS1_SCHEMA = pa.schema([
    ('mz', pa.float32()),
    ('i', pa.float32()),
    ('rt', pa.float32()),
    ('polarity', pa.int16()),
])

MS2_SCHEMA = pa.schema([
    ('mz', pa.float32()),
    ('i', pa.float32()),
    ('rt', pa.float32()),
    ('polarity', pa.int16()),
    ('precursor_MZ', pa.float32()),
    ('precursor_intensity', pa.float32()),
    ('collision_energy', pa.float32()),
])


def set_file_permissions(file_path, mode=0o660):
    """Set file permissions with error handling."""
    try:
        path = Path(file_path)
        os.chown(path, -1, os.stat(path.parent).st_gid)
        os.chmod(path, mode)
    except Exception as e:
        logger.warning(f"Could not set permissions for {file_path}: {e}")


def raw_to_mzml(raw_file):
    """
    Convert .raw file to .mzML using ThermoRawFileParser.
    Returns (success, mzml_path, error_message)
    """
    raw_path = Path(raw_file).resolve()
    mzml_path = raw_path.with_suffix('.mzML')
    progress_path = raw_path.with_suffix('.progress')
    failed_path = raw_path.with_suffix('.failed')
    
    # Check if already converted or failed
    if mzml_path.exists():
        logger.info(f"mzML already exists: {mzml_path}")
        return True, str(mzml_path), ""
    
    if failed_path.exists():
        logger.info(f"Previous conversion failed: {raw_file}")
        return False, "", "Previous conversion failed"
    
    # Create progress file
    progress_path.touch()
    set_file_permissions(progress_path, 0o640)
    
    try:
        logger.info(f"Converting {raw_file} to mzML")
        
        # Run ThermoRawFileParser
        subprocess.run(["shifterimg", "pull", RAW_IMAGE], check=True, capture_output=True)
        shifter_cmd = ["shifter", f"--image={RAW_IMAGE}", "--clearenv", "--module=none"]
        shifter_cmd.extend(["ThermoRawFileParser.sh", f"-i={raw_path}", f"-o={raw_path.parent}", "-f=1"])
        result = subprocess.run(shifter_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = f"ThermoRawFileParser failed: {result.stderr}"
            logger.error(error_msg)
            progress_path.rename(failed_path)
            return False, "", error_msg
        
        if not mzml_path.exists() or mzml_path.stat().st_size == 0:
            error_msg = "mzML file not created or empty"
            logger.error(error_msg)
            progress_path.rename(failed_path)
            return False, "", error_msg
        
        progress_path.unlink()
        logger.info(f"Successfully converted to mzML: {mzml_path}")
        return True, str(mzml_path), ""
        
    except Exception as e:
        error_msg = f"Error during raw to mzML conversion: {e}"
        logger.error(error_msg)
        if progress_path.exists():
            progress_path.rename(failed_path)
        return False, "", error_msg


def read_spectrum(spectrum):
    """
    Read a single spectrum from pymzml.
    Returns (data, info) where info is [rt, polarity, ms_level]
    """
    try:
        # Determine polarity
        if spectrum.get('negative scan'):
            polarity = 0
        elif spectrum.get('positive scan'):
            polarity = 1
        else:
            return None, None
        
        ms_level = spectrum.ms_level
        rt = spectrum.scan_time_in_minutes()
        info = [rt, polarity, ms_level]
        
        if ms_level == 1:
            data = [(mz, i, rt, polarity) for (mz, i) in spectrum.peaks('centroided')]
        else:
            prec = spectrum.selected_precursors
            
            if len(prec) != 1:
                return None, None
            
            prec = prec[0]
            collision_energy = spectrum.get('collision energy', 0.0)
            precursor_intensity = prec.get('i', 0.0)
            precursor_mz = prec.get('mz', 0.0)
            
            data = [(mz, i, rt, polarity, precursor_mz, precursor_intensity, collision_energy) 
                    for (mz, i) in spectrum.peaks('centroided')]
        
        return data, info
        
    except Exception as e:
        logger.warning(f"Error reading spectrum: {e}")
        return None, None


def remove_easyic_signal(spectrum_data, easyic_mzs={1: 202.07770, 0: 202.07880}, mz_tolerance=0.001):
    """Remove Easy-IC lock mass signal from spectral data."""
    if not spectrum_data:
        return spectrum_data
    
    targeted_mz = easyic_mzs[spectrum_data[0][3]]
    return [peak for peak in spectrum_data if abs(peak[0] - targeted_mz) > mz_tolerance]


def parse_filename_for_expected_parquet(filename):
    """
    Determine which parquet files should exist based on filename conventions.
    Returns list of expected parquet file suffixes like ['ms1_pos', 'ms2_pos'].
    
    Filename format: parts separated by '_', where:
    - 10th position (index 9): polarity (POS, NEG, FPS)
    - 11th position (index 10): MS level (MS1, MS2)
    """
    parts = Path(filename).stem.split('_')
    
    if len(parts) < 14:
        logger.warning(f"Filename doesn't match expected format: {filename}")
        return []
    
    polarity_str = parts[9]
    ms_level_str = parts[10]
    
    # Determine polarities
    polarity_map = {
        'FPS': ['pos', 'neg'],
        'POS': ['pos'],
        'NEG': ['neg']
    }
    polarities = polarity_map.get(polarity_str, ['pos', 'neg'])
    if polarity_str not in polarity_map:
        logger.warning(f"Unknown polarity '{polarity_str}' in {filename}, assuming both")
    
    # Determine MS levels (MS2 runs include MS1)
    ms_level_map = {
        'MS2': ['ms1', 'ms2'],
        'MS1': ['ms1']
    }
    ms_levels = ms_level_map.get(ms_level_str, ['ms1', 'ms2'])
    if ms_level_str not in ms_level_map:
        logger.warning(f"Unknown MS level '{ms_level_str}' in {filename}, assuming both")
    
    # Build expected file list
    return [f"{ms_level}_{polarity}" for ms_level in ms_levels for polarity in polarities]


def find_unconverted_files(base_dir):
    """
    Find .raw files that don't have corresponding .mzML, expected .parquet files, or .failed files.
    """
    logger.info(f"Searching for unconverted files in {base_dir}")
    
    # Find all relevant files
    cmd = (
        f'find "{base_dir}" -mindepth 2 -maxdepth 2 -type f '
        r'\( -name "*.raw" -o -name "*.mzML" -o -name "*_ms1_pos.parquet" '
        r'-o -name "*_ms1_neg.parquet" -o -name "*_ms2_pos.parquet" '
        r'-o -name "*_ms2_neg.parquet" -o -name "*.failed" \)'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Find command failed: {result.stderr}")
        return []
    
    all_files = [f for f in result.stdout.strip().split('\n') if f]
    
    # Group files by base name
    file_groups = {}  # base_path -> {raw, mzml, parquet_files, failed}
    
    for file_path in all_files:
        path = Path(file_path)
        
        # Determine base path (without any suffix)
        base_path = str(path)
        for suffix in ['_ms1_pos.parquet', '_ms1_neg.parquet', '_ms2_pos.parquet', '_ms2_neg.parquet']:
            if base_path.endswith(suffix):
                base_path = base_path.replace(suffix, '')
                break
        else:
            base_path = str(path.with_suffix(''))
        
        if base_path not in file_groups:
            file_groups[base_path] = {
                'raw': None,
                'mzml': False,
                'parquet_files': set(),
                'failed': False
            }
        
        if file_path.endswith('.raw'):
            file_groups[base_path]['raw'] = file_path
        elif file_path.endswith('.mzML'):
            file_groups[base_path]['mzml'] = True
        elif file_path.endswith('.failed'):
            file_groups[base_path]['failed'] = True
        elif '.parquet' in file_path:
            # Extract which parquet file this is
            for parquet_type in ['ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg']:
                if f'_{parquet_type}.parquet' in file_path:
                    file_groups[base_path]['parquet_files'].add(parquet_type)
                    break
    
    # Find unconverted files
    unconverted = []
    
    for base_path, files in file_groups.items():
        # Skip if no raw file or conversion failed
        if not files['raw'] or files['failed']:
            continue
        
        # Determine expected parquet files from filename
        expected_parquet = set(parse_filename_for_expected_parquet(base_path))
        has_all_parquet = expected_parquet.issubset(files['parquet_files'])
        
        # File needs conversion if missing mzml or expected parquet files
        if not (files['mzml'] and has_all_parquet):
            unconverted.append(files['raw'])
            logger.debug(f"Need conversion: {files['raw']} - mzml:{files['mzml']}, "
                        f"parquet:{files['parquet_files']} (expected:{expected_parquet})")
    
    unconverted = sorted(unconverted)
    logger.info(f"Found {len(unconverted)} unconverted .raw files")
    
    return unconverted


def mzml_to_parquet(mzml_file, filter_easyic=True):
    """
    Convert mzML to Parquet files based on filename expectations.
    Returns (success, error_message)
    """
    logger.info(f"Converting mzML to Parquet: {mzml_file}")
    
    mzml_path = Path(mzml_file)
    output_prefix = str(mzml_path.with_suffix(''))
    failed_path = mzml_path.with_suffix('.failed')
    
    output_files = {
        'ms1_pos': f'{output_prefix}_ms1_pos.parquet',
        'ms1_neg': f'{output_prefix}_ms1_neg.parquet',
        'ms2_pos': f'{output_prefix}_ms2_pos.parquet',
        'ms2_neg': f'{output_prefix}_ms2_neg.parquet',
    }
    
    # Determine expected files from filename
    expected_files = set(parse_filename_for_expected_parquet(str(mzml_file)))
    
    if not expected_files:
        error_msg = "Cannot parse filename format - unable to determine expected parquet files"
        logger.error(error_msg)
        failed_path.touch()
        return False, error_msg
    
    # Check if expected parquet files already exist
    existing_expected = [k for k in expected_files if Path(output_files[k]).exists()]
    if len(existing_expected) == len(expected_files):
        logger.info(f"All expected Parquet files already exist: {expected_files}")
        return True, ""
    
    try:
        mzml_reader = pymzml.run.Reader(str(mzml_file))
    except Exception as e:
        error_msg = f"Error opening mzML file: {e}"
        logger.error(error_msg)
        failed_path.touch()
        return False, error_msg
    
    # Accumulate data
    data_buffers = {key: [] for key in output_files}
    
    try:
        for spectrum in mzml_reader:
            data, info = read_spectrum(spectrum)
            
            if data is None or not data:
                continue
            
            if filter_easyic:
                data = remove_easyic_signal(data)
                if not data:
                    continue
            
            ms_level = info[2]
            polarity = info[1]
            
            key = f"ms{ms_level}_{'pos' if polarity else 'neg'}"
            data_buffers[key].extend(data)
        
        # Write only expected Parquet files
        files_created = []
        for key in expected_files:
            buffer = data_buffers[key]
            output_path = output_files[key]
            
            # Prepare data dictionary
            if key.startswith('ms1'):
                schema = MS1_SCHEMA
                if len(buffer) == 0:
                    data_dict = {'mz': [], 'i': [], 'rt': [], 'polarity': []}
                    logger.warning(f"No data found for expected file type: {key}")
                else:
                    data_dict = {
                        'mz': [row[0] for row in buffer],
                        'i': [row[1] for row in buffer],
                        'rt': [row[2] for row in buffer],
                        'polarity': [row[3] for row in buffer],
                    }
            else:
                schema = MS2_SCHEMA
                if len(buffer) == 0:
                    data_dict = {
                        'mz': [], 'i': [], 'rt': [], 'polarity': [],
                        'precursor_MZ': [], 'precursor_intensity': [], 'collision_energy': []
                    }
                    logger.warning(f"No data found for expected file type: {key}")
                else:
                    data_dict = {
                        'mz': [row[0] for row in buffer],
                        'i': [row[1] for row in buffer],
                        'rt': [row[2] for row in buffer],
                        'polarity': [row[3] for row in buffer],
                        'precursor_MZ': [row[4] for row in buffer],
                        'precursor_intensity': [row[5] for row in buffer],
                        'collision_energy': [row[6] for row in buffer],
                    }
            
            # Create and sort table
            table = pa.Table.from_pydict(data_dict, schema=schema)
            
            if len(buffer) > 0:
                sorted_indices = pa.compute.sort_indices(table, sort_keys=[('mz', 'ascending')])
                table = table.take(sorted_indices)
            
            # Write Parquet file
            pq.write_table(
                table,
                output_path,
                compression='snappy',
                use_dictionary=True,
                row_group_size=100_000,
                write_statistics=True,
                data_page_size=1024*1024,
            )
            
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"  {key}: {len(buffer):,} peaks, {file_size_mb:.2f} MB")
            files_created.append(key)
        
        logger.info(f"Successfully converted to Parquet files: {files_created}")
        return True, ""
        
    except Exception as e:
        error_msg = f"Error during mzML to Parquet conversion: {e}"
        logger.error(error_msg)
        failed_path.touch()
        
        # Cleanup partial parquet files
        for key in expected_files:
            output_path = Path(output_files[key])
            if output_path.exists():
                output_path.unlink()
        
        return False, error_msg


def process_single_file(raw_file):
    """
    Process a single .raw file through the complete pipeline.
    Returns a dictionary with results and any errors.
    """
    result = {
        'raw_file': raw_file,
        'success': False,
        'mzml_created': False,
        'parquet_created': False,
        'errors': []
    }
    
    logger.info(f"Processing: {raw_file}")
    
    # Step 1: raw to mzML
    success, mzml_file, error = raw_to_mzml(raw_file)
    if not success:
        result['errors'].append(f"raw->mzML: {error}")
        return result
    
    result['mzml_created'] = True
    
    # Step 2: mzML to Parquet
    success, error = mzml_to_parquet(mzml_file)
    if not success:
        result['errors'].append(f"mzML->Parquet: {error}")
    else:
        result['parquet_created'] = True
    
    result['success'] = result['parquet_created']
    
    return result


def main():
    """Main entry point for the consolidated conversion script."""
    parser = argparse.ArgumentParser(
        description='Convert .raw files to mzML and Parquet formats'
    )
    parser.add_argument(
        'directory',
        choices=['jgi', 'egsb'],
        help='Search directory (jgi or egsb)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be processed without processing them'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = f"/Users/BKieft/Metabolomics/metatlas2/data/test_data/projects/{args.directory}"
    log_file = f"/Users/BKieft/Metabolomics/metatlas2/data/test_data/file_conversion_logs/{args.directory}.log"
    #base_dir = f"/global/cfs/cdirs/metatlas/raw_data/{args.directory}"
    #log_file = f"/global/cfs/cdirs/m2650/file_converter_logs/{args.directory}.log"

    # Configure logging to write ONLY to file
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info(f"Starting conversion for directory: {args.directory}")
    logger.info(f"Max workers: {args.max_workers}")
    
    # Find unconverted files
    unconverted_files = find_unconverted_files(base_dir)
    
    if not unconverted_files:
        logger.info("No files to convert")
        return 0
    
    if args.dry_run:
        logger.info(f"Would process {len(unconverted_files)} files:")
        for f in unconverted_files:
            logger.info(f"  {f}")
        return 0
    
    # Process files in parallel
    results = []
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, f): f for f in unconverted_files}
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                if not result['success']:
                    failed_files.append(file)
                    logger.error(f"Failed to process {file}: {result['errors']}")
                
            except Exception as e:
                logger.error(f"Exception processing {file}: {e}")
                failed_files.append(file)
    
    # Summary
    successful = len([r for r in results if r['success']])
    logger.info("=" * 80)
    logger.info(f"Conversion complete: {successful}/{len(unconverted_files)} files processed successfully")
    
    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)}):")
        for f in failed_files:
            logger.warning(f"  {f}")
    
    logger.info("=" * 80)
    
    return 0 if not failed_files else 1


if __name__ == '__main__':
    sys.exit(main())