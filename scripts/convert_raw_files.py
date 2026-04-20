#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tables

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pymzml.run import Reader

import metatlas2.logging_config as lcf
logger = lcf.get_logger('analysis_summary')

# Vars
_DATA_DIR = os.environ.get("METATLAS_DATA_DIR")
if _DATA_DIR is None:
    raise EnvironmentError(
        "METATLAS_DATA_DIR environment variable is not set. "
        "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it."
    )
RAW_FILES_BASE = f"{_DATA_DIR}/lcmsruns/"
LOG_FILE_BASE = f"{RAW_FILES_BASE}/file_conversion_logs"
RAW_IMAGE = "quay.io/biocontainers/thermorawfileparser@sha256:3b930ef774b3d4e0d559f38903da2390f9b24b96a016a1761805b88ae78c2b40"
FORMAT_VERSION = 5
METATLAS_VERSION = "2.0.0"
SCHEMA_DEFINITIONS = {
    'MS1': [
        ('mz', 'float32'),
        ('i', 'float32'),
        ('rt', 'float32'),
        ('polarity', 'int16'),
    ],
    'MS2': [
        ('mz', 'float32'),
        ('i', 'float32'),
        ('rt', 'float32'),
        ('polarity', 'int16'),
        ('precursor_MZ', 'float32'),
        ('precursor_intensity', 'float32'),
        ('collision_energy', 'float32'),
    ]
}

# Generate PyArrow schemas
def _create_pa_schema(schema_def):
    """Convert schema definition to PyArrow schema."""
    type_map = {
        'float32': pa.float32(),
        'int16': pa.int16(),
    }
    return pa.schema([(name, type_map[dtype]) for name, dtype in schema_def])

# Generate PyTables classes
def _create_tables_class(schema_def, base_class=None):
    """Convert schema definition to PyTables description class."""
    type_map = {
        'float32': tables.Float32Col,
        'int16': tables.Int16Col,
    }
    attrs = {}
    for pos, (name, dtype) in enumerate(schema_def):
        attrs[name] = type_map[dtype](pos=pos)
    
    if base_class:
        return type(f'Generated{base_class.__name__}', (base_class,), attrs)
    return type('GeneratedDescription', (tables.IsDescription,), attrs)

# Create objects for the two MS levels for each file output type
parquet_ms1_schema = _create_pa_schema(SCHEMA_DEFINITIONS['MS1'])
parquet_ms2_schema = _create_pa_schema(SCHEMA_DEFINITIONS['MS2'])
h5_ms1_schema = _create_tables_class(SCHEMA_DEFINITIONS['MS1'])
h5_ms2_schema = _create_tables_class(SCHEMA_DEFINITIONS['MS2'])

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
    
    # Determine project directory (parent of raw/ subdirectory)
    project_dir = raw_path.parent.parent
    
    # Output mzML to mzML subdirectory in project
    mzml_dir = project_dir / "mzML"
    mzml_dir.mkdir(parents=True, exist_ok=True)
    mzml_path = mzml_dir / raw_path.with_suffix('.mzML').name
    
    # Progress and failed files go to project directory
    progress_path = project_dir / raw_path.with_suffix('.progress').name
    failed_path = project_dir / raw_path.with_suffix('.failed').name
    
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
        # Run ThermoRawFileParser
        shifter_cmd = ["shifter", f"--image={RAW_IMAGE}", "--clearenv", "--module=none"]
        shifter_cmd.extend(["ThermoRawFileParser.sh", f"-i={raw_path}", f"-o={mzml_dir}", "-f=1"])
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
        logger.info(f"Successfully converted .raw to .mzML: {mzml_path.name}")
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


def mzml_to_h5_and_parquet(mzml_file, filter_easyic=True):
    """
    Convert mzML to both HDF5 and Parquet formats in a single pass.
    Returns (success, error_messages)
    """
    #logger.info(f"Converting mzML to H5 and Parquet: {mzml_file}")
    
    mzml_path = Path(mzml_file)
    
    # Determine project directory (parent of mzML/ subdirectory)
    project_dir = mzml_path.parent.parent
    
    # Output directories in project
    h5_dir = project_dir / "h5"
    parquet_dir = project_dir / "parquet"
    h5_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    h5_path = h5_dir / mzml_path.with_suffix('.h5').name
    base_name = mzml_path.stem
    output_prefix = str(parquet_dir / base_name)
    
    # Failed file goes to project directory
    failed_path = project_dir / mzml_path.with_suffix('.failed').name
    errors = []
    
    # Define output files
    parquet_files = {
        'ms1_pos': f'{output_prefix}_ms1_pos.parquet',
        'ms1_neg': f'{output_prefix}_ms1_neg.parquet',
        'ms2_pos': f'{output_prefix}_ms2_pos.parquet',
        'ms2_neg': f'{output_prefix}_ms2_neg.parquet',
    }
    
    # Determine expected parquet files from filename
    expected_parquet = set(parse_filename_for_expected_parquet(str(mzml_file)))
    
    if not expected_parquet:
        error_msg = "Cannot parse filename format - unable to determine expected parquet files"
        logger.error(error_msg)
        failed_path.touch()
        return False, [error_msg]
    
    # Check what already exists
    h5_exists = h5_path.exists()
    existing_parquet = {k for k in expected_parquet if Path(parquet_files[k]).exists()}
    all_parquet_exist = len(existing_parquet) == len(expected_parquet)
    
    if h5_exists and all_parquet_exist:
        logger.info(f"All output files already exist")
        return True, []
    
    # Open mzML file
    try:
        mzml_reader = Reader(str(mzml_file), use_index = False, build_index_from_scratch = True)
    except Exception as e:
        error_msg = f"Error opening mzML file: {e}"
        logger.error(error_msg)
        failed_path.touch()
        return False, [error_msg]
    
    # Setup HDF5 file (if needed)
    h5_file = None
    h5_tables = {}
    h5_data = {}
    if not h5_exists:
        try:
            FILTERS = tables.Filters(complib='blosc', complevel=1)
            h5_file = tables.open_file(str(h5_path), "w", filters=FILTERS)
            h5_tables = {
                'ms1_neg': h5_file.create_table('/', 'ms1_neg', description=h5_ms1_schema),
                'ms1_pos': h5_file.create_table('/', 'ms1_pos', description=h5_ms1_schema),
                'ms2_neg': h5_file.create_table('/', 'ms2_neg', description=h5_ms2_schema),
                'ms2_pos': h5_file.create_table('/', 'ms2_pos', description=h5_ms2_schema),
            }
            h5_data = {k: [] for k in h5_tables}
        except Exception as e:
            error_msg = f"Error creating HDF5 file: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            if h5_file:
                h5_file.close()
            h5_file = None
    
    # Setup Parquet buffers (if needed)
    parquet_buffers = None
    if not all_parquet_exist:
        parquet_buffers = {key: [] for key in parquet_files}
    
    # Process spectra (single pass through mzML)
    try:
        for spectrum in mzml_reader:
            try:
                data, info = read_spectrum(spectrum)
            except (KeyError, TypeError):
                continue
            except Exception as e:
                sys.stdout.write(f'Read spectrum error: {e}\n')
                sys.stdout.flush()
                continue
            
            if not data:
                continue
            
            if filter_easyic:
                data = remove_easyic_signal(data)
                if not data:
                    continue
            
            # Determine table/buffer key
            ms_level = info[2]
            polarity = info[1]
            key = f"ms{ms_level}_{'pos' if polarity else 'neg'}"
            
            # Accumulate for HDF5
            if h5_file and key in h5_data:
                h5_data[key].extend(data)
            
            # Accumulate for Parquet
            if parquet_buffers is not None:
                parquet_buffers[key].extend(data)

        # Batch write accumulated HDF5 data
        if h5_file:
            for key, rows in h5_data.items():
                if rows:
                    h5_tables[key].append(rows)

        # Finalize HDF5
        if h5_file:
            try:
                for name in ['ms1_neg', 'ms2_neg', 'ms1_pos', 'ms2_pos']:
                    table = h5_file.get_node('/' + name)
                    table.cols.mz.create_csindex()
                    table.copy(sortby='mz', newname=name + '_mz')
                    table.cols.mz.remove_index()
                h5_file.set_node_attr('/', 'format_version', FORMAT_VERSION)
                h5_file.set_node_attr('/', 'metatlas_version', METATLAS_VERSION)
                h5_file.close()
                logger.info(f"Successfully converted .mzML to H5 file: {h5_path.name}")
            except Exception as e:
                error_msg = f"Error finalizing HDF5 file: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                h5_file.close()
                if h5_path.exists():
                    h5_path.unlink()
        
        # Write Parquet files
        if parquet_buffers is not None:
            files_created = []
            for key in expected_parquet:
                try:
                    buffer = parquet_buffers[key]
                    parquet_output_path = parquet_files[key]
                    
                    # Prepare data dictionary
                    if key.startswith('ms1'):
                        schema = parquet_ms1_schema
                        if len(buffer) == 0:
                            data_dict = {'mz': [], 'i': [], 'rt': [], 'polarity': []}
                            logger.warning(f"No data found for expected file type: {key}")
                        else:
                            arr = np.array(buffer, dtype=np.float32)
                            data_dict = {
                                'mz': arr[:, 0],
                                'i': arr[:, 1],
                                'rt': arr[:, 2],
                                'polarity': arr[:, 3].astype(np.int16),
                            }
                    else:
                        schema = parquet_ms2_schema
                        if len(buffer) == 0:
                            data_dict = {
                                'mz': [], 'i': [], 'rt': [], 'polarity': [],
                                'precursor_MZ': [], 'precursor_intensity': [], 'collision_energy': []
                            }
                            logger.warning(f"No data found for expected file type: {key}")
                        else:
                            arr = np.array(buffer, dtype=np.float32)
                            data_dict = {
                                'mz': arr[:, 0],
                                'i': arr[:, 1],
                                'rt': arr[:, 2],
                                'polarity': arr[:, 3].astype(np.int16),
                                'precursor_MZ': arr[:, 4],
                                'precursor_intensity': arr[:, 5],
                                'collision_energy': arr[:, 6],
                            }
                    
                    # Create and sort table
                    table = pa.Table.from_pydict(data_dict, schema=schema)
                    
                    if len(buffer) > 0:
                        sorted_indices = pa.compute.sort_indices(table, sort_keys=[('mz', 'ascending')])
                        table = table.take(sorted_indices)
                    
                    # Write Parquet file
                    pq.write_table(
                        table,
                        parquet_output_path,
                        compression='snappy',
                        use_dictionary=True,
                        row_group_size=100_000,
                        write_statistics=True,
                        data_page_size=1024*1024,
                    )
                    
                    file_size_mb = Path(parquet_output_path).stat().st_size / (1024 * 1024)
                    logger.info(f"Successfully converted .mzML to Parquet file: {Path(parquet_output_path).name} - {key}: {len(buffer):,} peaks, {file_size_mb:.2f} MB")
                    files_created.append(key)
                    
                except Exception as e:
                    error_msg = f"Error writing Parquet file {key}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    # Cleanup partial file
                    parquet_output_path = Path(parquet_files[key])
                    if parquet_output_path.exists():
                        parquet_output_path.unlink()
            
            if not files_created:
                logger.warning(f"Failed to create any Parquet files for: {mzml_file.name}")
        
        # Mark as failed if any errors occurred
        if errors:
            failed_path.touch()
            return False, errors
        
        return True, []
        
    except Exception as e:
        error_msg = f"Error during mzML conversion: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        
        # Cleanup
        if h5_file:
            h5_file.close()
        if h5_path.exists():
            h5_path.unlink()
        for key in expected_parquet:
            parquet_output_path = Path(parquet_files[key])
            if parquet_output_path.exists():
                parquet_output_path.unlink()
        
        failed_path.touch()
        return False, errors


def find_unconverted_files(top_dir):
    """
    Find .raw files in all project subdirectories that don't have corresponding 
    .mzML, .h5, expected .parquet files, or .failed files.
    """
    logger.info(f"Searching for unconverted files in {top_dir}")
    
    # Find all project directories (directories containing a 'raw' subdirectory)
    project_dirs = []
    for item in Path(top_dir).iterdir():
        if item.is_dir() and (item / "raw").exists():
            project_dirs.append(item)
    
    logger.info(f"Found {len(project_dirs)} project directories")
    
    all_unconverted = []
    
    for project_dir in sorted(project_dirs):
        logger.debug(f"Scanning project: {project_dir.name}")
        
        raw_dir = project_dir / "raw"
        mzml_dir = project_dir / "mzML"
        h5_dir = project_dir / "h5"
        parquet_dir = project_dir / "parquet"
        
        # Find all .raw files in this project
        raw_files = list(raw_dir.glob("*.raw"))
        
        # Find all existing output files
        mzml_files = set()
        h5_files = set()
        parquet_files = {}  # base_name -> set of parquet types
        failed_files = set()
        
        # Find mzML files
        if mzml_dir.exists():
            for mzml_file in mzml_dir.glob("*.mzML"):
                mzml_files.add(mzml_file.stem)
        
        # Find h5 files
        if h5_dir.exists():
            for h5_file in h5_dir.glob("*.h5"):
                h5_files.add(h5_file.stem)
        
        # Find parquet files
        if parquet_dir.exists():
            for parquet_file in parquet_dir.glob("*.parquet"):
                # Extract base name and parquet type
                name = parquet_file.stem
                for parquet_type in ['ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg']:
                    if name.endswith(f'_{parquet_type}'):
                        base_name = name[:-len(f'_{parquet_type}')]
                        if base_name not in parquet_files:
                            parquet_files[base_name] = set()
                        parquet_files[base_name].add(parquet_type)
                        break
        
        # Find failed files in project directory
        for failed_file in project_dir.glob("*.failed"):
            failed_files.add(failed_file.stem)
        
        # Find unconverted files in this project
        for raw_file in raw_files:
            base_name = raw_file.stem
            
            # Skip if conversion failed
            if base_name in failed_files:
                continue
            
            # Check for mzML, h5, and expected parquet files
            has_mzml = base_name in mzml_files
            has_h5 = base_name in h5_files
            
            expected_parquet = set(parse_filename_for_expected_parquet(str(raw_file)))
            existing_parquet = parquet_files.get(base_name, set())
            has_all_parquet = expected_parquet.issubset(existing_parquet)
            
            # File needs conversion if missing any output
            if not (has_mzml and has_h5 and has_all_parquet):
                all_unconverted.append(str(raw_file))
                logger.debug(f"Need conversion: {raw_file.name} - mzml:{has_mzml}, "
                            f"h5:{has_h5}, parquet:{existing_parquet} (expected:{expected_parquet})")
    
    all_unconverted = sorted(all_unconverted)
    logger.info(f"Found {len(all_unconverted)} unconverted .raw files across all projects")
    
    return all_unconverted


def process_single_file(raw_file):
    """
    Process a single .raw file through the complete pipeline:
    raw -> mzML -> (h5 + parquet)
    
    Returns a dictionary with results and any errors.
    """
    result = {
        'raw_file': raw_file,
        'success': False,
        'mzml_created': False,
        'h5_created': False,
        'parquet_created': False,
        'errors': []
    }
    
    #logger.info(f"Processing: {raw_file}")
    
    # Step 1: raw to mzML
    success, mzml_file, error = raw_to_mzml(raw_file)
    if not success:
        result['errors'].append(f"raw->mzML: {error}")
        return result
    
    result['mzml_created'] = True
    
    # Step 2: mzML to H5 and Parquet (single pass)
    success, errors = mzml_to_h5_and_parquet(mzml_file)
    
    if not success:
        result['errors'].extend(errors)
    else:
        # Check which files were actually created
        mzml_path = Path(mzml_file)
        project_dir = mzml_path.parent.parent
        h5_dir = project_dir / "h5"
        parquet_dir = project_dir / "parquet"
        
        h5_path = h5_dir / mzml_path.with_suffix('.h5').name
        result['h5_created'] = h5_path.exists()
        
        expected_parquet = set(parse_filename_for_expected_parquet(str(mzml_file)))
        base_name = mzml_path.stem
        output_prefix = str(parquet_dir / base_name)
        parquet_files = {
            'ms1_pos': f'{output_prefix}_ms1_pos.parquet',
            'ms1_neg': f'{output_prefix}_ms1_neg.parquet',
            'ms2_pos': f'{output_prefix}_ms2_pos.parquet',
            'ms2_neg': f'{output_prefix}_ms2_neg.parquet',
        }
        existing_parquet = {k for k in expected_parquet if Path(parquet_files[k]).exists()}
        result['parquet_created'] = len(existing_parquet) == len(expected_parquet)
    
    result['success'] = result['h5_created'] and result['parquet_created']
    
    return result


def main():
    """Main entry point for the consolidated conversion script."""
    parser = argparse.ArgumentParser(
        description='Convert .raw files to mzML, HDF5, and Parquet formats'
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
    top_dir = f"{RAW_FILES_BASE}/{args.directory}"
    log_file = f"{LOG_FILE_BASE}/{args.directory}.log"

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
    
    # Find unconverted files across all projects
    unconverted_files = find_unconverted_files(top_dir)
    
    if not unconverted_files:
        logger.info("No files to convert")
        return 0
    
    if args.dry_run:
        logger.info(f"Would process {len(unconverted_files)} files:")
        for f in unconverted_files:
            logger.info(f"  {f}")
        return 0
    
    # Pull shifter image once before spawning workers
    subprocess.run(["shifterimg", "pull", RAW_IMAGE], check=True, capture_output=True)

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
    logger.info(f"Conversion complete: {successful}/{len(unconverted_files)} files processed successfully")
    
    if failed_files:
        logger.warning(f"Failed files ({len(failed_files)}):")
        for f in failed_files:
            logger.warning(f"  {f}")
    
    logger.info("Conversion process finished.")
    return 0 if not failed_files else 1


if __name__ == '__main__':
    sys.exit(main())