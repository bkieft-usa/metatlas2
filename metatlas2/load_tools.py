import pandas as pd
import numpy as np
import os
import sys
import csv
import yaml
import ast
from pathlib import Path
from typing import Dict, Any
import grp
import subprocess
import joblib
from matchms import Spectrum
from tqdm.auto import tqdm

import metatlas2.logging_config as lcf
logger = lcf.get_logger('load_tools')

import ast
import json
import pandas as pd
from pathlib import Path

PRECURSOR_FILTER_OFFSET_DA = 2.5  # keep peaks below precursor + 2.5 Da (covers M+2 isotope)

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

def tsv_to_jsonl(tsv_path: str, jsonl_path: str) -> None:
    """Convert a legacy .tab msms refs file to .jsonl format."""
    
    input_file_path = Path(tsv_path)
    output_file_path = Path(jsonl_path)

    col_names = [
        'ix', 'database', 'id', 'name', 'spectrum', 'decimal', 'precursor_mz', 'polarity', 'adduct', 'fragmentation_method', 
        'collision_energy', 'instrument', 'instrument_type', 'formula', 'exact_mass', 'inchi_key', 'inchi', 'smiles'
    ]
    df = pd.read_csv(input_file_path, sep='\t', header=None, names=col_names, low_memory=False)

    n_written = 0
    n_skipped = 0

    with output_file_path.open('w') as out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting spectra to JSONL", disable=should_disable_tqdm()):
            try:
                spectrum_data = ast.literal_eval(row['spectrum'])
                mz_list, int_list = spectrum_data[0], spectrum_data[1]
                assert len(mz_list) == len(int_list)
            except Exception:
                n_skipped += 1
                continue

            def _float_or_none(val):
                try:
                    f = float(val)
                    return None if pd.isna(f) else f
                except (TypeError, ValueError):
                    return None

            def _str_or_none(val):
                s = str(val).strip()
                return None if s in ('', 'nan', 'None') else s
            
            def _int_or_none(val):
                try:
                    i = int(val)
                    return None if pd.isna(i) else i
                except (TypeError, ValueError):
                    return None

            rec = {
                'ix': _int_or_none(row['ix']),
                'database': _str_or_none(row['database']),
                'id': _str_or_none(row['id']),
                'name': _str_or_none(row['name']),
                'decimal': _float_or_none(row['decimal']),
                'inchi_key': _str_or_none(row['inchi_key']),
                'precursor_mz': _float_or_none(row['precursor_mz']),
                'polarity': _str_or_none(row['polarity']),
                'adduct': _str_or_none(row['adduct']),
                'fragmentation_method': _str_or_none(row['fragmentation_method']),
                'collision_energy': _str_or_none(row['collision_energy']),
                'instrument': _str_or_none(row['instrument']),
                'instrument_type': _str_or_none(row['instrument_type']),
                'formula': _str_or_none(row['formula']),
                'mono_isotopic_molecular_weight': _float_or_none(row['exact_mass']),
                'inchi': _str_or_none(row['inchi']),
                'smiles': _str_or_none(row['smiles']),
                'mz': [round(x, 5) for x in mz_list],
                'intensities': [round(x, 5) for x in int_list],
            }
            out.write(json.dumps(rec) + '\n')
            n_written += 1

    print(f"Wrote {n_written} spectra, skipped {n_skipped}.")

def load_msms_refs_file(
    file_path: str,
    database_filter: str | None = None,
    inchi_keys: list[str] | None = None,
) -> dict[str, list[Spectrum]]:
    """
    Load an msms refs file (.jsonl format) and return a dict mapping
    inchi_key -> list of matchms Spectrum objects.
    Results are cached to a .pkl file and reloaded on subsequent calls
    if the source file has not been modified.

    Args:
        file_path: Path to the msms_refs.jsonl file
        database_filter: If provided, only load spectra where database == this string
        inchi_keys: If provided, only load spectra for the specified inchi_keys
    Returns:
        dict mapping inchi_key (str) -> list of matchms Spectrum objects
    """

    file_path = Path(file_path)
    logger.info(
        f"Loading reference spectra from {file_path}"
        + (f" (database='{database_filter}')" if database_filter else "")
        + (f" (filtered to {len(inchi_keys)} inchi_keys)" if inchi_keys is not None else "")
        + "..."
    )


    refs_by_inchi_key: dict[str, list[Spectrum]] = {}
    n_skipped = 0
    inchi_key_set = set(inchi_keys) if inchi_keys is not None else None

    with file_path.open('r') as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"  Line {line_num}: JSON parse error — {e}")
                n_skipped += 1
                continue


            if database_filter and rec.get('database') != database_filter:
                continue

            rec_inchi_key = rec.get('inchi_key', '')
            if inchi_key_set is not None and rec_inchi_key not in inchi_key_set:
                continue

            mz_list = rec.get('mz')
            int_list = rec.get('intensities')
            if not mz_list or not int_list or len(mz_list) != len(int_list):
                n_skipped += 1
                continue

            # Direct construction from JSON arrays — no eval, no string parsing
            mz = np.array(mz_list, dtype=np.float32)
            intensities = np.array(int_list, dtype=np.float32)

            precursor_mz = rec.get('precursor_mz')
            try:
                precursor_mz = float(rec.get('precursor_mz')) if rec.get('precursor_mz') is not None else None
            except (TypeError, ValueError):
                precursor_mz = None
            if precursor_mz is not None and not np.isnan(precursor_mz):
                mask = mz < precursor_mz + PRECURSOR_FILTER_OFFSET_DA
                mz = mz[mask]
                intensities = intensities[mask]

            if len(mz) == 0:
                n_skipped += 1
                continue

            inchi_key = rec_inchi_key
            # matchms requires m/z ascending; defensively sort
            if len(mz) > 1 and not np.all(mz[:-1] <= mz[1:]):
                order = np.argsort(mz, kind='stable')
                mz = mz[order]
                intensities = intensities[order]
            spec = Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={
                    'precursor_mz': float(precursor_mz) if precursor_mz is not None else 0.0,
                    'database': str(rec.get('database', '')),
                    'id': str(rec.get('id', '')),
                    'name': str(rec.get('name', '')),
                    'inchi_key': inchi_key,
                }
            )
            refs_by_inchi_key.setdefault(inchi_key, []).append(spec)

    if not refs_by_inchi_key:
        if database_filter:
            raise ValueError(
                f"No spectra matched database_filter={database_filter!r} in {file_path}. "
                f"Check the filter value or the 'database' field in the source file."
            )
        raise ValueError(f"Reference file {file_path} is empty after parsing.")

    total = sum(len(v) for v in refs_by_inchi_key.values())
    logger.info(f"  Loaded {total} reference spectra for {len(refs_by_inchi_key)} unique InChI keys.")
    if n_skipped > 0:
        logger.warning(f"  Skipped {n_skipped} rows due to unparseable or empty spectra.")

    return refs_by_inchi_key

def load_metatlas2_config(config_path: str) -> Dict[str, Any]:
    """Load and validate new metatlas2 configuration from YAML file with type enforcement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define expected top-level structure
    required_sections = ['WORKFLOWS']
    required_subsections = ["RT_ALIGNMENT", "TARGETED_ANALYSES"]
    
    # Validate required sections exist
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate required subsections in WORKFLOWS
    for subsection in required_subsections:
        if subsection not in config['WORKFLOWS']:
            raise ValueError(f"Missing required WORKFLOWS subsection: {subsection}")
    
    # Validate RT_ALIGNMENT section if present
    if 'RT_ALIGNMENT' in config['WORKFLOWS']:
        rt_alignment = config['WORKFLOWS']['RT_ALIGNMENT']
        for chromatography, chrom_config in rt_alignment.items():
            if 'ATLAS' not in chrom_config:
                raise ValueError(f"RT_ALIGNMENT {chromatography} missing ATLAS section")
            
            if 'uid' not in chrom_config['ATLAS']:
                raise ValueError(f"RT_ALIGNMENT {chromatography} missing ATLAS uid field")
            
            uid = chrom_config['ATLAS']['uid']
            chrom_config['ATLAS']['uid'] = str(uid) if uid else None
            
            if 'PARAMS' in chrom_config:
                params = chrom_config['PARAMS']
                params['ppm_error'] = float(params.get('ppm_error', 20.0))
                params['extra_time'] = float(params.get('extra_time', 5.0))
                params['polynomial_degree'] = int(params.get('polynomial_degree', 2))
                params['min_observations_per_compound'] = int(params.get('min_observations_per_compound', 1))
                params['min_compounds_for_modeling'] = int(params.get('min_compounds_for_modeling', 2))
                params['r2_threshold'] = float(params.get('r2_threshold', 0.7))
                params['apply_model_to_min_max'] = bool(params.get('apply_model_to_min_max', True))
                params['use_existing_rt_alignment'] = bool(params.get('use_existing_rt_alignment', False))

                params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else []
                params['exclude_lcmsruns'] = list(params['exclude_lcmsruns']) if params.get('exclude_lcmsruns') else []

                params['exclude_inchikeys'] = list(params['exclude_inchikeys']) if params.get('exclude_inchikeys') else []

    if 'TARGETED_ANALYSES' in config['WORKFLOWS']:
        targeted = config['WORKFLOWS']['TARGETED_ANALYSES']
        for chromatography, chrom_config in targeted.items():
            for polarity, pol_config in chrom_config.items():
                for analysis_name, analysis_config in pol_config.items():
                    if 'ATLAS' not in analysis_config:
                        raise ValueError(f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name} missing ATLAS section")
                    if 'uid' not in analysis_config['ATLAS']:
                        raise ValueError(f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name} missing ATLAS uid field")
                    uid = analysis_config['ATLAS']['uid']
                    analysis_config['ATLAS']['uid'] = str(uid) if uid else None
                    if 'PARAMS' in analysis_config:
                        params = analysis_config['PARAMS']
                        params['do_alignment'] = bool(params.get('do_alignment', True))
                        params['create_curation_notebooks'] = bool(params.get('create_curation_notebooks', True))
                        params['upload_to_gdrive'] = bool(params.get('upload_to_gdrive', True))
                        params['remove_unided_compounds'] = bool(params.get('remove_unided_compounds', True))
                        params['remove_flagged_compounds'] = bool(params.get('remove_flagged_compounds', True))
                        params['ppm_error'] = float(params.get('ppm_error', 5.0))
                        params['extra_time'] = float(params.get('extra_time', 5.0))
                        params['ms1_min_peak_intensity'] = float(params.get('ms1_min_peak_intensity', 1e5))
                        params['ms1_min_num_points'] = int(params.get('ms1_min_num_points', 5))
                        params['ms2_min_score'] = float(params.get('ms2_min_score', 0.1))
                        params['ms2_min_matching_frags'] = int(params.get('ms2_min_matching_frags', 1))
                        params['ms2_frag_mz_tolerance'] = float(params.get('ms2_frag_mz_tolerance', 0.05))
                        params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else []
                        excl = params.get('exclude_lcmsruns')
                        if excl is None:
                            params['exclude_lcmsruns'] = {}
                        elif isinstance(excl, dict):
                            params['exclude_lcmsruns'] = {
                                step: list(runs) if runs else []
                                for step, runs in excl.items()
                            }
                        elif isinstance(excl, list):
                            params['exclude_lcmsruns'] = {'data_extraction': list(excl)}
                        else:
                            raise ValueError(
                                f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name}: "
                                f"exclude_lcmsruns must be a dict or list"
                            )
                        params['gui_require_all_evaluated'] = bool(params.get('gui_require_all_evaluated', True))
                        params['gui_top_n_hits'] = int(params.get('gui_top_n_hits', 20))
                        gui_colors = params.get('gui_lcmsruns_colors')
                        params['gui_lcmsruns_colors'] = dict(gui_colors) if gui_colors else {}

                        # Handle note_options_overrides: allow blank, None, or 'default' to mean use GUI defaults
                        note_overrides = params.get('note_options_overrides')
                        if not isinstance(note_overrides, dict):
                            params['note_options_overrides'] = {}
                        else:
                            clean_overrides = {}
                            for note_type in ['ms1_notes', 'ms2_notes', 'other_notes']:
                                val = note_overrides.get(note_type, None)
                                if val is None:
                                    continue  # No override for this note type
                                if isinstance(val, dict):
                                    clean_overrides[note_type] = {str(k): str(v) for k, v in val.items()}
                                    continue
                            params['note_options_overrides'] = clean_overrides
    return config

def load_atlas_config(atlas_config_path: str) -> Dict[str, Any]:
    """Load and validate atlas configuration from YAML file with type enforcement."""
    with open(atlas_config_path, 'r') as f:
        atlas_config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['ATLASES']
    for field in required_fields:
        if field not in atlas_config:
            raise ValueError(f"Missing required atlas configuration field: {field}")

    # Validate ATLASES section structure    
    for chromatography, chrom_config in atlas_config['ATLASES'].items():
        if not isinstance(chrom_config, dict):
            raise ValueError(f"Invalid chromatography configuration for {chromatography}")
        
        for polarity, pol_config in chrom_config.items():
            if not isinstance(pol_config, dict):
                raise ValueError(f"Invalid polarity configuration for {chromatography}/{polarity}")
            
            for atlas_type, atlas_info in pol_config.items():
                if isinstance(atlas_info, dict):
                    atlas_entries = [atlas_info]
                elif isinstance(atlas_info, list):
                    atlas_entries = atlas_info
                else:
                    raise ValueError(f"Invalid atlas configuration for {chromatography}/{polarity}/{atlas_type}")

                normalized_entries = []
                for atlas_entry in atlas_entries:
                    if not isinstance(atlas_entry, dict):
                        raise ValueError(f"Invalid atlas entry for {chromatography}/{polarity}/{atlas_type}")

                    # Check for required atlas fields (path, name, desc) but allow None/empty
                    required_atlas_fields = ['path', 'name', 'desc']
                    for field in required_atlas_fields:
                        if field not in atlas_entry:
                            raise ValueError(
                                f"Missing required field '{field}' in {chromatography}/{polarity}/{atlas_type}"
                            )

                    # Convert to strings and handle None/empty values
                    atlas_entry['path'] = str(atlas_entry['path']) if atlas_entry['path'] else None
                    atlas_entry['name'] = str(atlas_entry['name']) if atlas_entry['name'] else None
                    atlas_entry['desc'] = str(atlas_entry['desc']) if atlas_entry['desc'] else None

                    # # Optional label for logging or downstream filtering
                    # if 'label' in atlas_entry and atlas_entry['label']:
                    #     atlas_entry['label'] = str(atlas_entry['label'])

                    # Only check file existence if path is not None
                    if atlas_entry['path'] and not Path(atlas_entry['path']).exists():
                        raise FileNotFoundError(
                            f"Atlas file not found: {atlas_entry['path']} for {chromatography}/{polarity}/{atlas_type}"
                        )

                    normalized_entries.append(atlas_entry)

                atlas_info = normalized_entries if len(normalized_entries) > 1 else normalized_entries[0]
                pol_config[atlas_type] = atlas_info

    logger.info(f"Loaded atlas configuration from {atlas_config_path}")
    
    return atlas_config

def load_compound_config(compound_config_path: str) -> Dict[str, Any]:
    """Load and validate compound configuration from YAML file with type enforcement."""
    with open(compound_config_path, 'r') as f:
        compound_config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['PARAMS', 'COMPOUNDS']
    for field in required_fields:
        if field not in compound_config:
            raise ValueError(f"Missing required compound configuration field: {field}")

    # Validate PARAMS section
    params = compound_config['PARAMS']
    params['use_pubchem_cache'] = bool(params.get('use_pubchem_cache', True))
    params['update_pubchem_cache'] = bool(params.get('update_pubchem_cache', False))

    # Validate COMPOUNDS section
    for chromatography, chrom_config in compound_config['COMPOUNDS'].items():
        if not isinstance(chrom_config, dict):
            raise ValueError(f"Invalid chromatography configuration for {chromatography}")

        for polarity, pol_config in chrom_config.items():
            if not isinstance(pol_config, dict):
                raise ValueError(f"Invalid polarity configuration for {chromatography}/{polarity}")

            if 'PATHS' not in pol_config:
                raise ValueError(f"Missing PATHS in {chromatography}/{polarity}")

            if not isinstance(pol_config['PATHS'], list):
                raise ValueError(f"PATHS must be a list in {chromatography}/{polarity}")

            # Normalize paths: filter None/empty, convert to strings, warn if missing
            validated_paths = []
            for path in pol_config['PATHS']:
                if not path:
                    continue
                path_str = str(path)
                if not Path(path_str).exists():
                    raise FileNotFoundError(f"Compound input file not found: {path_str} for {chromatography}/{polarity}")
                validated_paths.append(path_str)

            pol_config['PATHS'] = validated_paths

    logger.info(f"Loaded compound configuration from {compound_config_path}")

    return compound_config

def load_compound_input(file_path: str) -> pd.DataFrame:
    """Load compound input file (TSV/CSV) and validate required columns."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Compound input file not found: {file_path}")
    
    # Try to read as TSV first, then CSV
    if file_path.suffix.lower() != '.csv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    # Check for required columns (need to use label if it's present)
    if 'compound_name' in df.columns and 'label' in df.columns:
        df.rename(columns={'compound_name': 'compound_name_input', 'label': 'compound_name'}, inplace=True)
    elif 'compound_name' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'compound_name'})
    required_columns = ['inchi_key', 'compound_name']
    check_missing_columns(df, required_columns)
    
    logger.info(f"Loaded {len(df)} compounds from {file_path}")
    return df

def detect_atlas_input_chromatography(df: pd.DataFrame) -> str:
    """Detect chromatography type from atlas input data."""
    if 'chromatography' in df.columns:
        chrom_values = df['chromatography'].dropna().unique()
        return str(chrom_values[0])
    
    return 'Unknown'

def detect_atlas_input_polarity(df: pd.DataFrame) -> str:
    """Detect polarity from atlas input data."""
    if 'polarity' in df.columns:
        pol_values = df['polarity'].dropna().unique()
        if len(pol_values) > 0:
            return str(pol_values[0])
    
    # Try to infer from adduct information
    if 'adduct' in df.columns:
        adducts = ' '.join(df['adduct'].dropna().astype(str))
        if '+' in adducts and '[M+H]+' in adducts:
            return 'positive'
        elif '-' in adducts and '[M-H]-' in adducts:
            return 'negative'
    
    return 'positive'  # Default

def check_missing_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Check for missing required columns and raise error if any are missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def load_atlas_input(file_path: str) -> pd.DataFrame:
    """Load atlas input file and validate required columns."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Atlas input file not found: {file_path}")
    
    # Try to read as TSV first, then CSV
    if file_path.suffix.lower() != '.csv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    # Check for required columns for atlas creation (need to use label if it's present)
    if 'compound_name' in df.columns and 'label' in df.columns:
        df.rename(columns={'compound_name': 'compound_name_input', 'label': 'compound_name'}, inplace=True)
    elif 'compound_name' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'compound_name'})
    required_columns = ['inchi_key', 'compound_name', 'rt_peak', 'mz', 'adduct']
    check_missing_columns(df, required_columns)
    
    # Add default values for optional columns
    if 'rt_min' not in df.columns:
        df['rt_min'] = df['rt_peak'] - 0.5
    if 'rt_max' not in df.columns:
        df['rt_max'] = df['rt_peak'] + 0.5
    if 'mz_tolerance' not in df.columns:
        df['mz_tolerance'] = 5.0
    
    logger.info(f"Loaded {len(df)} atlas entries from {file_path}")
    
    return df

def save_atlas_data_to_csv(atlas_obj: "Atlas", output_path: str) -> None:
    """Save Atlas data to CSV file."""
    logger.info(f"Saving Atlas data to {output_path}...")
    atlas_info = atlas_obj.to_dict()
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=atlas_info.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(atlas_info)

def load_atlas_data_from_csv(file_path: str) -> pd.DataFrame:
    """Load Atlas data from CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Atlas data file not found: {file_path}")
    return pd.read_csv(file_path)

def change_ownership_to_metatlas_group(project_dir_path: str) -> None:
    """Change ownership of project directory to metatlas group for HPC environments."""

    group_name = 'metatlas'

    try:
        grp.getgrnam(group_name)
    except KeyError:
        logger.warning(f"Group '{group_name}' not found. Skipping ownership change.")
        return

    try:
        subprocess.run(['chgrp', '-R', group_name, project_dir_path], check=True)
        logger.info(f"Changed group of {project_dir_path} to {group_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to change group: {e}")