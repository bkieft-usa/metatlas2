import pandas as pd
import numpy as np
import os
import csv
import yaml
import ast
from pathlib import Path
from typing import Dict, Any
import grp
import subprocess

import metatlas2.logging_config as lcf
logger = lcf.get_logger('load_tools')

def load_msms_refs_file(file_path):
    """
    Load the msms_refs.tab file and convert it to a DataFrame format suitable for MS2 matching.
    
    Args:
        file_path: Path to the msms_refs.tab file
        
    Returns:
        DataFrame with columns: ['database', 'id', 'name', 'spectrum', 'collision_energy', 
                                'precursor_mz', 'polarity', 'adduct', 'formula', 'mono_isotopic_molecular_weight', 
                                'inchi_key', 'inchi', 'smiles']
    """
    
    logger.info(f"Loading reference spectra from {file_path}...")

    # Read the tab-separated file with explicit column names
    df = pd.read_csv(file_path, sep='\t', header=None, names=[
        'id', 'database', 'compound_id', 'name', 'spectrum', 'collision_energy', 
        'precursor_mz', 'polarity', 'adduct', 'fragmentation_method', 'other_id', 
        'experiment', 'instrument', 'formula', 'mono_isotopic_molecular_weight', 'inchi_key', 'inchi', 'smiles'
    ])

    # Convert spectrum strings to numpy arrays using ast.literal_eval
    def parse_spectrum(spec_str):
        try:
            spectrum = ast.literal_eval(spec_str)
            if len(spectrum) == 2 and len(spectrum[0]) == len(spectrum[1]):
                return np.array(spectrum)
            else:
                return None
        except:
            return None

    df['spectrum_parsed'] = df['spectrum'].apply(parse_spectrum)

    # Remove rows with unparseable spectra
    df = df.dropna(subset=['spectrum_parsed'])
    df['spectrum'] = df['spectrum_parsed']
    df = df.drop('spectrum_parsed', axis=1)

    # Clean up data types
    df['precursor_mz'] = pd.to_numeric(df['precursor_mz'], errors='coerce')
    df['collision_energy'] = pd.to_numeric(df['collision_energy'], errors='coerce') 
    df['mono_isotopic_molecular_weight'] = pd.to_numeric(df['mono_isotopic_molecular_weight'], errors='coerce')

    if not df.empty:
        logger.info(f"    Number of total references: {df.shape[0]}")
        logger.info(f"    Number of unique InChI keys: {df['inchi_key'].nunique()}")
        return df
    else:
        raise ValueError("    Reference DataFrame is empty")

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
            
            # Check for uid field but allow None/empty
            if 'uid' not in chrom_config['ATLAS']:
                raise ValueError(f"RT_ALIGNMENT {chromatography} missing ATLAS uid field")
            
            # Convert uid to string or None
            uid = chrom_config['ATLAS']['uid']
            chrom_config['ATLAS']['uid'] = str(uid) if uid else None
            
            # Validate and convert RT alignment parameters if present
            if 'PARAMS' in chrom_config:
                params = chrom_config['PARAMS']
                params['ppm_error'] = float(params.get('ppm_error', 20.0))
                params['extra_time'] = float(params.get('extra_time', 1.0))
                params['polynomial_degree'] = int(params.get('polynomial_degree', 2))
                params['min_observations_per_compound'] = int(params.get('min_observations_per_compound', 1))
                params['min_compounds_for_modeling'] = int(params.get('min_compounds_for_modeling', 2))
                params['r2_threshold'] = float(params.get('r2_threshold', 0.7))
                params['apply_model_to_min_max'] = bool(params.get('apply_model_to_min_max', True))
                params['use_existing_rt_alignment'] = bool(params.get('use_existing_rt_alignment', False))

                # include/exclude lcmsruns: list or None
                params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else []
                params['exclude_lcmsruns'] = list(params['exclude_lcmsruns']) if params.get('exclude_lcmsruns') else []

                # exclude_inchikeys: list or empty
                params['exclude_inchikeys'] = list(params['exclude_inchikeys']) if params.get('exclude_inchikeys') else []

    # Validate TARGETED_ANALYSES section if present
    if 'TARGETED_ANALYSES' in config['WORKFLOWS']:
        targeted = config['WORKFLOWS']['TARGETED_ANALYSES']
        for chromatography, chrom_config in targeted.items():
            for polarity, pol_config in chrom_config.items():
                for analysis_name, analysis_config in pol_config.items():
                    if 'ATLAS' not in analysis_config:
                        raise ValueError(f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name} missing ATLAS section")
                    
                    # Check for uid field but allow None/empty
                    if 'uid' not in analysis_config['ATLAS']:
                        raise ValueError(f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name} missing ATLAS uid field")
                    
                    # Convert uid to string or None
                    uid = analysis_config['ATLAS']['uid']
                    analysis_config['ATLAS']['uid'] = str(uid) if uid else None
                    
                    # Validate and convert analysis parameters if present
                    if 'PARAMS' in analysis_config:
                        params = analysis_config['PARAMS']

                        # Boolean workflow flags
                        params['do_alignment'] = bool(params.get('do_alignment', True))
                        params['create_curation_notebooks'] = bool(params.get('create_curation_notebooks', True))
                        params['remove_unided_compounds'] = bool(params.get('remove_unided_compounds', True))
                        params['remove_flagged_compounds'] = bool(params.get('remove_flagged_compounds', True))

                        # MS1 parameters
                        params['ppm_error'] = float(params.get('ppm_error', 5.0))
                        params['extra_time'] = float(params.get('extra_time', 0.0))
                        params['ms1_min_peak_intensity'] = float(params.get('ms1_min_peak_intensity', 1e5))
                        params['ms1_min_num_points'] = int(params.get('ms1_min_num_points', 5))

                        # MS2 parameters
                        params['ms2_min_score'] = float(params.get('ms2_min_score', 0.1))
                        params['ms2_min_matching_frags'] = int(params.get('ms2_min_matching_frags', 1))
                        params['ms2_frag_mz_tolerance'] = float(params.get('ms2_frag_mz_tolerance', 0.05))

                        # include_lcmsruns: flat list
                        params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else []

                        # exclude_lcmsruns: nested dict of lists (keyed by workflow step)
                        # e.g. {data_extraction: [...], gui: [...], ...}
                        excl = params.get('exclude_lcmsruns')
                        if excl is None:
                            params['exclude_lcmsruns'] = {}
                        elif isinstance(excl, dict):
                            params['exclude_lcmsruns'] = {
                                step: list(runs) if runs else []
                                for step, runs in excl.items()
                            }
                        elif isinstance(excl, list):
                            # Backwards compatibility: flat list -> wrap under 'data_extraction'
                            logger.warning(
                                f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name}: "
                                f"exclude_lcmsruns is a flat list; wrapping under 'data_extraction'"
                            )
                            params['exclude_lcmsruns'] = {'data_extraction': list(excl)}
                        else:
                            raise ValueError(
                                f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_name}: "
                                f"exclude_lcmsruns must be a dict or list"
                            )

                        # GUI parameters
                        params['gui_require_all_evaluated'] = bool(params.get('gui_require_all_evaluated', True))
                        params['gui_top_n_hits'] = int(params.get('gui_top_n_hits', 20))
                        gui_colors = params.get('gui_lcmsruns_colors')
                        params['gui_lcmsruns_colors'] = dict(gui_colors) if gui_colors else {}
    
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
                if not isinstance(atlas_info, dict):
                    raise ValueError(f"Invalid atlas configuration for {chromatography}/{polarity}/{atlas_type}")
                
                # Check for required atlas fields (path, name, desc) but allow None/empty
                required_atlas_fields = ['path', 'name', 'desc']
                for field in required_atlas_fields:
                    if field not in atlas_info:
                        raise ValueError(f"Missing required field '{field}' in {chromatography}/{polarity}/{atlas_type}")
                
                # Convert to strings and handle None/empty values
                atlas_info['path'] = str(atlas_info['path']) if atlas_info['path'] else None
                atlas_info['name'] = str(atlas_info['name']) if atlas_info['name'] else None
                atlas_info['desc'] = str(atlas_info['desc']) if atlas_info['desc'] else None
                
                # Only check file existence if path is not None
                if atlas_info['path'] and not Path(atlas_info['path']).exists():
                    raise FileNotFoundError(f"Atlas file not found: {atlas_info['path']} for {chromatography}/{polarity}/{atlas_type}")

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
    
    # Check for required columns
    required_columns = ['inchi_key', 'compound_name']
    check_missing_columns(df, required_columns)
    
    logger.info(f"Loaded {len(df)} compounds from {file_path}")
    return df

def detect_atlas_input_chromatography(df: pd.DataFrame) -> str:
    """Detect chromatography type from atlas input data."""
    if 'chromatography' in df.columns:
        chrom_values = df['chromatography'].dropna().unique()
        if len(chrom_values) > 0:
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
    
    # Check for required columns for atlas creation
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