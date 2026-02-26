import pandas as pd
import numpy as np
import yaml
import ast
import getpass
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
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
        logger.info("    Reference DataFrame is empty")
        return None

def load_metatlas2_config(config_path: str) -> Dict[str, Any]:
    """Load and validate new metatlas2 configuration from YAML file with type enforcement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define expected top-level structure
    required_sections = ['ENV', 'WORKFLOWS']
    
    # Validate required sections exist
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate ENV section
    if 'PATHS' not in config['ENV']:
        raise ValueError("Missing 'PATHS' section in ENV configuration")
    required_paths = ['projects_dir', 'main_database', 'msms_refs']
    for path_key in required_paths:
        if path_key not in config['ENV']['PATHS']:
            raise ValueError(f"Missing required path: {path_key}")
    
    # Validate WORKFLOWS section structure
    workflows = config['WORKFLOWS']
    
    # Validate RT_ALIGNMENT section if present
    if 'RT_ALIGNMENT' in workflows:
        rt_alignment = workflows['RT_ALIGNMENT']
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
    
    # Validate TARGETED_ANALYSES section if present
    if 'TARGETED_ANALYSES' in workflows:
        targeted = workflows['TARGETED_ANALYSES']
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
                        params['use_existing_hits'] = bool(params.get('use_existing_hits', False))
                        params['use_existing_analysis'] = bool(params.get('use_existing_analysis', False))
                        params['default_ppm_error'] = float(params.get('default_ppm_error', 5.0))
                        params['min_peak_intensity'] = float(params.get('min_peak_intensity', 100000.0))
                        params['extra_time'] = float(params.get('extra_time', 0.0))
                        params['ms2_min_score'] = float(params.get('ms2_min_score', 0.1))
                        params['ms2_min_matches'] = int(params.get('ms2_min_matches', 1))
    
    # Add config path for reference
    config['ENV']['PATHS']['config_path'] = str(Path(config_path).resolve())
    
    logger.info(f"Loaded metatlas2 configuration from {config_path}")
    
    return config

def load_atlas_config(atlas_config_path: str) -> Dict[str, Any]:
    """Load and validate atlas configuration from YAML file with type enforcement."""
    with open(atlas_config_path, 'r') as f:
        atlas_config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['ENV', 'ATLASES']
    for field in required_fields:
        if field not in atlas_config:
            raise ValueError(f"Missing required atlas configuration field: {field}")

    # Validate ENV section
    if 'PATHS' not in atlas_config['ENV']:
        raise ValueError("Missing 'PATHS' section in ENV configuration")
    required_paths = ['main_database']
    for path_key in required_paths:
        if path_key not in atlas_config['ENV']['PATHS']:
            raise ValueError(f"Missing required path: {path_key}")

    # Validate ATLASES section structure
    atlases = atlas_config['ATLASES']
    
    for chromatography, chrom_config in atlases.items():
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
                    logger.warning(f"Atlas file not found: {atlas_info['path']} for {chromatography}/{polarity}/{atlas_type}")

    logger.info(f"Loaded atlas configuration from {atlas_config_path}")
    
    return atlas_config

def load_compound_config(compound_config_path: str) -> Dict[str, Any]:
    """Load and validate compound configuration from YAML file with type enforcement."""
    with open(compound_config_path, 'r') as f:
        compound_config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['ENV', 'COMPOUNDS']
    for field in required_fields:
        if field not in compound_config:
            raise ValueError(f"Missing required compound configuration field: {field}")

    # Validate ENV section
    if 'PATHS' not in compound_config['ENV']:
        raise ValueError("Missing 'PATHS' section in ENV configuration")
    required_paths = ['main_database', 'pubchem_cache']
    for path_key in required_paths:
        if path_key not in compound_config['ENV']['PATHS']:
            raise ValueError(f"Missing required path: {path_key}")

    return compound_config

def get_provenance():
    """Get provenance information for database records."""
    return {
        "analyst": getpass.getuser(),
        "timestamp": datetime.now().isoformat()
    }

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