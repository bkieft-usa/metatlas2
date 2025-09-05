import pandas as pd
import numpy as np
import yaml
import ast
import getpass
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
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
        logger.info(f"    Reference DataFrame shape: {df.shape}")
        logger.info(f"    Number of unique InChI keys: {df['inchi_key'].nunique()}")
        return df
    else:
        logger.info("    Reference DataFrame is empty")
        return None

def load_metatlas_config(config_path: str) -> Dict[str, Any]:
    """Load and validate metatlas configuration from YAML file with type enforcement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define expected structure and types
    required_sections = ['paths', 'rt_alignment', 'analysis_settings', 'database_options']
    
    # Validate required sections exist
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate and convert data types
    if 'tolerances' in config['rt_alignment']:
        tolerances = config['rt_alignment']['tolerances']
        tolerances['mz'] = float(tolerances.get('mz', 10.0))
        tolerances['rt'] = float(tolerances.get('rt', 0.5))
        tolerances['i'] = float(tolerances.get('i', 1000.0))
    
    if 'analysis_settings' in config:
        settings = config['analysis_settings']
        settings['default_ppm_error'] = float(settings.get('default_ppm_error', 20.0))
        settings['extra_time'] = float(settings.get('extra_time', 0.1))
    
    return config

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
    try:
        df = pd.read_csv(file_path, sep='\t')
    except:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")
    
    # Check for required columns
    required_columns = ['inchi_key', 'label']
    check_missing_columns(df, required_columns)
    
    logger.info(f"Loaded {len(df)} compounds from {file_path}")
    return df

def detect_atlas_input_chromatography(df: pd.DataFrame) -> str:
    """Detect chromatography type from atlas input data."""
    if 'chromatography' in df.columns:
        chrom_values = df['chromatography'].dropna().unique()
        if len(chrom_values) > 0:
            return str(chrom_values[0])
    
    # Try to infer from compound names or other columns
    compound_names = ' '.join(df.get('label', df.get('compound_name', [''])).astype(str))
    if 'HILIC' in compound_names.upper():
        return 'HILIC'
    elif 'C18' in compound_names.upper():
        return 'C18'
    
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
    try:
        df = pd.read_csv(file_path, sep='\t')
    except:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")
    
    # Check for required columns for atlas creation
    required_columns = ['inchi_key', 'label', 'rt_peak', 'mz', 'adduct']
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