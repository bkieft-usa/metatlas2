import pandas as pd
import numpy as np
from datetime import datetime
import getpass
import yaml
from typing import Dict, Any, List, Union
import ast

def load_msms_refs_file(file_path):
    """
    Load the msms_refs.tab file and convert it to a DataFrame format suitable for MS2 matching.
    
    Args:
        file_path: Path to the msms_refs.tab file
        
    Returns:
        DataFrame with columns: ['database', 'id', 'name', 'spectrum', 'collision_energy', 
                                'precursor_mz', 'polarity', 'adduct', 'formula', 'exact_mass', 
                                'inchi_key', 'inchi', 'smiles']
    """
    
    print(f"Loading reference spectra from {file_path}...")

    # Read the tab-separated file
    df = pd.read_csv(file_path, sep='\t', header=None, names=[
        'id', 'database', 'compound_id', 'name', 'spectrum', 'collision_energy', 
        'precursor_mz', 'polarity', 'adduct', 'fragmentation_method', 'other_id', 
        'experiment', 'instrument', 'formula', 'exact_mass', 'inchi_key', 'inchi', 'smiles'
    ])

    # Convert spectrum strings to numpy arrays
    def parse_spectrum(spec_str):
        try:
            # Parse the string representation of the spectrum
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
    df['exact_mass'] = pd.to_numeric(df['exact_mass'], errors='coerce')

    if not df.empty:
        print(f"    Reference DataFrame shape: {df.shape}")
        print(f"    Number of unique InChI keys: {df['inchi_key'].nunique()}")
        return df
    else:
        print("    Reference DataFrame is empty")
        return None

def load_metatlas_config(config_path: str) -> Dict[str, Any]:
    """Load and validate metatlas configuration from YAML file with type enforcement."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define expected structure and types
    expected_schema = {
        'paths': {
            'projects_dir': str,
            'main_database': str,
            'msms_refs': str,
            'pubchem_cache': str
        },
        'rt_alignment': {
            'tolerances': {
                'i': (int, float),
                'mz': (int, float),
                'rt': (int, float)
            },
            'model': {
                'polynomial_degree': int,
                'min_observations_per_compound': int,
                'min_compounds_for_modeling': int,
                'r2_threshold': (int, float),
                'exclude_inchikeys': list
            }
        },
        'analysis_settings': {
            'default_ppm_error': (int, float),
            'min_peak_intensity': (int, float),
            'extra_time': (int, float),
            'ms2_min_score': (int, float),
            'ms2_min_matches': int
        },
        'database_options': {
            'overwrite_existing_main_db': bool,
            'overwrite_existing_project_db': bool,
            'add_compound_duplicates': bool,
            'force_pubchem_cache_update': bool
        },
        'plot_settings': {
            'file_color_mapping': dict
        }
    }
    
    def validate_config_section(config_data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> None:
        """Recursively validate configuration against expected schema."""
        for key, expected_type in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config_data:
                raise ValueError(f"Missing required config key: {current_path}")
            
            value = config_data[key]
            
            if isinstance(expected_type, dict):
                # Nested dictionary - recurse
                if not isinstance(value, dict):
                    raise TypeError(f"Config key '{current_path}' should be a dictionary, got {type(value).__name__}")
                validate_config_section(value, expected_type, current_path)
            
            elif isinstance(expected_type, tuple):
                # Multiple allowed types
                if not isinstance(value, expected_type):
                    type_names = [t.__name__ for t in expected_type]
                    raise TypeError(f"Config key '{current_path}' should be one of {type_names}, got {type(value).__name__}")
            
            else:
                # Single type
                if not isinstance(value, expected_type):
                    raise TypeError(f"Config key '{current_path}' should be {expected_type.__name__}, got {type(value).__name__}")
                
                # Additional validation for specific types
                if expected_type == list and key == 'exclude_inchikeys':
                    # Validate that all items in exclude_inchikeys are strings
                    for i, item in enumerate(value):
                        if not isinstance(item, str):
                            raise TypeError(f"Config key '{current_path}[{i}]' should be str, got {type(item).__name__}")
    
    # Validate the loaded config
    validate_config_section(config, expected_schema)
    
    print(f"Successfully loaded and validated config from: {config_path}")
    return config

def get_provenance():
    return {
        "timestamp": datetime.now().isoformat(),
        "analyst": getpass.getuser()
    }

def load_compound_input(file_path: str) -> pd.DataFrame:
    """Load compound data from a CSV or TSV file."""
    print(f"Loading input data from: {file_path}")
    delimiter = '\t' if file_path.endswith(('.tsv', '.tab', '.txt')) else ','
    df = pd.read_csv(file_path, sep=delimiter)
    
    check_missing_columns(df, ['inchi_key', 'label'])

    print(f"    Loaded {len(df)} rows from input table")
    return df

def detect_atlas_input_chromatography(df: pd.DataFrame) -> str:
    """Detect the chromatography method from the input DataFrame."""
    chrom = df['chromatography'].dropna()
    if chrom.empty:
        raise ValueError("No chromatography information found.")
    if len(chrom.unique()) > 1:
        raise ValueError(f"Multiple chromatography methods found: {chrom.unique()}")
    return chrom.iloc[0]

def detect_atlas_input_polarity(df: pd.DataFrame) -> str:
    """Detect the polarity from the input DataFrame."""
    pol = df['polarity'].dropna()
    if pol.empty:
        raise ValueError("No polarity information found.")
    if len(pol.unique()) > 1:
        raise ValueError(f"Multiple polarities found: {pol.unique()}")
    return pol.iloc[0]

def check_missing_columns(df: pd.DataFrame, required_columns: list) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def load_atlas_input(file_path: str) -> pd.DataFrame:
    """Load compound data from a CSV or TSV file."""
    print(f"Loading input data from: {file_path}")
    delimiter = '\t' if file_path.endswith(('.tsv', '.tab', '.txt')) else ','
    df = pd.read_csv(file_path, sep=delimiter)

    if 'label' not in df.columns:
        print("    Missing 'label' column, so adding it by copying 'inchi_key' column")
        df['label'] = df['inchi_key']

    check_missing_columns(df, ['inchi_key', 'chromatography', 'polarity', 'rt_peak', 'mz', 'adduct', 'label'])

    detected_chrom = detect_atlas_input_chromatography(df)
    detected_pol = detect_atlas_input_polarity(df)

    print(f"    Detected chromatography: {detected_chrom}")
    print(f"    Detected polarity: {detected_pol}")

    print(f"    Loaded {len(df)} rows from input table")
    return df