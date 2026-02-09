import glob
import os
import re
import sys
import pandas as pd
import numpy as np
from pymzml.run import Reader
from tqdm.notebook import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('lcmsrun_tools')

def get_project_files(project_path: str) -> dict:
    """
    Scan project directory for LCMS files and organize by chromatography/polarity/analysis type.
    
    Args:
        project_path: Path to directory containing .mzML files
        
    Returns:
        Nested dictionary: {chromatography: {polarity: {analysis_type: [file_paths]}}}
    """
    project_path = Path(project_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    # Find all .mzML files
    mzML_files = list(project_path.glob("*.mzML"))
    if not mzML_files:
        raise ValueError(f"No .mzML files found in {project_path}")
    
    logger.info(f"Found {len(mzML_files)} .mzML files in {project_path}")
    
    # Initialize nested dictionary
    files_by_group = {}
    
    for file_path in tqdm(mzML_files, desc="Getting project files"):
        filename = file_path.name
        
        # Infer chromatography from filename
        if any(x in filename.upper() for x in ['HILIC', 'HILICZ']):
            chromatography = 'HILIC'
        elif any(x in filename.upper() for x in ['C18', 'RP']):
            chromatography = 'C18'
        else:
            chromatography = 'Unknown'
        
        # Infer polarity from filename
        if any(x in filename.upper() for x in ['POS', 'POSITIVE']):
            polarity = 'positive'
        elif any(x in filename.upper() for x in ['NEG', 'NEGATIVE']):
            polarity = 'negative'
        elif 'FPS' in filename.upper():
            polarity = 'FPS'
        else:
            polarity = 'Unknown'
        
        # Infer analysis type from filename
        if any(x in filename.upper() for x in ['QC']):
            analysis_type = 'qc'
        elif any(x in filename.upper() for x in ['ISTD', 'STD']):
            analysis_type = 'istd'
        elif any(x in filename.upper() for x in ['BLANK', 'BLK']):
            analysis_type = 'injbl'
        elif any(x in filename.upper() for x in ['CTRL', 'CONTROL']):
            analysis_type = 'exctrl'
        else:
            analysis_type = 'experimental'
        
        # Initialize nested structure if needed
        if chromatography not in files_by_group:
            files_by_group[chromatography] = {}
        if polarity not in files_by_group[chromatography]:
            files_by_group[chromatography][polarity] = {}
        if analysis_type not in files_by_group[chromatography][polarity]:
            files_by_group[chromatography][polarity][analysis_type] = []
        
        # Add file to appropriate category
        files_by_group[chromatography][polarity][analysis_type].append(str(file_path))
    
    return files_by_group

# def combine_dfs_across_files(file_list: List[str], key: str) -> pd.DataFrame:
#     """
#     Combine DataFrames from multiple HDF5 files based on a common key.
    
#     Args:
#         file_list: List of file paths to .mzML files
#         key: Key name to extract from each file (e.g., 'ms1_pos', 'ms1_neg')
        
#     Returns:
#         Combined pandas DataFrame
#     """
#     combined_df = pd.DataFrame()
    
#     for file in file_list:
#         try:
#             df = read_hdf_file(file, key)
#             if df is not None and not df.empty:
#                 combined_df = pd.concat([combined_df, df], ignore_index=True)
#         except Exception as e:
#             logger.info(f"Error processing file {file}: {e}")
    
#     return combined_df

def read_hdf_file(filename, desired_key=None):
    """
    Read data from HDF5 file, returning specified key or attempting common keys.
    
    Args:
        filename: Path to .h5 file
        desired_key: Specific key to extract (e.g., 'ms1_pos', 'ms1_neg')
        
    Returns:
        pandas DataFrame or None if no data found
    """
    try:
        with h5py.File(filename, 'r') as f:
            if desired_key and desired_key in f:
                # Read specific key
                data = pd.read_hdf(filename, key=desired_key)
                return data
            elif desired_key:
                # Desired key not found
                return None
            else:
                # Try common keys
                common_keys = ['ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg']
                for key in common_keys:
                    if key in f:
                        data = pd.read_hdf(filename, key=key)
                        return data
                return None
    except Exception as e:
        logger.info(f"Error reading {filename}: {e}")
        return None

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from the filename using regex.
    
    Args:
        filename: The name of the file (with extension)
        
    Returns:
        Dictionary with extracted metadata (e.g., {'chromatography': 'HILIC', 'polarity': 'positive', ...})
    """
    # Define regex pattern for extracting metadata
    pattern = r"(?P<chromatography>HILIC|C18|RP|Unknown)_(?P<polarity>POSITIVE|NEGATIVE|FPS|Unknown)_(?P<analysis_type>QC|ISTD|STD|BLANK|BLK|CTRL|CONTROL|experimental)"
    
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.groupdict()
    else:
        return {'chromatography': 'Unknown', 'polarity': 'Unknown', 'analysis_type': 'Unknown'}

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names in the DataFrame for consistency.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def filter_dataframe(df: pd.DataFrame, min_intensity: float = 0) -> pd.DataFrame:
    """
    Filter the DataFrame to remove low-intensity signals.
    
    Args:
        df: Input pandas DataFrame
        min_intensity: Minimum intensity threshold (default: 0)
        
    Returns:
        Filtered DataFrame
    """
    return df[df['intensity'] >= min_intensity]

def pivot_dataframe(df: pd.DataFrame, index_cols: List[str], value_col: str) -> pd.DataFrame:
    """
    Pivot the DataFrame to wide format.
    
    Args:
        df: Input pandas DataFrame
        index_cols: List of columns to use as index in the pivoted table
        value_col: Column containing values to fill in the pivoted table
        
    Returns:
        Pivoted DataFrame
    """
    return df.pivot_table(index=index_cols, values=value_col, aggfunc='sum').reset_index()

def save_dataframe_to_hdf(df: pd.DataFrame, file_path: str, key: str, mode: str = 'w'):
    """
    Save the DataFrame to an HDF5 file.
    
    Args:
        df: pandas DataFrame to save
        file_path: Path to the output .mzML file
        key: Key name under which to store the DataFrame
        mode: File mode ('w' for write, 'a' for append)
    """
    try:
        with pd.HDFStore(file_path, mode) as store:
            store.put(key, df, format='table', data_columns=True)
    except Exception as e:
        logger.info(f"Error saving DataFrame to {file_path}: {e}")

def load_dataframe_from_hdf(file_path: str, key: str) -> pd.DataFrame:
    """
    Load a DataFrame from an HDF5 file.
    
    Args:
        file_path: Path to the .mzML file
        key: Key name under which the DataFrame is stored
        
    Returns:
        Loaded pandas DataFrame
    """
    try:
        with pd.HDFStore(file_path, 'r') as store:
            return store[key]
    except Exception as e:
        logger.info(f"Error loading DataFrame from {file_path}: {e}")
        return pd.DataFrame()