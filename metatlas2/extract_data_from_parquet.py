"""
Parquet-based feature extraction tools - drop-in replacements for feature_tools.py HDF5 functions.
"""

from __future__ import absolute_import, print_function
import os
from typing import Dict, Optional, Union
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from scipy import interpolate


def df_container_from_parquet_file(
    filename: str, 
    desired_key: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Read Parquet file(s) and return dataframe(s).
    Drop-in replacement for df_container_from_metatlas_file().
    
    Parameters
    ----------
    filename : str
        Path to parquet file OR prefix for multiple parquet files
        Examples:
          - '/path/to/sample_ms1_pos.parquet'  # Single file
          - '/path/to/sample'                  # Prefix for 4 files
    desired_key : str, optional
        One of: 'ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg'
        If None, returns dict with all 4 keys
        
    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        If desired_key is specified: single DataFrame
        If desired_key is None: dict with keys 'ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg'
    """
    # Determine if filename is a prefix or a full file path
    if filename.endswith('.parquet'):
        # Full file path provided
        prefix = filename.replace('_ms1_pos.parquet', '')\
                        .replace('_ms1_neg.parquet', '')\
                        .replace('_ms2_pos.parquet', '')\
                        .replace('_ms2_neg.parquet', '')
    else:
        # Prefix provided
        prefix = filename
    
    if desired_key is not None:
        # Return single dataframe for desired key
        parquet_file = f'{prefix}_{desired_key}.parquet'
        if not os.path.exists(parquet_file):
            # Return empty dataframe with correct schema
            if desired_key.startswith('ms1'):
                return pd.DataFrame(columns=['mz', 'i', 'rt', 'polarity'])
            else:
                return pd.DataFrame(columns=[
                    'mz', 'i', 'rt', 'polarity', 
                    'precursor_MZ', 'precursor_intensity', 'collision_energy'
                ])
        
        return pq.read_table(parquet_file).to_pandas()
    
    else:
        # Return dict of all dataframes
        df_container = {}
        for key in ['ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg']:
            parquet_file = f'{prefix}_{key}.parquet'
            if os.path.exists(parquet_file):
                df_container[key] = pq.read_table(parquet_file).to_pandas()
            else:
                # Empty dataframe with correct schema
                if key.startswith('ms1'):
                    df_container[key] = pd.DataFrame(columns=['mz', 'i', 'rt', 'polarity'])
                else:
                    df_container[key] = pd.DataFrame(columns=[
                        'mz', 'i', 'rt', 'polarity',
                        'precursor_MZ', 'precursor_intensity', 'collision_energy'
                    ])
        
        return df_container


def read_parquet_with_filters(
    filename: str,
    desired_key: str,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Read Parquet file with efficient filtering using predicate pushdown.
    This is much faster than loading everything and filtering in pandas.
    
    Parameters
    ----------
    filename : str
        Path to parquet file or prefix
    desired_key : str
        One of: 'ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg'
    mz_min, mz_max : float, optional
        m/z range filter
    rt_min, rt_max : float, optional
        Retention time range filter
    columns : list, optional
        Specific columns to read (enables column pruning)
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    # Determine file path
    if filename.endswith('.parquet'):
        prefix = filename.replace('_ms1_pos.parquet', '')\
                        .replace('_ms1_neg.parquet', '')\
                        .replace('_ms2_pos.parquet', '')\
                        .replace('_ms2_neg.parquet', '')
    else:
        prefix = filename
    
    parquet_file = f'{prefix}_{desired_key}.parquet'
    
    if not os.path.exists(parquet_file):
        if desired_key.startswith('ms1'):
            return pd.DataFrame(columns=['mz', 'i', 'rt', 'polarity'])
        else:
            return pd.DataFrame(columns=[
                'mz', 'i', 'rt', 'polarity',
                'precursor_MZ', 'precursor_intensity', 'collision_energy'
            ])
    
    # Build filter expressions for predicate pushdown
    filters = []
    if mz_min is not None:
        filters.append(('mz', '>=', mz_min))
    if mz_max is not None:
        filters.append(('mz', '<=', mz_max))
    if rt_min is not None:
        filters.append(('rt', '>=', rt_min))
    if rt_max is not None:
        filters.append(('rt', '<=', rt_max))
    
    # Read with filters and column pruning
    table = pq.read_table(
        parquet_file,
        filters=filters if filters else None,
        columns=columns
    )
    
    return table.to_pandas()


def get_atlas_data_from_parquet_file(
    filename: str,
    atlas: pd.DataFrame,
    desired_key: str = 'ms1_pos'
) -> pd.DataFrame:
    """
    Extract atlas-matched data from Parquet file.
    Drop-in replacement for get_atlas_data_from_file().
    
    Parameters
    ----------
    filename : str
        Path to parquet file or prefix
    atlas : pd.DataFrame
        Atlas dataframe with columns: label, mz, rt_min, rt_max, rt_peak, 
        group_index, extra_time, ppm_tolerance
    desired_key : str
        One of: 'ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg'
        
    Returns
    -------
    pd.DataFrame
        Filtered data with columns:
        - MS1: ['label', 'rt', 'mz', 'i', 'in_feature']
        - MS2: ['label', 'rt', 'in_feature', 'i', 'mz', 'precursor_MZ', 
                'precursor_intensity', 'collision_energy']
    """
    # Use fast filtered read for initial data load
    # Calculate global mz and rt bounds from atlas
    mz_min = (atlas['mz'] * (1 - atlas['ppm_tolerance'].max() / 1e6)).min()
    mz_max = (atlas['mz'] * (1 + atlas['ppm_tolerance'].max() / 1e6)).max()
    rt_min = (atlas['rt_min'] - atlas['extra_time'].max()).min()
    rt_max = (atlas['rt_max'] + atlas['extra_time'].max()).max()
    
    msdata = read_parquet_with_filters(
        filename,
        desired_key,
        mz_min=mz_min,
        mz_max=mz_max,
        rt_min=rt_min,
        rt_max=rt_max
    )
    
    if msdata.empty:
        # Return empty dataframe with correct schema
        if 'ms2' in desired_key:
            return pd.DataFrame(columns=[
                'label', 'rt', 'in_feature', 'i', 'mz', 
                'precursor_MZ', 'precursor_intensity', 'collision_energy'
            ])
        else:
            return pd.DataFrame(columns=['label', 'rt', 'mz', 'i', 'in_feature'])
    
    if 'ms2' in desired_key:
        # For MS2: initially work with deduplicated data
        msdata_dedup = msdata[['rt', 'precursor_MZ']].drop_duplicates('rt')
        msdata_dedup = msdata_dedup.rename(columns={'precursor_MZ': 'mz'})
        
        # Map to atlas groups
        msdata_dedup['group_index'] = map_mzgroups_to_data(
            atlas['mz'].values,
            atlas['group_index'].values,
            msdata_dedup['mz'].values
        )
        
        # Filter using atlas
        df = filter_raw_data_using_atlas(atlas, msdata_dedup)
        
        # Keep essential columns
        df = df[['label', 'rt', 'in_feature']]
        
        # Merge back full MS2 data
        mcols = ['rt', 'i', 'mz', 'precursor_MZ', 'precursor_intensity', 'collision_energy']
        df = pd.merge(df, msdata[mcols], left_on='rt', right_on='rt', how='left')
        
        return df.reset_index(drop=True)
    
    else:
        # For MS1: direct processing
        msdata['group_index'] = map_mzgroups_to_data(
            atlas['mz'].values,
            atlas['group_index'].values,
            msdata['mz'].values
        )
        
        df = filter_raw_data_using_atlas(atlas, msdata)
        df = df[['label', 'rt', 'mz', 'i', 'in_feature']]
        
        return df.reset_index(drop=True)


def get_bpc_from_parquet(
    filename: str,
    dataset: str = 'ms1_pos',
    integration: str = 'bpc'
) -> pd.DataFrame:
    """
    Get base peak chromatogram (BPC) or total ion chromatogram (TIC) from Parquet file.
    Drop-in replacement for get_bpc().
    
    Parameters
    ----------
    filename : str
        Path to parquet file or prefix
    dataset : str
        One of: 'ms1_pos', 'ms1_neg', 'ms2_pos', 'ms2_neg'
    integration : str
        'bpc' for base peak chromatogram, 'tic' for total ion chromatogram
        
    Returns
    -------
    pd.DataFrame
        Chromatogram data with columns ['rt', 'i', 'mz'] (bpc) or ['rt', 'i'] (tic)
    """
    df = df_container_from_parquet_file(filename, desired_key=dataset)
    
    if df.empty:
        return pd.DataFrame(columns=['rt', 'i'])
    
    if integration == 'bpc':
        # Base peak: max intensity at each RT
        bpc = df.sort_values('i', ascending=False)\
                .groupby('rt', as_index=False)\
                .first()\
                .sort_values('rt', ascending=True)
        return bpc[['rt', 'i', 'mz']]
    else:
        # Total ion: sum intensity at each RT
        tic = df[['rt', 'i']]\
                .groupby('rt', as_index=False)\
                .sum()\
                .sort_values('rt', ascending=True)
        return tic


# Helper functions (same as in feature_tools.py, but included for completeness)

def map_mzgroups_to_data(mz_atlas, mz_group_indices, mz_data):
    """
    Map raw data m/z values to atlas group indices.
    
    Parameters
    ----------
    mz_atlas : array
        m/z values from atlas
    mz_group_indices : array
        Integer indices from group_consecutive()
    mz_data : array
        m/z values from raw data
        
    Returns
    -------
    array
        Index list mapping raw features to atlas features by m/z
    """
    f = interpolate.interp1d(
        mz_atlas,
        np.arange(mz_atlas.size),
        kind='nearest',
        bounds_error=False,
        fill_value='extrapolate'
    )
    idx = f(mz_data)
    idx = idx.astype('int')
    
    return mz_group_indices[idx]


def filter_raw_data_using_atlas(atlas, msdata):
    """
    Filter raw MS data using atlas constraints.
    
    Parameters
    ----------
    atlas : pd.DataFrame
        Must contain: label, mz, rt_min, rt_max, group_index, 
                     ppm_tolerance, extra_time
    msdata : pd.DataFrame
        Must contain: mz, rt, group_index
        
    Returns
    -------
    pd.DataFrame
        Filtered data with 'in_feature' column indicating if point is 
        within RT window (True) or in extra_time padding (False)
    """
    # Merge atlas info with raw data based on group_index
    merged_data = pd.merge(
        atlas[['label', 'mz', 'rt_min', 'rt_max', 'group_index', 
               'ppm_tolerance', 'extra_time']],
        msdata,
        left_on='group_index',
        right_on='group_index',
        how='right',
        suffixes=('_atlas', '_data')
    )
    
    # Filter by m/z tolerance
    mz_delta = np.abs(merged_data['mz_data'] - merged_data['mz_atlas'])
    mz_tolerance_da = merged_data['mz_atlas'] * merged_data['ppm_tolerance'] / 1e6
    mz_condition = mz_delta <= mz_tolerance_da
    
    # Filter by RT range (including extra_time padding)
    rt_min_condition = merged_data['rt'] >= (
        merged_data['rt_min'] - merged_data['extra_time']
    )
    rt_max_condition = merged_data['rt'] <= (
        merged_data['rt_max'] + merged_data['extra_time']
    )
    
    # Apply filters
    merged_data_filtered = merged_data[
        mz_condition & rt_min_condition & rt_max_condition
    ].reset_index(drop=True)
    
    merged_data_filtered['in_feature'] = True
    
    # Label datapoints in extra_time padding as not in_feature
    if merged_data_filtered['extra_time'].max() > 0.0:
        cond_rt = (
            (merged_data_filtered['rt'] < merged_data_filtered['rt_min']) |
            (merged_data_filtered['rt'] > merged_data_filtered['rt_max'])
        )
        merged_data_filtered.loc[cond_rt, 'in_feature'] = False
    
    # Rename mz_data back to mz
    merged_data_filtered = merged_data_filtered.rename(columns={'mz_data': 'mz'})
    
    return merged_data_filtered


def extract_eic_from_parquet(
    filename: str,
    mz_target: float,
    ppm_tolerance: float,
    rt_min: float,
    rt_max: float,
    polarity: str = 'positive',
    ms_level: int = 1
) -> pd.DataFrame:
    """
    Extract a single EIC (Extracted Ion Chromatogram) efficiently from Parquet.
    
    This uses Parquet's predicate pushdown for fast extraction without
    loading the entire file.
    
    Parameters
    ----------
    filename : str
        Path to parquet file or prefix
    mz_target : float
        Target m/z value
    ppm_tolerance : float
        m/z tolerance in ppm
    rt_min, rt_max : float
        Retention time window
    polarity : str
        'positive' or 'negative'
    ms_level : int
        1 or 2
        
    Returns
    -------
    pd.DataFrame
        EIC data with columns ['rt', 'i', 'mz']
    """
    # Calculate m/z range
    mz_delta = mz_target * ppm_tolerance / 1e6
    mz_min = mz_target - mz_delta
    mz_max = mz_target + mz_delta
    
    # Determine file key
    pol_key = 'pos' if polarity == 'positive' else 'neg'
    desired_key = f'ms{ms_level}_{pol_key}'
    
    # Read with filters
    eic_data = read_parquet_with_filters(
        filename,
        desired_key,
        mz_min=mz_min,
        mz_max=mz_max,
        rt_min=rt_min,
        rt_max=rt_max,
        columns=['rt', 'i', 'mz']  # Only read needed columns
    )
    
    return eic_data.sort_values('rt')


# Utility function for batch EIC extraction
def extract_multiple_eics_from_parquet(
    filename: str,
    atlas: pd.DataFrame,
    polarity: str = 'positive',
    ms_level: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Extract multiple EICs efficiently from Parquet file.
    
    Parameters
    ----------
    filename : str
        Path to parquet file or prefix
    atlas : pd.DataFrame
        Atlas with columns: label, mz, rt_min, rt_max, ppm_tolerance, extra_time
    polarity : str
        'positive' or 'negative'
    ms_level : int
        1 or 2
        
    Returns
    -------
    dict
        Dictionary mapping label to EIC DataFrame
    """
    eics = {}
    
    # Filter atlas by polarity if it has that column
    if 'detected_polarity' in atlas.columns:
        atlas_filtered = atlas[atlas['detected_polarity'] == polarity]
    else:
        atlas_filtered = atlas
    
    for _, row in atlas_filtered.iterrows():
        eic = extract_eic_from_parquet(
            filename,
            mz_target=row['mz'],
            ppm_tolerance=row['ppm_tolerance'],
            rt_min=row['rt_min'] - row.get('extra_time', 0.1),
            rt_max=row['rt_max'] + row.get('extra_time', 0.1),
            polarity=polarity,
            ms_level=ms_level
        )
        eics[row['label']] = eic
    
    return eics