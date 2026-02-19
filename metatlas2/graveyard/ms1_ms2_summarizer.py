import sys
import pandas as pd
from typing import Dict

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('ms1_ms2_summarizer')

def create_ms_summaries(exp_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], only_ms_level: int = None) -> pd.DataFrame:
    """
    Create summary statistics for MS2 data and hits.
    
    Returns DataFrame with one row containing summary statistics.
    """
    for inchi_key, file_dict in exp_data.items():
        for filename, data_dict in file_dict.items():
            ms1_df = data_dict.get('ms1_data', pd.DataFrame())
            ms2_df = data_dict.get('ms2_data', pd.DataFrame())
            ms2_hits_df = data_dict.get('ms2_hits', pd.DataFrame())

            if only_ms_level is None or only_ms_level == 1:
                ms1_summary = _calculate_ms1_summary(ms1_df)

            if only_ms_level is None or only_ms_level == 2:
                ms2_summary = _calculate_ms2_summary(ms2_df, ms2_hits_df)

            exp_data[inchi_key][filename]['ms1_summary'] = ms1_summary
            exp_data[inchi_key][filename]['ms2_summary'] = ms2_summary

    return exp_data

def _calculate_ms1_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary properties for features from MS1 data.
    """
    summary = {
        'label': [],
        'adduct': [],
        'num_datapoints': [], 
        'peak_area': [], 
        'peak_height': [], 
        'mz_centroid': [],
        'rt_peak': []
    }

    if df.empty:
        summary['label'].append('')
        summary['adduct'].append('')
        summary['num_datapoints'].append(0)
        summary['peak_area'].append(0.0)
        summary['peak_height'].append(0.0)
        summary['mz_centroid'].append(0.0)
        summary['rt_peak'].append(0.0)
        return pd.DataFrame(summary)

    for label_group, label_data in df.groupby('label'):
        summary['label'].append(label_group)
        summary['adduct'].append(label_data['adduct'].iloc[0])
        summary['num_datapoints'].append(label_data['i'].count())
        sum_intensity = label_data['i'].sum()
        summary['peak_area'].append(sum_intensity)
        
        if sum_intensity > 0:
            idx = label_data['i'].idxmax()
            summary['peak_height'].append(label_data.loc[idx, 'i'])
            summary['mz_centroid'].append(sum(label_data['i'] * label_data['mz']) / sum_intensity)
            summary['rt_peak'].append(label_data.loc[idx, 'rt'])
        else:
            summary['peak_height'].append(0.0)
            summary['mz_centroid'].append(0.0)
            summary['rt_peak'].append(0.0)

    ms1_summary_output = pd.DataFrame(summary)
    if len(ms1_summary_output) > 1:
        raise ValueError(f"Expected only one MS1 summary row but found {len(ms1_summary_output)} features for label {ms1_summary_output['label'].iloc[0]}")

    return ms1_summary_output

def _calculate_ms2_summary(ms2_df: pd.DataFrame, ms2_hits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary properties for MS2 data and hits.
    Returns DataFrame with consistent schema even when inputs are empty.
    """
    # Define schema for consistent output
    summary_schema = {
        'label': '',
        'adduct': '',
        'num_scans': 0,
        'num_fragments': 0,
        'best_ms2_rt': 0.0,
        'best_ms2_mz': 0.0,
        'best_ms2_intensity': 0.0,
        'num_hits': 0,
        'best_hit_score': 0.0,
        'best_hit_database': '',
        'best_hit_ref_id': '',
        'best_hit_ref_name': '',
        'best_hit_num_matches': 0
    }
    
    # If both inputs are empty, return empty DataFrame with correct schema
    if ms2_df.empty and ms2_hits_df.empty:
        return pd.DataFrame([summary_schema])
    
    summary = summary_schema.copy()
    summary['label'] = ms2_df['label'].iloc[0] if not ms2_df.empty else ''
    summary['adduct'] = ms2_df['adduct'].iloc[0] if not ms2_df.empty else ''
    
    # Get best MS2 scan (highest precursor intensity)
    if not ms2_df.empty:
        best_scan_idx = ms2_df.groupby('rt')['precursor_intensity'].first().idxmax()
        best_scan_data = ms2_df[ms2_df['rt'] == best_scan_idx].iloc[0]
        summary['best_ms2_rt'] = float(best_scan_data['rt'])
        summary['best_ms2_mz'] = float(best_scan_data['precursor_MZ'])
        summary['best_ms2_intensity'] = float(best_scan_data['precursor_intensity'])
        summary['num_scans'] = ms2_df['rt'].nunique()
        summary['num_fragments'] = len(ms2_df)
    
    # Get best hit if available
    if not ms2_hits_df.empty:
        best_hit_idx = ms2_hits_df['score'].idxmax()
        best_hit = ms2_hits_df.loc[best_hit_idx]
        summary['num_hits'] = len(ms2_hits_df)
        summary['best_hit_score'] = float(best_hit['score'])
        summary['best_hit_database'] = str(best_hit['database'])
        summary['best_hit_ref_id'] = str(best_hit['ref_id'])
        summary['best_hit_ref_name'] = str(best_hit['ref_name'])
        summary['best_hit_num_matches'] = int(best_hit['num_matches'])

    return pd.DataFrame([summary])
