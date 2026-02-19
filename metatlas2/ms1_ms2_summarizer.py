import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('ms1_ms2_summarizer')


def create_ms_summaries(
    exp_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    atlas_dataframe: pd.DataFrame,
    only_ms_level: Optional[int] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Create comprehensive summary structure combining experimental data with atlas metadata.
    
    Returns:
        Dict with structure:
        {
            inchi_key: {
                adduct: {
                    'compound_info': pd.DataFrame (1 row with compound-level data),
                    'file_summaries': {
                        filename: {
                            'ms1_summary': pd.DataFrame (1 row),
                            'ms2_summary': pd.DataFrame (1 row),
                            'ms2_hits': pd.DataFrame (multiple rows, one per hit)
                        }
                    }
                }
            }
        }
    """
    results = {}
    
    # Iterate through atlas entries (ensures all compounds are included)
    for _, atlas_row in atlas_dataframe.iterrows():
        inchi_key = atlas_row['inchi_key']
        adduct = atlas_row['adduct']
        
        # Initialize nested dict structure if needed
        if inchi_key not in results:
            results[inchi_key] = {}
        
        # Create compound-level summary from atlas
        compound_info = _create_compound_info(atlas_row)
        
        # Get experimental data for this inchi_key (may be empty)
        file_dict = exp_data.get(inchi_key, {})
        
        # Create file-level summaries for this adduct
        file_summaries = {}
        for filename, data_dict in file_dict.items():
            ms1_df = data_dict.get('ms1_data', pd.DataFrame())
            ms2_df = data_dict.get('ms2_data', pd.DataFrame())
            ms2_hits_df = data_dict.get('ms2_hits', pd.DataFrame())
            
            # Filter data for this specific adduct
            if not ms1_df.empty and 'adduct' in ms1_df.columns:
                ms1_df = ms1_df[ms1_df['adduct'] == adduct]
            if not ms2_df.empty and 'adduct' in ms2_df.columns:
                ms2_df = ms2_df[ms2_df['adduct'] == adduct]
            if not ms2_hits_df.empty and 'adduct' in ms2_hits_df.columns:
                ms2_hits_df = ms2_hits_df[ms2_hits_df['adduct'] == adduct]
            
            file_summary = {}
            
            if only_ms_level is None or only_ms_level == 1:
                file_summary['ms1_summary'] = _calculate_ms1_summary(
                    ms1_df, 
                    compound_info
                )
                # Store raw EIC data for RT suggestions
                file_summary['ms1_raw_data'] = ms1_df
            
            if only_ms_level is None or only_ms_level == 2:
                file_summary['ms2_summary'] = _calculate_ms2_summary(
                    ms2_df, 
                    ms2_hits_df,
                    compound_info
                )
                file_summary['ms2_hits'] = ms2_hits_df
            
            file_summaries[filename] = file_summary
        
        # Calculate best EIC across all files for this adduct
        compound_info = _calculate_best_eic_across_files(compound_info, file_summaries)
        
        results[inchi_key][adduct] = {
            'compound_info': compound_info,
            'file_summaries': file_summaries
        }
    
    # Add isomers and RT suggestions
    logger.info("Adding isomer detection to summaries...")
    add_isomers_to_summaries(results, atlas_dataframe)
    
    logger.info("Adding RT bound suggestions to summaries...")
    add_rt_suggestions_to_summaries(results)
    
    logger.info("Calculating analysis summary statistics")
    print_summary_statistics(results)

    return results


def _create_compound_info(atlas_row: pd.Series) -> pd.DataFrame:
    """
    Create compound-level information DataFrame from atlas entry.
    Returns single-row DataFrame with compound metadata and placeholders for analysis results.
    Adduct is NOT included as a column since it's now a dict key.
    """
    compound_data = {
        # Core identifiers
        'inchi_key': atlas_row['inchi_key'],
        'compound_uid': atlas_row['compound_uid'],
        'compound_name': atlas_row.get('compound_name', atlas_row.get('label', '')),
        'formula': atlas_row.get('formula', ''),
        'polarity': atlas_row.get('polarity', ''),
        'chromatography': atlas_row.get('chromatography', ''),
        
        # m/z and tolerances
        'mz': atlas_row.get('mz', 0.0),
        'mz_tolerance': atlas_row.get('mz_tolerance', 5.0),
        
        # Original RT bounds from atlas
        'original_rt_peak': atlas_row.get('rt_peak', 0.0),
        'original_rt_min': atlas_row.get('rt_min', 0.0),
        'original_rt_max': atlas_row.get('rt_max', 0.0),
        
        # Current RT bounds (modifiable)
        'rt_peak': atlas_row.get('rt_peak', 0.0),
        'rt_min': atlas_row.get('rt_min', 0.0),
        'rt_max': atlas_row.get('rt_max', 0.0),
        
        # Analysis annotations (compound-level)
        'ms1_notes': 'keep',
        'ms2_notes': 'no selection',
        'analyst_notes': '',
        'identification_notes': '',
        
        # Modification tracking
        'is_rt_modified': False,
        'is_annotation_modified': False,
        
        # Placeholders for calculated fields (filled later)
        'total_files_detected': 0,
        'ms2_files_with_data': 0,
        
        # Best EIC results (across all files)
        'best_eic_file': '',
        'best_eic_rt': 0.0,
        'best_eic_mz': 0.0,
        'best_eic_intensity': 0.0,
        'best_eic_ppm_error': 0.0,
        'best_eic_rt_error': 0.0,
        
        # Isomers and RT suggestions
        'isomers': None,
        'suggested_rt_min': 0.0,
        'suggested_rt_max': 0.0,
        'suggested_rt_peak': 0.0,
        'rt_suggestion_confidence': 0.0
    }
    
    return pd.DataFrame([compound_data])


def _calculate_ms1_summary(df: pd.DataFrame, compound_info: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary properties for MS1 data from a single file.
    Returns single-row DataFrame with MS1 metrics.
    """
    inchi_key = compound_info['inchi_key'].iloc[0]
    atlas_mz = compound_info['mz'].iloc[0]
    atlas_rt = compound_info['rt_peak'].iloc[0]
    
    summary_schema = {
        'inchi_key': inchi_key,
        'num_datapoints': 0,
        'peak_area': 0.0,
        'peak_height': 0.0,
        'mz_centroid': 0.0,
        'rt_peak': 0.0,
        'ppm_error': 0.0,
        'rt_error': 0.0
    }
    
    if df.empty:
        return pd.DataFrame([summary_schema])
    
    sum_intensity = df['i'].sum()
    summary_schema['num_datapoints'] = int(df['i'].count())
    summary_schema['peak_area'] = float(sum_intensity)
    
    if sum_intensity > 0:
        idx = df['i'].idxmax()
        peak_height = df.loc[idx, 'i']
        peak_mz = df.loc[idx, 'mz']
        peak_rt = df.loc[idx, 'rt']
        
        mz_centroid = (df['i'] * df['mz']).sum() / sum_intensity
        
        summary_schema['peak_height'] = float(peak_height)
        summary_schema['mz_centroid'] = float(mz_centroid)
        summary_schema['rt_peak'] = float(peak_rt)
        summary_schema['ppm_error'] = float((mz_centroid - atlas_mz) / atlas_mz * 1e6)
        summary_schema['rt_error'] = float(peak_rt - atlas_rt)
    
    return pd.DataFrame([summary_schema])


def _calculate_ms2_summary(
    ms2_df: pd.DataFrame,
    ms2_hits_df: pd.DataFrame,
    compound_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate summary properties for MS2 data from a single file.
    Returns single-row DataFrame with MS2 scan metrics.
    """
    inchi_key = compound_info['inchi_key'].iloc[0]
    
    summary_schema = {
        'inchi_key': inchi_key,
        'num_scans': 0,
        'num_fragments': 0,
        'best_scan_rt': 0.0,
        'best_scan_precursor_mz': 0.0,
        'best_scan_precursor_intensity': 0.0,
        'total_hits': 0
    }
    
    if ms2_df.empty:
        return pd.DataFrame([summary_schema])
    
    summary_schema['num_scans'] = int(ms2_df['rt'].nunique())
    summary_schema['num_fragments'] = int(len(ms2_df))
    
    best_scan_idx = ms2_df.groupby('rt')['precursor_intensity'].first().idxmax()
    best_scan = ms2_df[ms2_df['rt'] == best_scan_idx].iloc[0]
    
    summary_schema['best_scan_rt'] = float(best_scan['rt'])
    summary_schema['best_scan_precursor_mz'] = float(best_scan['precursor_MZ'])
    summary_schema['best_scan_precursor_intensity'] = float(best_scan['precursor_intensity'])
    
    if not ms2_hits_df.empty:
        summary_schema['total_hits'] = int(len(ms2_hits_df))
    
    return pd.DataFrame([summary_schema])


def _calculate_best_eic_across_files(
    compound_info: pd.DataFrame,
    file_summaries: Dict[str, Dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Calculate best EIC metrics across all files for this compound+adduct.
    Updates compound_info DataFrame with best_eic_* fields.
    """
    compound_info = compound_info.copy()
    
    best_intensity = 0.0
    
    for filename, summaries in file_summaries.items():
        ms1_summary = summaries.get('ms1_summary')
        if ms1_summary is None or ms1_summary.empty:
            continue
        
        intensity = ms1_summary['peak_height'].iloc[0]
        if intensity > best_intensity:
            best_intensity = intensity
            
            compound_info.loc[0, 'best_eic_file'] = filename
            compound_info.loc[0, 'best_eic_rt'] = float(ms1_summary['rt_peak'].iloc[0])
            compound_info.loc[0, 'best_eic_mz'] = float(ms1_summary['mz_centroid'].iloc[0])
            compound_info.loc[0, 'best_eic_intensity'] = float(intensity)
            compound_info.loc[0, 'best_eic_ppm_error'] = float(ms1_summary['ppm_error'].iloc[0])
            compound_info.loc[0, 'best_eic_rt_error'] = float(ms1_summary['rt_error'].iloc[0])
    
    # Update file counts
    compound_info.loc[0, 'total_files_detected'] = len([
        f for f, s in file_summaries.items() 
        if 'ms1_summary' in s and not s['ms1_summary'].empty
        and s['ms1_summary']['num_datapoints'].iloc[0] > 0
    ])
    
    compound_info.loc[0, 'ms2_files_with_data'] = len([
        f for f, s in file_summaries.items()
        if 'ms2_summary' in s and not s['ms2_summary'].empty 
        and s['ms2_summary']['num_scans'].iloc[0] > 0
    ])
    
    return compound_info


def add_isomers_to_summaries(
    results: Dict[str, Dict[str, Dict]],
    atlas_dataframe: pd.DataFrame
) -> None:
    """
    Add isomer information to compound_info DataFrames.
    Modifies results in-place.
    """
    isomer_dict = _build_isomer_dict(atlas_dataframe)
    
    for inchi_key, adduct_dict in results.items():
        isomer_list = isomer_dict.get(inchi_key, [])
        for adduct, data in adduct_dict.items():
            compound_info = data['compound_info']
            compound_info.at[0, 'isomers'] = isomer_list


def _build_isomer_dict(atlas_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return a dict: inchi_key → list of isomer dicts (empty list if none).
    Isomers are defined as:
      - mz or mono_isotopic_molecular_weight within 0.005
      - OR inchi_key prefix (before '-') identical
    """
    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    
    for _, row in atlas_df.iterrows():
        mz = row["mz"]
        mono_isotopic_molecular_weight = row.get("mono_isotopic_molecular_weight", None)
        inchi_prefix = row["inchi_key"].split("-")[0]
        
        def is_isomer(r):
            if r["inchi_key"] == row["inchi_key"]:
                return False
            mz_close = abs(r["mz"] - mz) <= 0.005
            mass_close = (
                mono_isotopic_molecular_weight is not None and 
                r.get("mono_isotopic_molecular_weight", None) is not None and
                abs(r["mono_isotopic_molecular_weight"] - mono_isotopic_molecular_weight) <= 0.005
            )
            prefix_match = r["inchi_key"].split("-")[0] == inchi_prefix
            return mz_close or mass_close or prefix_match
        
        isomers = atlas_df[atlas_df.apply(is_isomer, axis=1)]
        isomer_dict[row["inchi_key"]] = [
            {
                "inchi_key": r["inchi_key"],
                "compound_name": r.get("compound_name", r.get("label", "")),
                "rt": r["rt_peak"],
                "mz": r["mz"],
                "mz_tolerance": r.get("mz_tolerance_ppm", r.get("mz_tolerance", 10.0)),
            }
            for _, r in isomers.iterrows()
        ]
    
    return isomer_dict

def add_rt_suggestions_to_summaries(
    results: Dict[str, Dict[str, Dict]]
) -> None:
    """
    Add RT bound suggestions to compound_info DataFrames based on EIC data.
    Modifies results in-place.
    """
    for inchi_key, adduct_dict in results.items():
        for adduct, data in adduct_dict.items():
            compound_info = data['compound_info']
            file_summaries = data['file_summaries']
            
            # Collect EIC data across files
            eic_data = {}
            for filename, summaries in file_summaries.items():
                ms1_raw_data = summaries.get('ms1_raw_data')
                if ms1_raw_data is not None and not ms1_raw_data.empty:
                    eic_data[filename] = {
                        'rt_vals': ms1_raw_data['rt'].values,
                        'i_vals': ms1_raw_data['i'].values
                    }
            
            suggestion = _suggest_rt_bounds_from_eic(
                eic_data,
                atlas_rt_peak=compound_info['original_rt_peak'].iloc[0],
                atlas_rt_min=compound_info['original_rt_min'].iloc[0],
                atlas_rt_max=compound_info['original_rt_max'].iloc[0]
            )
            
            if suggestion is not None:
                compound_info.at[0, 'suggested_rt_min'] = suggestion['rt_min']
                compound_info.at[0, 'suggested_rt_max'] = suggestion['rt_max']
                compound_info.at[0, 'suggested_rt_peak'] = suggestion['rt_peak']
                compound_info.at[0, 'rt_suggestion_confidence'] = suggestion['confidence']
            else:
                compound_info.at[0, 'suggested_rt_min'] = compound_info['original_rt_min'].iloc[0]
                compound_info.at[0, 'suggested_rt_max'] = compound_info['original_rt_max'].iloc[0]
                compound_info.at[0, 'suggested_rt_peak'] = compound_info['original_rt_peak'].iloc[0]
                compound_info.at[0, 'rt_suggestion_confidence'] = 0.0

def _suggest_rt_bounds_from_eic(
    eic_data: Dict[str, Dict[str, Any]],
    atlas_rt_peak: float,
    atlas_rt_min: float,
    atlas_rt_max: float
) -> Optional[Dict[str, float]]:
    """
    Compute RT bounds from the *average* extracted-ion chromatogram (EIC)
    of many LC-MS/MS files.
    """
    import warnings
    
    if not eic_data:
        return None

    # Select top 50 files by intensity
    sorted_files = sorted(
        eic_data.items(),
        key=lambda kv: np.max(kv[1].get("i_vals", [0])) if len(kv[1].get("i_vals", [])) > 0 else 0,
        reverse=True,
    )
    selected = sorted_files[:50]

    # Check EICs for bad data
    rt_lists: List[np.ndarray] = []
    int_lists: List[np.ndarray] = []
    weights: List[float] = []

    for fname, trace in selected:
        rt_raw = trace.get("rt_vals", [])
        i_raw  = trace.get("i_vals", [])

        rt_arr = np.asarray(rt_raw, dtype=np.float64)
        i_arr  = np.asarray(i_raw,  dtype=np.float64)

        valid = (~np.isnan(rt_arr)) & (~np.isnan(i_arr)) & (i_arr >= 0)
        rt_arr = rt_arr[valid]
        i_arr  = i_arr[valid]

        if rt_arr.size < 5:
            continue

        rt_lists.append(rt_arr)
        int_lists.append(i_arr)

        w = float(np.max(i_arr)) if i_arr.size > 0 else 1.0
        if np.isnan(w) or w <= 0:
            w = 1.0
        weights.append(w)

    if not rt_lists:
        return None

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    # Combine samples
    global_min = min(rt.min() for rt in rt_lists)
    global_max = max(rt.max() for rt in rt_lists)

    # Choose a step size
    all_spacings = np.concatenate(
        [np.diff(rt) for rt in rt_lists if rt.size > 1]
    )
    step = np.median(all_spacings) if all_spacings.size else 0.01

    # Guard against a zero step
    if step <= 0:
        step = 0.01

    common_rt = np.arange(global_min, global_max + step, step)

    # Make a common grid by interpolating
    interpolated = []
    for rt, intensity in zip(rt_lists, int_lists):
        if np.array_equal(rt, common_rt):
            interp_i = intensity
        else:
            f = interp1d(rt, intensity, kind="linear",
                         bounds_error=False, fill_value=0.0)
            interp_i = f(common_rt)
        interpolated.append(interp_i)

    intensity_matrix = np.vstack(interpolated)

    # Average EICs
    weighted_avg = np.average(intensity_matrix, axis=0, weights=weights)
    ma_window = 5
    smoothed = _moving_average(weighted_avg, window=ma_window)
    if ma_window > 1:
        pad = (ma_window - 1) // 2
        smoothed = np.pad(smoothed, (pad, pad), mode="edge")
        smoothed = smoothed[:common_rt.size]

    # Peak detection
    if np.max(smoothed) <= 0:
        return None

    # Use a modest height / prominence threshold
    max_int = np.max(smoothed)
    min_height = max_int * 0.10
    min_prom   = max_int * 0.05

    peaks, _ = find_peaks(
        smoothed,
        height=min_height,
        prominence=min_prom,
        distance=5
    )

    if peaks.size == 0:
        peaks = np.array([np.argmax(smoothed)])

    # Choose best peak
    best_idx = np.argmax(smoothed[peaks])
    best_peak = peaks[best_idx]
    best_rt = common_rt[best_peak]
    best_int = smoothed[best_peak]

    # Calculate peak bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominences = peak_prominences(smoothed, [best_peak])[0]
    
    if len(prominences) == 0 or prominences[0] == 0.0:
        left_ips = [0]
        right_ips = [len(smoothed) - 1]
        widths = [right_ips[0] - left_ips[0]]
    else:
        widths, _, left_ips, right_ips = peak_widths(
            smoothed,
            [best_peak],
            rel_height=0.5
        )
    
    # Convert from index space to RT space
    left_idx  = int(np.floor(left_ips[0]))
    right_idx = int(np.ceil(right_ips[0]))
    left_idx  = max(0, left_idx)
    right_idx = min(len(common_rt) - 1, right_idx)

    rt_left  = common_rt[left_idx]
    rt_right = common_rt[right_idx]

    # Add padding
    width_rt = rt_right - rt_left
    pad = max(0.05, width_rt * 0.05)
    rt_min = rt_left - pad
    rt_max = rt_right + pad

    # Intensity score
    intensity_score = min(1.0, best_int / max(max_int, 1e3))

    # Prominence score
    if len(prominences) == 0 or prominences[0] == 0.0:
        prominence_score = 0.0
    else:
        prominence_score = min(1.0, prominences[0] / max(best_int * 0.5, 1e3))

    # Width score
    optimal_width_pts = 20
    width_score = 1.0 - min(1.0, abs(widths[0] - optimal_width_pts) / optimal_width_pts)

    # RT proximity score
    max_expected_dev = max(abs(atlas_rt_max - atlas_rt_min), 1.0)
    rt_dev = abs(best_rt - atlas_rt_peak)
    rt_score = max(0.0, 1.0 - (rt_dev / max_expected_dev))

    # Shape symmetry score
    left_tail  = best_peak - left_idx
    right_tail = right_idx - best_peak
    asym = abs(left_tail - right_tail) / max(left_tail + right_tail, 1)
    shape_score = max(0.0, 1.0 - asym)

    # Weight scores
    confidence = (
        0.30 * intensity_score +
        0.20 * prominence_score +
        0.15 * width_score +
        0.20 * rt_score +
        0.15 * shape_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Apply penalties
    if confidence < 0.1:
        return None
    if width_rt > (atlas_rt_max - atlas_rt_min) * 2:
        confidence *= 0.5
    if rt_dev > max_expected_dev * 2:
        confidence *= 0.3

    return {
        "rt_min": float(rt_min),
        "rt_max": float(rt_max),
        "rt_peak": float(best_rt),
        "confidence": float(confidence),
    }

def _moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple moving‑average.  window=1 returns the original array."""
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def print_summary_statistics(results: Dict[str, Dict[str, Dict]]) -> None:
    """
    Print summary statistics from MS summaries results.
    
    Args:
        results: Output from create_ms_summaries()
    """
    total_compounds = 0
    total_adducts = 0
    compounds_with_eic = 0
    compounds_with_ms2 = 0
    modified_compounds = 0
    total_files_analyzed = set()
    
    # Iterate through all compounds and adducts
    for inchi_key, adduct_dict in results.items():
        for adduct, data in adduct_dict.items():
            total_adducts += 1
            compound_info = data['compound_info']
            file_summaries = data['file_summaries']
            
            # Track unique compounds (by inchi_key)
            if adduct == list(adduct_dict.keys())[0]:  # Count once per inchi_key
                total_compounds += 1
            
            # Check if compound has EIC data
            if compound_info['total_files_detected'].iloc[0] > 0:
                compounds_with_eic += 1
            
            # Check if compound has MS2 data
            if compound_info['ms2_files_with_data'].iloc[0] > 0:
                compounds_with_ms2 += 1
            
            # Check if modified
            if (compound_info['is_rt_modified'].iloc[0] or 
                compound_info['is_annotation_modified'].iloc[0]):
                modified_compounds += 1
            
            # Track all files
            total_files_analyzed.update(file_summaries.keys())
    
    # Print summary
    logger.info("Analysis Summary")
    logger.info(f"  Total unique compounds (InChI keys): {total_compounds}")
    logger.info(f"  Total compound+adduct entries: {total_adducts}")
    logger.info(f"  Entries with MS1/EIC data: {compounds_with_eic} ({compounds_with_eic/total_adducts*100:.1f}%)")
    logger.info(f"  Entries with MS2 data: {compounds_with_ms2} ({compounds_with_ms2/total_adducts*100:.1f}%)")
    logger.info(f"  Entries with modifications: {modified_compounds}")
    logger.info(f"  Total files analyzed: {len(total_files_analyzed)}")
    
    return