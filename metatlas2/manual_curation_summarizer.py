import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

import metatlas2.logging_config as lcf
logger = lcf.get_logger('curation_creator')

def create_manual_curation_obj(
    auto_id_obj: "AutoIdentification",
) -> "ManualCuration":
    """
    Create ManualCuration object from CompoundMZRT.
    """
    from metatlas2.workflow_objects import ManualCuration

    for atlas_compound_mzrt in auto_id_obj.pre_autoid_atlas_obj.compound_mzrts.values():

        compound_data = pd.DataFrame([{
            'compound_uid': atlas_compound_mzrt.compound_uid,
            'inchi_key': atlas_compound_mzrt.inchi_key,
            'adduct': atlas_compound_mzrt.adduct,
            'rt_alignment_number': auto_id_obj.rt_alignment_number,
            'analysis_number': auto_id_obj.analysis_number,
            'compound_name': getattr(atlas_compound_mzrt, 'compound_name', ''),
            'auto_ided': False,
            'polarity': getattr(atlas_compound_mzrt, 'polarity', ''),
            'chromatography': getattr(atlas_compound_mzrt, 'chromatography', ''),
            'mz_tolerance': getattr(atlas_compound_mzrt, 'mz_tolerance', 5.0),
            'atlas_mz': getattr(atlas_compound_mzrt, 'mz', 0.0),
            'atlas_rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'atlas_rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'atlas_rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'original_rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'original_rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'original_rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'ms1_notes': getattr(atlas_compound_mzrt, 'ms1_notes', 'keep'),
            'ms2_notes': getattr(atlas_compound_mzrt, 'ms2_notes', 'no selection'),
            'other_notes': getattr(atlas_compound_mzrt, 'other_notes', 'no selection'),
            'identification_notes': getattr(atlas_compound_mzrt, 'identification_notes', ''),
            'analyst_notes': getattr(atlas_compound_mzrt, 'analyst_notes', ''),
            'best_ms1_file': '',
            'best_ms1_rt': 0.0,
            'best_ms1_mz': 0.0,
            'best_ms1_intensity': 0.0,
            'best_ms1_ppm_error': 0.0,
            'best_ms1_rt_error': 0.0,
            'isomers': None,
            'suggested_rt_min': 0.0,
            'suggested_rt_max': 0.0,
            'suggested_rt_peak': 0.0,
            'rt_suggestion_confidence': 0.0
        }])

        manual_curation_obj = ManualCuration(
            inchi_key=atlas_compound_mzrt.inchi_key, 
            adduct=atlas_compound_mzrt.adduct, 
            data=compound_data
        )

        _fill_best_ms1_to_manual_curation(
            manual_curation_obj,
            atlas_compound_mzrt,
            auto_id_obj.experimental_data
        )

        _add_isomers_to_manual_curation_obj(
            manual_curation_obj,
            auto_id_obj.pre_autoid_atlas_obj
        )

        _add_rt_suggestions_to_manual_curation_obj(
            manual_curation_obj, 
            auto_id_obj.experimental_data
        )

        auto_id_obj.experimental_data.manual_curation.append(manual_curation_obj)

    return manual_curation_obj

def _fill_best_ms1_to_manual_curation(
    manual_curation_obj: "ManualCuration",
    atlas_compound_mzrt: "CompoundMZRT",
    exp_data_obj: "ExperimentalData"
) -> None:
    """
    For a given atlas compound, find the best MS1 file (highest intensity)
    and fill manual_curation_obj.data with its info.
    """
    all_files = []
    for ms1 in exp_data_obj.ms1_data:
        if ms1.inchi_key == atlas_compound_mzrt.inchi_key and ms1.adduct == atlas_compound_mzrt.adduct:
            if ms1.data is not None and not ms1.data.empty:
                sum_intensity = ms1.data['i'].sum()
                if sum_intensity > 0:
                    idx = ms1.data['i'].idxmax()
                    peak_height = ms1.data.loc[idx, 'i']
                    peak_mz = ms1.data.loc[idx, 'mz']
                    peak_rt = ms1.data.loc[idx, 'rt']
                    mz_centroid = (ms1.data['i'] * ms1.data['mz']).sum() / sum_intensity
                    ppm_error = (mz_centroid - atlas_compound_mzrt.mz) / atlas_compound_mzrt.mz * 1e6
                    rt_error = peak_rt - atlas_compound_mzrt.rt_peak
                    all_files.append({
                        'filename': ms1.filename,
                        'rt_peak': peak_rt,
                        'mz_centroid': mz_centroid,
                        'peak_height': peak_height,
                        'ppm_error': ppm_error,
                        'rt_error': rt_error
                    })
                
    if all_files:
        manual_curation_obj.data.loc[0, 'auto_ided'] = True
        best = max(all_files, key=lambda x: x['peak_height'])
        manual_curation_obj.data.loc[0, 'best_ms1_file'] = best['filename']
        manual_curation_obj.data.loc[0, 'best_ms1_rt'] = float(best['rt_peak'])
        manual_curation_obj.data.loc[0, 'best_ms1_mz'] = float(best['mz_centroid'])
        manual_curation_obj.data.loc[0, 'best_ms1_intensity'] = float(best['peak_height'])
        manual_curation_obj.data.loc[0, 'best_ms1_ppm_error'] = float(best['ppm_error'])
        manual_curation_obj.data.loc[0, 'best_ms1_rt_error'] = float(best['rt_error'])

def _add_isomers_to_manual_curation_obj(
    manual_curation_obj: "ManualCuration",
    atlas_obj: "Atlas"
) -> None:
    """
    Add isomer information to a ManualCuration object for its inchi_key.
    Modifies manual_curation_obj.data in-place.
    """
    inchi_key = manual_curation_obj.inchi_key
    isomer_dict = _build_isomer_dict(atlas_obj)
    isomer_list = isomer_dict.get(inchi_key, [])
    manual_curation_obj.data.at[0, 'isomers'] = isomer_list

def _build_isomer_dict(
    atlas_obj: "Atlas"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return a dict: inchi_key → list of isomer dicts (empty list if none).
    Isomers are defined as:
      - mz or mono_isotopic_molecular_weight within 0.005
      - OR inchi_key prefix (before '-') identical
    """
    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    atlas_df = atlas_obj.to_dataframe()
    for _, row in atlas_df.iterrows():
        mz = row["mz"]
        mono_isotopic_molecular_weight = row.get("mono_isotopic_molecular_weight", None)
        inchi_prefix = row["inchi_key"].split("-")[0]
        
        def is_isomer(atlas_row):
            if atlas_row["inchi_key"] == row["inchi_key"]:
                return False
            mz_close = abs(atlas_row["mz"] - mz) <= 0.005
            mass_close = (
                mono_isotopic_molecular_weight is not None and 
                atlas_row.get("mono_isotopic_molecular_weight", None) is not None and
                abs(atlas_row["mono_isotopic_molecular_weight"] - mono_isotopic_molecular_weight) <= 0.005
            )
            prefix_match = atlas_row["inchi_key"].split("-")[0] == inchi_prefix
            return mz_close or mass_close or prefix_match
        
        isomers = atlas_df[atlas_df.apply(is_isomer, axis=1)]
        isomer_dict[row["inchi_key"]] = [
            {
                "inchi_key": atlas_row["inchi_key"],
                "adduct": atlas_row["adduct"],
                "compound_name": atlas_row.get("compound_name", ""),
                "rt": atlas_row["rt_peak"],
                "mz": atlas_row["mz"],
            }
            for _, atlas_row in isomers.iterrows()
        ]
    
    return isomer_dict

def _add_rt_suggestions_to_manual_curation_obj(
    manual_curation_obj: "ManualCuration",
    exp_data_obj: "ExperimentalData"
) -> None:
    """
    Add RT bound suggestions to a ManualCuration object based on ms1 data from exp_data_obj.
    Modifies manual_curation_obj.data in-place.
    """
    df = manual_curation_obj.data
    inchi_key = manual_curation_obj.inchi_key
    adduct = manual_curation_obj.adduct
    ms1_data_dict = {}
    for ms1_data in getattr(exp_data_obj, 'ms1_data', []):
        if getattr(ms1_data, 'inchi_key', None) == inchi_key and getattr(ms1_data, 'adduct', None) == adduct:
            ms1_df = ms1_data.data
            if ms1_df is not None and not ms1_df.empty:
                ms1_data_dict[ms1_data.filename] = {
                    'rt_vals': ms1_df['rt'].values,
                    'i_vals': ms1_df['i'].values
                }
    suggestion = _suggest_rt_bounds_from_ms1(
        ms1_data_dict,
        atlas_rt_peak=df['atlas_rt_peak'].iloc[0],
        atlas_rt_min=df['atlas_rt_min'].iloc[0],
        atlas_rt_max=df['atlas_rt_max'].iloc[0]
    )
    if suggestion is not None:
        df.at[0, 'suggested_rt_min'] = suggestion['rt_min']
        df.at[0, 'suggested_rt_max'] = suggestion['rt_max']
        df.at[0, 'suggested_rt_peak'] = suggestion['rt_peak']
        df.at[0, 'rt_suggestion_confidence'] = suggestion['confidence']
    else:
        df.at[0, 'suggested_rt_min'] = df['atlas_rt_min'].iloc[0]
        df.at[0, 'suggested_rt_max'] = df['atlas_rt_max'].iloc[0]
        df.at[0, 'suggested_rt_peak'] = df['atlas_rt_peak'].iloc[0]
        df.at[0, 'rt_suggestion_confidence'] = 0.0

def _suggest_rt_bounds_from_ms1(
    ms1_data: Dict[str, Dict[str, Any]],
    atlas_rt_peak: float,
    atlas_rt_min: float,
    atlas_rt_max: float
) -> Optional[Dict[str, float]]:
    """
    Compute RT bounds from the *average* extracted-ion chromatogram (ms1)
    of many LC-MS/MS files.
    """
    import warnings
    
    if not ms1_data:
        return None

    # Select top 50 files by intensity
    sorted_files = sorted(
        ms1_data.items(),
        key=lambda kv: np.max(kv[1].get("i_vals", [0])) if len(kv[1].get("i_vals", [])) > 0 else 0,
        reverse=True,
    )
    selected = sorted_files[:50]

    # Check ms1s for bad data
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

    # Average ms1s
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