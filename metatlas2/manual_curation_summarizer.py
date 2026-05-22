import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from tqdm.auto import tqdm
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.signal._peak_finding_utils import PeakPropertyWarning

import metatlas2.logging_config as lcf
logger = lcf.get_logger('curation_creator')

def create_manual_curation_obj(auto_id_obj: "AutoIdentification") -> "ManualCuration":
    from metatlas2.workflow_objects import ManualCuration

    logger.info("Building indices and isomer map...")
    isomer_dict = _build_isomer_dict(auto_id_obj.pre_autoid_atlas_obj)
    ms1_index = _build_ms1_index(auto_id_obj.experimental_data)
    
    wp = auto_id_obj.workflow_params
    apply_bounds = wp.get('apply_suggested_bounds', False)

    logger.info("Processing compounds...")
    for atlas_compound_mzrt in tqdm(auto_id_obj.pre_autoid_atlas_obj.compound_mzrts.values(), desc="Creating manual curation objects"):
        
        # 1. Start with a dictionary instead of a 1-row DataFrame
        # This is 100x faster than pd.DataFrame([{}])
        meta = {
            'mz_rt_uid': atlas_compound_mzrt.mz_rt_uid,
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
            'mz': 0.0,
            'atlas_rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'atlas_rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'atlas_rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'initial_rt_min': 0.0,
            'initial_rt_max': 0.0,
            'rt_error': 0.0,
            'mz_error': 0.0,
            'ms1_notes': '', 'ms2_notes': '', 'other_notes': '',
            'identification_notes': getattr(atlas_compound_mzrt, 'identification_notes', ''),
            'analyst_notes': '',
            'best_ms1_file': '', 'best_ms1_rt': 0.0, 'best_ms1_mz': 0.0,
            'best_ms1_intensity': 0.0, 'best_ms1_ppm_error': 0.0, 'best_ms1_rt_error': 0.0,
            'max_eic_rt': [], 'max_eic_intensity': [],
            'isomers': isomer_dict.get(atlas_compound_mzrt.mz_rt_uid, []),
            'suggested_rt_min': 0.0, 'suggested_rt_max': 0.0, 'suggested_rt_peak': 0.0,
            'rt_suggestion_confidence': 0.0
        }

        # Finds best MS1 data and creates suggested rt bounds
        ms1_results = _analyze_ms1_full(
            atlas_compound_mzrt, 
            ms1_index.get(atlas_compound_mzrt.mz_rt_uid, []), 
            apply_bounds=apply_bounds
        )
        if ms1_results:
            meta.update(ms1_results)
            meta['auto_ided'] = True

        # Create object
        manual_curation_obj = ManualCuration(
            mz_rt_uid=atlas_compound_mzrt.mz_rt_uid,
            compound_uid=atlas_compound_mzrt.compound_uid,
            inchi_key=atlas_compound_mzrt.inchi_key,
            adduct=atlas_compound_mzrt.adduct,
            data=meta
        )
        auto_id_obj.experimental_data.manual_curation.append(manual_curation_obj)

    return auto_id_obj.experimental_data.manual_curation

def _analyze_ms1_full(atlas_mzrt, ms1_list, apply_bounds=False) -> Dict:
    """Consolidated function to handle all MS1-based metadata extraction."""
    if not ms1_list:
        return {}

    # State for best file, RT suggestion, and EIC
    best_file = None
    max_height = -np.inf
    
    rt_arrays, int_arrays, mz_arrays = [], [], []
    window_mz, window_int, window_rt = [], [], []
    
    atlas_mz = getattr(atlas_mzrt, 'mz', 0.0)
    atlas_rt_peak = getattr(atlas_mzrt, 'rt_peak', 0.0)
    atlas_rt_min = getattr(atlas_mzrt, 'rt_min', 0.0)
    atlas_rt_max = getattr(atlas_mzrt, 'rt_max', 0.0)

    # One single loop over MS1 files
    for ms1 in ms1_list:
        if ms1.data is None or ms1.data.empty:
            continue
        
        # Convert to numpy once
        df = ms1.data
        rts, ints, mzs = df['rt'].to_numpy(), df['i'].to_numpy(), df['mz'].to_numpy()
        
        # A. Track Best File
        sum_int = ints.sum()
        if sum_int > 0:
            idx = np.argmax(ints)
            h = ints[idx]
            if h > max_height:
                max_height = h
                centroid = (ints * mzs).sum() / sum_int
                best_file = {
                    'best_ms1_file': ms1.filename,
                    'best_ms1_rt': float(rts[idx]),
                    'best_ms1_mz': float(centroid),
                    'best_ms1_intensity': float(h),
                    'best_ms1_ppm_error': float((centroid - atlas_mz) / atlas_mz * 1e6) if atlas_mz else 0.0,
                    'best_ms1_rt_error': float(rts[idx] - atlas_rt_peak)
                }

        # B. Collect for EIC and Window Metrics
        rt_arrays.append(rts)
        int_arrays.append(ints)
        
        mask = (rts >= atlas_rt_min) & (rts <= atlas_rt_max) & np.isfinite(rts) & np.isfinite(ints)
        if np.any(mask):
            window_mz.extend(mzs[mask])
            window_int.extend(ints[mask])
            window_rt.extend(rts[mask])

    if not rt_arrays:
        return {}

    # Calculate EIC
    all_rts = np.unique(np.concatenate(rt_arrays))
    # Use a generator expression inside np.max for memory efficiency
    max_eic_intensity = np.max([np.interp(all_rts, r, i, left=0, right=0) 
                                 for r, i in zip(rt_arrays, int_arrays)], axis=0)

    # Calculate Window Metrics
    if window_int:
        idx_max = np.argmax(window_int)
        win_rt_peak = float(window_rt[idx_max])
        win_mz_mean = float(np.mean(window_mz))
        rt_err = win_rt_peak - atlas_rt_peak
        mz_err = (win_mz_mean - atlas_mz) / atlas_mz * 1e6 if atlas_mz else 0.0
    else:
        win_rt_peak, win_mz_mean, rt_err, mz_err = atlas_rt_peak, atlas_mz, 0.0, 0.0

    # RT Suggestion logic (uses the best trace found)
    # Find the trace that provided 'best_file' to avoid another loop
    best_trace_data = None
    if best_file:
        for ms1 in ms1_list:
            if ms1.filename == best_file['best_ms1_file']:
                best_trace_data = {'rt': ms1.data['rt'].values, 'i': ms1.data['i'].values}
                break

    suggestion = _suggest_rt_bounds_from_ms1(
        best_trace_data, atlas_rt_peak, atlas_rt_min, atlas_rt_max
    ) if best_trace_data else None

    # Assemble result dictionary
    res = best_file.copy() if best_file else {}
    res.update({
        'mz': win_mz_mean,
        'rt_peak': win_rt_peak,
        'mz_error': mz_err,
        'rt_error': rt_err,
        'max_eic_rt': all_rts.tolist(),
        'max_eic_intensity': max_eic_intensity.tolist(),
    })

    if suggestion:
        res.update({
            'suggested_rt_min': suggestion['rt_min'],
            'suggested_rt_max': suggestion['rt_max'],
            'suggested_rt_peak': suggestion['rt_peak'],
            'rt_suggestion_confidence': suggestion['confidence'],
        })
        if suggestion['confidence'] > 0.75 and apply_bounds:
            res['rt_min'] = suggestion['rt_min']
            res['rt_max'] = suggestion['rt_max']
            res['rt_peak'] = 0.5 * (suggestion['rt_min'] + suggestion['rt_max'])
            res['initial_rt_min'] = res['rt_min']
            res['initial_rt_max'] = res['rt_max']
    else:
        res.update({
            'suggested_rt_min': atlas_rt_min, 'suggested_rt_max': atlas_rt_max,
            'suggested_rt_peak': atlas_rt_peak, 'rt_suggestion_confidence': 0.0,
            'initial_rt_min': atlas_rt_min, 'initial_rt_max': atlas_rt_max
        })

    return res

def _build_isomer_dict(atlas_obj: "Atlas") -> Dict[str, List[Dict[str, Any]]]:
    atlas_df = atlas_obj.to_dataframe().reset_index(drop=True)
    n = len(atlas_df)
    mzs = atlas_df["mz"].to_numpy(dtype=float)
    masses = atlas_df["mono_isotopic_molecular_weight"].to_numpy(dtype=float) if "mono_isotopic_molecular_weight" in atlas_df.columns else np.full(n, np.nan)
    inchi_prefixes = np.array([str(ik).split("-")[0] for ik in atlas_df["inchi_key"]], dtype=object)
    uids = atlas_df["mz_rt_uid"].to_numpy()

    # Vectorized distance matrices
    mz_sim = np.abs(mzs[:, None] - mzs[None, :]) <= 0.005
    mass_sim = np.abs(masses[:, None] - masses[None, :]) <= 0.005
    prefix_match = inchi_prefixes[:, None] == inchi_prefixes[None, :]
    
    isomer_mask = (mz_sim | mass_sim | prefix_match)
    np.fill_diagonal(isomer_mask, False) # Remove self

    isomer_dict = {}
    # Optimization: instead of iterrows, use boolean indices to slice the dataframe once per compound
    for i in range(n):
        match_indices = np.where(isomer_mask[i])[0]
        if match_indices.size > 0:
            matches = atlas_df.iloc[match_indices]
            isomer_dict[uids[i]] = [
                {
                    "mz_rt_uid": row.mz_rt_uid, "inchi_key": row.inchi_key,
                    "adduct": row.adduct, "compound_name": row.get("compound_name", ""),
                    "rt": row.rt_peak, "mz": row.mz,
                } for row in matches.itertuples()
            ]
        else:
            isomer_dict[uids[i]] = []
            
    return isomer_dict

def _build_ms1_index(exp_data_obj: "ExperimentalData") -> Dict[str, List]:
    ms1_index = {}
    for ms1 in exp_data_obj.ms1_data:
        uid = getattr(ms1, 'mz_rt_uid', None)
        if uid:
            ms1_index.setdefault(uid, []).append(ms1)
    return ms1_index

def _suggest_rt_bounds_from_ms1(best_trace_data, atlas_rt_peak, atlas_rt_min, atlas_rt_max):
    if not best_trace_data:
        return None
    
    rt = np.asarray(best_trace_data.get("rt"), dtype=np.float64)
    intensity = np.asarray(best_trace_data.get("i"), dtype=np.float64)

    # Match legacy filtering: keep positive intensities and finite values.
    valid_mask = (intensity > 0) & (~np.isnan(rt)) & (~np.isnan(intensity))
    if not np.any(valid_mask):
        return None
    rt = rt[valid_mask]
    intensity = intensity[valid_mask]

    if rt.size < 5:
        return None

    theoretical_rt = atlas_rt_peak

    smoothed_intensity = gaussian_filter1d(intensity, sigma=0.8)
    max_intensity = float(np.max(smoothed_intensity))
    min_height = max_intensity * 0.10
    min_prominence = max_intensity * 0.05

    peaks, properties = find_peaks(
        smoothed_intensity,
        height=min_height,
        prominence=min_prominence,
        distance=3,
    )

    if peaks.size == 0:
        peaks = np.array([int(np.argmax(smoothed_intensity))])
        properties = {
            "prominences": np.array([max_intensity * 0.5]),
            "widths": np.array([10.0]),
            "left_bases": np.array([max(0, peaks[0] - 5)]),
            "right_bases": np.array([min(len(rt) - 1, peaks[0] + 5)]),
        }

    # Match legacy selection: choose detected peak closest to expected RT.
    peak_rts = rt[peaks]
    closest_peak_idx = int(np.argmin(np.abs(peak_rts - theoretical_rt)))
    main_peak_idx = int(peaks[closest_peak_idx])
    peak_rt = float(rt[main_peak_idx])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PeakPropertyWarning)
            _, _, left_ips, right_ips = peak_widths(
                smoothed_intensity,
                [main_peak_idx],
                rel_height=0.5,
            )
        left_idx = int(np.floor(left_ips[0]))
        right_idx = int(np.ceil(right_ips[0]))
        left_idx = max(0, left_idx)
        right_idx = min(len(rt) - 1, right_idx)
    except Exception:
        left_idx = int(properties["left_bases"][closest_peak_idx])
        right_idx = int(properties["right_bases"][closest_peak_idx])

    rt_left = float(rt[left_idx])
    rt_right = float(rt[right_idx])
    width_rt = rt_right - rt_left
    pad = max(0.05, width_rt * 0.05)
    rt_min_suggested = rt_left - pad
    rt_max_suggested = rt_right + pad

    # Match legacy bound constraints.
    rt_min_suggested = max(rt_min_suggested, float(rt[0]))
    rt_max_suggested = min(rt_max_suggested, float(rt[-1]))

    final_width = rt_max_suggested - rt_min_suggested
    if final_width < 0.08:
        rt_min_suggested = peak_rt - 0.04
        rt_max_suggested = peak_rt + 0.04
    elif final_width > 0.6:
        rt_min_suggested = peak_rt - 0.3
        rt_max_suggested = peak_rt + 0.3

    prominence = float(properties["prominences"][closest_peak_idx])
    peak_intensity = float(smoothed_intensity[main_peak_idx])

    intensity_score = min(1.0, peak_intensity / max(max_intensity, 1e3))
    prominence_score = min(1.0, prominence / max(peak_intensity * 0.5, 1e3))

    optimal_width = 0.3
    width_score = 1.0 - min(1.0, abs(final_width - optimal_width) / optimal_width)

    max_expected_dev = max(abs(atlas_rt_max - atlas_rt_min), 0.5)
    rt_dev = abs(peak_rt - theoretical_rt)
    rt_score = max(0.0, 1.0 - (rt_dev / max_expected_dev))

    left_tail = main_peak_idx - left_idx
    right_tail = right_idx - main_peak_idx
    if left_tail + right_tail > 0:
        asym = abs(left_tail - right_tail) / (left_tail + right_tail)
        shape_score = max(0.0, 1.0 - asym)
    else:
        shape_score = 0.5

    confidence = (
        0.30 * intensity_score +
        0.20 * prominence_score +
        0.15 * width_score +
        0.20 * rt_score +
        0.15 * shape_score
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < 0.1:
        return None
    if final_width > (atlas_rt_max - atlas_rt_min) * 1.5:
        confidence *= 0.6
    if rt_dev > max_expected_dev * 1.5:
        confidence *= 0.5

    return {
        "rt_min": float(rt_min_suggested),
        "rt_max": float(rt_max_suggested),
        "rt_peak": float(peak_rt),
        "confidence": float(confidence),
    }