import json
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Optional, List, Any
from tqdm.auto import tqdm
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.signal._peak_finding_utils import PeakPropertyWarning

import metatlas2.logging_config as lcf
logger = lcf.get_logger('curation_creator')

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

def create_manual_curation_obj(auto_id_obj) -> pd.DataFrame:
    """
    Builds the manual curation metadata table using the Tidy DataFrame architecture.
    Returns a DataFrame where each row is a compound's curation metadata.
    """
    logger.info("Loading experimental data and atlas for curation metadata creation...")
    atlas_df = auto_id_obj.auto_ided_atlas_obj.to_dataframe()
    ms1_df = auto_id_obj.experimental_data.ms1_df

    logger.info("Building indices and isomer map...")
    isomer_dict = _build_isomer_dict(auto_id_obj.auto_ided_atlas_obj)

    logger.info(f"Creating curation container from identified compounds starting from {len(atlas_df)} atlas compounds...")
    curation_records = []    
    ms1_groups = ms1_df.groupby("mz_rt_uid")
    for uid, atlas_row in tqdm(atlas_df.set_index("mz_rt_uid").iterrows(), total=len(atlas_df), desc="Building curation container", disable=should_disable_tqdm()):
        curation_entry = {
            'mz_rt_uid': uid,
            'compound_uid': atlas_row.get('compound_uid', ''),
            'inchi_key': atlas_row.get('inchi_key', ''),
            'adduct': atlas_row.get('adduct', ''),
            'compound_name': atlas_row.get('compound_name', ''),
            'passed_autoid': False,
            'passed_curation': False,
            'polarity': atlas_row.get('polarity', ''),
            'chromatography': atlas_row.get('chromatography', ''),
            'mz_tolerance': atlas_row.get('mz_tolerance', 5.0),
            'atlas_mz': atlas_row.get('mz', 0.0),
            'atlas_rt_peak': atlas_row.get('rt_peak', 0.0),
            'atlas_rt_min': atlas_row.get('rt_min', 0.0),
            'atlas_rt_max': atlas_row.get('rt_max', 0.0),
            'mz': atlas_row.get('mz', 0.0),
            'rt_peak': atlas_row.get('rt_peak', 0.0),
            'rt_min': atlas_row.get('rt_min', 0.0),
            'rt_max': atlas_row.get('rt_max', 0.0),
            'initial_rt_min': atlas_row.get('rt_min', 0.0),
            'initial_rt_max': atlas_row.get('rt_max', 0.0),
            'rt_error': 0.0,
            'mz_error': 0.0,
            'ms1_notes': '',
            'ms2_notes': '',
            'other_notes': '',
            'identification_notes': atlas_row.get('identification_notes', ''),
            'analyst_notes': '',
            'best_ms1_file': '',
            'best_ms1_rt': 0.0,
            'best_ms1_mz': 0.0,
            'best_ms1_intensity': 0.0,
            'best_ms1_ppm_error': 0.0,
            'best_ms1_rt_error': 0.0,
            'max_eic_rt': [],
            'max_eic_intensity': [],
            'isomers': isomer_dict.get(uid, []),
            'suggested_rt_min': 0.0,
            'suggested_rt_max': 0.0,
            'suggested_rt_peak': 0.0,
            'rt_suggestion_confidence': 0.0,
            # PubChem / chemical metadata — carried from the atlas (compounds table join)
            'formula': atlas_row.get('formula', ''),
            'smiles': atlas_row.get('smiles', ''),
            'inchi': atlas_row.get('inchi', ''),
            'pubchem_cid': atlas_row.get('pubchem_cid', ''),
            'mono_isotopic_molecular_weight': atlas_row.get('mono_isotopic_molecular_weight', 0.0),
            'iupac_name': atlas_row.get('iupac_name', ''),
        }

        try:
            compound_ms1 = ms1_groups.get_group(uid)
            ms1_summary = analyze_ms1(
                atlas_row,
                compound_ms1,
                apply_bounds_cutoff=auto_id_obj.ta.params.get('suggested_min_conf', None)
            )
            if ms1_summary:
                curation_entry.update(ms1_summary)
                curation_entry['passed_autoid'] = True
        except KeyError: # no MS1 data for this compound
            pass

        if curation_entry['passed_autoid'] is False and auto_id_obj.ta.params.get('remove_unided_compounds', False) is True:
            continue

        curation_records.append(curation_entry)

    manual_curation_df = pd.DataFrame(curation_records)
    manual_curation_df['isomers'] = manual_curation_df['isomers'].apply(json.dumps)
    logger.info(f"Summary: manual_curation_df contains {manual_curation_df['mz_rt_uid'].nunique()} unique mz_rt_uids built from {atlas_df.shape[0]} atlas entries.")
    
    auto_id_obj.experimental_data.curation_df = manual_curation_df

    return

def analyze_ms1(atlas_row, compound_ms1_df, stage="manual_curation_creator",apply_bounds_cutoff=None) -> dict:
    """
    Analyzes MS1 data using wide format (one row per file, lists of rts, intensities, mzs, in_feature).
    Aggregates across all files for the compound.
    """
    if compound_ms1_df.empty:
        return {}

    # Setup variables
    best_file_info = None
    max_height = -np.inf
    atlas_mz = atlas_row.get('mz', 0.0)
    atlas_rt_peak = atlas_row.get('rt_peak', 0.0)
    atlas_rt_min = atlas_row.get('rt_min', 0.0)
    atlas_rt_max = atlas_row.get('rt_max', 0.0)

    # 1. Per-file analysis for Best File and EIC (wide format)
    rt_arrays, int_arrays = [], []  # For all data (for EIC)
    in_feature_mz, in_feature_int, in_feature_rt = [], [], []  # For in_feature only
    per_file_peak_rts = []  # RT of the highest-intensity in_feature point per file

    # Each row is one file for this compound, with lists in columns
    for _, row in compound_ms1_df.iterrows():
        filename = row.get('filename', None)
        # Wide format columns: spec_rts, spec_ints, spec_mzs, in_feature
        rt_list = row.get('spec_rts', [])
        intensity_list = row.get('spec_ints', [])
        mz_list = row.get('spec_mzs', [])
        in_feature_mask = row.get('in_feature', [True]*len(rt_list))

        # Convert to numpy arrays for easier indexing
        rt_arr = np.asarray(rt_list)
        int_arr = np.asarray(intensity_list)
        mz_arr = np.asarray(mz_list)
        mask = np.asarray(in_feature_mask, dtype=bool)

        # Store for EIC calculation (all data)
        if stage == "manual_curation_creator":
            if len(rt_arr) > 0 and len(int_arr) == len(rt_arr):
                rt_arrays.append(rt_arr)
                int_arrays.append(int_arr)

        # Store for in_feature calculations
        if np.any(mask):
            in_feature_ints = int_arr[mask]
            in_feature_rts = rt_arr[mask]
            in_feature_mzs = mz_arr[mask]
            in_feature_mz.extend(in_feature_mzs.tolist())
            in_feature_int.extend(in_feature_ints.tolist())
            in_feature_rt.extend(in_feature_rts.tolist())
            sum_int = in_feature_ints.sum()
            if sum_int > 0:
                idx = np.argmax(in_feature_ints)
                h = in_feature_ints[idx]
                per_file_peak_rts.append(float(in_feature_rts[idx]))
                if h > max_height:
                    max_height = h
                    centroid = (in_feature_ints * in_feature_mzs).sum() / sum_int
                    best_file_info = {
                        'best_ms1_file': filename,
                        'best_ms1_rt': float(in_feature_rts[idx]),
                        'best_ms1_mz': float(centroid),
                        'best_ms1_intensity': float(h),
                        'best_ms1_ppm_error': float((centroid - atlas_mz) / atlas_mz * 1e6) if atlas_mz else 0.0,
                        'best_ms1_rt_error': float(in_feature_rts[idx] - atlas_rt_peak)
                    }

    if stage == "manual_curation_creator":
        if not rt_arrays or not in_feature_rt:
            return {}

    if not in_feature_rt or not in_feature_int:
        return {}

    if stage == "manual_curation_creator":
        # 2. EIC Calculation (Vectorized, all data)
        all_rts = np.unique(np.concatenate(rt_arrays))
        max_eic_intensity = np.max([
            np.interp(all_rts, r, i, left=0, right=0)
            for r, i in zip(rt_arrays, int_arrays)
        ], axis=0)

    # 3. Window Metrics (in_feature only)
    # rt_peak = mean of each file's highest-intensity in_feature RT point
    # mz      = mean of all in_feature mzs across all files
    win_rt_peak = float(np.mean(per_file_peak_rts)) if per_file_peak_rts else float(in_feature_rt[np.argmax(in_feature_int)])
    win_mz_mean = float(np.mean(in_feature_mz))
    rt_err = win_rt_peak - atlas_rt_peak
    mz_err = (win_mz_mean - atlas_mz) / atlas_mz * 1e6 if atlas_mz else 0.0

    if stage == "manual_curation_creator":
        # 4. RT Suggestion (using average EIC across all files)
        avg_trace_data = None
        if rt_arrays:
            all_rts = np.unique(np.concatenate(rt_arrays))  # already computed above
            avg_eic_intensity = np.mean([
                np.interp(all_rts, r, i, left=0, right=0)
                for r, i in zip(rt_arrays, int_arrays)
            ], axis=0)
            avg_trace_data = {
                'rt': all_rts,
                'i': avg_eic_intensity
            }

        suggestion = _suggest_rt_bounds_from_ms1(
            avg_trace_data, atlas_rt_peak, atlas_rt_min, atlas_rt_max
        ) if avg_trace_data else None

    # Assemble result
    res = best_file_info.copy() if best_file_info else {}
    if stage == "manual_curation_creator":
        res.update({
            'mz': win_mz_mean,
            'rt_peak': win_rt_peak,
            'mz_error': mz_err,
            'rt_error': rt_err,
            'max_eic_rt': all_rts.tolist(),
            'max_eic_intensity': max_eic_intensity.tolist(),
        })
    elif stage == "post_curation_summary":
        res.update({
            'mz': win_mz_mean,
            'rt_peak': win_rt_peak,
            'mz_error': mz_err,
            'rt_error': rt_err,
        })

    # Always set rt_min and rt_max based on atlas bounds and rt_peak
    if stage == "manual_curation_creator":
        atlas_rt_peak = atlas_row.get('rt_peak', 0.0)
        atlas_rt_min = atlas_row.get('rt_min', 0.0)
        atlas_rt_max = atlas_row.get('rt_max', 0.0)
        rt_peak = win_rt_peak
        rt_min = max(0.0, rt_peak - (atlas_rt_peak - atlas_rt_min))
        rt_max = rt_peak + (atlas_rt_max - atlas_rt_peak)
        res['rt_min'] = rt_min
        res['rt_max'] = rt_max
        res['initial_rt_min'] = rt_min
        res['initial_rt_max'] = rt_max

    if stage == "manual_curation_creator":
        if suggestion:
            res.update({
                'suggested_rt_min': suggestion['rt_min'],
                'suggested_rt_max': suggestion['rt_max'],
                'suggested_rt_peak': suggestion['rt_peak'],
                'rt_suggestion_confidence': suggestion['confidence'],
            })
            if apply_bounds_cutoff is not None and suggestion['confidence'] > apply_bounds_cutoff:
                res.update({
                    'rt_min': suggestion['rt_min'],
                    'rt_max': suggestion['rt_max'],
                    'rt_peak': win_rt_peak,
                    'initial_rt_min': suggestion['rt_min'],
                    'initial_rt_max': suggestion['rt_max']
                })
        else:
            res.update({
                'suggested_rt_min': atlas_rt_min, 
                'suggested_rt_max': atlas_rt_max,
                'suggested_rt_peak': atlas_rt_peak, 
                'rt_suggestion_confidence': 0.0,
                'initial_rt_min': atlas_rt_min,
                'initial_rt_max': atlas_rt_max
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
                    "adduct": row.adduct, "compound_name": row.compound_name,
                    "rt": row.rt_peak, "mz": row.mz,
                } for row in matches.itertuples()
            ]
        else:
            isomer_dict[uids[i]] = []
            
    return isomer_dict

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