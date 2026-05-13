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

def create_manual_curation_obj(
    auto_id_obj: "AutoIdentification",
) -> "ManualCuration":
    """
    Create ManualCuration object from CompoundMZRT.
    """
    from metatlas2.workflow_objects import ManualCuration

    logger.info("Creating ManualCuration objects for each Compound in the Atlas...")
    
    logger.info("Building MS1 index and isomer dictionary...")
    isomer_dict = _build_isomer_dict(auto_id_obj.pre_autoid_atlas_obj)
    ms1_index = _build_ms1_index(auto_id_obj.experimental_data)
    
    logger.info("Starting Compound loop...")
    for atlas_compound_mzrt in tqdm(auto_id_obj.pre_autoid_atlas_obj.compound_mzrts.values(), desc="Creating manual curation objects"):

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
            'rt_peak': getattr(atlas_compound_mzrt, 'rt_peak', 0.0),
            'rt_min': getattr(atlas_compound_mzrt, 'rt_min', 0.0),
            'rt_max': getattr(atlas_compound_mzrt, 'rt_max', 0.0),
            'ms1_notes': '',
            'ms2_notes': '',
            'other_notes': '',
            'identification_notes': getattr(atlas_compound_mzrt, 'identification_notes', ''),
            'analyst_notes': '',
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
            ms1_index
        )

        isomer_list = isomer_dict.get(atlas_compound_mzrt.inchi_key, [])
        manual_curation_obj.data.at[0, 'isomers'] = isomer_list

        _add_rt_suggestions_to_manual_curation_obj(
            manual_curation_obj,
            atlas_compound_mzrt,
            ms1_index
        )

        auto_id_obj.experimental_data.manual_curation.append(manual_curation_obj)

    return manual_curation_obj

def _build_ms1_index(
    exp_data_obj: "ExperimentalData"
) -> Dict[tuple, List]:
    """
    Build an index of MS1 data by (inchi_key, adduct) for fast lookup.
    Returns a dict: (inchi_key, adduct) -> list of MS1 _SpecData objects
    """
    ms1_index = {}
    for ms1 in exp_data_obj.ms1_data:
        key = (ms1.inchi_key, ms1.adduct)
        if key not in ms1_index:
            ms1_index[key] = []
        ms1_index[key].append(ms1)
    return ms1_index

def _fill_best_ms1_to_manual_curation(
    manual_curation_obj: "ManualCuration",
    atlas_compound_mzrt: "CompoundMZRT",
    ms1_index: Dict[tuple, List]
) -> None:
    """
    For a given atlas compound, find the best MS1 file (highest intensity)
    and fill manual_curation_obj.data with its info.
    Uses pre-built ms1_index for fast lookup.
    """
    all_files = []
    key = (atlas_compound_mzrt.inchi_key, atlas_compound_mzrt.adduct)
    ms1_list = ms1_index.get(key, [])
    
    for ms1 in ms1_list:
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

def _build_isomer_dict(
    atlas_obj: "Atlas"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return a dict: inchi_key → list of isomer dicts (empty list if none).
    Isomers are defined as:
      - mz or mono_isotopic_molecular_weight within 0.005
      - OR inchi_key prefix (before '-') identical
    """
    atlas_df = atlas_obj.to_dataframe().reset_index(drop=True)
    n = len(atlas_df)
    mzs = atlas_df["mz"].to_numpy(dtype=float)
    # Some rows may not have mono_isotopic_molecular_weight
    if "mono_isotopic_molecular_weight" in atlas_df.columns:
        masses = atlas_df["mono_isotopic_molecular_weight"].to_numpy(dtype=float)
    else:
        masses = np.full(n, np.nan)
    inchi_keys = atlas_df["inchi_key"].astype(str).to_numpy()
    inchi_prefixes = np.array([ik.split("-")[0] for ik in inchi_keys], dtype=object)

    # Pairwise comparisons
    mz_i, mz_j = mzs[:, None], mzs[None, :]
    mz_valid = (~np.isnan(mz_i)) & (~np.isnan(mz_j))
    mz_similar = mz_valid & (np.abs(mz_i - mz_j) <= 0.005)

    m_i, m_j = masses[:, None], masses[None, :]
    mass_valid = (~np.isnan(m_i)) & (~np.isnan(m_j))
    mass_similar = mass_valid & (np.abs(m_i - m_j) <= 0.005)

    prefix_i, prefix_j = inchi_prefixes[:, None], inchi_prefixes[None, :]
    prefix_match = prefix_i == prefix_j

    # Exclude self-matches for isomer detection
    self_mask = np.eye(n, dtype=bool)

    isomer_mask = (mz_similar | mass_similar | prefix_match) & (~self_mask)

    isomer_dict: Dict[str, List[Dict[str, Any]]] = {}
    for idx, row in atlas_df.iterrows():
        isomer_idxs = np.where(isomer_mask[idx])[0]
        isomer_dict[row["inchi_key"]] = [
            {
                "inchi_key": atlas_df.iloc[j]["inchi_key"],
                "adduct": atlas_df.iloc[j]["adduct"],
                "compound_name": atlas_df.iloc[j].get("compound_name", ""),
                "rt": atlas_df.iloc[j]["rt_peak"],
                "mz": atlas_df.iloc[j]["mz"],
            }
            for j in isomer_idxs
        ]
    return isomer_dict

def _add_rt_suggestions_to_manual_curation_obj(
    manual_curation_obj: "ManualCuration",
    atlas_compound_mzrt: "CompoundMZRT",
    ms1_index: Dict[tuple, List]
) -> None:
    """
    Add RT bound suggestions to a ManualCuration object based on ms1 data.
    Uses pre-built ms1_index for fast lookup.
    """
    df = manual_curation_obj.data
    key = (atlas_compound_mzrt.inchi_key, atlas_compound_mzrt.adduct)
    ms1_list = ms1_index.get(key, [])
    
    ms1_data_dict = {}
    for ms1_data in ms1_list:
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
    Compute RT bounds from the highest-intensity ms1 trace.
    This matches the legacy suggest_rt_bounds() mechanism.
    """
    if not ms1_data:
        return None

    # Select the single file with highest ms1 intensity.
    best_trace: Optional[Dict[str, Any]] = None
    best_max_intensity = -np.inf
    for trace in ms1_data.values():
        i_vals = np.asarray(trace.get("i_vals", []), dtype=np.float64)
        if i_vals.size == 0:
            continue
        i_vals = i_vals[~np.isnan(i_vals)]
        if i_vals.size == 0:
            continue
        max_i = float(np.max(i_vals))
        if max_i > best_max_intensity:
            best_max_intensity = max_i
            best_trace = trace

    if best_trace is None:
        return None

    rt = np.asarray(best_trace.get("rt_vals", []), dtype=np.float64)
    intensity = np.asarray(best_trace.get("i_vals", []), dtype=np.float64)

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