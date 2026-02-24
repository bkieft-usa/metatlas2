import sys
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from typing import Dict, Optional, List, Any

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, peak_prominences

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('ms1_ms2_summarizer')

def create_ms_summaries(
    exp_data_obj: "ExperimentalData",
    atlas_obj: "Atlas",
) -> "ExperimentalData":
    """
    Create MS1Summary and MS2Summary objects for each compound/adduct/file and add them to ExperimentalData.
    Returns:
        Updated ExperimentalData object.
    """

    from workflow_objects import MS1Summary, MS2Summary, ManualCuration

    logger.info("Calculating MS1 and MS2 summary statistics for each compound and file...")

    logger.info("Building lookups for each data type...")
    ms1_lookup = {}
    for ms1 in exp_data_obj.ms1_data:
        key = (ms1.inchi_key, ms1.adduct, ms1.filename)
        ms1_lookup[key] = ms1.data
    logger.info(f"  MS1 lookup built with {len(ms1_lookup)} entries.")

    ms2_lookup = {}
    for ms2 in exp_data_obj.ms2_data:
        key = (ms2.inchi_key, ms2.adduct, ms2.filename)
        ms2_lookup[key] = ms2.data
    logger.info(f"  MS2 lookup built with {len(ms2_lookup)} entries.")

    ms2_hits_lookup = {}
    for ms2_hit in exp_data_obj.ms2_hits:
        key = (ms2_hit.inchi_key, ms2_hit.adduct, ms2_hit.filename)
        ms2_hits_lookup[key] = ms2_hit.data
    logger.info(f"  MS2 hits lookup built with {len(ms2_hits_lookup)} entries.")

    logger.info("Creating MS1 and MS2 summaries...")
    atlas_df = atlas_obj.to_dataframe()
    for atlas_compound_mzrt in tqdm(atlas_obj.compound_mzrts.values(), desc="Processing atlas compounds", unit="compound"):
        atlas_inchi_key = atlas_compound_mzrt.inchi_key
        atlas_adduct = atlas_compound_mzrt.adduct

        # Find all filenames for this compound/adduct
        filenames = set(
            fname for exp_data_inchi, exp_data_adduct, fname in ms1_lookup.keys()
            if exp_data_inchi == atlas_inchi_key and exp_data_adduct == atlas_adduct
        )
        filenames.update(
            fname for exp_data_inchi, exp_data_adduct, fname in ms2_lookup.keys()
            if exp_data_inchi == atlas_inchi_key and exp_data_adduct == atlas_adduct
        )
        filenames.update(
            fname for exp_data_inchi, exp_data_adduct, fname in ms2_hits_lookup.keys()
            if exp_data_inchi == atlas_inchi_key and exp_data_adduct == atlas_adduct
        )

        for filename in filenames:
            ms1_df = ms1_lookup.get((atlas_inchi_key, atlas_adduct, filename), pd.DataFrame())
            ms2_df = ms2_lookup.get((atlas_inchi_key, atlas_adduct, filename), pd.DataFrame())
            ms2_hits_df = ms2_hits_lookup.get((atlas_inchi_key, atlas_adduct, filename), pd.DataFrame())

            ms1_summary_df = _calculate_ms1_summary(ms1_df, atlas_compound_mzrt)
            ms2_summary_df = _calculate_ms2_summary(ms2_df, ms2_hits_df, atlas_compound_mzrt)

            ms1_summary_obj = MS1Summary(
                inchi_key=atlas_inchi_key,
                adduct=atlas_adduct,
                filename=filename,
                data=ms1_summary_df
            )
            ms2_summary_obj = MS2Summary(
                inchi_key=atlas_inchi_key,
                adduct=atlas_adduct,
                filename=filename,
                data=ms2_summary_df
            )

            exp_data_obj.ms1_summaries.append(ms1_summary_obj)
            exp_data_obj.ms2_summaries.append(ms2_summary_obj)

        manual_curation_obj = _create_manual_curation_obj(
            atlas_compound_mzrt=atlas_compound_mzrt,
            exp_data_obj=exp_data_obj,
            whole_atlas_df=atlas_df
        )
        exp_data_obj.manual_curation.append(manual_curation_obj)

    logger.info("Finished processing all atlas compounds:")
    print_summary_statistics(exp_data_obj)

    return exp_data_obj

def _create_manual_curation_obj(
    atlas_compound_mzrt: "CompoundMZRT",
    exp_data_obj: "ExperimentalData",
    whole_atlas_df: pd.DataFrame
) -> "ManualCuration":
    """
    Create ManualCuration object from CompoundMZRT.
    """
    from workflow_objects import ManualCuration

    compound_data = {
        'compound_uid': atlas_compound_mzrt.compound_uid,
        'compound_name': getattr(atlas_compound_mzrt, 'compound_name', getattr(atlas_compound_mzrt, 'label', '')),
        'formula': getattr(atlas_compound_mzrt, 'formula', ''),
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
        'identification_notes': getattr(atlas_compound_mzrt, 'identification_notes', ''),
        'ms1_notes': 'keep',
        'ms2_notes': 'no selection',
        'analyst_notes': '',
        'is_rt_modified': False,
        'is_annotation_modified': False,
        'total_files_detected': 0,
        'ms2_files_with_data': 0,
        'ms2_files_with_hits': 0,
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
    }

    manual_curation_obj = ManualCuration(
        inchi_key=atlas_compound_mzrt.inchi_key, 
        adduct=atlas_compound_mzrt.adduct, 
        data=pd.DataFrame([compound_data])
    )

    _calculate_best_ms1_across_files(manual_curation_obj, exp_data_obj)
    _add_isomers_to_manual_curation_obj(manual_curation_obj, whole_atlas_df)
    _add_rt_suggestions_to_manual_curation_obj(manual_curation_obj, exp_data_obj)

    return manual_curation_obj


def _calculate_ms1_summary(
    df: pd.DataFrame, 
    atlas_compound_mzrt: "CompoundMZRT"
) -> pd.DataFrame:
    inchi_key = atlas_compound_mzrt.inchi_key
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
        return pd.DataFrame(columns=summary_schema.keys())
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
        summary_schema['ppm_error'] = float((mz_centroid - atlas_compound_mzrt.mz) / atlas_compound_mzrt.mz * 1e6)
        summary_schema['rt_error'] = float(peak_rt - atlas_compound_mzrt.rt_peak)
    return pd.DataFrame([summary_schema])

def _calculate_ms2_summary(
    ms2_df: pd.DataFrame,
    ms2_hits_df: pd.DataFrame,
    atlas_compound_mzrt: "CompoundMZRT"
) -> pd.DataFrame:
    inchi_key = atlas_compound_mzrt.inchi_key
    adduct = getattr(atlas_compound_mzrt, "adduct", "")
    # Unified schema for all cases
    columns = [
        "inchi_key", "adduct", "num_scans", "num_fragments", "best_scan_rt", "best_scan_precursor_mz",
        "best_scan_precursor_intensity", "rt_error", "ppm_error", "total_hits",
        "ref_id", "ref_name", "database", "score", "num_matches", "mz_theoretical", "mz_measured",
        "ppm_error_hit", "rt_measured", "qry_intensity_peak", "ref_frags", "data_frags",
        "matched_fragments", "qry_frag_colors", "qry_spectrum", "ref_spectrum",
        "qry_spectrum_original", "ref_spectrum_original"
    ]

    # If no MS2 data or hits, return empty DataFrame with schema
    if ms2_df.empty and ms2_hits_df.empty:
        return pd.DataFrame(columns=columns)

    # If MS2 data but no hits, summarize best scan and leave hit fields blank
    elif not ms2_df.empty and ms2_hits_df.empty:
        num_scans = int(ms2_df['rt'].nunique())
        num_fragments = int(len(ms2_df))
        best_scan_idx = ms2_df.groupby('rt')['precursor_intensity'].first().idxmax()
        best_scan = ms2_df[ms2_df['rt'] == best_scan_idx].iloc[0]
        best_rt = float(best_scan['rt'])
        fragment_mz = ms2_df[ms2_df['rt'] == best_rt]['mz'].values
        fragment_intensity = ms2_df[ms2_df['rt'] == best_rt]['i'].values
        row = {
            "inchi_key": inchi_key,
            "adduct": adduct,
            "num_scans": num_scans,
            "num_fragments": num_fragments,
            "best_scan_rt": best_rt,
            "best_scan_precursor_mz": float(best_scan['precursor_MZ']),
            "best_scan_precursor_intensity": float(best_scan['precursor_intensity']),
            "rt_error": float(best_rt - atlas_compound_mzrt.rt_peak),
            "ppm_error": float((best_scan['precursor_MZ'] - atlas_compound_mzrt.mz) / atlas_compound_mzrt.mz * 1e6),
            "total_hits": 0,
            # Hit fields blank/None
            "ref_id": None,
            "ref_name": None,
            "database": None,
            "score": None,
            "num_matches": None,
            "mz_theoretical": None,
            "mz_measured": None,
            "ppm_error_hit": None,
            "rt_measured": None,
            "qry_intensity_peak": None,
            "ref_frags": None,
            "data_frags": None,
            "matched_fragments": None,
            "qry_frag_colors": None,
            "qry_spectrum": None,
            "ref_spectrum": None,
            "qry_spectrum_original": [fragment_mz.tolist(), fragment_intensity.tolist()],
            "ref_spectrum_original": None,
        }
        return pd.DataFrame([row], columns=columns)

    # If MS2 hits exist, return all hits, filling summary fields for each hit
    elif not ms2_df.empty and not ms2_hits_df.empty:
        num_scans = int(ms2_df['rt'].nunique()) if not ms2_df.empty else None
        num_fragments = int(len(ms2_df)) if not ms2_df.empty else None
        best_scan_idx = ms2_df.groupby('rt')['precursor_intensity'].first().idxmax() if not ms2_df.empty else None
        best_scan = ms2_df[ms2_df['rt'] == best_scan_idx].iloc[0] if not ms2_df.empty else None
        best_rt = float(best_scan['rt']) if best_scan is not None else None
        best_precursor_mz = float(best_scan['precursor_MZ']) if best_scan is not None else None
        best_precursor_intensity = float(best_scan['precursor_intensity']) if best_scan is not None else None
        rt_error = float(best_rt - atlas_compound_mzrt.rt_peak) if best_scan is not None else None
        ppm_error = float((best_precursor_mz - atlas_compound_mzrt.mz) / atlas_compound_mzrt.mz * 1e6) if best_scan is not None else None

        rows = []
        for _, hit in ms2_hits_df.iterrows():
            row = {
                "inchi_key": inchi_key,
                "adduct": adduct,
                "num_scans": num_scans,
                "num_fragments": num_fragments,
                "best_scan_rt": best_rt,
                "best_scan_precursor_mz": best_precursor_mz,
                "best_scan_precursor_intensity": best_precursor_intensity,
                "rt_error": rt_error,
                "ppm_error": ppm_error,
                "total_hits": len(ms2_hits_df),
                # Hit fields
                "ref_id": hit.get("ref_id", None),
                "ref_name": hit.get("ref_name", None),
                "database": hit.get("database", None),
                "score": hit.get("score", None),
                "num_matches": hit.get("num_matches", None),
                "mz_theoretical": hit.get("mz_theoretical", None),
                "mz_measured": hit.get("mz_measured", None),
                "ppm_error_hit": hit.get("ppm_error", None),
                "rt_measured": hit.get("rt_measured", None),
                "qry_intensity_peak": hit.get("qry_intensity_peak", None),
                "ref_frags": hit.get("ref_frags", None),
                "data_frags": hit.get("data_frags", None),
                "matched_fragments": hit.get("matched_fragments", None),
                "qry_frag_colors": hit.get("qry_frag_colors", None),
                "qry_spectrum": hit.get("qry_spectrum", None),
                "ref_spectrum": hit.get("ref_spectrum", None),
                "qry_spectrum_original": hit.get("qry_spectrum_original", None),
                "ref_spectrum_original": hit.get("ref_spectrum_original", None),
            }
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    elif ms2_df.empty and not ms2_hits_df.empty:
        raise ValueError("MS2 data should not be absent if there are MS2 hits.")
    else:
        raise ValueError("Unhandled case in MS2 summary calculation.")

def _calculate_best_ms1_across_files(
    manual_curation_obj: "ManualCuration",
    exp_data_obj: "ExperimentalData"
) -> None:
    """
    Fill best MS1 metrics for this compound+adduct using exp_data_obj.
    Updates manual_curation_obj.data DataFrame in place.
    """
    df = manual_curation_obj.data
    best_intensity = 0.0
    inchi_key = manual_curation_obj.inchi_key
    adduct = manual_curation_obj.adduct
    ms1_summaries = [s for s in exp_data_obj.ms1_summaries if s.inchi_key == inchi_key and s.adduct == adduct]
    ms2_summaries = [s for s in exp_data_obj.ms2_summaries if s.inchi_key == inchi_key and s.adduct == adduct]
    ms2_hits_data = [
        (d.filename, d.data)
        for d in getattr(exp_data_obj, 'ms2_hits', [])
        if getattr(d, 'inchi_key', None) == inchi_key and getattr(d, 'adduct', None) == adduct
    ]

    for ms1_summary in ms1_summaries:
        ms1_df = ms1_summary.data
        if ms1_df is None or ms1_df.empty:
            continue
        intensity = ms1_df['peak_height'].iloc[0]
        if intensity > best_intensity:
            best_intensity = intensity
            df.loc[0, 'best_ms1_file'] = ms1_summary.filename
            df.loc[0, 'best_ms1_rt'] = float(ms1_df['rt_peak'].iloc[0])
            df.loc[0, 'best_ms1_mz'] = float(ms1_df['mz_centroid'].iloc[0])
            df.loc[0, 'best_ms1_intensity'] = float(intensity)
            df.loc[0, 'best_ms1_ppm_error'] = float(ms1_df['ppm_error'].iloc[0])
            df.loc[0, 'best_ms1_rt_error'] = float(ms1_df['rt_error'].iloc[0])

    # Update file counts
    df.loc[0, 'total_files_detected'] = len([
        s for s in ms1_summaries if not s.data.empty and s.data['num_datapoints'].iloc[0] > 0
    ])
    df.loc[0, 'ms2_files_with_data'] = len([
        s for s in ms2_summaries if not s.data.empty and s.data['num_scans'].iloc[0] > 0
    ])
    df.loc[0, 'ms2_files_with_hits'] = len([
        1 for fname, hits_df in ms2_hits_data if hits_df is not None and not hits_df.empty
    ])


def _add_isomers_to_manual_curation_obj(
    manual_curation_obj: "ManualCuration",
    atlas_dataframe: pd.DataFrame
) -> None:
    """
    Add isomer information to a ManualCuration object for its inchi_key.
    Modifies manual_curation_obj.data in-place.
    """
    inchi_key = manual_curation_obj.inchi_key
    isomer_dict = _build_isomer_dict(atlas_dataframe)
    isomer_list = isomer_dict.get(inchi_key, [])
    manual_curation_obj.data.at[0, 'isomers'] = isomer_list


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

def print_summary_statistics(exp_data_obj: "ExperimentalData") -> None:
    """
    Print comprehensive summary statistics from an ExperimentalData object.
    """
    total_compounds = len(getattr(exp_data_obj, 'manual_curation', []))
    ms1_summary_count = len(getattr(exp_data_obj, 'ms1_summaries', []))
    ms2_summary_count = len(getattr(exp_data_obj, 'ms2_summaries', []))

    ms1_files = set()
    ms2_files = set()
    ms2_hits_files = set()
    all_files = set()
    total_ms2_hits = 0

    for ms1_summary in getattr(exp_data_obj, 'ms1_summaries', []):
        ms1_files.add(ms1_summary.filename)
        all_files.add(ms1_summary.filename)
    for ms2_summary in getattr(exp_data_obj, 'ms2_summaries', []):
        ms2_files.add(ms2_summary.filename)
        all_files.add(ms2_summary.filename)
        # Count MS2 hits from summary DataFrame
        df = ms2_summary.data
        if not df.empty and "total_hits" in df.columns:
            total_ms2_hits += df["total_hits"].sum()
            if df["total_hits"].sum() > 0:
                ms2_hits_files.add(ms2_summary.filename)

    number_w_ms1 = 0
    number_w_ms2 = 0
    number_w_ms2_hits = 0
    for manual_curation_obj in getattr(exp_data_obj, 'manual_curation', []):
        df = manual_curation_obj.data
        has_ms1 = df['total_files_detected'].iloc[0] > 0
        has_ms2 = df['ms2_files_with_data'].iloc[0] > 0
        has_ms2_hits = df['ms2_files_with_hits'].iloc[0] > 0
        if has_ms1:
            number_w_ms1 += 1
        if has_ms2:
            number_w_ms2 += 1
        if has_ms2_hits:
            number_w_ms2_hits += 1

    logger.info(f"Summary statistics:")
    logger.info(f"  Total compounds: {total_compounds}")
    logger.info(f"  MS1 summaries: {ms1_summary_count}")
    logger.info(f"  MS2 summaries: {ms2_summary_count}")
    logger.info(f"  Unique MS1 files: {len(ms1_files)}")
    logger.info(f"  Unique MS2 files: {len(ms2_files)}")
    logger.info(f"  Files with MS2 hits: {len(ms2_hits_files)}")
    logger.info(f"  Total MS2 hits: {int(total_ms2_hits)}")
    logger.info(f"  Compounds with MS1: {number_w_ms1}")
    logger.info(f"  Compounds with MS2: {number_w_ms2}")
    logger.info(f"  Compounds with MS2 hits: {number_w_ms2_hits}")