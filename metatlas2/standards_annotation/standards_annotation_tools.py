from __future__ import annotations

import glob
import os
import uuid
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Any

import metatlas2.shared_tools.file_and_project_format as fpf
import metatlas2.shared_tools.lcmsruns_tools as lrt
import metatlas2.shared_tools.logging_config as lcf
import metatlas2.shared_tools.extract_data_from_h5 as edh
import metatlas2.shared_tools.workflow_objects as wfo
from metatlas2.standards_annotation.standards_annotation_checks import validate_neutral_inchi_key
from metatlas2.shared_tools.chem_info import (
    DEFAULT_ADDUCTS,
    calc_exact_mass_from_smiles_or_inchi,
    get_precursor_mz,
    load_adducts_dict,
    resolve_inchi_and_inchi_key,
    smiles_to_inchi,
)

logger = lcf.get_logger("standards_annotation_tools")

STANDARD_CE_LABELS = ("CE102040", "CE205060")


def _make_compound_uid(compound_name: str, adduct: str) -> str:
    """Return a deterministic UUID for a (compound_name, adduct) pair.

    Uses uuid5 so the same compound+adduct always produces the same UID
    across runs, enabling safe resumability without re-generating IDs.

    Args:
        compound_name: Compound name string.
        adduct:        Adduct string (e.g. ``"[M+H]+"``)

    Returns:
        32-character hex string (UUID without hyphens).
    """
    _COMPOUND_UID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL
    return str(uuid.uuid5(_COMPOUND_UID_NAMESPACE, f"{compound_name}|{adduct}")).replace("-", "")


# ---------------------------------------------------------------------------
# Input table expansion
# ---------------------------------------------------------------------------

def _find_h5_files_for_row(
    row: pd.Series,
    include_polarities: list[str],
    raw_data_root: str,
) -> list[dict[str, str]]:
    """Find all h5 files matching a compound row's project_name/group_name/polarity.

    The raw h5 files live at ``<raw_data_root>/<project_name>/*.h5``.
    The ``project_name`` column in the input CSV is the experiment folder name.
    The ``group_name`` column matches the ``sample_name`` field (position 13)
    of the h5 filename.

    Args:
        row:               One row from the standards input CSV.
        include_polarities: List of polarity strings (e.g. ``["POS", "NEG"]``).
        raw_data_root:     Root directory containing experiment sub-folders.

    Returns:
        List of dicts with keys ``file_path``, ``filename``, ``polarity``,
        ``chromatography``, ``file_format``, ``file_type``.
    """
    project_name = str(row.get("project_name", "")).strip()
    group_name = str(row.get("group_name", "")).strip()
    if not project_name or not group_name:
        return []

    experiment_dir = Path(raw_data_root) / project_name
    pattern = str(experiment_dir / "*.h5")
    all_files = glob.glob(pattern)
    if not all_files:
        logger.debug(
            f"No h5 files found for project '{project_name}' at {pattern}"
        )
        return []
    logger.info(
        f"Found {len(all_files)} h5 files for project '{project_name}'"
    )

    matched = []
    logger.info(f"Matching h5 files for project '{project_name}' to group '{group_name}'")
    for file_path in all_files:
        try:
            fields = fpf.parse_file_name(os.path.basename(file_path))
        except Exception:
            continue

        # Match sample_name (position 13 of filename) to group_name
        sample_name = fields.get("sample_name", "")
        if sample_name.lower() != group_name.lower():
            logger.debug(
                f"Skipping file '{file_path}' because sample_name '{sample_name}' "
                f"does not match group_name '{group_name}'"
            )
            continue

        # Match polarity
        file_polarity = fields.get("polarity", "").upper()
        if file_polarity not in [p.upper() for p in include_polarities]:
            logger.debug(
                f"Skipping file '{file_path}' because polarity '{file_polarity}' "
                f"is not in include_polarities {include_polarities}"
            )
            continue

        matched.append({
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "polarity": file_polarity,
            "chromatography": fpf.canonical_chromatography(
                fields.get("chromatography", "")
            ),
            "file_format": "h5",
            "file_type": lrt._classify_file_type(fields, fields.get("owner", "")),
        })

    logger.info(
        f"Found {len(matched)} matching h5 files for project '{project_name}' "
        f"and group '{group_name}'"
    )
    return matched


def build_standards_lcmsrun_table(
    config: dict[str, Any],
    raw_data_root: str | None = None,
    output_dir: str | None = None
) -> pd.DataFrame:
    """Build the expanded compound x file x adduct table.

    Reads the input CSV, resolves chemical structure info, computes exact
    masses, globs matching h5 files, and expands to one row per
    ``(compound, file, adduct)`` with:

    - ``compound_uid``: stable uuid5 per ``(compound_name, adduct)`` — used
      for grouping in the DB and GUI.
    - ``mz_rt_uid``: random uuid4 per ``(compound_name, adduct, file)`` — used
      as the extraction engine's join key.

    Args:
        config:        Parsed config dict from
                       :func:`metatlas2.shared_tools.load_tools.load_standards_config`.
        raw_data_root: Optional override for the raw data root directory.

    Returns:
        DataFrame with one row per ``(compound, file, adduct)`` and columns:
        ``compound_uid``, ``mz_rt_uid``, ``compound_name``, ``inchi_key``,
        ``smiles``, ``inchi``, ``exact_mass``, ``adduct``, ``mz_theoretical``,
        ``polarity``, ``chromatography``, ``rt_min``, ``rt_max``,
        ``file_path``, ``filename``, ``annotation_notes``.
    """
    project = config["project"]
    include_polarities = project.get("include_polarities", ["POS", "NEG"])
    include_adducts = project.get("include_adducts", DEFAULT_ADDUCTS)

    # Read input table
    input_df = pd.read_csv(
        project["standards_input_file"],
        keep_default_na=False,
        dtype=str,
    )
    logger.info(
        f"Loaded {len(input_df)} compounds from {project['standards_input_file']}"
    )

    # Resolve exact masses and chemical identifiers
    exact_masses = []
    smiles_resolved = []
    inchi_resolved = []
    inchi_keys_resolved = []

    for _, row in input_df.iterrows():
        smiles = str(row.get("smiles", "")).strip()
        inchi = str(row.get("inchi", "")).strip()
        inchi_key = str(row.get("inchi_key", "")).strip()

        mass = float("nan")
        if smiles:
            mass = calc_exact_mass_from_smiles_or_inchi(smiles)
        if np.isnan(mass) and inchi:
            mass = calc_exact_mass_from_smiles_or_inchi(inchi)
        if np.isnan(mass) and inchi_key:
            try:
                from metatlas2.shared_tools.pubchem_retrieval import fetch_pubchem_entry
                from metatlas2.shared_tools.utils import get_provenance
                prov = get_provenance()
                data = fetch_pubchem_entry(inchi_key, prov["timestamp"])
                if data:
                    if not smiles and data.get("smiles"):
                        smiles = data["smiles"]
                    if not inchi and data.get("inchi"):
                        inchi = data["inchi"]
                    if smiles:
                        mass = calc_exact_mass_from_smiles_or_inchi(smiles)
                    elif inchi:
                        mass = calc_exact_mass_from_smiles_or_inchi(inchi)
            except Exception as exc:
                logger.warning(
                    f"PubChem lookup failed for {inchi_key}: {exc}"
                )

        if np.isnan(mass):
            logger.warning(
                f"Could not determine exact mass for compound "
                f"'{row.get('compound_name', 'unknown')}'; rows will be skipped."
            )

        # Derive InChI and InChIKey from SMILES/InChI when not provided in CSV
        smiles, inchi, inchi_key = resolve_inchi_and_inchi_key(smiles, inchi, inchi_key)
        validate_neutral_inchi_key(str(row.get("compound_name", "unknown")), inchi, inchi_key)
        if smiles:
            try:
                smiles_inchi = smiles_to_inchi(smiles)
            except ValueError as exc:
                raise ValueError(f"Invalid SMILES for '{row.get('compound_name', 'unknown')}': {exc}") from exc
            if smiles_inchi != inchi:
                validate_neutral_inchi_key(str(row.get("compound_name", "unknown")), smiles_inchi, inchi_key)

        exact_masses.append(mass)
        smiles_resolved.append(smiles)
        inchi_resolved.append(inchi)
        inchi_keys_resolved.append(inchi_key)

    input_df = input_df.copy()
    input_df["exact_mass"] = exact_masses
    input_df["smiles"] = smiles_resolved
    input_df["inchi"] = inchi_resolved
    input_df["inchi_key"] = inchi_keys_resolved

    # Drop rows where we couldn't get a mass
    n_before = len(input_df)
    input_df = input_df[~input_df["exact_mass"].apply(np.isnan)].reset_index(drop=True)
    if len(input_df) < n_before:
        logger.warning(
            f"Dropped {n_before - len(input_df)} compound(s) with unknown exact mass"
        )

    # Expand: one row per (compound, file)
    expanded_rows = []
    for _, row in input_df.iterrows():
        files = _find_h5_files_for_row(row, include_polarities, raw_data_root)
        if not files:
            raise ValueError(
                f"No matching h5 files found for compound "
                f"'{row.get('compound_name', 'unknown')}'"
            )
        for file_info in files:
            expanded_rows.append({**row.to_dict(), **file_info})

    if not expanded_rows:
        raise ValueError(
            "No matching h5 files found for any compound. "
            "Check experiment/group columns and raw_data_root "
            f"(currently: {raw_data_root})."
        )

    expanded_df = pd.DataFrame(expanded_rows)
    logger.info(
        f"Expanded to {len(expanded_df)} compound x file rows "
        f"across {expanded_df['file_path'].nunique()} unique files"
    )

    # Load adduct metadata once (matchms-based)
    all_adduct_meta = load_adducts_dict()

    # Expand: one row per (compound, file, adduct)
    adduct_rows = []
    for _, row in expanded_df.iterrows():
        file_polarity = str(row.get("polarity", "")).upper()
        ionmode = "positive" if file_polarity == "POS" else "negative"
        compound_name = str(row.get("compound_name", ""))

        for adduct in include_adducts:
            adduct_info = all_adduct_meta.get(adduct)
            if adduct_info is None:
                logger.warning(f"Unknown adduct '{adduct}' (not in matchms); skipping")
                continue
            if adduct_info["ionmode"] != ionmode:
                continue  # wrong polarity for this file

            mz = get_precursor_mz(float(row["exact_mass"]), adduct)
            if np.isnan(mz) or mz <= 0:
                continue

            # RT window: use rt_min/rt_max columns if present, else full range
            rt_min_val = str(row.get("rt_min", "")).strip()
            rt_max_val = str(row.get("rt_max", "")).strip()
            try:
                rt_min = float(rt_min_val) if rt_min_val else 0.0
            except ValueError:
                rt_min = 0.0
            try:
                rt_max = float(rt_max_val) if rt_max_val else 20.0
            except ValueError:
                rt_max = 20.0

            # compound_uid: stable per (compound_name, adduct) — same across runs
            compound_uid = _make_compound_uid(compound_name, adduct)

            # mz_rt_uid: unique per (compound_name, adduct, file) — extraction join key
            mz_rt_uid = str(uuid.uuid4()).replace("-", "")

            # Extract CE label from the filename's run_metadata field
            _filename = str(row.get("filename", ""))
            ce_label = fpf.extract_collision_energy_label(_filename)

            adduct_rows.append({
                "compound_uid":          compound_uid,
                "mz_rt_uid":             mz_rt_uid,
                "compound_name":         compound_name,
                "inchi_key":             str(row.get("inchi_key", "")),
                "smiles":                str(row.get("smiles", "")),
                "inchi":                 str(row.get("inchi", "")),
                "exact_mass":            float(row["exact_mass"]),
                "adduct":                adduct,
                "mz_theoretical":        float(mz),
                "polarity":              file_polarity,
                "chromatography":        str(row.get("chromatography", "")),
                "rt_min":                rt_min,
                "rt_max":                rt_max,
                "file_path":             str(row.get("file_path", "")),
                "filename":              _filename,
                "file_type":             str(row.get("file_type", "")),
                "annotation_notes":      str(row.get("annotation_notes", "")),
                "collision_energy_label": ce_label,
            })

    if not adduct_rows:
        raise ValueError("No valid compound x file x adduct combinations found.")

    result = pd.DataFrame(adduct_rows)
    logger.info(
        f"Final expanded table: {len(result)} rows "
        f"({result['compound_name'].nunique()} compounds, "
        f"{result['adduct'].nunique()} unique adducts, "
        f"{result['file_path'].nunique()} unique files)"
    )

    if output_dir is not None:
        input_stem = Path(project["standards_input_file"]).stem
        expanded_table_path = os.path.join(output_dir, f"{input_stem}_expanded.csv")
        result.to_csv(expanded_table_path, index=False)
        logger.info(f"Expanded table saved to {expanded_table_path}")

    return result


def _build_lcmsruns_list(expanded_df: pd.DataFrame) -> list:
    """Build a list of LCMSRun-compatible objects from the expanded table.

    Args:
        expanded_df: Expanded compound x file x adduct DataFrame.

    Returns:
        List of :class:`metatlas2.shared_tools.workflow_objects.LCMSRun` objects, one per
        unique h5 file path.
    """
    from metatlas2.shared_tools.workflow_objects import LCMSRun

    seen = set()
    runs = []
    for _, row in expanded_df.iterrows():
        fp = str(row["file_path"])
        if fp in seen:
            continue
        seen.add(fp)
        try:
            fields = fpf.parse_file_name(os.path.basename(fp))
            ms_level = fields.get("ms_level", "").lower()
            polarity = fields.get("polarity", "").lower()
            chrom = fpf.canonical_chromatography(fields.get("chromatography", ""))
            file_type = lrt._classify_file_type(fields, fields.get("owner", ""))
        except Exception:
            raise ValueError(f"Could not parse file name '{fp}'")

        runs.append(
            LCMSRun(
                file_path=fp,
                filename=os.path.basename(fp),
                file_format="h5",
                file_type=file_type,
                chromatography=chrom,
                ms_level=ms_level,
                polarity=polarity,
                created_by="",
                created_date="",
            )
        )
    return runs


# ---------------------------------------------------------------------------
# TIC computation from already-extracted MS1 data
# ---------------------------------------------------------------------------

def build_tic_from_ms1_df(ms1_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build a TIC (total ion chromatogram) per file from the extracted MS1 data.

    The TIC is computed by exploding the list columns and **summing**
    intensities across all ``mz_rt_uid`` entries for each ``(filename, rt)``
    pair.  RT values are rounded to 6 decimal places before grouping to avoid
    float representation drift after the DuckDB REAL[] round-trip.

    Note: because the MS1 data only contains ions within the atlas m/z windows,
    this is technically a "target compound TIC" rather than a full-file TIC.
    It is used for visual context in the GUI.

    Args:
        ms1_df: Wide-format MS1 DataFrame with list columns ``spec_rts`` and
                ``spec_ints``, one row per ``(mz_rt_uid, filename)``.

    Returns:
        Dict mapping ``filename`` → ``DataFrame(rt, tic_intensity)``.
    """
    if ms1_df is None or ms1_df.empty:
        return {}

    tic_by_file: dict[str, pd.DataFrame] = {}

    for filename, file_df in ms1_df.groupby("filename"):
        rows = []
        for _, row in file_df.iterrows():
            rts = row.get("spec_rts", [])
            ints = row.get("spec_ints", [])
            if rts is None or ints is None:
                continue
            for rt, intensity in zip(rts, ints):
                # Round RT to avoid float drift from REAL[] storage
                rows.append({"rt": round(float(rt), 6), "intensity": float(intensity)})

        if not rows:
            continue

        tic_df = (
            pd.DataFrame(rows)
            .groupby("rt", sort=True)["intensity"]
            .sum()          # sum = true TIC; max would be BPC
            .reset_index()
        )
        tic_df.columns = ["rt", "tic_intensity"]
        tic_by_file[str(filename)] = tic_df

    logger.info(f"Built TIC for {len(tic_by_file)} files from extracted MS1 data")
    return tic_by_file


# ---------------------------------------------------------------------------
# Peak detection and MS2 association
# ---------------------------------------------------------------------------

def _frag_list(v) -> list[float]:
    """Coerce a frag m/z or intensity cell (list or JSON string) to list[float]."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    try:
        return [float(x) for x in list(v)]
    except (TypeError, ValueError):
        return []


def pick_best_ms2_scan(scans: pd.DataFrame, rt_peak: float) -> pd.Series | None:
    """Return the MS2 scan closest in RT to ``rt_peak``.

    Scans without a valid ``scan_rt`` are ignored.
    """
    if scans is None or scans.empty or "scan_rt" not in scans.columns:
        return None
    srts = pd.to_numeric(scans["scan_rt"], errors="coerce")
    cand = scans[srts.notna()].copy()
    if cand.empty:
        return None
    cand["_rt_diff"] = (pd.to_numeric(cand["scan_rt"], errors="coerce") - float(rt_peak)).abs()
    return cand.sort_values("_rt_diff", kind="stable").iloc[0]


def associate_peak_ms2(peak: dict, expanded_table: pd.DataFrame, ms2_df: pd.DataFrame) -> list[dict]:
    """Associate the nearest usable MS2 scan from each file for a peak."""
    targets = expanded_table[
        (expanded_table["compound_name"] == peak["compound_name"])
        & (expanded_table["chromatography"] == peak["chromatography"])
        & (expanded_table["polarity"] == peak["polarity"])
        & (expanded_table["adduct"] == peak["adduct"])
    ]
    associations = []
    for _, target in targets.iterrows():
        uid = str(target["mz_rt_uid"])
        filename = str(target["filename"])
        scans = ms2_df[
            (ms2_df["mz_rt_uid"] == uid) & (ms2_df["filename"] == filename)
        ] if not ms2_df.empty else pd.DataFrame()
        scan = pick_best_ms2_scan(scans, peak["rt_peak"])
        if scan is None:
            logger.warning(f"No usable MS2 scan in {peak['polarity']} mode for {peak['compound_name']} ({peak['adduct']}) in {filename}")
            continue
        if not _frag_list(scan.get("frag_mzs")) or not _frag_list(scan.get("frag_ints")):
            logger.warning(f"Closest MS2 scan in {peak['polarity']} mode for {peak['compound_name']} ({peak['adduct']}) in {filename} has no fragment data")
        precursor_col = "precursor_MZ" if "precursor_MZ" in scans.columns else "precursor_mz"
        associations.append({
            "peak_uid": peak["peak_uid"],
            "scan_rt": float(scan["scan_rt"]),
            "precursor_mz": float(scan.get(precursor_col, float("nan"))),
            "precursor_intensity": float(scan.get("precursor_intensity", float("nan"))),
            "collision_energy": float(scan.get("collision_energy", float("nan"))),
            "frag_mzs": _frag_list(scan["frag_mzs"]),
            "frag_ints": _frag_list(scan["frag_ints"]),
            "filename": filename,
        })
    return associations


def detect_and_save_peaks(
    data: "wfo.StandardsAnnotationData",
    peak_params: dict | None = None,
) -> None:
    """Select one MS1 file per compound group and associate MS2 scans from each file.

    Detection strategy: candidates are found on lightly smoothed data (noise-robust,
    stable apexes), then each candidate's base window is re-scanned on RAW data so
    closely spaced doublets are not collapsed onto the saddle. Reported RT/intensity
    always come from raw data.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    from metatlas2.standards_annotation.standards_annotation_db import save_detected_peaks_to_db

    if peak_params is None:
        peak_params = {}

    gaussian_sigma = float(peak_params.get("gaussian_sigma", 1.0))
    min_height_frac = float(peak_params.get("min_height_fraction", 0.15))
    min_prominence_frac = float(peak_params.get("min_prominence_fraction", 0.03))
    min_distance = int(peak_params.get("min_distance", 2))
    min_width = float(peak_params.get("min_width", 2))
    prominence_snr = float(peak_params.get("prominence_snr", 5.0))

    RT_PAD_MIN = 0.5  # hard-coded: rt_min/rt_max = apex +/- 0.5 min
    APEX_REFINE = 2   # scan points: raw-apex search radius around a smoothed apex

    ms1_df = data.ms1_df
    ms2_df = data.ms2_df
    expanded_table = data.expanded_table
    db_path = data.project_db_path

    group_cols = ["compound_name", "chromatography", "polarity"]
    groups = list(expanded_table.groupby(group_cols, sort=True))
    for group, targets in groups:
        filenames = sorted(targets["filename"].astype(str).unique())
        if len(filenames) > 2:
            raise ValueError(f"More than two files for {group}: {filenames}")
        if len(filenames) == 1:
            logger.warning(f"Only one file for {group}: {filenames[0]}")

    if ms1_df is None or ms1_df.empty:
        logger.warning("No MS1 data available for peak detection.")
        return

    # Deduplicate ms1_df by (mz_rt_uid, filename) — the sa_ms1 primary key.
    if "mz_rt_uid" in ms1_df.columns and "filename" in ms1_df.columns:
        ms1_df = ms1_df.drop_duplicates(
            subset=["mz_rt_uid", "filename"]
        ).reset_index(drop=True)

    # Build lookup: mz_rt_uid → row from expanded_table
    uid_to_meta: dict[str, dict] = {}
    for _, row in expanded_table.iterrows():
        uid = str(row["mz_rt_uid"])
        if uid in uid_to_meta:
            logger.warning(
                f"Duplicate mz_rt_uid '{uid}' in expanded_table — "
                f"this should not happen. Keeping first occurrence."
            )
            continue
        uid_to_meta[uid] = row.to_dict()

    all_peaks: list[dict] = []

    # Process each MS1 row (one per mz_rt_uid x filename)
    for _, ms1_row in ms1_df.iterrows():
        uid = str(ms1_row.get("mz_rt_uid", ""))
        filename = str(ms1_row.get("filename", ""))
        meta = uid_to_meta.get(uid)
        if meta is None:
            logger.warning(
                f"mz_rt_uid '{uid}' from ms1_df not found in expanded_table — skipping."
            )
            continue

        rts = ms1_row.get("spec_rts", [])
        ints = ms1_row.get("spec_ints", [])
        mzs = ms1_row.get("spec_mzs", [])
        if rts is None or ints is None or len(rts) < 5:
            continue

        rt_arr = np.asarray(rts, dtype=np.float64)
        int_arr = np.asarray(ints, dtype=np.float64)
        mz_arr = np.asarray(mzs, dtype=np.float64) if mzs is not None else np.array([])

        # Robust noise estimate: for white noise, std(diff) = noise*sqrt(2), and
        # 1.4826*MAD ≈ std. If the trace is mostly zeros this underestimates and
        # gracefully falls back to the fraction-of-max floor below.
        diffs = np.diff(int_arr)
        noise = (
            1.4826 * float(np.median(np.abs(diffs))) / np.sqrt(2.0)
            if diffs.size
            else 0.0
        )

        smoothed = gaussian_filter1d(int_arr, sigma=gaussian_sigma)
        max_int = float(np.max(smoothed))
        if max_int <= 0:
            continue

        # Prominence floor: noise-referenced OR fraction-of-max, whichever is larger.
        prom_floor = max(noise * prominence_snr, max_int * min_prominence_frac)
        height_thr = max_int * min_height_frac

        find_peaks_kwargs: dict = dict(
            height=height_thr,
            prominence=prom_floor,
            distance=min_distance,
        )
        if min_width > 0:
            find_peaks_kwargs["width"] = min_width

        # Pass 1: candidates on smoothed data.
        peaks_idx, props = find_peaks(smoothed, **find_peaks_kwargs)
        if peaks_idx.size == 0:
            continue

        # Pass 2: re-detect on RAW data inside each candidate's base window.
        # A doublet merged by smoothing appears as two raw local maxima here.
        refined: list[int] = []
        for p, lb, rb in zip(peaks_idx, props["left_bases"], props["right_bases"]):
            p, lb, rb = int(p), int(lb), int(rb)
            seg = int_arr[lb : rb + 1]
            if seg.size >= 3:
                sub, _ = find_peaks(
                    seg,
                    height=height_thr,
                    prominence=prom_floor,
                    distance=min_distance,
                )
                if sub.size > 0:
                    # Raw sub-peak(s) found — use them directly (they are already
                    # exact raw local maxima; no further refinement needed).
                    refined.extend((lb + sub).tolist())
                    continue
            # No raw sub-peak passed the thresholds: snap the smoothed apex to the
            # raw maximum within +/- APEX_REFINE points.
            lo = max(0, p - APEX_REFINE)
            hi = min(len(int_arr), p + APEX_REFINE + 1)
            refined.append(lo + int(np.argmax(int_arr[lo:hi])))

        # Overlapping base windows can re-detect the same raw apex; unique() collapses
        # those duplicates.
        peaks_idx = np.unique(np.asarray(refined, dtype=np.int64))
        peaks_idx_sorted = peaks_idx[np.argsort(rt_arr[peaks_idx])]

        file_peaks: list[dict] = []
        for peak_num, peak_idx in enumerate(peaks_idx_sorted, start=1):
            # All reported values come from raw data at a raw local maximum.
            rt_peak = float(rt_arr[peak_idx])
            peak_intensity = float(int_arr[peak_idx])

            # Hard-coded +/- 0.5 min window, clamped to the file's RT range.
            rt_min = max(float(rt_arr[0]), rt_peak - RT_PAD_MIN)
            rt_max = min(float(rt_arr[-1]), rt_peak + RT_PAD_MIN)

            # Compute mz_observed from scan points within the peak RT window
            if mz_arr.size > 0:
                mask = (rt_arr >= rt_min) & (rt_arr <= rt_max)
                if np.any(mask):
                    # Weight by intensity for a more accurate observed m/z
                    mz_in = mz_arr[mask]
                    int_in = int_arr[mask]
                    total_int = float(np.sum(int_in))
                    if total_int > 0:
                        mz_observed = float(np.sum(mz_in * int_in) / total_int)
                    else:
                        mz_observed = float(np.mean(mz_in))
                else:
                    mz_observed = float("nan")
            else:
                mz_observed = float("nan")

            peak_uid = str(uuid.uuid4()).replace("-", "")
            file_peaks.append({
                "peak_uid":               peak_uid,
                "compound_uid":           str(meta.get("compound_uid", "")),
                "compound_name":          str(meta.get("compound_name", "")),
                "inchi_key":              str(meta.get("inchi_key", "")),
                "adduct":                 str(meta.get("adduct", "")),
                "polarity":               str(meta.get("polarity", "")),
                "chromatography":         str(meta.get("chromatography", "")),
                "filename":               filename,
                "mz_rt_uid":              uid,
                "mz_theoretical":         float(meta.get("mz_theoretical", float("nan"))),
                "rt_peak":                rt_peak,
                "rt_min":                 rt_min,
                "rt_max":                 rt_max,
                "peak_intensity":         peak_intensity,
                "mz_observed":            mz_observed,
                "peak_num":               peak_num,
                "is_manual":              False,
                "collision_energy_label": str(meta.get("collision_energy_label", "")),
            })

        all_peaks.extend(file_peaks)

    peaks_by_group: dict[tuple, list[dict]] = {}
    for peak in all_peaks:
        peaks_by_group.setdefault(tuple(peak[c] for c in group_cols), []).append(peak)
    sources = []
    for group, targets in groups:
        filenames = sorted(targets["filename"].astype(str).unique())
        group_peaks = peaks_by_group.get(group, [])
        if group_peaks:
            winner = min(group_peaks, key=lambda p: (-p["peak_intensity"], p["filename"]))["filename"]
        else:
            raw_max = {}
            for filename in filenames:
                uids = targets.loc[targets["filename"] == filename, "mz_rt_uid"]
                rows = ms1_df[ms1_df["mz_rt_uid"].isin(uids)]
                values = [float(np.max(v)) for v in rows["spec_ints"] if v is not None and len(v)]
                raw_max[filename] = max(values, default=float("-inf"))
            if all(value == float("-inf") for value in raw_max.values()):
                raise ValueError(f"No MS1 intensity data available for {group}: {filenames}")
            winner = min(filenames, key=lambda f: (-raw_max[f], f))
            logger.warning("No peaks detected for %s; choosing MS1 file %s by raw intensity", group, winner)
        sources.append((*group, winner))
    source_by_group = {tuple(row[:3]): row[3] for row in sources}
    all_peaks = [
        p for p in all_peaks
        if p["filename"] == source_by_group[tuple(p[c] for c in group_cols)]
    ]

    seen_peak_keys: set[tuple] = set()
    deduped_peaks: list[dict] = []

    for peak in all_peaks:
        key = (
            str(peak.get("compound_uid", "")),
            str(peak.get("adduct", "")),
            str(peak.get("filename", "")),
            float(peak.get("rt_peak", 0.0)),
        )
        if key in seen_peak_keys:
            continue
        seen_peak_keys.add(key)
        deduped_peaks.append(peak)

    association_cols = group_cols + ["adduct"]
    target_groups = dict(tuple(expanded_table.groupby(association_cols, sort=False)))
    ms2_chunks: dict[tuple, list[pd.DataFrame]] = {}
    if ms2_df is not None and not ms2_df.empty:
        for uid, scans in ms2_df.groupby("mz_rt_uid", sort=False):
            meta = uid_to_meta.get(str(uid))
            if meta is not None:
                key = tuple(meta[c] for c in association_cols)
                ms2_chunks.setdefault(key, []).append(scans)
    ms2_groups = {key: pd.concat(chunks, ignore_index=True) for key, chunks in ms2_chunks.items()}
    deduped_ms2 = []
    for peak in deduped_peaks:
        key = tuple(peak[c] for c in association_cols)
        deduped_ms2.extend(associate_peak_ms2(
            peak, target_groups[key], ms2_groups.get(key, pd.DataFrame())
        ))

    n_removed = len(all_peaks) - len(deduped_peaks)
    if n_removed > 0:
        logger.info(
            f"Removed {n_removed} duplicate peak(s) (same compound_uid+adduct+file+RT "
            f"from multiple extraction passes)."
        )

    logger.info(
        f"Peak detection complete: {len(deduped_peaks)} peaks detected, "
        f"{len(deduped_ms2)} MS2 associations"
    )
    save_detected_peaks_to_db(db_path, deduped_peaks, deduped_ms2, sources)
    selected_uids = {
        str(row["mz_rt_uid"]) for _, row in expanded_table.iterrows()
        if str(row["filename"]) == source_by_group[tuple(row[c] for c in group_cols)]
    }
    data.ms1_df = ms1_df[ms1_df["mz_rt_uid"].astype(str).isin(selected_uids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main extraction orchestrator
# ---------------------------------------------------------------------------

def run_standards_extraction(
    config: dict[str, Any],
    raw_data_root: str | None = None,
    db_path: str | None = None,
) -> "wfo.StandardsAnnotationData":
    """Orchestrate the full standards annotation data extraction.

    Steps:
    1. Build the expanded compound x file x adduct table (with compound_uid
       and mz_rt_uid).
    2. Build a :class:`wfo.StandardsAnnotationAutoIdentification` per
       (polarity, chromatography) group.
    3. Call :func:`metatlas2.shared_tools.extract_data_from_h5.extract_data_from_raw`
       for each group.
    4. Build the TIC from the extracted MS1 data.
    5. Persist all data to the project DuckDB.
    6. Detect MS1 peaks per file and associate best MS2 scans.
    7. Return a :class:`wfo.StandardsAnnotationData` container.

    Args:
        config:        Parsed config dict.
        raw_data_root: Optional override for the raw data root directory.
        db_path:       Path to the project DuckDB.

    Returns:
        :class:`wfo.StandardsAnnotationData` with all extracted data.
    """
    from metatlas2.standards_annotation.standards_annotation_db import (
        is_peaks_complete,
        save_extraction_to_db,
        is_injbl_extraction_complete,
        save_injbl_ms1_to_db,
        load_injbl_ms1,
    )

    expanded_table = build_standards_lcmsrun_table(config, raw_data_root, output_dir=os.path.dirname(db_path))
    logger.info(
        f"Running extraction on {len(expanded_table)} compound x adduct x file rows"
    )

    extraction_cfg = config.get("extraction", {})
    max_workers = int(extraction_cfg.get("max_workers", 4))

    all_ms1_chunks: list[pd.DataFrame] = []
    all_ms2_chunks: list[pd.DataFrame] = []

    # Group by (polarity, chromatography) — one extraction pass per group
    group_cols = ["polarity", "chromatography"]
    for (polarity, chromatography), group_df in expanded_table.groupby(group_cols):
        logger.info(
            f"Extracting data for polarity={polarity}, chromatography={chromatography} "
            f"({len(group_df)} compound x adduct rows, "
            f"{group_df['file_path'].nunique()} files)"
        )

        lcmsruns = _build_lcmsruns_list(group_df)

        if not lcmsruns:
            logger.warning(
                f"No LCMS runs found for polarity={polarity}, "
                f"chromatography={chromatography}; skipping"
            )
            continue

        standards_annotation_params = {
            "atlas_extra_time": 0.0,
            "extract_extra_time": None,
            "ms1_mz_tolerance_ppm": float(extraction_cfg.get("ms1_mz_tolerance_ppm", 5.0)),
            "only_keep_data_in_feature": False,
            "ms1_min_num_points": int(extraction_cfg.get("ms1_min_num_points", 0)),
            "ms1_min_peak_intensity": int(extraction_cfg.get("ms1_min_peak_intensity", 0)),
            "ms2_min_num_scans": int(extraction_cfg.get("ms2_min_num_scans", 0)),
            "ms2_min_precursor_intensity": int(extraction_cfg.get("ms2_min_precursor_intensity", 0)),
            "max_workers": max_workers,
        }

        sa_auto_id_obj = wfo.StandardsAnnotationAutoIdentification(
            standards_annotation_atlas_df=group_df,
            standards_annotation_lcmsruns=lcmsruns,
            standards_annotation_params=standards_annotation_params,
        )

        try:
            edh.extract_data_from_raw(sa_auto_id_obj, stage="standards_annotation")
        except Exception as exc:
            logger.error(
                f"Extraction failed for polarity={polarity}, "
                f"chromatography={chromatography}: {exc}"
            )
            continue

        ms1 = sa_auto_id_obj.experimental_data.ms1_df
        ms2 = sa_auto_id_obj.experimental_data.ms2_df

        if ms1 is not None and not ms1.empty:
            all_ms1_chunks.append(ms1)
        if ms2 is not None and not ms2.empty:
            all_ms2_chunks.append(ms2)

    final_ms1 = (
        pd.concat(all_ms1_chunks, ignore_index=True)
        if all_ms1_chunks
        else pd.DataFrame()
    )
    final_ms2 = (
        pd.concat(all_ms2_chunks, ignore_index=True)
        if all_ms2_chunks
        else pd.DataFrame()
    )

    logger.info(
        f"Extraction complete: {len(final_ms1)} MS1 rows, {len(final_ms2)} MS2 rows"
    )

    tic_by_file = build_tic_from_ms1_df(final_ms1)

    data = wfo.StandardsAnnotationData(
        ms1_df=final_ms1,
        ms2_df=final_ms2,
        tic_by_file=tic_by_file,
        expanded_table=expanded_table,
        config=config,
        project_db_path=db_path,
    )

    save_extraction_to_db(db_path, data)

    # InjBl background extraction
    if not is_injbl_extraction_complete(db_path):
        logger.info("Searching for InjBl files and extracting background EICs...")
        injbl_map = find_injbl_files_for_targets(expanded_table, raw_data_root)
        if not injbl_map.empty:
            injbl_ms1_df = _extract_injbl_ms1(
                injbl_map, expanded_table, extraction_cfg, max_workers
            )
            data.injbl_ms1_df = injbl_ms1_df
            save_injbl_ms1_to_db(db_path, injbl_ms1_df)
            logger.info(
                f"InjBl extraction complete: {len(injbl_ms1_df)} EIC rows "
                f"across {injbl_ms1_df['injbl_filename'].nunique() if not injbl_ms1_df.empty else 0} "
                f"InjBl files."
            )
        else:
            logger.info(
                "No InjBl files found in any project directory; "
                "skipping InjBl extraction."
            )
    else:
        logger.info("InjBl extraction already complete; loading from DB.")
        data.injbl_ms1_df = load_injbl_ms1(db_path)

    # Detect peaks and associate best MS2 scans (stored in sa_peaks / sa_peak_ms2)
    if not is_peaks_complete(db_path):
        logger.info("Detecting MS1 peaks and associating best MS2 scans...")
        peak_params = extraction_cfg.get("peak_finding", {})
        detect_and_save_peaks(data, peak_params=peak_params)
    else:
        logger.info("Peak detection already complete; skipping.")

    return data


# ---------------------------------------------------------------------------
# InjBl file discovery and extraction
# ---------------------------------------------------------------------------

def find_injbl_files_for_targets(
    expanded_table: pd.DataFrame,
    raw_data_root: str,
) -> pd.DataFrame:
    """Find the closest InjBl file for each target sample file.

    For each unique ``(project_dir, polarity, chromatography)`` group in
    *expanded_table*, globs the project directory for h5 files classified as
    ``"injbl"`` by :func:`metatlas2.shared_tools.lcmsruns_tools._classify_file_type`.
    For each target file in the group, finds the single InjBl file with the
    minimum ``|injbl_run_number - target_run_number|`` distance.  Ties are
    broken by taking the lower run_number.

    Args:
        expanded_table: One row per ``(compound, adduct, file)`` as produced
                        by :func:`build_standards_lcmsrun_table`.  Must have
                        ``file_path``, ``filename``, ``polarity``,
                        ``chromatography``, ``mz_rt_uid``, ``compound_uid``,
                        ``compound_name``, ``adduct`` columns.
        raw_data_root:  Root directory containing project sub-folders.

    Returns:
        DataFrame with columns ``mz_rt_uid``, ``compound_uid``,
        ``compound_name``, ``adduct``, ``polarity``, ``chromatography``,
        ``target_filename``, ``injbl_filename``, ``injbl_file_path``.
        Empty DataFrame if no InjBl files are found for any group.
    """
    import re as _re

    def _run_number_int(filename: str) -> int | None:
        """Extract the integer run number from a filename, or None on failure."""
        try:
            fields = fpf.parse_file_name(os.path.basename(filename))
            rn = fields.get("run_number", "")
            rn_str = _re.sub(r"^[Rr]un", "", str(rn)).strip()
            return int(rn_str) if rn_str.isdigit() else None
        except Exception:
            return None

    output_rows: list[dict] = []

    # Group by (project_dir, polarity, chromatography) - project_dir is the parent directory of file_path
    expanded_table = expanded_table.copy()
    expanded_table["_project_dir"] = expanded_table["file_path"].apply(
        lambda fp: str(Path(fp).parent)
    )

    group_cols = ["_project_dir", "polarity", "chromatography"]
    for (project_dir, polarity, chromatography), group_df in expanded_table.groupby(
        group_cols
    ):
        # Discover all InjBl h5 files in this project directory
        project_path = Path(project_dir)
        if not project_path.exists():
            logger.warning(
                f"Project directory does not exist: {project_dir}; "
                f"skipping InjBl search."
            )
            continue

        injbl_candidates: list[tuple[int, str, str]] = []  # (run_num, filename, file_path)
        for h5_path in project_path.glob("*.h5"):
            try:
                fields = fpf.parse_file_name(h5_path.name)
            except Exception:
                continue
            if lrt._classify_file_type(fields, fields.get("owner", "")) != "injbl":
                continue
            #if fields.get("polarity", "").upper() != polarity.upper():
            #    continue
            if fpf.canonical_chromatography(fields.get("chromatography", "")) != chromatography:
                continue
            rn = _run_number_int(h5_path.name)
            if rn is None:
                continue
            injbl_candidates.append((rn, h5_path.name, str(h5_path)))

        if not injbl_candidates:
            logger.debug(
                f"No InjBl files found for project_dir={project_dir}, "
                f"polarity={polarity}, chromatography={chromatography}"
            )
            continue

        logger.info(
            f"Found {len(injbl_candidates)} InjBl candidate(s) for "
            f"{project_dir} / {polarity} / {chromatography}"
        )

        # For each target row, find the closest InjBl by run_number
        for _, row in group_df.iterrows():
            target_rn = _run_number_int(str(row["filename"]))
            if target_rn is None:
                logger.warning(
                    f"Could not parse run_number from target file "
                    f"'{row['filename']}'; skipping InjBl match."
                )
                continue

            # Sort by (distance, run_number) — ties broken by lower run_number
            best = min(
                injbl_candidates,
                key=lambda c: (abs(c[0] - target_rn), c[0]),
            )
            output_rows.append({
                "mz_rt_uid":       str(row["mz_rt_uid"]),
                "compound_uid":    str(row["compound_uid"]),
                "compound_name":   str(row["compound_name"]),
                "adduct":          str(row["adduct"]),
                "polarity":        str(row["polarity"]),
                "chromatography":  str(row["chromatography"]),
                "target_filename": str(row["filename"]),
                "injbl_filename":  best[1],
                "injbl_file_path": best[2],
            })

    if not output_rows:
        logger.info("find_injbl_files_for_targets: no InjBl matches found.")
        return pd.DataFrame()

    result = pd.DataFrame(output_rows)
    logger.info(
        f"InjBl mapping: {len(result)} (mz_rt_uid, injbl_filename) pairs, "
        f"{result['injbl_filename'].nunique()} unique InjBl files."
    )
    return result


def _extract_injbl_ms1(
    injbl_map: pd.DataFrame,
    expanded_table: pd.DataFrame,
    extraction_cfg: dict,
    max_workers: int,
) -> pd.DataFrame:
    """Extract MS1 EIC data from InjBl files for all compound+adduct m/z windows.

    Groups *injbl_map* by ``(polarity, chromatography)``.  For each group,
    builds a **deduplicated** atlas (one row per ``compound_uid``) so that all
    InjBl files share the same atlas entries.  The extraction engine is told
    NOT to filter the atlas per-file (``per_file_atlas=False``) because InjBl
    files are not in the target ``expanded_table`` and therefore have no
    per-file atlas rows.

    After extraction the wide-format MS1 rows (keyed by ``compound_uid``) are
    joined back to *injbl_map* to produce one output row per
    ``(mz_rt_uid, injbl_filename)`` pair — matching the ``sa_injbl_ms1``
    schema.  This allows the GUI to look up InjBl EIC data by the target
    file's ``mz_rt_uid``.

    Args:
        injbl_map:      Output of :func:`find_injbl_files_for_targets`.
        expanded_table: Full expanded compound x file x adduct table.
        extraction_cfg: ``config["extraction"]`` dict.
        max_workers:    Number of parallel worker processes.

    Returns:
        Wide-format MS1 DataFrame with columns ``mz_rt_uid``,
        ``injbl_filename``, ``inchi_key``, ``adduct``, ``spec_rts``,
        ``spec_ints``, ``spec_mzs``.  Empty DataFrame if extraction fails
        for all groups.
    """
    all_chunks: list[pd.DataFrame] = []

    group_cols = ["polarity", "chromatography"]
    for (polarity, chromatography), group_df in injbl_map.groupby(group_cols):
        et_group = expanded_table[
            (expanded_table["polarity"] == polarity) &
            (expanded_table["chromatography"] == chromatography)
        ]
        if et_group.empty:
            logger.warning(
                f"No expanded_table rows for InjBl group "
                f"polarity={polarity}, chromatography={chromatography}; skipping."
            )
            continue

        dedup = (
            et_group
            .drop_duplicates(subset=["compound_uid", "adduct"])
            .copy()
            .reset_index(drop=True)
        )
        dedup["mz_rt_uid"] = dedup["compound_uid"]

        # Build LCMSRun list from unique InjBl file paths in this group
        unique_injbl = group_df[["injbl_filename", "injbl_file_path"]].drop_duplicates()
        injbl_runs = []
        for _, injbl_row in unique_injbl.iterrows():
            fp = str(injbl_row["injbl_file_path"])
            fn = str(injbl_row["injbl_filename"])
            try:
                fields = fpf.parse_file_name(fn)
                ms_level = fields.get("ms_level", "ms1").lower()
                pol = fields.get("polarity", "pos").lower()
                chrom = fpf.canonical_chromatography(fields.get("chromatography", ""))
                file_type = lrt._classify_file_type(fields, fields.get("owner", ""))
            except Exception:
                logger.error(
                    f"InjBl filename '{fn}' could not be parsed. "
                    f"Polarity/chromatography cannot be determined. Skipping."
                )
                continue
            injbl_runs.append(
                wfo.LCMSRun(
                    file_path=fp,
                    filename=fn,
                    file_format="h5",
                    file_type=file_type,
                    chromatography=chrom,
                    ms_level=ms_level,
                    polarity=pol,
                    created_by="",
                    created_date="",
                )
            )

        if not injbl_runs:
            continue

        params = {
            "atlas_extra_time": 0.0,
            "extract_extra_time": None,
            "ms1_mz_tolerance_ppm": float(extraction_cfg.get("ms1_mz_tolerance_ppm", 5.0)),
            "only_keep_data_in_feature": False,
            "ms1_min_num_points": 0,
            "ms1_min_peak_intensity": 0,
            "ms2_min_num_scans": 0,
            "ms2_min_precursor_intensity": 0,
            "max_workers": max_workers,
        }

        sa_obj = wfo.StandardsAnnotationAutoIdentification(
            standards_annotation_atlas_df=dedup,
            standards_annotation_lcmsruns=injbl_runs,
            standards_annotation_params=params,
            per_file_atlas=False, # InjBl files are not in expanded_table so the extraction engine must NOT filter the atlas per-file.
        )

        try:
            edh.extract_data_from_raw(sa_obj, stage="standards_annotation")
        except Exception as exc:
            logger.error(
                f"InjBl extraction failed for polarity={polarity}, "
                f"chromatography={chromatography}: {exc}"
            )
            continue

        ms1 = sa_obj.experimental_data.ms1_df
        if ms1 is None or ms1.empty:
            continue

        # ms1 is keyed by (compound_uid, injbl_filename).
        # Join back to injbl_map to expand to (mz_rt_uid, injbl_filename) pairs.
        # Build compound_uid -> [mz_rt_uid, ...] lookup from group_df
        cuid_to_uids: dict[str, list[str]] = {}
        for _, row in group_df.iterrows():
            cuid = str(row["compound_uid"])
            uid = str(row["mz_rt_uid"])
            cuid_to_uids.setdefault(cuid, [])
            if uid not in cuid_to_uids[cuid]:
                cuid_to_uids[cuid].append(uid)

        # Also build compound_uid -> adduct lookup from dedup
        cuid_adduct = {
            str(r["compound_uid"]): str(r["adduct"])
            for _, r in dedup.iterrows()
        }

        expanded_rows: list[pd.DataFrame] = []
        for _, ms1_row in ms1.iterrows():
            # ms1_row["mz_rt_uid"] is actually compound_uid (we swapped above)
            cuid = str(ms1_row.get("mz_rt_uid", ""))
            injbl_fn = str(ms1_row.get("filename", ""))
            target_uids = cuid_to_uids.get(cuid, [])
            if not target_uids:
                continue
            for target_uid in target_uids:
                new_row = ms1_row.to_dict()
                new_row["mz_rt_uid"] = target_uid
                new_row["injbl_filename"] = injbl_fn
                expanded_rows.append(new_row)

        if not expanded_rows:
            continue

        chunk = pd.DataFrame(expanded_rows)
        all_chunks.append(chunk)
        logger.info(
            f"InjBl MS1 extracted: {len(chunk)} (mz_rt_uid, injbl_filename) rows for "
            f"polarity={polarity}, chromatography={chromatography}"
        )

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks, ignore_index=True)
    # Keep only the columns needed for sa_injbl_ms1
    keep = ["mz_rt_uid", "injbl_filename", "inchi_key", "adduct",
            "spec_rts", "spec_ints", "spec_mzs"]
    result = result[[c for c in keep if c in result.columns]]
    return result
