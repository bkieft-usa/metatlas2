import pandas as pd
import numpy as np
import os
import sys
import csv
import yaml
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import grp
import subprocess
import joblib
from matchms import Spectrum
from tqdm.auto import tqdm

import metatlas2.logging_config as lcf
logger = lcf.get_logger('load_tools')

import ast
import json
import pandas as pd
from pathlib import Path

DEFAULT_EXCLUDE_LCMSRUNS = {
    'gui': ['INJBL', 'BLANK'],
    'id_sheet': ['INJBL', 'BLANK', 'REFSTD'],
    'chromatograms': ['INJBL', 'BLANK'],
    'id_plots': ['INJBL', 'BLANK', 'REFSTD'],
    'data_sheets': ['INJBL', 'BLANK'],
}

DEFAULT_INCLUDE_LCMSRUNS_ANALYSES = ['EXPERIMENTAL', 'ISTD', 'EXCTRL', 'REFSTD', 'INJBLK']

DEFAULT_INCLUDE_LCMSRUNS_RT_ALIGNMENT = ['QC']

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

def load_msms_refs_file(
    file_path: str,
    database_filter: str | None = None,
    polarity: str | None = None,
    inchi_keys: list[str] | None = None,
) -> dict[str, list[Spectrum]]:
    """
    Load an msms refs file (.jsonl format) and return a dict mapping
    inchi_key -> list of matchms Spectrum objects.
    Results are cached to a .pkl file and reloaded on subsequent calls
    if the source file has not been modified.

    Args:
        file_path: Path to the msms_refs.jsonl file
        database_filter: If provided, only load spectra where database == this string
        polarity: If provided, only load spectra for the specified polarity
        inchi_keys: If provided, only load spectra for the specified inchi_keys
    Returns:
        dict mapping inchi_key (str) -> list of matchms Spectrum objects
    """

    file_path = Path(file_path)
    logger.info(
        f"Loading reference spectra from {file_path}"
        + (f" (database='{database_filter}')" if database_filter else "")
        + (f" (polarity='{polarity}')" if polarity else "")
        + (f" (filtered to {len(inchi_keys)} inchi_keys)" if inchi_keys is not None else "")
        + "..."
    )


    refs_by_inchi_key: dict[str, list[Spectrum]] = {}
    n_skipped = 0
    inchi_key_set = set(inchi_keys) if inchi_keys is not None else None

    with file_path.open('r') as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"  Line {line_num}: JSON parse error — {e}")
                n_skipped += 1
                continue


            if database_filter and rec.get('database') != database_filter:
                continue

            if polarity and rec.get('polarity') != polarity:
                continue

            rec_inchi_key = rec.get('inchi_key', '')
            if inchi_key_set is not None and rec_inchi_key not in inchi_key_set:
                continue

            mz_list = rec.get('mz')
            int_list = rec.get('intensities')
            if not mz_list or not int_list or len(mz_list) != len(int_list):
                n_skipped += 1
                continue

            # Direct construction from JSON arrays
            mz = np.array(mz_list, dtype=np.float32)
            intensities = np.array(int_list, dtype=np.float32)

            precursor_mz = rec.get('precursor_mz')
            try:
                precursor_mz = float(rec.get('precursor_mz')) if rec.get('precursor_mz') is not None else None
            except (TypeError, ValueError):
                precursor_mz = None
            if precursor_mz is not None and not np.isnan(precursor_mz):
                mask = mz < precursor_mz + 2.5
                mz = mz[mask]
                intensities = intensities[mask]

            if len(mz) == 0:
                n_skipped += 1
                continue

            inchi_key = rec_inchi_key
            # matchms requires m/z ascending; defensively sort
            if len(mz) > 1 and not np.all(mz[:-1] <= mz[1:]):
                order = np.argsort(mz, kind='stable')
                mz = mz[order]
                intensities = intensities[order]
            spec = Spectrum(
                mz=mz,
                intensities=intensities,
                metadata={
                    'precursor_mz': float(precursor_mz) if precursor_mz is not None else 0.0,
                    'database': str(rec.get('database', '')),
                    'id': str(rec.get('id', '')),
                    'name': str(rec.get('name', '')),
                    'inchi_key': inchi_key,
                }
            )
            refs_by_inchi_key.setdefault(inchi_key, []).append(spec)

    if not refs_by_inchi_key:
        if database_filter:
            raise ValueError(
                f"No spectra matched database_filter={database_filter!r} in {file_path}. "
                f"Check the filter value or the 'database' field in the source file."
            )
        raise ValueError(f"Reference file {file_path} is empty after parsing.")

    total = sum(len(v) for v in refs_by_inchi_key.values())
    logger.info(f"  Loaded {total} reference spectra for {len(refs_by_inchi_key)} unique InChI keys.")
    if n_skipped > 0:
        logger.warning(f"  Skipped {n_skipped} rows due to unparseable or empty spectra.")

    return refs_by_inchi_key

def _validate_rt_alignment_params(params: Dict[str, Any], location: str) -> Dict[str, Any]:
    """Validate and coerce a single PARAMS block from RT_ALIGNMENT.
    """
    params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else DEFAULT_INCLUDE_LCMSRUNS_RT_ALIGNMENT
    params['exclude_lcmsruns'] = list(params['exclude_lcmsruns']) if params.get('exclude_lcmsruns') else []
    params['use_existing_rt_alignment'] = bool(params.get('use_existing_rt_alignment', False))
    params['remove_unided_compounds'] = bool(params.get('remove_unided_compounds', False))
    params['only_keep_data_in_feature'] = bool(params.get('only_keep_data_in_feature', True))
    params['atlas_extra_time'] = float(params.get('atlas_extra_time', 2.0))
    params['ms1_min_peak_intensity'] = float(params.get('ms1_min_peak_intensity', 0.0))
    params['ms1_min_num_points'] = int(params.get('ms1_min_num_points', 0))
    params['ms1_mz_tolerance_ppm'] = float(params.get('ms1_mz_tolerance_ppm', 5.0))
    params['apply_model_to_min_max'] = bool(params.get('apply_model_to_min_max', True))
    params['polynomial_degree'] = int(params.get('polynomial_degree', 2))
    params['min_observations_per_compound'] = int(params.get('min_observations_per_compound', 1))
    params['min_compounds_for_modeling'] = int(params.get('min_compounds_for_modeling', 2))
    params['r2_threshold'] = float(params.get('r2_threshold', 0.5))
    params['exclude_inchikeys'] = list(params['exclude_inchikeys']) if params.get('exclude_inchikeys') else []
    return params


def _validate_targeted_analysis_params(params: Dict[str, Any], location: str) -> Dict[str, Any]:
    """Validate and coerce a single PARAMS block from TARGETED_ANALYSES.
    """
    params['include_lcmsruns'] = list(params['include_lcmsruns']) if params.get('include_lcmsruns') else DEFAULT_INCLUDE_LCMSRUNS_ANALYSES
    excl = params.get('exclude_lcmsruns')
    if excl is None:
        params['exclude_lcmsruns'] = {step: list(runs) for step, runs in DEFAULT_EXCLUDE_LCMSRUNS.items()}
    elif isinstance(excl, dict):
        params['exclude_lcmsruns'] = {step: list(runs) if runs else [] for step, runs in excl.items()}
    elif isinstance(excl, list):
        params['exclude_lcmsruns'] = {'data_extraction': list(excl)}
    else:
        raise ValueError(f"{location}: exclude_lcmsruns must be a dict or list")
    params['do_alignment'] = bool(params.get('do_alignment', True))
    params['remove_unided_compounds'] = bool(params.get('remove_unided_compounds', True))
    params['remove_flagged_compounds'] = bool(params.get('remove_flagged_compounds', True))
    params['only_keep_data_in_feature'] = bool(params.get('only_keep_data_in_feature', False))
    params['apply_istd_curation_to_ema'] = bool(params.get('apply_istd_curation_to_ema', True))
    params['apply_cross_polarity_curation'] = bool(params.get('apply_cross_polarity_curation', True))
    params['suggested_min_conf'] = float(params.get('suggested_min_conf', 0.75))
    params['atlas_extra_time'] = float(params.get('atlas_extra_time', 0.5))
    params['ms1_min_peak_intensity'] = float(params.get('ms1_min_peak_intensity', 1e5))
    params['ms1_min_num_points'] = int(params.get('ms1_min_num_points', 5))
    params['ms1_mz_tolerance_ppm'] = float(params.get('ms1_mz_tolerance_ppm', 5.0))
    params['ms2_min_num_scans'] = int(params.get('ms2_min_num_scans', 1))
    params['ms2_min_precursor_intensity'] = float(params.get('ms2_min_precursor_intensity', 0.0))
    params['ms2_min_score'] = float(params.get('ms2_min_score', 0.25))
    params['ms2_min_matching_frags'] = int(params.get('ms2_min_matching_frags', 1))
    params['ms2_mz_tolerance_ppm'] = float(params.get('ms2_mz_tolerance_ppm', 20.0))
    params['ms2_frag_mz_tolerance'] = float(params.get('ms2_frag_mz_tolerance', 0.05))
    params['gui_require_all_evaluated'] = bool(params.get('gui_require_all_evaluated', True))
    params['gui_top_n_hits'] = int(params.get('gui_top_n_hits', 10))
    gui_colors = params.get('gui_lcmsruns_colors')
    params['gui_lcmsruns_colors'] = dict(gui_colors) if gui_colors else {}
    note_overrides = params.get('note_options_overrides')
    if not isinstance(note_overrides, dict):
        params['note_options_overrides'] = {}
    else:
        clean_overrides = {}
        for note_type in ['ms1_notes', 'ms2_notes', 'other_notes']:
            val = note_overrides.get(note_type, None)
            if val is None:
                continue
            if isinstance(val, dict):
                clean_overrides[note_type] = {str(k): str(v) for k, v in val.items()}
        params['note_options_overrides'] = clean_overrides
    params['create_curation_notebooks'] = bool(params.get('create_curation_notebooks', True))
    params['upload_to_gdrive'] = bool(params.get('upload_to_gdrive', True))
    # skip_outputs is a free-form field; pass through as-is (None or list)
    params['skip_outputs'] = params.get('skip_outputs', None)
    return params


def load_metatlas2_config(config_path: str) -> "Metatlas2Config":
    """Load and validate a metatlas2 YAML config file.

    The config uses ``yaml.safe_load`` with a structure where analysis name
    is a mapping key, so each unique analysis is unambiguous:

    .. code-block:: yaml

        TARGETED_ANALYSES:
          HILICZ:
            POS:
              EMA:
                ANALYSIS-NAME-1:
                  ATLAS:
                    uid: atl-ref-ema-hilicz-pos-...
                  PARAMS:
                    ...
                ANALYSIS-NAME-2:
                  ATLAS:
                    uid: atl-ref-ema-hilicz-pos-...
                  PARAMS:
                    ...

    Returns a :class:`~metatlas2.workflow_objects.Metatlas2Config` instance
    whose ``targeted_analyses`` attribute is a flat list of
    :class:`~metatlas2.workflow_objects.TargetedAnalysis` objects — one per
    unique ``chrom/pol/analysis_type/name`` combination.
    """
    from metatlas2.workflow_objects import Metatlas2Config, TargetedAnalysis

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    # ── top-level structure ────────────────────────────────────────────────
    if 'WORKFLOWS' not in raw:
        raise ValueError("Missing required configuration section: WORKFLOWS")
    for subsection in ("RT_ALIGNMENT", "TARGETED_ANALYSES"):
        if subsection not in raw['WORKFLOWS']:
            raise ValueError(f"Missing required WORKFLOWS subsection: {subsection}")

    # ── RT_ALIGNMENT ───────────────────────────────────────────────────────
    rt_alignment_config: Dict[str, Any] = {}
    for chromatography, chrom_cfg in raw['WORKFLOWS']['RT_ALIGNMENT'].items():
        location = f"RT_ALIGNMENT {chromatography}"
        if 'ATLAS' not in chrom_cfg:
            raise ValueError(f"{location} missing ATLAS section")
        if 'uid' not in chrom_cfg['ATLAS']:
            raise ValueError(f"{location} missing ATLAS uid field")
        uid = chrom_cfg['ATLAS']['uid']
        chrom_cfg['ATLAS']['uid'] = str(uid) if uid else None
        chrom_cfg['PARAMS'] = _validate_rt_alignment_params(
            dict(chrom_cfg.get('PARAMS') or {}), location
        )
        rt_alignment_config[chromatography] = chrom_cfg

    # ── TARGETED_ANALYSES → flat list of TargetedAnalysis objects ──────────
    # Structure: chrom -> polarity -> analysis_type -> name -> {ATLAS, PARAMS}
    targeted_analyses: list = []
    for chromatography, chrom_cfg in raw['WORKFLOWS']['TARGETED_ANALYSES'].items():
        for polarity, pol_cfg in chrom_cfg.items():
            for analysis_type, named_entries in pol_cfg.items():
                if not isinstance(named_entries, dict):
                    raise ValueError(
                        f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_type} "
                        f"must be a dict mapping analysis name -> {{ATLAS, PARAMS}}"
                    )
                for name, entry in named_entries.items():
                    name = str(name)
                    location = f"TARGETED_ANALYSES {chromatography}/{polarity}/{analysis_type}/{name}"
                    if not isinstance(entry, dict):
                        raise ValueError(f"{location} must be a dict with ATLAS and PARAMS keys")
                    if 'ATLAS' not in entry:
                        raise ValueError(f"{location} missing ATLAS section")
                    if 'uid' not in entry['ATLAS']:
                        raise ValueError(f"{location} missing ATLAS uid field")
                    atlas_uid = entry['ATLAS']['uid']
                    atlas_uid = str(atlas_uid) if atlas_uid else None
                    params = _validate_targeted_analysis_params(
                        dict(entry.get('PARAMS') or {}), location
                    )
                    targeted_analyses.append(TargetedAnalysis(
                        chromatography=chromatography,
                        polarity=polarity,
                        analysis_type=analysis_type,
                        name=name,
                        atlas_uid=atlas_uid,
                        params=params,
                    ))

    paths_config: Dict[str, Any] = dict(raw['WORKFLOWS'].get('PATHS') or {})

    logger.info(
        f"Loaded config from {config_path}: "
        f"{len(rt_alignment_config)} RT alignment chromatographies, "
        f"{len(targeted_analyses)} targeted analyses"
    )
    return Metatlas2Config(
        paths_config=paths_config,
        rt_alignment_config=rt_alignment_config,
        targeted_analyses=targeted_analyses,
    )

def load_compound_input(file_path: str) -> pd.DataFrame:
    """Load compound input file (TSV/CSV) and validate required columns."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Compound input file not found: {file_path}")
    
    # Try to read as TSV first, then CSV
    if file_path.suffix.lower() != '.csv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    # Check for required columns (need to use label if it's present)
    if 'compound_name' in df.columns and 'label' in df.columns:
        df.rename(columns={'compound_name': 'compound_name_input', 'label': 'compound_name'}, inplace=True)
    elif 'compound_name' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'compound_name'})
    required_columns = ['inchi_key', 'compound_name']
    check_missing_columns(df, required_columns)
    
    logger.info(f"Loaded {len(df)} compounds from {file_path}")
    return df

def detect_atlas_input_chromatography(df: pd.DataFrame) -> str:
    """Detect chromatography type from atlas input data."""
    if 'chromatography' in df.columns:
        chrom_values = df['chromatography'].dropna().unique()
        return str(chrom_values[0])
    
    return 'Unknown'

def detect_atlas_input_polarity(df: pd.DataFrame) -> str:
    """Detect polarity from atlas input data."""
    if 'polarity' in df.columns:
        pol_values = df['polarity'].dropna().unique()
        if len(pol_values) > 0:
            return str(pol_values[0])
    
    # Try to infer from adduct information
    if 'adduct' in df.columns:
        adducts = ' '.join(df['adduct'].dropna().astype(str))
        if '+' in adducts and '[M+H]+' in adducts:
            return 'positive'
        elif '-' in adducts and '[M-H]-' in adducts:
            return 'negative'
    
    return 'positive'  # Default

def check_missing_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Check for missing required columns and raise error if any are missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

def load_atlas_input(file_path: str) -> pd.DataFrame:
    """Load atlas input file and validate required columns."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Atlas input file not found: {file_path}")
    
    # Try to read as TSV first, then CSV
    if file_path.suffix.lower() != '.csv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.read_csv(file_path)

    # Check for required columns for atlas creation (need to use label if it's present)
    if 'compound_name' in df.columns and 'label' in df.columns:
        df.rename(columns={'compound_name': 'compound_name_input', 'label': 'compound_name'}, inplace=True)
    elif 'compound_name' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'compound_name'})
    required_columns = ['inchi_key', 'compound_name', 'rt_peak', 'mz', 'adduct']
    check_missing_columns(df, required_columns)
    
    # Add default values for optional columns
    if 'rt_min' not in df.columns:
        df['rt_min'] = df['rt_peak'] - 0.5
    if 'rt_max' not in df.columns:
        df['rt_max'] = df['rt_peak'] + 0.5
    if 'mz_tolerance' not in df.columns:
        df['mz_tolerance'] = 5.0
    
    logger.info(f"Loaded {len(df)} atlas entries from {file_path}")
    
    return df

def save_atlas_metadata_to_csv(atlas_obj: "Atlas", output_path: str) -> None:
    """Save Atlas metadata to CSV file."""
    logger.info(f"Saving Atlas data to {output_path}...")
    atlas_info = atlas_obj.to_dict()
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=atlas_info.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(atlas_info)

def save_atlas_data_to_tsv(atlas_obj: "Atlas", output_path: str) -> None:
    atlas_df = atlas_obj.to_dataframe()
    output_file = f"{atlas_obj.atlas_uid}.tsv"
    atlas_df.to_csv(f"{output_path}/{output_file}", index=False, sep='\t')

def load_atlas_data_from_csv(file_path: str) -> pd.DataFrame:
    """Load Atlas data from CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Atlas data file not found: {file_path}")
    return pd.read_csv(file_path)

def change_ownership_to_metatlas_group(project_dir_path: str) -> None:
    """Change ownership of project directory to metatlas group for HPC environments."""

    group_name = 'metatlas'

    try:
        grp.getgrnam(group_name)
    except KeyError:
        logger.warning(f"Group '{group_name}' not found. Skipping ownership change.")
        return

    try:
        subprocess.run(['chgrp', '-R', group_name, project_dir_path], check=True)
        logger.info(f"Changed group of {project_dir_path} to {group_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to change group: {e}")


def log_filter_table(steps, starting_entries, starting_compounds, entries_label="Entries", title=None):
    """Log a fixed-width summary table of sequential filtering steps.

    Args:
        steps: list of (label, count, compounds) tuples; first entry should be the 'start' row.
        starting_entries: row count before any filtering (denominator for % columns).
        starting_compounds: unique compound count before any filtering (denominator for % columns).
        entries_label: column header for the row-count column (e.g. 'Entries' or 'Scans').
        title: optional log message title; defaults to '<entries_label> filtering summary'.
    """
    if title is None:
        title = f"{entries_label} filtering summary"
    pct_label = f"{entries_label} %"
    col_w = [25, 10, 12, 10, 12]
    header = (
        f"{'Step':<{col_w[0]}} {entries_label:>{col_w[1]}} {pct_label:>{col_w[2]}} "
        f"{'Compounds':>{col_w[3]}} {'Compounds %':>{col_w[4]}}"
    )
    sep = "-" * sum(col_w + [4])  # 4 spaces between 5 columns
    rows = [header, sep]
    for label, entries, compounds in steps:
        pct_e = entries / starting_entries * 100 if starting_entries > 0 else 0.0
        pct_c = compounds / starting_compounds * 100 if starting_compounds > 0 else 0.0
        rows.append(
            f"{label:<{col_w[0]}} {entries:>{col_w[1]}} {pct_e:>{col_w[2] - 1}.1f}% "
            f"{compounds:>{col_w[3]}} {pct_c:>{col_w[4] - 1}.1f}%"
        )
    rows.append(sep)
    logger.info("%s:\n%s", title, "\n".join(rows))