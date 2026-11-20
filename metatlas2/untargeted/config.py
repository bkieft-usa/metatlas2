"""Configuration for the untargeted pipeline.

Defaults live in :data:`DEFAULT_CONFIG` (so the pipeline works inside the
container without any config file). A YAML file with the same UPPERCASE
sections can override any subset of keys, and CLI flags override both.

The YAML at ``configs/untargeted/untargeted_pipeline.yaml`` is a fully
annotated copy of these defaults.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SECTIONS = ("paths", "lims", "project_selection", "mzmine", "filtering", "gnps2", "gdrive")

#: Names accepted by ``--skip_steps`` (order matters: index == step number - 1).
STEP_NAMES = (
    "Sync",
    "MZmine_Status",
    "MZmine_Submit",
    "FBMN_Status",
    "FBMN_Submit",
    "FBMN_Download",
    "Zip_and_Upload",
)

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "paths": {
        "raw_data_dir": "/global/cfs/cdirs/metatlas/raw_data",
        "output_dir": "/global/cfs/cdirs/metatlas/projects/untargeted_tasks",
        "download_dir": "/global/cfs/cdirs/metatlas/projects/untargeted_outputs",
        "log_file": "/global/cfs/cdirs/m2650/untargeted_logs/untargeted_pipeline.log",
        # Force a raw_data subdirectory (e.g. "jgi"); null = infer from project name.
        "raw_data_subdir": None,
        # Subdirectories of raw_data_dir searched when the inferred one has no project.
        "fallback_raw_data_subdirs": [
            "jgi", "egsb", "akuftin", "agolini", "kblouie", "lzhan",
            "smkosina", "jsjordan", "mtvu", "tharwood", "bpb",
        ],
        # Unix group applied (chgrp -R) to outputs; null to skip.
        "group": "metatlas",
    },
    "lims": {
        "server": "metatlas.jgi.doe.gov",
        "container": "LIMS/",
        "api_key_file": "/global/cfs/cdirs/metatlas/labkey_user.txt",
        "schema": "lists",
        "tasks_table": "untargeted_tasks",
        "files_table": "lcmsrun_plus",
        "timeout": 10000,
        "max_rows": 1000000,
        "batch_size": 1000,
    },
    "project_selection": {
        # Newest file in a project must be at least this old before it is picked up.
        "min_file_age_hours": 3,
        "validate_names": False,
        "skip_blank_filter": False,
        # Sample-group designator(s) of extraction controls (field 12 of filenames).
        "background_designator": ["ExCtrl"],
    },
    "mzmine": {
        "executable": "/global/common/software/m2650/mzmine_parameters/MZmine/MZmine-3.7.2/bin/MZmine",
        "extra_args": "-threads auto -t /tmp",
        "batch_file_dir": "/global/common/software/m2650/mzmine_parameters/batch_files",
        # String in the template XML that is replaced by the polarity output prefix.
        "output_placeholder": "/Users/bpb/Downloads/mzmine_outputs",
        # Instrument (project field 6) -> template. First rule whose substring matches wins.
        "template_rules": [
            {"match": ["iqx", "idx"], "template": "IQX-mzmine-3.7.2-batchparams.xml", "parameter_row": 5},
            {"match": ["exp", "exploris", "qe"], "template": "mzmine-3.7.2-batchparams.xml", "parameter_row": 2},
        ],
        "default_template": "IQX-mzmine-3.7.2-batchparams.xml",
        "default_parameter_row": 5,
        # Optional full path to a single custom batch XML used for every project.
        "custom_batch_params": None,
        "slurm": {
            "nodes": 1,
            "exclusive": True,
            "account": "m2650",
            "constraint": "cpu",
            "qos": "regular",
            "time": "3:00:00",
            "retry_time": "6:00:00",
        },
        # Automatically resubmit MZmine jobs in "09 error" (only those this
        # pipeline queued itself, i.e. with a *_mzmine-queued.txt marker).
        "auto_resubmit_errors": True,
        "max_retries": 1,
        # A RUNNING job older than its time limit + this grace is declared failed.
        "status_grace_hours": 1,
        # A queued job with no RUNNING marker after this long is declared failed.
        "max_pending_hours": 48,
    },
    "filtering": {
        "background_ratio": 3,
        "zero_value": 2 / 3,
        "polar_solvent_front": 0.8,
        "polar_solvent_end": 17.5,
        "nonpolar_solvent_front": 0.5,
        "nonpolar_solvent_end": 10,
    },
    "gnps2": {
        "username": "bpbowen",
        "credentials_file": "/global/cfs/cdirs/metatlas/gnps2/gnps2_bpbowen.txt",
        "base_url": "https://gnps2.org",
        "sftp_host": "sftp.gnps2.org",
        "sftp_port": 443,
        "sftp_retry_wait_seconds": 30,
        "request_timeout_seconds": 300,
        "mirror_raw_data": True,
        "mirror_mzmine_results": True,
        "submit_everything_bagel": True,
        # Internal calibrant m/z removed from MGF files before FBMN submission.
        "calibrant_mz": {"positive": 202.07770, "negative": 202.07880},
        "calibrant_ppm": 5,
        "fbmn_params": {
            "workflowname": "feature_based_molecular_networking_workflow",
            "featurefindingtool": "MZMINE",
            "input_libraries": "LIBRARYLOCATION/LC/LIBRARY",
            "input_supplemental_edges": "",
            "library_analog_search": "0",
            "library_min_cosine": "0.7",
            "library_min_matched_peaks": "3",
            "library_topk": "20",
            "min_peak_intensity": "0.0",
            "networking_max_shift": "1999",
            "networking_min_cosine": "0.7",
            "networking_min_matched_peaks": "3",
            "normalization": "None",
            "pm_tolerance": "0.01",
            "fragment_tolerance": "0.01",
            "precursor_filter": "1",
            "api": "no",
        },
        "everything_bagel_params": {
            "workflowname": "everything_bagel_workflow",
            "detection_preset": "rare",
            "experimentdescription": "",
            "feature_finder_engine": "v1",
            "filter_precursor": "1",
            "filter_window": "1",
            "filtertostructures": "0",
            "formula_prediction_method": "buddy_fiddle",
            "fragment_tolerance": "0.05",
            "input_spectral_library": "LIBRARYLOCATION/LC/LIBRARY",
            "library_min_matched_peaks": "3",
            "library_min_similarity": "0.7",
            "mode": "fbmn",
            "noise_threshold_mode": "standard",
            "pm_tolerance": "0.05",
            "run_peak_resolving_challenger": "no",
            "topk": "20",
        },
    },
    "gdrive": {
        "upload": True,
        "remote": "ben_lbl_gdrive:/untargeted_outputs",
        "add_documentation": True,
        "doc_name": "Untargeted_metabolomics_GNPS2_Guide.docx",
        "abridged_filenames": True,
        # Project is zipped only if total features across polarities exceeds this.
        "min_features": 0,
        # Suffixes stripped from project names before building abridged file names.
        "abridged_strip_suffixes": ["_lenient", "_strict", "_custom", "_test"],
    },
}

_REQUIRED_PATH_KEYS = ("raw_data_dir", "output_dir", "download_dir")


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge *updates* into *base* (in place) and return *base*."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_list(value: Any) -> list[str]:
    """Coerce a comma-separated string or list into a list of stripped strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def normalize_number(value: Any) -> Any:
    """Return ints for integral floats (``3.0`` -> ``3``) so file names match legacy output."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load_untargeted_config(
    config_path: str | Path | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the untargeted pipeline configuration.

    Args:
        config_path: Optional YAML file whose UPPERCASE sections override the defaults.
        overrides: Optional ``{section: {key: value}}`` mapping (from CLI flags);
            ``None`` values are ignored so unset flags keep config/default values.

    Returns:
        Config dict with lowercase section keys (see :data:`DEFAULT_CONFIG`).

    Raises:
        FileNotFoundError: If *config_path* is given but does not exist.
        ValueError: If the YAML has unknown sections or required keys are empty.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file {config_path} must contain a mapping of sections.")
        user = {str(k).lower(): v for k, v in raw.items()}
        unknown = sorted(set(user) - set(SECTIONS))
        if unknown:
            raise ValueError(f"Unknown section(s) in {config_path}: {unknown}. Valid: {list(SECTIONS)}")
        for section, values in user.items():
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"Section {section.upper()} must be a mapping.")
            _deep_update(config[section], values)

    for section, values in (overrides or {}).items():
        if section not in config:
            raise ValueError(f"Unknown override section: {section}")
        _deep_update(config[section], {k: v for k, v in values.items() if v is not None})

    # Normalise list-valued and numeric keys.
    sel = config["project_selection"]
    sel["background_designator"] = _as_list(sel["background_designator"]) or ["ExCtrl"]
    config["paths"]["fallback_raw_data_subdirs"] = _as_list(config["paths"]["fallback_raw_data_subdirs"])
    for key, value in list(config["filtering"].items()):
        config["filtering"][key] = normalize_number(value)

    missing = [k for k in _REQUIRED_PATH_KEYS if not config["paths"].get(k)]
    if missing:
        raise ValueError(f"PATHS is missing required key(s): {missing}")
    for key in ("tasks_table", "files_table", "server", "container"):
        if not config["lims"].get(key):
            raise ValueError(f"LIMS.{key} is required")
    return config


def write_default_config(path: str | Path) -> Path:
    """Write :data:`DEFAULT_CONFIG` to *path* as YAML with UPPERCASE sections."""
    path = Path(path)
    data = {k.upper(): v for k, v in DEFAULT_CONFIG.items()}
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path
