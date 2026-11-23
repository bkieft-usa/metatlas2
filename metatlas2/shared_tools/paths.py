"""Filesystem path helpers shared by targeted, standards-annotation and DB tools.

All paths are derived from the ``METATLAS_DATA_DIR`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path


def get_project_db_path(project_name: str) -> str:
    """Derive the project database path from ``project_name`` alone.

    Scans ``{METATLAS_DATA_DIR}/projects/targeted_outputs/`` for a file named
    ``{project_name}.duckdb`` under any ``owner/user`` subdirectory.  This
    allows async stages (GUI, summary) to locate the project DB without needing
    the config YAML on disk.

    Args:
        project_name: The full project name string.

    Returns:
        Absolute path to the ``.duckdb`` file.

    Raises:
        FileNotFoundError: If no matching database file is found.
        ValueError: If ``METATLAS_DATA_DIR`` is not set.
    """
    data_dir = os.environ.get("METATLAS_DATA_DIR")
    if data_dir is None:
        raise ValueError(
            "METATLAS_DATA_DIR environment variable is not set. "
            "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it."
        )
    base = Path(data_dir) / "projects" / "targeted_outputs"
    target = f"*/*/{project_name}/{project_name}.duckdb"
    matches = list(base.glob(target))
    if not matches:
        raise FileNotFoundError(
            f"No project database found for '{project_name}' under {base}. "
            "Run project setup first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple project databases found for '{project_name}': {matches}. "
            "Specify the path explicitly."
        )
    return str(matches[0])


def set_up_paths(
    config: "Metatlas2Config | None",
    project_name: str | None = None,
    rt_alignment_number: int | None = None,
    analysis_number: int | None = None,
) -> "PathsConfig":
    """Build all workflow paths and create all output directories for a run.

    Returns a :class:`~metatlas2.shared_tools.workflow_objects.PathsConfig` TypedDict
    containing every filesystem path needed by the workflow.  The dict is
    structurally compatible with plain ``dict[str, str]`` so existing code
    that accesses paths by key continues to work unchanged.
    """

    data_dir = os.environ.get("METATLAS_DATA_DIR")
    if data_dir is None:
        raise EnvironmentError(
            "METATLAS_DATA_DIR environment variable is not set. "
            "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it."
        )
    lcmsruns_path = f"{data_dir}/raw_data/"
    main_db_path = f"{data_dir}/databases/main_db/metatlas.duckdb"
    pubchem_cache_path = f"{data_dir}/databases/pubchem_cache/pubchem_global_cache.json"
    project_output_path = f"{data_dir}/projects/targeted_outputs/"
    modelseed_table_path = f"{data_dir}/databases/modelseed_db/modelseed.tsv"
    chebi_cache_path = f"{data_dir}/databases/chebi_cache/chebi_global_cache.json"
    parquet_output_path = f"{data_dir}/projects/parquet_outputs/"

    if project_name is None: # This is for converting files and adding compounds and atlases to main db
        return {
            "lcmsruns_directory": str(lcmsruns_path),
            "main_db_path": str(main_db_path),
            "pubchem_cache_path": str(pubchem_cache_path),
            "modelseed_table_path": str(modelseed_table_path),
            "chebi_cache_path": str(chebi_cache_path),
        }

    if config is None:
        raise ValueError("A config is required when project_name is specified.")
    owner = config.owner
    user = os.environ.get("USER", "other")
    if not owner:
        raise ValueError("Owner not specified in config under PATHS.owner")
    if user is None:
        raise ValueError("USER environment variable is not set")
    project_output_dir = Path(project_output_path) / owner / user / project_name
    parquet_output_dir = Path(parquet_output_path) # / owner / user / project_name
    try:
        project_short = str(project_name.split("_")[4]) + "_RTA" + str(rt_alignment_number) + "_TGA" + str(analysis_number)
    except Exception as e:
        raise ValueError(f"Could not parse short project name from full project name - is it malformed? Error: {e}")
    rta_dir = project_output_dir / f"RTA{rt_alignment_number}"
    analysis_dir = rta_dir / f"TGA{analysis_number}"

    paths = {
        "lcmsruns_directory": str(Path(lcmsruns_path) / owner / project_name),
        "project_directory": str(project_output_dir),
        "log_path": str(project_output_dir / f"{project_short}.log"),
        "project_db_path": str(project_output_dir / f"{project_name}.duckdb"),
        "main_db_path": str(main_db_path),
        "pubchem_cache_path": str(pubchem_cache_path),
        "modelseed_table_path": str(modelseed_table_path),
        "chebi_cache_path": str(chebi_cache_path),
        "parquet_output_dir": str(parquet_output_dir),
        "rt_alignment_output_dir": str(rta_dir),
        "rt_alignment_results_dir": str(rta_dir / "rt_alignment_results"),
        "aligned_atlases_store_file": str(rta_dir / "rt_aligned_atlases.csv"),
        "analysis_output_dir": str(analysis_dir),
        "auto_ided_atlases_store_file": str(analysis_dir / "auto_ided_atlases.csv"),
        "curated_atlases_store_file": str(analysis_dir / "curated_atlases.csv"),
    }

    if not Path(paths["lcmsruns_directory"]).exists():
        raise ValueError(f"Raw data directory not found: {paths['lcmsruns_directory']}")
    if not Path(paths["main_db_path"]).exists():
        raise ValueError(f"Main database not found: {paths['main_db_path']}")

    parquet_output_dir.mkdir(parents=True, exist_ok=True)
    project_output_dir.mkdir(parents=True, exist_ok=True)
    rta_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    Path(paths['rt_alignment_results_dir']).mkdir(parents=True, exist_ok=True)

    return paths
