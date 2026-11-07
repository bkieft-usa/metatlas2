"""
metatlas2/run_standards_annotation.py

CLI entrypoint for the standards-annotation workflow.

Runs background data extraction and generates a pre-filled Jupyter notebook
that the analyst opens to launch the interactive GUI.

Usage (via metatlas2.sh):
    metatlas2 standards-annotation --config_path /path/to/config.yaml

Usage (direct Python):
    python -m metatlas2.run_standards_annotation --config_path /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import metatlas2.logging_config as lcf


def run_standards_annotation(config_path: str) -> None:
    """Main entrypoint for the standards-annotation workflow.

    1. Loads the config YAML.
    2. Sets up the output directory and log file.
    3. Checks whether extraction has already been completed (resumability).
    4. Runs data extraction if needed and persists results to the project DB.
    5. Generates a pre-filled Jupyter notebook for the analyst to open.

    Args:
        config_path: Absolute or relative path to the standards-annotation
                     YAML config file.
    """
    from metatlas2.load_tools import load_standards_config
    from metatlas2.standards_annotation_db import (
        get_project_db_path,
        get_project_output_root,
        get_raw_data_root,
        create_project_db,
        is_extraction_complete,
        is_peaks_complete,
        load_extraction_from_db,
    )
    from metatlas2.standards_annotation_tools import (
        run_standards_extraction,
        detect_and_save_peaks,
    )
    from metatlas2.notebook_generator import generate_standards_annotation_notebook

    config_path = str(Path(config_path).resolve())
    config = load_standards_config(config_path)

    # Set up output directory and logging
    output_root = get_project_output_root(config)
    raw_data_root = get_raw_data_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "standards_annotation.log")

    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger("run_standards_annotation")
        logger.info(f"Standards annotation workflow starting for config: {config_path}")
        logger.info(f"Output root: {output_root}")

        db_path = get_project_db_path(config)
        run_name = config["project"]["standards_run_name"]

        # ------------------------------------------------------------------
        # Gate 1: MS1/MS2 extraction
        # Checks sa_extraction_targets (not sa_ms1) so a run that completed
        # extraction but crashed before peak detection is correctly resumed.
        # ------------------------------------------------------------------
        if is_extraction_complete(db_path):
            logger.info(
                f"Extraction already exists for standards annotation run "
                f"'{run_name}'; loading from DB."
            )
            data = load_extraction_from_db(db_path, config)
        else:
            logger.info(
                f"No existing extraction for standards annotation run "
                f"'{run_name}'. Creating project DB..."
            )
            create_project_db(db_path)
            logger.info("Running data extraction...")
            data = run_standards_extraction(config, raw_data_root, db_path)
            logger.info("Data extraction complete.")

        # ------------------------------------------------------------------
        # Gate 2: Peak detection (independent of extraction gate)
        # If extraction completed but the process crashed before peak
        # detection, this gate ensures peaks are detected on resume without
        # re-running the expensive extraction step.
        # ------------------------------------------------------------------
        if not is_peaks_complete(db_path):
            logger.info(
                "Peak detection not yet complete for this run. "
                "Running detect_and_save_peaks..."
            )
            extraction_cfg = config.get("extraction", {})
            peak_params = extraction_cfg.get("peak_finding", {})
            top_n_ms2 = int(extraction_cfg.get("ms2_top_n_per_peak", 5))
            detect_and_save_peaks(data, peak_params=peak_params, top_n_ms2=top_n_ms2)
            logger.info("Peak detection complete.")
        else:
            logger.info("Peak detection already complete; skipping.")

        nb_path = generate_standards_annotation_notebook(
            data=data,
            config=config,
            config_path=config_path,
            output_root=output_root,
        )

        logger.info("Standards annotation setup complete.")
        logger.info(f"Notebook generated at: {nb_path}")
        logger.info("Open the notebook in JupyterHub to launch the annotation GUI.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the metatlas2 standards-annotation workflow."
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the standards_annotation YAML config file.",
    )
    args = parser.parse_args()
    run_standards_annotation(args.config_path)