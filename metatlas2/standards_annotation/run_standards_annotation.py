from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import metatlas2.shared_tools.logging_config as lcf


def run_standards_annotation(config_path: str) -> None:
    """Main entrypoint for the standards-annotation workflow.

    1. Loads the config YAML.
    2. Sets up the output directory and log file.
    3. Creates a new project directory and database.
    4. Runs data extraction and persists results to the project DB.
    5. Generates a pre-filled Jupyter notebook for the analyst to open.

    Args:
        config_path: Absolute or relative path to the standards-annotation
                     YAML config file.
    """
    from metatlas2.shared_tools.load_tools import load_standards_config
    from metatlas2.standards_annotation.standards_annotation_db import (
        get_project_db_path,
        get_project_output_root,
        get_raw_data_root,
        create_project_db,
    )
    from metatlas2.standards_annotation.standards_annotation_tools import (
        run_standards_extraction,
    )
    from metatlas2.shared_tools.notebook_generator import generate_standards_annotation_notebook

    config_path = str(Path(config_path).resolve())
    config = load_standards_config(config_path)

    # Set up output directory and logging
    output_root = get_project_output_root(config)
    raw_data_root = get_raw_data_root(config)
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Standards annotation output already exists: {output_root}. "
            "Choose a different standards_run_name in the config or manually delete "
            "the entire existing project folder before starting again."
        ) from exc
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

        # Persist the config YAML and input standards table alongside results
        input_file = config["project"]["standards_input_file"]
        for src in (config_path, input_file):
            dest = shutil.copy2(src, output_root / Path(src).name)
            logger.info(f"Copied {src} to {dest}")

        db_path = get_project_db_path(config)
        run_name = config["project"]["standards_run_name"]

        logger.info(f"Creating project DB for standards annotation run '{run_name}'...")
        create_project_db(db_path)
        logger.info("Running data extraction...")
        data = run_standards_extraction(config, raw_data_root, db_path)
        logger.info("Data extraction complete.")

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