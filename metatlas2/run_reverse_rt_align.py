"""
metatlas2/run_reverse_rt_align.py

CLI entrypoint for the reverse RT alignment phase of the standards-annotation
workflow.

Reads analyst selections from the project DuckDB (sa_selections), builds a
reverse RT alignment model using QC h5 files and a QC atlas from the main
metatlas DuckDB, applies the model to the selections, and writes aligned rows
to sa_rt_aligned.  Also generates:
  - RTAlignment-Summary__{run_name}.pdf
  - RTAlignment-Table__{run_name}.xlsx
  - RefStd_Atlas__{chrom}__{pol}__{run_name}.csv  (one per chrom × polarity)
  - A pre-filled Jupyter notebook for reviewing results.

Usage (via metatlas2.sh):
    metatlas2 reverse-rt-align --config_path /path/to/config.yaml

Usage (direct Python):
    python -m metatlas2.run_reverse_rt_align --config_path /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import metatlas2.logging_config as lcf


def run_reverse_rt_align(config_path: str) -> None:
    """Main entrypoint for the reverse RT alignment workflow.

    Steps:
      1. Loads the config YAML.
      2. Sets up the output directory and log file.
      3. Runs the reverse RT alignment (or regenerates outputs if already done).
      4. Generates a pre-filled Jupyter notebook for reviewing results.

    Args:
        config_path: Absolute or relative path to the standards-annotation
                     YAML config file.
    """
    from metatlas2.load_tools import load_standards_config
    from metatlas2.standards_annotation_db import (
        get_project_db_path,
        get_project_output_root,
    )
    from metatlas2.standards_annotation_rt_align import run_reverse_rt_alignment

    config_path = str(Path(config_path).resolve())
    config = load_standards_config(config_path)

    # Set up output directory and logging
    output_root = get_project_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "reverse_rt_align.log")

    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger("run_reverse_rt_align")
        logger.info(f"Reverse RT alignment starting for config: {config_path}")

        db_path = get_project_db_path(config)
        output_path = output_root / "outputs"
        run_reverse_rt_alignment(config_path, db_path, output_path)

        logger.info("Reverse RT alignment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the metatlas2 reverse RT alignment phase of the "
            "standards-annotation workflow."
        )
    )
    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the standards_annotation YAML config file.",
    )
    args = parser.parse_args()
    run_reverse_rt_align(args.config_path)
