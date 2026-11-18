import os
from pathlib import Path
import logging
from metatlas2.workflow_objects import add_compounds_from_files
import metatlas2.run_targeted_analysis as rtg
import metatlas2.logging_config as lcf

def add_compounds_to_db(
    input_csv: str,
    overwrite_db: bool = False,
) -> None:
    """
    Create the main database if needed and load compounds from one CSV/TSV file.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Compound input file not found: {input_path}")
    if input_path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError(f"Compound input must be a CSV or TSV file: {input_path}")

    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    log_dir = Path(data_dir) / "logs" if data_dir else input_path.parent
    log_file = str(log_dir / "add_compounds_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding compounds from input file to database...")
        paths = rtg.set_up_paths(config=None)
        add_compounds_from_files([str(input_path)], paths, overwrite_db=overwrite_db)
