import os
from pathlib import Path
import logging
from metatlas2.workflow_objects import NewCompoundsConfig
import metatlas2.logging_config as lcf

def add_compounds_to_db(
    config_path: str,
    overwrite_db: bool = False,
) -> None:
    """
    Creates main database (if needed) and loads compounds from config file paths.
    """
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    log_dir = Path(data_dir) / "logs" if data_dir else Path(config_path).parent
    log_file = str(log_dir / "add_compounds_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding compounds from config file to database...")
        NewCompoundsConfig.from_yaml(config_path).execute(overwrite_db=overwrite_db)
