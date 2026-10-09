import os
from pathlib import Path
import logging
from metatlas2.workflow_objects import NewAtlasesConfig
import metatlas2.logging_config as lcf

def add_atlases_to_db(
    config_path: str,
) -> None:
    """
    Creates atlases from config file paths and saves them to the database.
    """
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    log_dir = Path(data_dir) / "logs" if data_dir else Path(config_path).parent
    log_file = str(log_dir / "add_atlases_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding atlases from config file to database...")
        NewAtlasesConfig.from_yaml(config_path).execute()
