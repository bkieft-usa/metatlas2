import os
from pathlib import Path
import logging
from metatlas2.workflow_objects import NewMsmsRefsConfig
import metatlas2.logging_config as lcf


def add_msms_refs_to_db(config_path: str) -> None:
    """Stream one or more ``.tsv`` MSMS-refs files and bulk-insert all spectra
    into the ``msms_refs`` table of the main metatlas DuckDB.
    """
    data_dir = os.environ.get("METATLAS_DATA_DIR", "")
    log_dir = Path(data_dir) / "logs" if data_dir else Path(config_path).parent
    log_file = str(log_dir / "add_msms_refs_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding MSMS reference spectra from config file to database...")
        NewMsmsRefsConfig.from_yaml(config_path).execute()
