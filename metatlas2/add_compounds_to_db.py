import argparse
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
    config_dir = Path(config_path).parent
    log_file = str(config_dir / "add_compounds_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding compounds from config file to database...")
        NewCompoundsConfig.from_yaml(config_path).execute(overwrite_db=overwrite_db)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add compounds to the database from a config file.')
    parser.add_argument('--config_path', type=str, help='Path to the compounds config file.')
    parser.add_argument('--overwrite_db', action='store_true', default=False, help='Overwrite the existing database if set.')
    args = parser.parse_args()
    add_compounds_to_db(args.config_path, args.overwrite_db)