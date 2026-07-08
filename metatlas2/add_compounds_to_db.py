import argparse
from pathlib import Path
import logging
from metatlas2.workflow_objects import Compound
import metatlas2.logging_config as lcf

def add_compounds_to_db(
    config_path: str,
    overwrite_db: bool = False,
    log_to_stdout: bool = True,
) -> None:
    """
    Creates main database (if needed) and loads compounds from config file paths.
    """
    config_dir = Path(config_path).parent
    log_file = str(config_dir / "add_compounds_to_db.log")
    lcf.setup_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=log_to_stdout,
        reconfigure_existing=False,
    )
    logger = lcf.get_logger('workflow_objects')
    logger.info("Adding compounds from config file to database...")
    Compound.create_from_config(
        config_path=config_path,
        overwrite_db=overwrite_db
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add compounds to the database from a config file.')
    parser.add_argument('--config_path', type=str, help='Path to the compounds config file.')
    parser.add_argument('--overwrite_db', action='store_true', default=False, help='Overwrite the existing database if set.')
    parser.add_argument('--no-log-to-stdout', action='store_false', dest='log_to_stdout', default=True, help='Disable stdout logging and write only to log file in the config directory')
    args = parser.parse_args()
    add_compounds_to_db(args.config_path, args.overwrite_db, log_to_stdout=args.log_to_stdout)