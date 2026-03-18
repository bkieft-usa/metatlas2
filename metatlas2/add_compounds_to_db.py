import argparse
import sys
sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf
from workflow_objects import Compound

logger = lcf.get_logger('workflow_objects')

def add_compounds_to_db(
    config_path: str,
    overwrite_db: bool = False,
) -> None:
    """
    Creates main database (if needed) and loads compounds from config file paths.
    """
    logger.info("Adding compounds from config file to database...")
    Compound.create_from_config(
        config_path=config_path,
        overwrite_db=overwrite_db
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add compounds to the database from a config file.')
    parser.add_argument('--config_path', type=str, help='Path to the compounds config file.')
    parser.add_argument('--overwrite_db', action='store_true', default=False, help='Overwrite the existing database if set.')
    args = parser.parse_args()
    add_compounds_to_db(args.config_path, args.overwrite_db)