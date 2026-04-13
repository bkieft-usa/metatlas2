import argparse
from metatlas2.workflow_objects import Atlas
import metatlas2.logging_config as lcf
logger = lcf.get_logger('workflow_objects')

def add_atlases_to_db(
    config_path: str
) -> None:
    """
    Creates atlases from config file paths and saves them to the database.
    """
    logger.info("Adding atlases from config file to database...")
    Atlas.create_from_config(
        config_path=config_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add atlases to the database from a config file.')
    parser.add_argument('--config_path', type=str, help='Path to the atlas config file.')
    args = parser.parse_args()
    add_atlases_to_db(args.config_path)