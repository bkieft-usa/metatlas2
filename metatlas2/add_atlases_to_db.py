import argparse
from pathlib import Path
from metatlas2.workflow_objects import Atlas
import metatlas2.logging_config as lcf

def add_atlases_to_db(
    config_path: str,
    log_to_stdout: bool = False
) -> None:
    """
    Creates atlases from config file paths and saves them to the database.
    """
    log_file = None
    if not log_to_stdout:
        config_dir = Path(config_path).parent
        log_file = str(config_dir / "add_atlases_to_db.log")
    lcf.setup_logging(log_level=None, log_file=log_file, log_to_stdout=log_to_stdout)
    logger = lcf.get_logger('workflow_objects')
    logger.info("Adding atlases from config file to database...")
    Atlas.create_from_config(
        config_path=config_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add atlases to the database from a config file.')
    parser.add_argument('--config_path', type=str, help='Path to the atlas config file.')
    parser.add_argument('--log-to-stdout', action='store_true', default=False, help='Write log output to stdout instead of a log file in the config directory')
    args = parser.parse_args()
    add_atlases_to_db(args.config_path, log_to_stdout=args.log_to_stdout)