import argparse
from pathlib import Path
import logging
from metatlas2.workflow_objects import NewMsmsRefsConfig
import metatlas2.logging_config as lcf


def add_msms_refs_to_db(config_path: str) -> None:
    """Stream one or more ``.jsonl`` MSMS-refs files and bulk-insert all spectra
    into the ``reference_fragmentation_data`` table of the main metatlas DuckDB.

    Args:
        config_path: Path to a YAML file with an ``MSMS_REFS`` top-level key.
    """
    config_dir = Path(config_path).parent
    log_file = str(config_dir / "add_msms_refs_to_db.log")
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=log_file,
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        logger = lcf.get_logger('workflow_objects')
        logger.info("Adding MSMS reference spectra from config file to database...")
        NewMsmsRefsConfig.from_yaml(config_path).execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add MSMS reference spectra to the main metatlas database from a config file.'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the MSMS refs config YAML file (must contain an MSMS_REFS key).',
    )
    args = parser.parse_args()
    add_msms_refs_to_db(args.config_path)
