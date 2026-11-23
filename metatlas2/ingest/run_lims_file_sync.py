"""CLI for the LIMS file-table sync (runs inside the metatlas2 container).

Usage::

    python -m metatlas2.ingest.run_lims_file_sync [--dry_run] [--skip_file_tables]
        [--skip_lcmsrun_names] [--skip_lcmsrun_links] [--log_file FILE | --log_to_stdout]

``scripts/metatlas2.sh lims-sync ...`` runs this in the container. Equivalent to
the legacy ``update_lcmsfiles_in_lims.py --update_file_tables True
--update_lcmsrun_names True --update_lcmsrun_items True``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

DEFAULT_LOG_FILE = "/global/cfs/cdirs/m2650/untargeted_logs/lims_update.log"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sync raw/mzML/h5 files on disk into the LabKey LIMS file tables.")
    p.add_argument("--raw_data_dir", default="/global/cfs/cdirs/metatlas/raw_data")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="Scan and compare, but write nothing to LIMS.")
    p.add_argument("--skip_file_tables", action="store_true", help="Skip step 1 (file tables).")
    p.add_argument("--skip_lcmsrun_names", action="store_true", help="Skip step 2 (new lcmsrun names).")
    p.add_argument("--skip_lcmsrun_links", action="store_true", help="Skip step 3 (lcmsrun file links).")
    p.add_argument("--lims_server", default="metatlas.jgi.doe.gov")
    p.add_argument("--lims_container", default="LIMS/")
    p.add_argument("--lims_api_key_file", default="/global/cfs/cdirs/metatlas/labkey_user.txt")
    p.add_argument("--lims_max_rows", type=int, default=5_000_000,
                   help="Abort if a table read returns this many rows (possible truncation).")
    p.add_argument("--log_file", default=DEFAULT_LOG_FILE)
    p.add_argument("--log_to_stdout", action="store_true")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import metatlas2.shared_tools.logging_config as lcf

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    lcf.setup_logging(log_level=level, log_file=None if args.log_to_stdout else args.log_file,
                      log_to_stdout=args.log_to_stdout)
    from metatlas2.ingest.lims_file_sync import run_lims_file_sync
    from metatlas2.shared_tools.lims import LimsClient

    logger = logging.getLogger("metatlas2.ingest.lims_file_sync")
    logger.info(f"Successfully started LIMS file sync on {datetime.now()}:")
    if args.dry_run:
        logger.info("DRY RUN: nothing will be written to LIMS.")
    lims = LimsClient(args.lims_server, args.lims_container, args.lims_api_key_file,
                      max_rows=args.lims_max_rows, dry_run=args.dry_run)
    try:
        run_lims_file_sync(
            lims, raw_data_dir=args.raw_data_dir,
            update_file_tables=not args.skip_file_tables,
            update_lcmsrun_names=not args.skip_lcmsrun_names,
            update_lcmsrun_links=not args.skip_lcmsrun_links,
        )
    except Exception:
        logger.exception("ERROR: LIMS file sync failed")
        return 1
    finally:
        logger.info(f"Successfully completed LIMS file sync on {datetime.now()}:")
    return 0


if __name__ == "__main__":
    sys.exit(main())
