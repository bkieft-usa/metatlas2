"""Command-line entry point for the untargeted pipeline.

Subcommands::

    run          Run Steps 1-7 (what the scheduled job calls).
    status       Print the tab-separated project status table.
    mirror-raw   Hard/soft mirror a project's mzML files to GNPS2.
    write-config Write the default config YAML to a file.

Flag names match the legacy ``run_untargeted_pipeline.py``. Every flag left
unset keeps the value from ``--config`` (or the built-in defaults).

In the container, MZmine jobs are handed to the host through
``--sbatch_queue_file`` (``scripts/metatlas2.sh untargeted run`` sets this).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from metatlas2.untargeted.config import STEP_NAMES, load_untargeted_config, write_default_config


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Untargeted config YAML (defaults built in).")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="Read-only: no LIMS writes, job submissions, GNPS2/Drive uploads, or file changes.")
    p.add_argument("--raw_data_dir", default=None, help="Path to the raw data directory.")
    p.add_argument("--raw_data_subdir", default=None,
                   help="raw_data subdirectory (e.g. jgi, egsb); inferred from project field 1 if unset.")
    p.add_argument("--output_dir", default=None, help="Untargeted task (per-polarity) output directory.")
    p.add_argument("--download_dir", default=None, help="Directory for project zip archives.")
    p.add_argument("--lims_tasks_table", default=None, help="LIMS list holding untargeted tasks.")


def _add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--log_file", default=None, help="Log file (default from config).")
    p.add_argument("--log_level", default="INFO", help="DEBUG, INFO, WARNING, ERROR or CRITICAL.")
    p.add_argument("--log_to_stdout", action="store_true", help="Log to stdout instead of the log file.")


def _bool_pair(p: argparse.ArgumentParser, name: str, help_on: str, help_off: str) -> None:
    p.add_argument(f"--{name}", dest=name, action="store_const", const=True, default=None, help=help_on)
    p.add_argument(f"--no_{name}", dest=name, action="store_const", const=False, help=help_off)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="metatlas2 untargeted pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run pipeline Steps 1-7.")
    _add_config_args(run)
    _add_logging_args(run)
    run.add_argument("--direct_input", type=_csv, default=None,
                     help="Comma-separated project names; bypasses status filters.")
    run.add_argument("--skip_steps", type=_csv, default=None, help=f"Comma-separated subset of {list(STEP_NAMES)}.")
    run.add_argument("--background_designator", type=_csv, default=None, help="Control sample group(s), e.g. ExCtrl.")
    run.add_argument("--sbatch_queue_file", default=None,
                     help="Append MZmine sbatch jobs here for host submission (set by metatlas2.sh).")
    run.add_argument("--force_reprocess", action="store_true",
                     help="With --direct_input: recreate project directories regardless of LIMS status.")
    # Step 1
    run.add_argument("--validate_names", action="store_const", const=True, default=None,
                     help="Validate mzML file and project names.")
    run.add_argument("--custom_mzmine_batch_params", default=None, help="Full path to a custom MZmine batch XML.")
    run.add_argument("--skip_blank_filter", action="store_const", const=True, default=None,
                     help="Keep files with 'Blank' in the name.")
    # Step 2
    run.add_argument("--background_ratio", type=float, default=None, help="Sample/control ratio for filtering.")
    run.add_argument("--zero_value", type=float, default=None, help="Fraction of lowest nonzero value used for zeros.")
    run.add_argument("--polar_solvent_front", type=float, default=None, help="HILIC solvent front (min).")
    run.add_argument("--nonpolar_solvent_front", type=float, default=None, help="C18 solvent front (min).")
    run.add_argument("--polar_solvent_end", type=float, default=None, help="HILIC solvent end (min).")
    run.add_argument("--nonpolar_solvent_end", type=float, default=None, help="C18 solvent end (min).")
    # Step 3
    run.add_argument("--overwrite_mzmine", action="store_true", help="Resubmit MZmine even if outputs exist.")
    # Step 5/6
    run.add_argument("--overwrite_fbmn", action="store_true", help="Resubmit/redownload FBMN even if outputs exist.")
    run.add_argument("--skip_mirror_raw_data", action="store_true", help="Do not mirror mzML files to GNPS2.")
    run.add_argument("--skip_mirror_mzmine_results", action="store_true", help="Do not mirror MZmine results to GNPS2.")
    run.add_argument("--skip_everything_bagel", action="store_true", help="Do not submit Everything Bagel jobs.")
    # Step 7
    run.add_argument("--min_features", type=int, default=None, help="Minimum total features required to zip.")
    run.add_argument("--gnps2_doc_name", default=None, help="GNPS2 guide file name added to zips.")
    run.add_argument("--overwrite_zip", action="store_true", help="Recreate existing zip archives.")
    run.add_argument("--overwrite_drive", action="store_true", help="Replace older zips on Google Drive.")
    _bool_pair(run, "gdrive_upload", "Upload zips to Google Drive.", "Do not upload zips to Google Drive.")
    _bool_pair(run, "add_gnps2_documentation", "Add GNPS2 guide to zips.", "Do not add GNPS2 guide to zips.")
    _bool_pair(run, "abridged_filenames", "Use abridged file names in zips.", "Keep full file names in zips.")

    status = sub.add_parser("status", help="Print project status table (tab-separated).")
    _add_config_args(status)
    status.add_argument("--direct_input", type=_csv, default=None, help="Comma-separated project names.")
    status.add_argument("--print_recent", type=int, default=50, help="Print the most recent N projects.")
    status.add_argument("--update_lims_table", action="store_true",
                        help="Run the MZmine and FBMN status steps (2 and 4) before reporting.")
    status.add_argument("--background_designator", type=_csv, default=None, help="Control sample group(s).")

    mirror = sub.add_parser("mirror-raw", help="Mirror project mzML files to GNPS2.")
    _add_config_args(mirror)
    _add_logging_args(mirror)
    mirror.add_argument("--direct_input", type=_csv, required=True, help="Comma-separated project names.")
    mirror.add_argument("--mode", choices=["hard", "soft"], required=True,
                        help="hard: overwrite everything; soft: upload new/changed files only.")

    wc = sub.add_parser("write-config", help="Write the default config YAML.")
    wc.add_argument("output", help="Destination YAML path.")
    return parser


def config_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Map CLI flags onto config sections (``None`` means "not given")."""
    g = lambda name: getattr(args, name, None)  # noqa: E731
    overrides: dict[str, dict[str, Any]] = {
        "paths": {
            "raw_data_dir": g("raw_data_dir"),
            "raw_data_subdir": g("raw_data_subdir"),
            "output_dir": g("output_dir"),
            "download_dir": g("download_dir"),
            "log_file": g("log_file"),
        },
        "lims": {"tasks_table": g("lims_tasks_table")},
        "project_selection": {
            "background_designator": g("background_designator"),
            "validate_names": g("validate_names"),
            "skip_blank_filter": g("skip_blank_filter"),
        },
        "mzmine": {"custom_batch_params": g("custom_mzmine_batch_params")},
        "filtering": {k: g(k) for k in (
            "background_ratio", "zero_value", "polar_solvent_front", "nonpolar_solvent_front",
            "polar_solvent_end", "nonpolar_solvent_end")},
        "gnps2": {
            "mirror_raw_data": False if g("skip_mirror_raw_data") else None,
            "mirror_mzmine_results": False if g("skip_mirror_mzmine_results") else None,
            "submit_everything_bagel": False if g("skip_everything_bagel") else None,
        },
        "gdrive": {
            "upload": g("gdrive_upload"),
            "add_documentation": g("add_gnps2_documentation"),
            "abridged_filenames": g("abridged_filenames"),
            "doc_name": g("gnps2_doc_name"),
            "min_features": g("min_features"),
        },
    }
    return overrides


def check_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Exit with a message on incompatible flags or missing directories."""
    if args.command == "run":
        if args.overwrite_drive and config["gdrive"]["upload"] is False:
            sys.exit("Incompatible flags. Cannot overwrite google drive if not uploading to google drive.")
        if args.force_reprocess and args.direct_input is None:
            sys.exit("--force_reprocess requires --direct_input.")
        for key in ("download_dir", "output_dir", "raw_data_dir"):
            if not os.path.exists(config["paths"][key]):
                sys.exit(f"{key} does not exist: {config['paths'][key]}. Please check flag/config.")
    if args.command == "mirror-raw" and not os.path.exists(config["paths"]["raw_data_dir"]):
        sys.exit(f"raw_data_dir does not exist: {config['paths']['raw_data_dir']}")


def _setup_logging(args: argparse.Namespace, config: dict[str, Any]) -> None:
    import metatlas2.shared_tools.logging_config as lcf

    level = getattr(logging, str(getattr(args, "log_level", "INFO")).upper(), logging.INFO)
    to_stdout = bool(getattr(args, "log_to_stdout", False))
    lcf.setup_logging(log_level=level, log_file=None if to_stdout else config["paths"]["log_file"],
                      log_to_stdout=to_stdout)


def _make_context(args: argparse.Namespace, config: dict[str, Any]):
    from metatlas2.untargeted.context import RunContext
    from metatlas2.untargeted.lims import LimsClient

    return RunContext(
        config=config,
        lims=LimsClient.from_config(config["lims"], dry_run=args.dry_run),
        dry_run=args.dry_run,
        direct_input=getattr(args, "direct_input", None),
        overwrite_mzmine=getattr(args, "overwrite_mzmine", False),
        overwrite_fbmn=getattr(args, "overwrite_fbmn", False),
        overwrite_zip=getattr(args, "overwrite_zip", False),
        overwrite_drive=getattr(args, "overwrite_drive", False),
        force_reprocess=getattr(args, "force_reprocess", False),
        sbatch_queue_file=getattr(args, "sbatch_queue_file", None),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-config":
        print(write_default_config(args.output))
        return 0

    config = load_untargeted_config(args.config, config_overrides(args))
    check_args(args, config)

    if args.command == "status":
        import metatlas2.shared_tools.logging_config as lcf
        from metatlas2.untargeted.status_report import format_status_table, get_untargeted_status

        # Keep stdout clean: the email scripts parse it. Warnings go to stderr.
        lcf.setup_logging(log_level=logging.WARNING, log_file=None, log_to_stdout=False)
        ctx = _make_context(args, config)
        if args.update_lims_table:
            from metatlas2.untargeted.fbmn import update_fbmn_status
            from metatlas2.untargeted.mzmine import update_mzmine_status

            update_mzmine_status(ctx)
            update_fbmn_status(ctx)
        table = get_untargeted_status(ctx)
        if table is None:
            return 1
        print_recent = None if args.direct_input is not None else args.print_recent
        sys.stdout.write(format_status_table(table, print_recent))
        return 0

    _setup_logging(args, config)
    from metatlas2.untargeted.pipeline import run_untargeted_pipeline

    ctx = _make_context(args, config)
    logger = logging.getLogger("metatlas2.untargeted.pipeline")
    if args.command == "mirror-raw":
        from metatlas2.untargeted.project_setup import find_raw_data_directory

        gnps2 = ctx.get_gnps2()
        failed = False
        for project in args.direct_input:
            logger.info(f"{args.mode.capitalize()} mirroring raw data to GNPS2 for project {project}...")
            local = find_raw_data_directory(project, config["paths"]["raw_data_dir"],
                                            config["paths"].get("raw_data_subdir"),
                                            config["paths"].get("fallback_raw_data_subdirs"))
            if local is None or not gnps2.mirror_raw_data(project, local, mode=args.mode):
                logger.error(f"Failed to mirror project {project}")
                failed = True
        return 1 if failed else 0

    logger.info(f"Arguments used: {vars(args)}")
    ok = run_untargeted_pipeline(ctx, skip_steps=args.skip_steps)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
