"""CLI for the raw -> mzML -> h5 converter. Runs on the host with plain Python 3.11.

Usage::

    python -m metatlas2.ingest.run_file_conversion convert {jgi|egsb} [options]
    python -m metatlas2.ingest.run_file_conversion convert-file PATH.raw [options]

``scripts/metatlas2.sh convert-raw ...`` runs this with ``/usr/bin/python3.11``
(standard library only, so it works for any account without a virtualenv) and
passes the metatlas2 image used for the mzML -> h5 step.

Only standard-library modules may be imported at module level here.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from metatlas2.ingest.conversion import TRFP_IMAGE, ConversionSettings, Converter, run_conversion

DEFAULT_LOG_DIR = "/global/cfs/cdirs/m2650/file_converter_logs"
DEFAULT_RAW_DATA_DIR = "/global/cfs/cdirs/metatlas/raw_data"
DEFAULT_IMAGE = "docker:ghcr.io/biorack/metatlas2:latest"


def _default_workers() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 2
    return max(1, min(cpus, 8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Thermo .raw files to .mzML and metatlas .h5")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--raw_data_dir", default=DEFAULT_RAW_DATA_DIR)
        p.add_argument("--max_workers", type=int, default=_default_workers(),
                       help="Files converted at once (default: CPUs available, max 8).")
        p.add_argument("--h5_image", default=os.environ.get("METATLAS2_IMAGE", DEFAULT_IMAGE),
                       help="Shifter image for the mzML->h5 step (set by metatlas2.sh).")
        p.add_argument("--h5_pythonpath", default=os.environ.get("METATLAS2_CONTAINER_PYTHONPATH", "/app"),
                       help="PYTHONPATH inside the container (metatlas2.sh --dev puts the repo first).")
        p.add_argument("--h5_inprocess", action="store_true",
                       help="Run the mzML->h5 step in this Python (needs pymzml/tables) instead of shifter.")
        p.add_argument("--trfp_image", default=TRFP_IMAGE, help="ThermoRawFileParser shifter image.")
        p.add_argument("--no_validate_names", action="store_true", help="Skip the minimal file-name checks.")
        p.add_argument("--step_timeout_minutes", type=float, default=None,
                       help="Fail a step that runs longer than this (default: no limit).")
        p.add_argument("--log_file", default=None, help="Run log (default: <log_dir>/<subdir>.log).")
        p.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
        p.add_argument("--log_to_stdout", action="store_true", help="Log to stdout instead of the log file.")
        p.add_argument("--log_level", default="INFO")

    conv = sub.add_parser("convert", help="Convert all pending .raw files under raw_data/<subdir>.")
    conv.add_argument("subdir", help="raw_data subdirectory, e.g. jgi or egsb")
    conv.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                      help="List the files that would be converted; change nothing.")
    common(conv)

    one = sub.add_parser("convert-file", help="Convert a single .raw file (ignores the scan).")
    one.add_argument("raw_file")
    common(one)
    return parser


def settings_from_args(args: argparse.Namespace) -> ConversionSettings:
    h5_command = None
    if not args.h5_inprocess:
        h5_command = [
            "shifter", f"--image={args.h5_image}", "--module=none", "--clearenv",
            "--env=HDF5_USE_FILE_LOCKING=FALSE", f"--env=PYTHONPATH={args.h5_pythonpath}",
            "/app/.venv/bin/python", "-m", "metatlas2.ingest.h5_writer",
        ]
    return ConversionSettings(
        raw_data_dir=args.raw_data_dir,
        trfp_image=args.trfp_image,
        h5_command=h5_command,
        max_workers=args.max_workers,
        validate_names=not args.no_validate_names,
        step_timeout_seconds=args.step_timeout_minutes * 60 if args.step_timeout_minutes else None,
        dry_run=getattr(args, "dry_run", False),
    )


def _setup_logging(args: argparse.Namespace, log_file: str | None) -> None:
    import metatlas2.shared_tools.logging_config as lcf

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    lcf.setup_logging(log_level=level, log_file=None if args.log_to_stdout else log_file,
                      log_to_stdout=args.log_to_stdout)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    if args.command == "convert":
        log_file = args.log_file or os.path.join(args.log_dir, f"{args.subdir}.log")
        _setup_logging(args, log_file)
        lock = os.path.join(args.log_dir, f".{args.subdir}.convert.lock")
        return run_conversion(args.subdir, settings, lock_path=lock)
    _setup_logging(args, args.log_file)
    log = logging.getLogger("metatlas2.ingest.conversion")
    res = Converter(settings).convert_file(args.raw_file)
    if res.ok:
        log.info(f"{res.stage}: {res.raw_file} {res.seconds} {res.h5_summary.get('rows', '')}")
        return 0
    log.error(f"FAILED at {res.stage}: {res.raw_file}: {res.error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
