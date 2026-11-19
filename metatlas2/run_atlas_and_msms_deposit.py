"""Command-line entrypoint for standards atlas deposition."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import metatlas2.logging_config as lcf
from metatlas2.standards_annotation_review import TIMESTAMP_PATTERN, review_directory


def run_atlas_and_msms_deposit(
    config_path: str, annotation_timestamp: str, alignment_timestamp: str,
    dry_run: bool = False, force_add: bool = False,
    skip_atlas_upload: bool = False, skip_msms_refs_deposit: bool = False,
) -> tuple[Path, Path]:
    """Prepare previews and optionally deposit the atlas and references."""
    from metatlas2.deposit_ref_standards_atlas_and_msms_refs import (
        deposit_ref_standards_atlas_and_msms_refs,
    )
    from metatlas2.load_tools import load_standards_config
    from metatlas2.gdrive_upload import upload_standards_outputs
    from metatlas2.standards_annotation_db import get_project_db_path, get_project_output_root

    config = load_standards_config(str(Path(config_path).resolve()))
    output_root = get_project_output_root(config)
    db_path = get_project_db_path(config)
    if not output_root.is_dir() or not Path(db_path).is_file():
        raise FileNotFoundError(f"Standards annotation project database not found: {db_path}")
    if not TIMESTAMP_PATTERN.fullmatch(alignment_timestamp):
        raise ValueError("Invalid alignment timestamp; use the timestamp printed by reverse-rt-align.")
    run_name = config["project"]["standards_run_name"]
    alignment_dir = review_directory(output_root, annotation_timestamp) / f"rt_align_{alignment_timestamp}"
    summary_path = alignment_dir / f"StandardsAnnotation_Summary_Aligned_{run_name}.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Aligned summary not found: {summary_path}")

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=str(log_dir / "atlas_and_msms_deposit.log"),
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        return deposit_ref_standards_atlas_and_msms_refs(
            summary_path, db_path, run_name, dry_run=dry_run, force_add=force_add,
            skip_atlas_upload=skip_atlas_upload, skip_msms_refs_deposit=skip_msms_refs_deposit,
            chem_source=config["project"].get("chem_source", ""),
            on_previews_created=lambda atlas, msms: upload_standards_outputs(
                config, alignment_dir.parent, [atlas, msms],
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare deposit previews and deposit the reviewed standards.")
    parser.add_argument("--config_path", required=True, help="Standards-annotation YAML config file.")
    parser.add_argument("--annotation_timestamp", dest="annotation_timestamp",
                        required=True, help="Annotation summary timestamp.")
    parser.add_argument("--alignment_timestamp", required=True, help="Reverse alignment timestamp.")
    parser.add_argument("--dry_run", action="store_true", help="Write previews without depositing.")
    parser.add_argument("--force_add", action="store_true", help="Add atlas rows and MS/MS references even when their identities already exist.")
    parser.add_argument("--skip_atlas_upload", action="store_true", help="Skip uploading atlas rows to Google Sheets.")
    parser.add_argument("--skip_msms_refs_deposit", action="store_true", help="Skip importing MS/MS references into the main database.")
    args = parser.parse_args()
    run_atlas_and_msms_deposit(
        args.config_path, args.annotation_timestamp, args.alignment_timestamp,
        dry_run=args.dry_run, force_add=args.force_add,
        skip_atlas_upload=args.skip_atlas_upload, skip_msms_refs_deposit=args.skip_msms_refs_deposit,
    )
