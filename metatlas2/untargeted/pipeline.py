"""Orchestrates the seven untargeted pipeline steps.

1. Sync            find new projects in LIMS, write MZmine inputs, insert LIMS rows
2. MZmine_Status   mark finished MZmine jobs complete (+ counts, background filtering)
3. MZmine_Submit   queue MZmine sbatch jobs (host wrapper submits them)
4. FBMN_Status     poll GNPS2 for FBMN task status
5. FBMN_Submit     mirror inputs to GNPS2, submit FBMN (+ Everything Bagel)
6. FBMN_Download   download FBMN graphml + library hits
7. Zip_and_Upload  zip outputs and upload to Google Drive

Every run is bracketed by "Successfully started/completed" log lines that the
warnings/errors email script uses to find the last N runs.
"""

from __future__ import annotations

from typing import Callable

import metatlas2.logging_config as lcf
from metatlas2.untargeted.config import STEP_NAMES
from metatlas2.untargeted.context import RunContext
from metatlas2.untargeted.fbmn import download_fbmn_results, submit_fbmn_jobs, update_fbmn_status
from metatlas2.untargeted.mzmine import submit_mzmine_jobs, update_mzmine_status
from metatlas2.untargeted.packaging import zip_and_upload_untargeted_results
from metatlas2.untargeted.project_setup import update_new_untargeted_tasks
from metatlas2.untargeted.status import end_message, start_message

logger = lcf.get_logger("untargeted.pipeline")

SCRIPT_NAME = "run_untargeted_pipeline.py"

_STEP_DESCRIPTIONS = {
    "Sync": "Step 1/7: Syncing LIMS and NERSC to identify new projects with raw data that are not yet in the "
            "untargeted task list...",
    "MZmine_Status": "Step 2/7: Checking and updating status of MZmine jobs in LIMS...",
    "MZmine_Submit": "Step 3/7: Submitting new MZmine jobs...",
    "FBMN_Status": "Step 4/7: Checking and updating status of FBMN jobs in LIMS...",
    "FBMN_Submit": "Step 5/7: Submitting new FBMN jobs to GNPS2...",
    "FBMN_Download": "Step 6/7: Checking for completed FBMN jobs and downloading results...",
    "Zip_and_Upload": "Step 7/7: Zipping up and (optionally) uploading output folders to gdrive...",
}


def parse_skip_steps(skip_steps: list[str] | str | None) -> set[str]:
    """Validate ``--skip_steps`` names (see :data:`STEP_NAMES`)."""
    if not skip_steps:
        return set()
    if isinstance(skip_steps, str):
        skip_steps = [s.strip() for s in skip_steps.split(",") if s.strip()]
    unknown = sorted(set(skip_steps) - set(STEP_NAMES))
    if unknown:
        raise ValueError(f"Unknown step name(s) in skip_steps: {unknown}. Valid: {list(STEP_NAMES)}")
    return set(skip_steps)


def run_untargeted_pipeline(ctx: RunContext, skip_steps: list[str] | str | None = None) -> bool:
    """Run Steps 1-7 in order.

    A step that raises is logged with its traceback and the remaining steps
    still run (each step re-reads LIMS, so later steps don't depend on
    in-memory results except Step 3 using Step 1's new projects).

    Returns:
        True if every executed step completed without raising.
    """
    skipped = parse_skip_steps(skip_steps)
    logger.info("\n")
    logger.info(start_message(SCRIPT_NAME))
    if ctx.dry_run:
        logger.info("DRY RUN: no LIMS writes, job submissions, GNPS2/Drive uploads, or file changes will be made.")

    state: dict[str, object] = {"new_projects": None}

    def step1() -> None:
        state["new_projects"] = update_new_untargeted_tasks(ctx)

    steps: list[tuple[str, Callable[[], None]]] = [
        ("Sync", step1),
        ("MZmine_Status", lambda: update_mzmine_status(ctx)),
        ("MZmine_Submit", lambda: submit_mzmine_jobs(ctx, state["new_projects"])),  # type: ignore[arg-type]
        ("FBMN_Status", lambda: update_fbmn_status(ctx)),
        ("FBMN_Submit", lambda: submit_fbmn_jobs(ctx)),
        ("FBMN_Download", lambda: download_fbmn_results(ctx)),
        ("Zip_and_Upload", lambda: zip_and_upload_untargeted_results(ctx)),
    ]
    ok = True
    for name, func in steps:
        if name in skipped:
            logger.info(f"Skipping {_STEP_DESCRIPTIONS[name]}")
            continue
        try:
            func()
        except Exception:
            ok = False
            logger.exception(f"ERROR: step {name} failed; continuing with the next step.")
    logger.info(end_message(SCRIPT_NAME))
    return ok
