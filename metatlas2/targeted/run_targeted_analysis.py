from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import os
import re
from pathlib import Path

from metatlas2.shared_tools.paths import get_project_db_path, set_up_paths  # noqa: F401  (re-exported)

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={project}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --constraint={constraint}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={analysis_output_dir}/auto_id_slurm_%j.log
#SBATCH --error={analysis_output_dir}/auto_id_slurm_%j.err

shifter --module=none \\
    --image=docker:ghcr.io/biorack/metatlas2:{image_tag} \\
    --env=METATLAS2_IMAGE_TAG={image_tag} \\
    --env=METATLAS_DATA_DIR={metatlas_data_dir} \\
    --env=HOME={home} \\
    --env=PYTHONPATH=/app \\
    /app/.venv/bin/python -m metatlas2.targeted.run_targeted_analysis run \\
    --config "{config}" \\
    --project "{project}" \\
    --rt-align-num {rt_align_num} \\
    --analysis-num {analysis_num} \\
    {extra_flags}
"""

def get_chromatographies_from_config(config: "Metatlas2Config") -> list:
    """Return all chromatography keys present under RT_ALIGNMENT."""
    return list(config.rt_alignment_config.keys())

def parse_args():
    parser = argparse.ArgumentParser(description="Run metatlas2 pre-curation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_args(p):
        p.add_argument("--config", required=True, help="Path to analysis.yaml")
        p.add_argument("--project", required=True, help="Project name")
        p.add_argument("--rt-align-num", type=int, default=0)
        p.add_argument("--analysis-num", type=int, default=0)
        p.add_argument("--analysis-subset",  type=lambda s: s.split(","), default=None, help="Comma-separated POL-TYPE-NAME list to filter which named analyses to run (e.g. 'POS-ISTD-default,POS-EMA-default,NEG-EMA-no-standards')")
        p.add_argument("--overwrite", action="store_true", default=False)
        p.add_argument("--skip-rt-align", action="store_true", default=False)
        p.add_argument("--skip-curation", action="store_true", default=False)
        p.add_argument("--log-to-stdout", action="store_true", default=False, help="Write log output to stdout in addition to the project log file.")

    run_parser = subparsers.add_parser("run", help="Execute the pre-curation workflow directly")
    add_shared_args(run_parser)

    submit_parser = subparsers.add_parser("submit", help="Write and immediately submit a Slurm job")
    add_shared_args(submit_parser)
    submit_parser.add_argument("--account", "--account", default="m2650", help="NERSC account/project to charge (e.g., m1234)")
    submit_parser.add_argument("--qos", default="regular")
    submit_parser.add_argument("--constraint", default="cpu", help="Node constraint (default: cpu for Perlmutter)")
    submit_parser.add_argument("--cpus", type=int, default=8)
    submit_parser.add_argument("--mem", default="128G")
    submit_parser.add_argument("--time", default="00:30:00")
    submit_parser.add_argument("--output", default=None, help="Override output path for the .sh script")
    submit_parser.add_argument(
        "--image",
        default=os.environ.get("METATLAS2_IMAGE_TAG", "latest"),
        help="Container image tag to embed in the SLURM script (default: METATLAS2_IMAGE_TAG env var or 'latest')",
    )
    submit_parser.add_argument(
        "--script-only",
        action="store_true",
        default=False,
        help="Write the SLURM script but do not call sbatch (used by the container wrapper)",
    )

    return parser.parse_args()

def generate_slurm_script(args, paths) -> str:
    """
    Populate the Slurm template with args and write it to disk.
    Script and logs are written to the analysis output directory.
    Returns the path of the written file.
    """

    extra_flags = []
    if args.overwrite: extra_flags.append("--overwrite")
    if args.skip_rt_align: extra_flags.append("--skip-rt-align")
    if args.skip_curation: extra_flags.append("--skip-curation")
    if args.analysis_subset: extra_flags.append("--analysis-subset " + ",".join(args.analysis_subset))
    extra_flags_str = " \\\n ".join(extra_flags)

    populated = SLURM_TEMPLATE.format(
        account = args.account,
        qos = args.qos,
        constraint = args.constraint,
        cpus = args.cpus,
        mem = args.mem,
        time = args.time,
        analysis_output_dir = paths["project_directory"],
        config = os.path.abspath(args.config),
        project = args.project,
        rt_align_num = args.rt_align_num,
        analysis_num = args.analysis_num,
        extra_flags = extra_flags_str,
        image_tag = args.image,
        metatlas_data_dir = os.environ.get("METATLAS_DATA_DIR", ""),
        home = os.environ.get("HOME", str(Path.home())),
    )

    if args.output:
        out_path = args.output
    else:
        out_path = paths["log_path"].replace(".log", ".sh")

    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        f.write(populated)
    os.chmod(out_path, 0o755)

    return out_path

def main():

    args = parse_args()

    import metatlas2.shared_tools.load_tools as ldt
    import metatlas2.targeted.workflows as wfs
    import metatlas2.shared_tools.logging_config as lcf

    config = ldt.load_metatlas2_config(args.config, analysis_subset=args.analysis_subset)

    paths = set_up_paths(
        config=config,
        project_name=args.project,
        rt_alignment_number=args.rt_align_num,
        analysis_number=args.analysis_num,
    )

    log_file = paths["log_path"]
    workflow_log_to_stdout = args.log_to_stdout if args.command == "run" else False
    with lcf.temporary_logging(log_level=logging.INFO, log_file=log_file, log_to_stdout=workflow_log_to_stdout, reconfigure_existing=True):
        logger = lcf.get_logger("run_targeted_analysis")
        logger.info("System set - starting pre-curation workflow")

        if args.command == "submit":
            out_path = generate_slurm_script(args, paths)
            logger.info(f"Slurm script written to: {out_path}")
            if args.script_only:
                return
            logger.info("Submitting slurm script...")
            result = subprocess.run(["sbatch", out_path], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Error submitting slurm script: {result.stderr.strip()}")
                sys.exit(result.returncode)
            else:
                submission_output = result.stdout.strip()
                logger.info(f"Slurm submission output: {submission_output}")

                job_id_match = re.search(r"Submitted batch job\s+(\d+)", submission_output)
                job_id = job_id_match.group(1) if job_id_match else "%j"
                slurm_stdout = Path(paths["project_directory"]) / f"auto_id_slurm_{job_id}.log"
                slurm_stderr = Path(paths["project_directory"]) / f"auto_id_slurm_{job_id}.err"
                logger.info(f"Expected SLURM stdout: {slurm_stdout}")
                logger.info(f"Expected SLURM stderr: {slurm_stderr}")
            return

        if not args.log_to_stdout and args.command == "run": print("======--- Setting up project specs...")
        logger.info("------------ Running Project Setup")
        wfs.run_project_setup(
            project_name=args.project,
            config=config,
            paths=paths,
            overwrite_existing=args.overwrite,
            rt_alignment_number=args.rt_align_num,
            analysis_number=args.analysis_num,
        )

        if not args.log_to_stdout and args.command == "run": print("=======-- Running RT alignment...")
        logger.info("------------ Running RT Alignment ...")
        wfs.run_rt_alignment(
            project_name=args.project,
            rt_alignment_number=args.rt_align_num,
            analysis_number=args.analysis_num,
            skip_alignment=args.skip_rt_align,
        )

        if not args.log_to_stdout and args.command == "run": print("========- Running Auto Identification...")
        logger.info("------------ Running Auto Identification")
        wfs.run_auto_identification(
            project_name=args.project,
            rt_alignment_number=args.rt_align_num,
            analysis_number=args.analysis_num,
            analysis_subset=args.analysis_subset,
            image_tag=os.environ.get("METATLAS2_IMAGE_TAG", "latest"),
        )

        if args.skip_curation:
            if not args.log_to_stdout and args.command == "run": print("========- Skipping curation GUI and running analysis summary...")
            logger.info("------------ Skipping curation GUI: applying auto-curation defaults and running analysis summary")

            import metatlas2.shared_tools.database_interact as dbi

            auto_ided_df = dbi.get_auto_ided_atlases_for_summary(
                project_db_path=paths["project_db_path"],
                rt_alignment_number=args.rt_align_num,
                analysis_number=args.analysis_num,
                analysis_subset=args.analysis_subset,
            )

            for _, row in auto_ided_df.iterrows():
                chrom = row["chromatography"]
                pol = row["polarity"]
                acat = row["atlas_category"]
                aname = row["atlas_name"]
                atlas_uid = row["atlas_uid"]

                logger.info(
                    f"Running auto-curated analysis summary for "
                    f"{chrom}-{pol}-{acat}-{aname} (atlas_uid={atlas_uid})..."
                )

                run_params = {
                    "project_name": args.project,
                    "rt_alignment_number": args.rt_align_num,
                    "analysis_number": args.analysis_num,
                    "chromatography": chrom,
                    "polarity": pol,
                    "atlas_category": acat,
                    "atlas_name": aname,
                    "input_atlas_uid": atlas_uid,
                }

                try:
                    wfs.run_analysis_summary(
                        run_parameters=run_params,
                        override_parameters=None,
                        overwrite=args.overwrite,
                        auto_curate=True,
                    )
                    logger.info(
                        f"Analysis summary complete for {chrom}-{pol}-{acat}-{aname}."
                    )
                except Exception as e:
                    logger.error(
                        f"Analysis summary failed for {chrom}-{pol}-{acat}-{aname}: {e}",
                        exc_info=True,
                    )

        if not args.log_to_stdout and args.command == "run": print("========= Pre-curation workflow complete!")
        logger.info("Pre-curation workflow complete. Open the generated notebooks to curate.")

if __name__ == "__main__":
    main()