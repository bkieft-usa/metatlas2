import argparse
import logging
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Dict, Any

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
#SBATCH --output={analysis_output_dir}/pre_curation_%j.log
#SBATCH --error={analysis_output_dir}/pre_curation_%j.err

shifter --module=none \\
    --image=docker:ghcr.io/bkieft-usa/metatlas2:{image_tag} \\
    --env=METATLAS2_IMAGE_TAG={image_tag} \\
    --env=METATLAS_DATA_DIR={metatlas_data_dir} \\
    --env=HOME={home} \\
    --env=PYTHONPATH=/app \\
    /app/.venv/bin/python -m metatlas2.run_targeted_analysis run \\
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
        p.add_argument("--skip-setup", action="store_true", default=False)
        p.add_argument("--skip-rt-align", action="store_true", default=False)
        p.add_argument("--skip-auto-id", action="store_true", default=False)
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
    submit_parser.add_argument("--mem", default="64G")
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
    if args.skip_setup: extra_flags.append("--skip-setup")
    if args.skip_rt_align: extra_flags.append("--skip-rt-align")
    if args.skip_auto_id: extra_flags.append("--skip-auto-id")
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

def get_project_db_path(project_name: str) -> str:
    """Derive the project database path from ``project_name`` alone.

    Scans ``{METATLAS_DATA_DIR}/projects/targeted_outputs/`` for a file named
    ``{project_name}.duckdb`` under any ``owner/user`` subdirectory.  This
    allows async stages (GUI, summary) to locate the project DB without needing
    the config YAML on disk.

    Args:
        project_name: The full project name string.

    Returns:
        Absolute path to the ``.duckdb`` file.

    Raises:
        FileNotFoundError: If no matching database file is found.
        ValueError: If ``METATLAS_DATA_DIR`` is not set.
    """
    data_dir = os.environ.get("METATLAS_DATA_DIR")
    if data_dir is None:
        raise ValueError(
            "METATLAS_DATA_DIR environment variable is not set. "
            "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it."
        )
    base = Path(data_dir) / "projects" / "targeted_outputs"
    target = f"{project_name}.duckdb"
    matches = list(base.rglob(target))
    if not matches:
        raise FileNotFoundError(
            f"No project database found for '{project_name}' under {base}. "
            "Run project setup first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple project databases found for '{project_name}': {matches}. "
            "Specify the path explicitly."
        )
    return str(matches[0])


def set_up_paths(
    config: "Metatlas2Config",
    project_name: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None,
) -> Dict[str, str]:
    """Build all workflow paths and create all output directories for a run."""

    data_dir = os.environ.get("METATLAS_DATA_DIR")
    if data_dir is None:
        raise EnvironmentError(
            "METATLAS_DATA_DIR environment variable is not set. "
            "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it."
        )
    lcmsruns_path = f"{data_dir}/raw_data/"
    main_db_path = f"{data_dir}/databases/main_db/metatlas.duckdb"
    pubchem_cache_path = f"{data_dir}/databases/pubchem_cache/pubchem_global_cache.json"
    project_output_path = f"{data_dir}/projects/targeted_outputs/"
    modelseed_table_path = f"{data_dir}/databases/modelseed_db/modelseed.tsv"

    if project_name is None: # This is for converting files and adding compounds and atlases to main db
        return {
            "lcmsruns_directory": str(lcmsruns_path),
            "main_db_path": str(main_db_path),
            "pubchem_cache_path": str(pubchem_cache_path),
            "modelseed_table_path": str(modelseed_table_path),
        }

    owner = config.owner
    user = os.environ.get("USER", "other")
    if not owner:
        raise ValueError("Owner not specified in config under WORKFLOWS.PATHS.owner")
    if user is None:
        raise ValueError("USER environment variable is not set")
    project_output_dir = Path(project_output_path) / owner / user / project_name
    try:
        project_short = str(project_name.split("_")[4]) + "_RTA" + str(rt_alignment_number) + "_TGA" + str(analysis_number)
    except Exception as e:
        raise ValueError(f"Could not parse short project name from full project name - is it malformed? Error: {e}")
    rta_dir = project_output_dir / f"RTA{rt_alignment_number}"
    analysis_dir = rta_dir / f"TGA{analysis_number}"

    # Resolve msms_refs_path relative to METATLAS_DATA_DIR if not absolute
    msms_refs_path_raw = config.msms_refs_path
    if msms_refs_path_raw and not Path(msms_refs_path_raw).is_absolute():
        msms_refs_path_resolved = str(Path(data_dir) / msms_refs_path_raw)
    else:
        msms_refs_path_resolved = str(msms_refs_path_raw) if msms_refs_path_raw else None

    paths = {
        "lcmsruns_directory": str(Path(lcmsruns_path) / owner / project_name),
        "project_directory": str(project_output_dir),
        "log_path": str(project_output_dir / f"{project_short}.log"),
        "project_db_path": str(project_output_dir / f"{project_name}.duckdb"),
        "main_db_path": str(main_db_path),
        "msms_refs_path": msms_refs_path_resolved,
        "pubchem_cache_path": str(pubchem_cache_path),
        "modelseed_table_path": str(modelseed_table_path),
        "rt_alignment_output_dir": str(rta_dir),
        "rt_alignment_results_dir": str(rta_dir / "rt_alignment_results"),
        "aligned_atlases_store_file": str(rta_dir / "rt_aligned_atlases.csv"),
        "analysis_output_dir": str(analysis_dir),
        "auto_ided_atlases_store_file": str(analysis_dir / "auto_ided_atlases.csv"),
        "curated_atlases_store_file": str(analysis_dir / "curated_atlases.csv"),
    }

    if not Path(paths["lcmsruns_directory"]).exists():
        raise ValueError(f"Raw data directory not found: {paths['lcmsruns_directory']}")
    if not Path(paths["main_db_path"]).exists():
        raise ValueError(f"Main database not found: {paths['main_db_path']}")

    project_output_dir.mkdir(parents=True, exist_ok=True)
    rta_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return paths

def main():

    args = parse_args()

    import metatlas2.load_tools as ldt
    import metatlas2.workflows as wfs
    import metatlas2.logging_config as lcf

    config = ldt.load_metatlas2_config(args.config)

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
                slurm_stdout = Path(paths["project_directory"]) / f"pre_curation_{job_id}.log"
                slurm_stderr = Path(paths["project_directory"]) / f"pre_curation_{job_id}.err"
                logger.info(f"Expected SLURM stdout: {slurm_stdout}")
                logger.info(f"Expected SLURM stderr: {slurm_stderr}")
            return

        if not args.skip_setup:
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

        if not args.skip_rt_align:
            if not args.log_to_stdout and args.command == "run": print("=======-- Running RT alignment...")
            logger.info("------------ Running RT Alignment ...")
            wfs.run_rt_alignment(
                project_name=args.project,
                rt_alignment_number=args.rt_align_num,
                analysis_number=args.analysis_num,
            )

        if not args.skip_auto_id:
            if not args.log_to_stdout and args.command == "run": print("========- Running Auto Identification...")
            logger.info("------------ Running Auto Identification")
            wfs.run_auto_identification(
                project_name=args.project,
                rt_alignment_number=args.rt_align_num,
                analysis_number=args.analysis_num,
                analysis_subset=args.analysis_subset,
                image_tag=os.environ.get("METATLAS2_IMAGE_TAG", "latest"),
            )

        ### NOT WORKING YET
        if args.skip_curation:
            logger.info("------------ Skipping curation step is not implemented yet.")
        #     if not args.log_to_stdout and args.command == "run": print("========- Skipping curation step and running summary...")
        #     logger.info("Skipping curation GUI and running summary.")

        #     wfs.run_analysis_summary(
        #         config_path=os.path.abspath(args.config),
        #         project_name=args.project,
        #         rt_alignment_number=args.rt_align_num,
        #         analysis_number=args.analysis_num,
        #         post_autoid_atlas=paths["auto_ided_atlases_store_file"],
        #     )

        if not args.log_to_stdout and args.command == "run": print("========= Pre-curation workflow complete!")
        logger.info("Pre-curation workflow complete. Open the generated notebooks to curate.")

if __name__ == "__main__":
    main()