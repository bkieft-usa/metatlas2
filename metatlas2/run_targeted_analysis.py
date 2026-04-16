import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={project}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={analysis_output_dir}/pre_curation_%j.log
#SBATCH --error={analysis_output_dir}/pre_curation_%j.err

export METATLAS2_IMAGE_TAG="{image_tag}"
shifter --image=docker:ghcr.io/bkieft-usa/metatlas2:{image_tag} \\
    python -m metatlas2.run_targeted_analysis run \\
    --config "{config}" \\
    --project "{project}" \\
    --rt-align-num {rt_align_num} \\
    --analysis-num {analysis_num} \\
    {extra_flags}
"""

def get_chromatographies_from_config(config: dict) -> list:
    """Return all chromatography keys present under RT_ALIGNMENT."""
    return list(config["WORKFLOWS"].get("RT_ALIGNMENT", {}).keys())

def parse_args():
    parser = argparse.ArgumentParser(description="Run metatlas2 pre-curation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_args(p):
        p.add_argument("--config", required=True, help="Path to analysis.yaml")
        p.add_argument("--project", required=True, help="Project name")
        p.add_argument("--rt-align-num", type=int, default=0)
        p.add_argument("--analysis-num", type=int, default=0)
        p.add_argument("--analysis-subset",  type=lambda s: s.split(","), default=None, help="Comma-separated polarity-analysis_type list to filter config for RT alignment and Auto ID (e.g. 'POS-ISTD,POS-EMA')")
        p.add_argument("--overwrite", action="store_true", default=False)
        p.add_argument("--skip-setup", action="store_true", default=False)
        p.add_argument("--skip-rt-align", action="store_true", default=False)
        p.add_argument("--skip-auto-id", action="store_true", default=False)
        p.add_argument("--log-to-stdout", action="store_true", default=False, help="Write log output to stdout instead of a log file in the project directory")

    run_parser = subparsers.add_parser("run", help="Execute the pre-curation workflow directly")
    add_shared_args(run_parser)

    submit_parser = subparsers.add_parser("submit", help="Write and immediately submit a Slurm job")
    add_shared_args(submit_parser)
    submit_parser.add_argument("--qos", default="regular")
    submit_parser.add_argument("--cpus", type=int, default=8)
    submit_parser.add_argument("--mem", default="64G")
    submit_parser.add_argument("--time", default="03:00:00")
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
    if args.log_to_stdout: extra_flags.append("--log-to-stdout")
    if args.analysis_subset: extra_flags.append("--analysis-subset " + ",".join(args.analysis_subset))
    extra_flags_str = " \\\n ".join(extra_flags)

    populated = SLURM_TEMPLATE.format(
        qos = args.qos,
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

def set_up_paths(
    config: Dict[str, Any],
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
    lcmsruns_path = f"{data_dir}/lcmsruns/"
    main_db_path = f"{data_dir}/databases/main_db/metatlas.duckdb"
    pubchem_cache_path = f"{data_dir}/databases/pubchem_cache/pubchem_global_cache.parquet"

    if project_name is None: # This is for converting files and adding compounds and atlases to main db
        return {
            "lcmsruns_directory": str(lcmsruns_path),
            "main_db_path": str(main_db_path),
            "pubchem_cache_path": str(pubchem_cache_path)
        }

    owner = config.get('WORKFLOWS').get('PATHS').get('owner', None).lower()
    project_output_dir = Path.home() / f"{owner}_metabolomics_data" / project_name
    project_short = str(project_name.split("_")[4]) + "_" + str(rt_alignment_number) + "_" + str(analysis_number)
    rta_dir = project_output_dir / f"RTA{rt_alignment_number}"
    analysis_dir = rta_dir / f"TGA{analysis_number}"

    paths = {
        "lcmsruns_directory": str(Path(lcmsruns_path) / owner / project_name),
        "project_directory": str(project_output_dir),
        "log_path": str(project_output_dir / f"{project_short}.log"),
        "project_db_path": str(project_output_dir / f"{project_name}.duckdb"),
        "main_db_path": str(main_db_path),
        "msms_refs_path": str(config.get('WORKFLOWS').get('PATHS').get('msms_refs_path', None)),
        "pubchem_cache_path": str(pubchem_cache_path),
        "rt_alignment_output_dir": str(rta_dir),
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

    print("Parsing arguments ...", flush=True)
    args = parse_args()

    print("Setting up logging...")
    import metatlas2.logging_config as lcf
    if args.log_to_stdout:
        log_file = None
    else:
        log_file = paths["log_path"]

    lcf.setup_logging(log_level=logging.INFO, log_file=log_file, log_to_stdout=args.log_to_stdout)
    logger = lcf.get_logger("run_targeted_analysis")

    print("Loading libraries ...", flush=True)
    logger.info("Loading libraries")
    import metatlas2.load_tools as ldt
    import metatlas2.workflows as wfs

    print("Loading config ...", flush=True)
    logger.info("Loading config")
    config = ldt.load_metatlas2_config(args.config)

    print("Setting up paths ...", flush=True)
    logger.info("Setting up paths")
    paths = set_up_paths(
        config=config,
        project_name=args.project,
        rt_alignment_number=args.rt_align_num,
        analysis_number=args.analysis_num,
    )

    if args.command == "submit":
        out_path = generate_slurm_script(args, paths)
        print(f"Slurm script written to: {out_path}")
        logger.info(f"Slurm script written to: {out_path}")
        if args.script_only:
            return
        result = subprocess.run(["sbatch", out_path], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            sys.exit(result.returncode)
        return

    if not args.skip_setup:
        print("Running project setup...")
        logger.info("Running Project Setup")
        wfs.run_project_setup(
            project_name=args.project,
            config=config,
            paths=paths,
            overwrite_existing=args.overwrite,
        )

    if not args.skip_rt_align:
        print("Running RT alignment...")
        logger.info("Running RT alignment ...")
        wfs.run_rt_alignment(
            project_name=args.project,
            rt_alignment_number=args.rt_align_num,
            config=config,
            paths=paths,
        )

    if not args.skip_auto_id:
        print("Running Auto Identification...")
        logger.info("Running Auto Identification")
        wfs.run_auto_identification(
            project_name=args.project,
            config=config,
            paths=paths,
            rt_alignment_number=args.rt_align_num,
            analysis_number=args.analysis_num,
            analysis_subset=args.analysis_subset,
            config_path=os.path.abspath(args.config),
            image_tag=os.environ.get("METATLAS2_IMAGE_TAG", "latest"),
        )

    print("Pre-curation workflow complete. Open the generated notebooks to curate.")
    logger.info("Pre-curation workflow complete. Open the generated notebooks to curate.")


if __name__ == "__main__":
    main()