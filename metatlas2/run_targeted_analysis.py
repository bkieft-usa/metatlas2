import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf
import load_tools as ldt
import workflows as wfs

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={project_short}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={analysis_output_dir}/pre_curation_%j.log
#SBATCH --error={analysis_output_dir}/pre_curation_%j.err

source ~/.bashrc
conda activate {conda_env}

python {script_path} run \\
    --config        "{config}" \\
    --project       "{project}" \\
    --rt-align-num  {rt_align_num} \\
    --analysis-num  {analysis_num} \\
    {extra_flags}
"""

def _get_project_dir(project_name: str) -> str:
    """
    Derive the project directory from config.
    Creates the directory if it does not exist.
    """
    _BASE_DATA_DIR = Path("/pscratch/sd/b/bkieft/metatlas_lite_data")
    _PROJECTS_DIR  = _BASE_DATA_DIR / "projects"
    _PROJECT_DIR = _PROJECTS_DIR / project_name
    os.makedirs(_PROJECT_DIR, exist_ok=True)
    return str(_PROJECT_DIR)

def _get_analysis_output_dir(config_path: str, project_name: str, rt_align_num: int, analysis_num: int) -> str:
    """
    Derive the analysis output directory from config, mirroring _set_up_paths logic.
    Creates the directory if it does not exist.
    """
    config = ldt.load_metatlas2_config(config_path)
    projects_dir = config["ENV"]["PATHS"]["projects_dir"]
    analysis_output_dir = os.path.join(
        projects_dir,
        project_name,
        f"{project_name}_RTA{rt_align_num}_TGA{analysis_num}"
    )
    os.makedirs(analysis_output_dir, exist_ok=True)
    return analysis_output_dir

def get_chromatographies_from_config(config: dict) -> list:
    """Return all chromatography keys present under RT_ALIGNMENT."""
    return list(config["WORKFLOWS"].get("RT_ALIGNMENT", {}).keys())

def parse_args():
    parser = argparse.ArgumentParser(description="Run metatlas2 pre-curation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_args(p):
        p.add_argument("--config",         required=True, help="Path to analysis.yaml")
        p.add_argument("--project",        required=True, help="Project name")
        p.add_argument("--rt-align-num",   type=int, default=0)
        p.add_argument("--analysis-num",   type=int, default=0)
        p.add_argument("--analysis-subset",  type=lambda s: s.split(","), default=None, help="Comma-separated polarity-analysis_type list to filter config for RT alignment and Auto ID (e.g. 'POS-ISTD,POS-EMA')")
        p.add_argument("--overwrite",      action="store_true", default=False)
        p.add_argument("--skip-setup",     action="store_true", default=False)
        p.add_argument("--skip-rt-align",  action="store_true", default=False)
        p.add_argument("--skip-auto-id",   action="store_true", default=False)
        p.add_argument("--log-to-stdout",   action="store_true", default=False, help="Write log output to stdout instead of a log file in the project directory")

    run_parser = subparsers.add_parser("run", help="Execute the pre-curation workflow directly")
    add_shared_args(run_parser)

    submit_parser = subparsers.add_parser("submit", help="Write and immediately submit a Slurm job")
    add_shared_args(submit_parser)
    submit_parser.add_argument("--qos",       default="regular")
    submit_parser.add_argument("--cpus",      type=int, default=8)
    submit_parser.add_argument("--mem",       default="64G")
    submit_parser.add_argument("--time",      default="03:00:00")
    submit_parser.add_argument("--conda-env", default="metatlas2")
    submit_parser.add_argument("--output",    default=None, help="Override output path for the .sh script")

    return parser.parse_args()

def generate_slurm_script(args) -> str:
    """
    Populate the Slurm template with args and write it to disk.
    Script and logs are written to the analysis output directory.
    Returns the path of the written file.
    """
    analysis_output_dir = _get_analysis_output_dir(
        config_path=args.config,
        project_name=args.project,
        rt_align_num=args.rt_align_num,
        analysis_num=args.analysis_num,
    )

    extra_flags = []
    if args.overwrite:        extra_flags.append("--overwrite")
    if args.skip_setup:       extra_flags.append("--skip-setup")
    if args.skip_rt_align:    extra_flags.append("--skip-rt-align")
    if args.skip_auto_id:     extra_flags.append("--skip-auto-id")
    if args.log_to_stdout:    extra_flags.append("--log-to-stdout")
    if args.analysis_subset:  extra_flags.append("--analysis-subset " + ",".join(args.analysis_subset))
    extra_flags_str = " \\\n    ".join(extra_flags)

    project_short = args.project[:30].replace(" ", "_")

    populated = SLURM_TEMPLATE.format(
        project_short       = project_short,
        qos                 = args.qos,
        cpus                = args.cpus,
        mem                 = args.mem,
        time                = args.time,
        analysis_output_dir = analysis_output_dir,
        conda_env           = args.conda_env,
        script_path         = os.path.abspath(__file__),
        config              = os.path.abspath(args.config),
        project             = args.project,
        rt_align_num        = args.rt_align_num,
        analysis_num        = args.analysis_num,
        extra_flags         = extra_flags_str,
    )

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(analysis_output_dir, f"{project_short}_pre_curation.sh")

    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        f.write(populated)
    os.chmod(out_path, 0o755)

    return out_path


def main():
    args = parse_args()

    if args.log_to_stdout:
        log_file = None
    else:
        print("Setting up logging...")
        project_dir = _get_project_dir(args.project)
        log_file = os.path.join(
            project_dir,
            f"RTA{args.rt_align_num}_TGA{args.analysis_num}.log"
        )

    lcf.setup_logging(log_level=logging.INFO, log_file=log_file, log_to_stdout=args.log_to_stdout)
    logger = lcf.get_logger("run_targeted_analysis")

    if args.command == "submit":
        out_path = generate_slurm_script(args)
        print(f"Slurm script written to: {out_path}")
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
            config_path=args.config,
            overwrite_existing=args.overwrite,
            rt_alignment_number=args.rt_align_num
        )

    if not args.skip_rt_align:
        print("Running RT alignment...")
        logger.info("Running RT alignment ...")
        wfs.run_rt_alignment(
            config_path=args.config,
            project_name=args.project,
            rt_alignment_number=args.rt_align_num
        )

    print("Running Auto Identification...")
    logger.info("Running Auto Identification")
    if not args.skip_auto_id:
        wfs.run_auto_identification(
            config_path=args.config,
            project_name=args.project,
            rt_alignment_number=args.rt_align_num,
            analysis_number=args.analysis_num,
            analysis_subset=args.analysis_subset,
        )

    print("Pre-curation workflow complete. Open the generated notebooks to curate.")
    logger.info("Pre-curation workflow complete. Open the generated notebooks to curate.")


if __name__ == "__main__":
    main()