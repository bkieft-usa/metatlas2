# Standalone Development Environment

The standalone development environment allows you to run metatlas2 on any machine with a supported container runtime, without NERSC access. This is ideal for:

- Testing code changes locally before deploying to production
- Developing new features with fast iteration
- Onboarding new team members who don't yet have NERSC access
- Working when NERSC is down or unavailable

## Quick Start

### Prerequisites

**Required:**
- **Container runtime** (one of the following):
  - **Docker Desktop** (macOS/Windows/Linux) - https://www.docker.com/products/docker-desktop
  - **Docker Engine** (Linux only, free): Standard Docker installation
- **If using Docker Desktop, use this OS-specific setup:**
  - **macOS**: Install Docker Desktop and ensure it is running before launching standalone mode.
  - **Linux**: Install Docker Desktop (or Docker Engine), sign in if prompted, and verify `docker` and `docker compose` are available in your shell.
  - **Windows + WSL2**: Install Docker Desktop on Windows, enable WSL 2 integration in Docker Desktop settings, enable your WSL distro (for example Ubuntu).
- ~2MB free disk space in `~/.metatlas2-dev/`
- Internet connection (for first-time data download)

- **GitHub authentication** for container registry access:
  - **Option 1 (Recommended)**: Set environment variable for automatic login
    1. Create token at https://github.com/settings/tokens/new?scopes=read:packages
    2. Add to `~/.bashrc` (or `~/.zshrc` on macOS):
      `export GITHUB_TOKEN='ghp_xxxxxxxxxxxx'`
    3. Reload shell:
      `source ~/.bashrc`
    4. Log into the container repository:
      `echo $GITHUB_TOKEN | docker login ghcr.io -u GITHUB_USERNAME --password-stdin`
  - **Option 2**: Interactive login when prompted (if you don't set the token env var)
    - Username: your GitHub username
    - Password: your GitHub Personal Access Token (not your password!)

### Launch Standalone Mode

```bash
# Clone or pull the latest repo
cd ~
git clone https://github.com/bkieft-usa/metatlas2.git
# OR
cd ~/metatlas2
git pull

# Launch standalone environment
./scripts/metatlas2.sh --standalone
```

This will automatically:
1. Download dev data to `~/.metatlas2-dev/` (if not already present or if it's an outdated data version)
2. Clean up any previous workflow outputs (move these elsewhere after a run if you don't want them to be removed)
3. Copy the workflow notebook to `~/.metatlas2-dev/` (keeps repo clean)
4. Install the notebook kernel (metatlas2 standalone)
5. Launch JupyterLab in a Docker container
6. Open the standalone workflow notebook in your browser at http://localhost:8889/lab

### Using the Notebook

The notebook `standalone_dev_workflow.ipynb` runs the current production workflow in explicit stages. These are all identical calls to the production metatlas2 version, so see the documentation for [running the pipeline](https://github.com/bkieft-usa/metatlas2/blob/main/docs/run_targeted_analysis.md) and [running the GUI](https://github.com/bkieft-usa/metatlas2/blob/main/docs/manual_curation_and_summary.md).

**NOTE:** Cells 6-8 only need to be run once per standalone session (these build the database and fill it with compounds and atlases). If you want to run the RT alignment/Auto ID/GUI/Summary tools multiple times (e.g., after updating local codebase), you can restart the kernel and skip those cells.

**Cell 1**: Title and execution note

**Cell 2**: Imports and logging setup
- Imports workflow modules and helper functions
- Configures logging for standalone execution

**Cell 3**: Environment setup
- Sets `METATLAS_DATA_DIR`

**Cell 4**: Project and config paths
- Sets project name, RT alignment number, analysis number
- Defines config file locations

**Cell 5**: Load analysis configuration

**Cell 6**: Add compounds to database
- Calls `add_compounds_to_db(config_path, overwrite_db=True)`

**Cell 7**: Create atlases
- Calls `add_atlases_to_db(config_path)`

**Cell 8**: Set atlas UIDs
- Paste atlas UIDs printed by the previous cell

**Cell 9**: Set up project paths
- Updates config atlas UIDs and calls `set_up_paths(...)`

**Cell 10**: Run project setup
- Calls `wfs.run_project_setup(...)`

**Cell 11**: Run RT alignment
- Calls `wfs.run_rt_alignment(...)`

**Cell 12**: Run auto-identification
- Calls `wfs.run_auto_identification(...)`
- Produces GUI notebooks for curation

**Total runtime**: ~1-2 minutes

## Directory Structure

After setup, the standalone environment lives at `~/.metatlas2-dev/`. Output directories from the workflow (listed as "generated") will show up in the tree, which is mounted to your local filesystem and will persist when the container is shut down.

```
~/.metatlas2-dev/
├── lcmsruns/
│   └── dev/
│       └── 20260101_JGI_XX_000000_STANDALONE-DEV_test_EXP000_HILICZ_TESTXXXX/
│           └── parquet/    # Pre-converted parquet subset (included)
├── databases/              # DuckDB database (generated)
│   └── main_db/
│       └── metatlas.duckdb
├── projects/
│   └── targeted_outputs/   # Analysis results (generated)
├── qc_compounds_pos.tsv    # QC compounds used for RT alignment
├── ema_compounds_pos.tsv   # EMA compounds used for targeted analysis
├── ms2_references.json      # MS2 reference spectra
├── configs/                # Configuration files
│   ├── compounds_config.yaml   # Compound paths
│   ├── atlases_config.yaml     # Atlas definitions
│   └── analysis_config.yaml    # Workflow parameters
├── standalone_dev_workflow.ipynb
└── .zenodo_version         # Downloaded package DOI
```

## Development Workflow

### Testing Code Changes

The standalone container mounts your local repository source code, so changes are immediately visible without restarting the container (though you'll still need to restart the notebook kernel):

1. Edit source code in `~/metatlas2/metatlas2/` (your local repository)
2. In JupyterLab: **Kernel → Restart Kernel** (or **Kernel → Restart Kernel and Clear Outputs**)
3. Re-run relevant notebook cells to test changes

### Resetting Environment

To start fresh:

```bash
# Stop current container

# Re-run to download and setup again
./scripts/metatlas2.sh --standalone
```

### Using Different Image Versions

```bash
# Use a specific tagged image
./scripts/metatlas2.sh --standalone --image v1.2.0

# Use latest (default)
./scripts/metatlas2.sh --standalone --image latest
```

## Data Package Details

The dev data package is hosted on Zenodo:
- **DOI**: https://doi.org/10.5281/zenodo.20142879
- **Contents**: Pre-converted parquet subset + atlas tables + MS2 references table + configs

Zenodo provides permanent, citable storage with unlimited bandwidth for research data.

### Source Data

Extracted from production project at NERSC:
- Project: `20260311_JGI_AE_511825_SorghAnth_final_EXP120B_HILICZ_USHXG03401`
- Included runs: 5 QC + 16 experimental = 21 runs
- Parquet files are pre-converted and copied as a positive-mode subset

### Compound Selection

- Two positive-mode compound tables are included:
  - `qc_compounds_pos.tsv` for RT alignment QC compounds
  - `ema_compounds_pos.tsv` for EMA targeted analysis compounds
- MS2 reference spectra are provided in `ms2_references.json`

See `local/prepare_dev_package.sh` for the complete run list and compound definitions.

## Creating New Dev Packages

To create an updated dev package (must be run at NERSC):

```bash
cd ~/metatlas2
./local/prepare_dev_package.sh
```