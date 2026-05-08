# Standalone Development Environment

The standalone development environment allows you to run metatlas2 on any Linux machine without NERSC access. This is ideal for:

- Testing code changes locally before deploying to production
- Developing new features with fast iteration
- Onboarding new team members who don't yet have NERSC access
- Working when NERSC is down or unavailable

## Features

- **Self-contained**: Includes all necessary data (pre-converted parquet files, atlases, MS2 refs)
- **Minimal dataset**: 15 runs (66 parquet files), 12 compounds (6 per polarity)
- **Complete workflow**: Database init → RT alignment → Auto-ID → GUI → Summaries
- **Production code**: Uses actual production functions (add_compounds_to_db, add_atlases_to_db, etc.)
- **Interactive notebook**: All stages executable from a single Jupyter notebook
- **Automated setup**: Downloads and configures everything on first run

## Quick Start

### Prerequisites

**Required:**
- **Container runtime** (one of the following):
  - **Docker Desktop** (macOS/Windows/Linux) - https://www.docker.com/products/docker-desktop
  - **Colima** (macOS, free/open-source alternative): `brew install colima docker && colima start`
  - **OrbStack** (macOS, fast alternative): `brew install orbstack`
  - **Docker Engine** (Linux only, free): Standard Docker installation
- ~1GB free disk space in `~/.metatlas2-dev/`
- Internet connection (for first-time data download)

**Optional (if container image is private):**
- **GitHub authentication** for container registry access:
  - **Option 1 (Recommended)**: Set environment variable for automatic login
    1. Create token at https://github.com/settings/tokens/new?scopes=read:packages
    2. Add to `~/.bashrc` (or `~/.zshrc` on macOS): `export GITHUB_TOKEN='ghp_xxxxxxxxxxxx'`
    3. Reload shell: `source ~/.bashrc`
  - **Option 2**: Interactive login when prompted
    - Username: your GitHub username
    - Password: your GitHub Personal Access Token (not your password!)

### Launch Standalone Mode

```bash
# Clone or pull the latest repo
cd ~/metatlas2
git pull

# Launch standalone environment
./scripts/metatlas2.sh --standalone
```

This will:
1. Download dev data to `~/.metatlas2-dev/` (if not already present or outdated)
2. Clean up any previous workflow outputs
3. Copy the workflow notebook to `~/.metatlas2-dev/` (keeps repo clean)
4. Launch JupyterLab in a Docker container
5. Open the standalone workflow notebook in your browser at http://localhost:8888

**Note**: The notebook is copied from the repo to `~/.metatlas2-dev/standalone_dev_workflow.ipynb` so your local repo stays clean. You can safely `git pull` updates without conflicts.

### Using the Notebook

The notebook `standalone_dev_workflow.ipynb` contains a minimal workflow with production function calls:

**Cell 1**: Imports
- Import all production metatlas2 functions

**Cell 2**: Environment setup
- Set DATA_DIR path

**Cell 3**: Add compounds to database
- Calls `add_compounds_to_db(config_path, overwrite_db=True)`
- Creates database and loads 12 compounds (6 per polarity)
- Loads MS2 reference spectra

**Cell 4**: Add atlases to database
- Calls `add_atlases_to_db(config_path)`
- Creates POS/NEG ISTD atlases and links compounds

**Cell 5**: Run targeted analysis
- Calls `run_targeted_analysis(config_path)`
- RT alignment using ISTD compounds
- Auto-identification with MS1/MS2 matching
- ~15-20 minutes

**Cell 6**: Generate GUI notebook
- Calls `generate_gui_notebook(config_path)`
- Creates interactive curation notebook

**Cell 7**: Generate summaries
- Calls `generate_summary(config_path)`
- Creates final reports and visualizations

**Total runtime**: ~20-30 minutes (parquet files pre-converted, no conversion time)

## Directory Structure

After setup, the standalone environment lives at `~/.metatlas2-dev/`:

```
~/.metatlas2-dev/
├── parquet/                # 130 pre-converted parquet files (included)
├── databases/              # DuckDB database (generated)
│   └── main_db/
│       └── metatlas.duckdb
├── analysis_output/        # Analysis results (generated)
│   ├── *_gui.ipynb         # Interactive GUI
│   └── summaries/          # Final reports
├── compounds_pos.tsv       # 6 positive mode compounds
├── compounds_neg.tsv       # 6 negative mode compounds
├── ms2_references.tsv      # 17 MS2 reference spectra
├── configs/                # Configuration files
│   ├── compounds_config.yaml   # Compound paths
│   ├── atlases_config.yaml     # Atlas definitions
│   └── analysis_config.yaml    # Workflow parameters
├── dev_environment.yaml    # Metadata
└── README.md               # Package documentation
```

## Development Workflow

### Testing Code Changes

The standalone container mounts your local repository source code, so changes are immediately visible without restarting the container:

1. Edit source code in `~/metatlas2/metatlas2/` (your local repository)
2. In JupyterLab: **Kernel → Restart Kernel** (or **Kernel → Restart Kernel and Clear Outputs**)
3. Re-run relevant notebook cells to test changes
4. Changes are immediately picked up because the source is mounted into the container
5. Iterate quickly - no need to restart the entire container!

**Note**: Leave the container running (`./scripts/metatlas2.sh --standalone`) and just restart the kernel. Only stop and restart if you need to change container settings or update the image.

### Resetting Environment

To start fresh:

```bash
# Remove dev environment (keeps source code)
rm -rf ~/.metatlas2-dev

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
- **DOI**: https://doi.org/10.5281/zenodo.20075571
- **Size**: 1.1GB compressed, ~3GB extracted
- **Contents**: 130 pre-converted parquet files + configs + compound definitions

Zenodo provides permanent, citable storage with unlimited bandwidth for research data.

### Source Data

Extracted from production project at NERSC:
- Project: `20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066`
- Runs: 5 ISTD + 3 QC + 14 experimental + 3 blank + 3 control = 28 runs
- Each run has multiple parquet files (ms1_pos, ms1_neg, ms2_pos, ms2_neg) = 130 total files
- HILIC positive and negative modes
- Parquet files are pre-converted (skips raw → mzML → parquet conversion)

### Compound Selection

- 12 specific compounds with defined adducts (6 POS + 6 NEG)
- Curated list of InChI key + adduct pairs hardcoded in extraction script
- Representative ISTD compounds for alignment and identification testing
- Spans different retention time ranges for comprehensive coverage

**Positive mode (6 compounds)**:
- 5 compounds with [M+H]+ adduct
- 1 compound with [M+Na]+ adduct

**Negative mode (6 compounds)**:
- All 6 compounds with [M-H]- adduct

See `scripts/prepare_dev_package.sh` for the complete list of InChI keys and run numbers.

## Creating New Dev Packages

To create an updated dev package (must be run at NERSC):

```bash
cd ~/metatlas2
./scripts/prepare_dev_package.sh

# Output: /global/cfs/cdirs/metatlas/databases/standalone_dev_data/metatlas2-dev-data.tar.gz
# Upload to the Zenodo page and get new/updated DOI to put into metatlas.sh
```