## Overview

**metatlas2** is a containerized Python package designed for high-throughput targeted metabolomics analysis. It provides an end-to-end workflow for processing LC-MS/MS data, from raw file conversion through automated peak detection, retention time correction, and interactive manual curation, to final summary exports.

### Key Features

- **Compound and Atlas Management**: Centralized DuckDB database for Compounds and reference Atlases with PubChem metadata integration. Also stores all completed Projects with metadata (e.g., Project database location on disk) to facilitate meta-analyses.
- **Automated RT Alignment**: Polynomial-based retention time correction using a subset of LCMSRun files (i.e., quality control samples) and a subset of Compounds (e.g., a pre-defined Atlas with historical RTs for quality controlled Compounds)
- **MS1 and MS2 Data Extraction**: Automated MS1 and MS2 data extraction and matching to user-defined reference compounds (i.e., an Atlas of findable Compounds)
- **MS2 Hit Detection**: Automated MS2 spectral matching against a reference libraries of MSMS data
- **Interactive Curation GUI**: JupyterLab-based interactive interface for manual peak review, validation, and notes storage
- **Batch Processing**: Support for SLURM cluster submission or real-time runs on a dedicated node
- **Container-Based Deployment**: Reproducible environment using Shifter/Docker containers for portability
- **Flexible Configuration**: Standardized, YAML-based configuration for creating new Compounds, Atlases, and Analyses

### Workflow Stages for a Typical Targeted Analysis

| # | Stage | Interface | Description |
|---|-------|-----------|-------------|
| 1 | **Project Setup** | CLI | Initialize Project file system locations and database, retrieve and store LCMSRun files from disk |
| 2 | **RT Alignment** | CLI | Fit and apply retention time correction models to empirical data |
| 3 | **Auto Identification** | CLI | Extract MS1/MS2 data from LCMSRun files and score MS2 hits against MSMS reference database |
| 4 | **Manual Curation** | JupyterLab NB | Interactive GUI-based peak review and validation to select retention time windows and vet MS2 data |
| 5 | **Analysis Summary** | JupyterLab NB | Export validated results and summary statistics after curation |

---

## Quick Start

### Standalone Development Mode

For local development and testing without NERSC access:

```bash
# Clone the repository
git clone https://github.com/bkieft-usa/metatlas2.git
cd metatlas2

# Launch standalone environment (auto-downloads 1.1GB dev data)
./scripts/metatlas2.sh --standalone
```

This opens a JupyterLab notebook that runs the complete metatlas2 pipeline using production code with a minimal dataset (28 runs, 12 compounds, ~20-30 min runtime).

See **[standalone_dev_environment.md](docs/standalone_dev_environment.md)** for complete details.

### First-Time Setup (NERSC Production)

For new users or new machines, complete the one-time setup:

```bash
# Set environment variables (ideally in .bashrc)
export METATLAS_DATA_DIR="/global/cfs/cdirs/metatlas/"
export PATH="/global/cfs/cdirs/metatlas/tools/metatlas2/scripts:${PATH}"

# Install Jupyter kernel specs
install_kernels.sh
```

See **[initial_setup.md](docs/initial_setup.md)** for complete setup instructions.

### Running a Workflow

```bash
# 1. Add compounds to the main database
metatlas2 add-compounds \
    --config_path /global/cfs/cdirs/metatlas/configs/compounds/<config_name>.yaml

# 2. Add reference atlases to the main database
metatlas2 add-atlases \
    --config_path /global/cfs/cdirs/metatlas/configs/atlases/<config_name>.yaml

# 3. Run targeted analysis (will generate notebooks for manual curation)
metatlas2 run \
    --config /global/cfs/cdirs/metatlas/configs/analyses/<config_name>.yaml
    --project MyProject \
    --rt-align-num 0 \ 
    --analysis-num 0
```

---

## Documentation

### Getting Started

| Document | Description |
|----------|-------------|
| **[initial_setup.md](docs/initial_setup.md)** | Complete first-time setup instructions for analysts and administrators |

### User Guides

| Document | Description |
|----------|-------------|
| **[add_compounds_to_db.md](docs/add_compounds_to_db.md)** | Add compounds to the central database with PubChem metadata retrieval |
| **[add_atlases_to_db.md](docs/add_atlases_to_db.md)** | Create and manage reference atlases for targeted analysis |
| **[run_targeted_analysis.md](docs/run_targeted_analysis.md)** | Main workflow execution: project setup, RT alignment, and auto identification |
| **[manual_curation_and_summary.md](docs/manual_curation_and_summary.md)** | Interactive GUI for manual peak curation and summary generation |

### Developer Resources

| Document | Description |
|----------|-------------|
| **[standalone_dev_environment.md](docs/standalone_dev_environment.md)** | Standalone development environment for testing on any Linux machine without NERSC access |
| **[codebase_overview.md](docs/codebase_overview.md)** | Programmer-oriented reference: module structure, workflow phases, and key classes |
| **[database_schema.md](docs/database_schema.md)** | Complete database schema documentation including table structures and relationships |
| **[parquet_file_structure.md](docs/parquet_file_structure.md)** | Parquet file format specification for MS1/MS2 data storage |
| **[development_and_testing.md](docs/development_and_testing.md)** | System test setup, fixture generation, CI/CD pipeline, and code quality tools |

---