## Overview

**metatlas2** is a containerized Python package designed for high-throughput targeted metabolomics analysis. It provides an end-to-end workflow for processing LC-MS/MS data, from raw file conversion through automated peak detection, retention time correction, and interactive manual curation, to final summary exports.

### Key Features

- **Compound and Atlas Management**: Centralized DuckDB database for compounds and reference atlases with PubChem metadata integration
- **Automated RT Alignment**: Polynomial-based retention time correction using quality control (QC) samples and Atlas compounds
- **MS2 Hit Detection**: Automated MS2 spectral matching against reference libraries
- **Interactive Curation GUI**: Jupyter-based interactive interface for manual peak review and validation
- **Batch Processing**: Support for SLURM cluster submission and parallel processing
- **Container-Based Deployment**: Reproducible environment using Shifter/Docker containers
- **Flexible Configuration**: YAML-based configuration for projects, compounds, and atlases

### Workflow Stages

1. **Project Setup** — Initialize project database and retrieve LC-MS run files
2. **RT Alignment** — Fit and apply retention time correction models
3. **Auto Identification** — Extract MS1/MS2 data and score MS2 hits
4. **Manual Curation** — Interactive GUI-based peak review and validation
5. **Analysis Summary** — Export validated results and summary statistics

---

## Quick Start

### First-Time Setup

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
    --config_path /path/to/create_compounds.yaml

# 2. Add reference atlases to the main database
metatlas2 add-atlases \
    --config_path /path/to/create_atlases.yaml

# 3. Run targeted analysis
metatlas2 run \
    --config /path/to/analysis.yaml \
    --project MyProject \
    --overwrite
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

### Technical Reference

| Document | Description |
|----------|-------------|
| **[codebase_overview.md](docs/codebase_overview.md)** | Programmer-oriented reference: module structure, workflow phases, and key classes |
| **[database_schema.md](docs/database_schema.md)** | Complete database schema documentation including table structures and relationships |
| **[parquet_file_structure.md](docs/parquet_file_structure.md)** | Parquet file format specification for MS1/MS2 data storage |
| **[development_and_testing.md](docs/development_and_testing.md)** | System test setup, fixture generation, CI/CD pipeline, and code quality tools |

---

## Development & Testing

### System Test

An end-to-end system test validates the full pipeline using synthetic fixtures in `tests/fixtures/data/`. Run it with:

```bash
nox -s system_test
```

The test auto-detects the environment and runs the pipeline inside a container:

| Environment | Detection | Container runtime |
|---|---|---|
| NERSC | `NERSC_HOST` or `SLURM_CLUSTER_NAME` env var | Shifter via `metatlas2.sh --dev` |
| GitHub Actions / local | neither env var set | Docker (`ghcr.io/bkieft-usa/metatlas2`) |

The Docker image tag defaults to `latest` and can be overridden:

```bash
METATLAS2_IMAGE_TAG=sha-abc1234 nox -s system_test
```

**Requirements:**
- NERSC: `metatlas2.sh` on PATH, Shifter access
- Local / CI: Docker installed and running, `uv` on PATH

**What is validated** (by `tests/test_system.py`):
- Output directory and all expected files exist (`<project>.duckdb`, `RTA0/`, `RTA0/TGA0/`, RT-aligned and auto-IDed atlas CSVs)
- Database schema and row counts match `tests/fixtures/expected_baseline.json`

The temporary data directory created during the test is deleted automatically after pytest completes.

### CI/CD

GitHub Actions (`.github/workflows/docker.yml`) runs on every push to `main`:
1. Builds and pushes the Docker image to `ghcr.io/bkieft-usa/metatlas2` (tagged `latest` and `sha-<short-sha>`)
2. Runs `nox -s system_test` against the freshly built image

---