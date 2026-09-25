# Running Metatlas2 Outside NERSC

This guide covers running the full metatlas2 targeted analysis pipeline on any machine that has Docker — a local laptop, a non-NERSC HPC cluster, or any Linux/macOS/WSL2 system.

The pipeline runs identically to NERSC. The only differences are:
- Docker is used instead of Shifter (detected automatically by `metatlas2.sh`)
- The main metatlas DuckDB is downloaded automatically from Zenodo on first run
- The ModelSEED reference table is downloaded automatically on first use (by `analysis_summary.py`)
- The curation GUI is accessed at `http://localhost:8050/` instead of a JupyterHub proxy URL

---

## Prerequisites

### 1. Docker

Install Docker and make sure it is running:

- **macOS / Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: [Docker Engine](https://docs.docker.com/engine/install/) or Docker Desktop
- **Windows + WSL2**: Install Docker Desktop on Windows, enable WSL 2 integration in Docker Desktop settings

Verify:
```bash
docker --version
docker compose version
```

### 2. GitHub container registry access

The metatlas2 container image is hosted on GitHub Container Registry (GHCR). Authenticate once:

```bash
# Create a token at https://github.com/settings/tokens/new?scopes=read:packages
export GITHUB_TOKEN='ghp_xxxxxxxxxxxx'
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Add the `export GITHUB_TOKEN=...` line to `~/.bashrc` (Linux) or `~/.zshrc` (macOS) so it persists.

### 3. Clone the repository

```bash
git clone https://github.com/bkieft-usa/metatlas2.git ~/metatlas2
```

### 4. Add `scripts/` to your PATH

```bash
echo 'export PATH="${HOME}/metatlas2/scripts:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

---

## One-time environment setup

### Set `METATLAS_DATA_DIR`

This is the single root directory for all metatlas2 data. All workflow paths (raw data, databases, project outputs) are derived from it.

```bash
export METATLAS_DATA_DIR="/path/to/your/metatlas_data"
mkdir -p "${METATLAS_DATA_DIR}"
```

Add this to `~/.bashrc` (or `~/.zshrc` on macOS) so it persists across sessions.

### Expected directory layout

```
$METATLAS_DATA_DIR/
├── raw_data/
│   └── <owner>/                        # matches GENERAL.owner in your config YAML
│       └── <project_name>/             # matches --project PROJECT_NAME exactly
│           ├── run1.h5
│           ├── run2.h5
│           └── ...
├── databases/
│   ├── main_db/
│   │   └── metatlas.duckdb             # auto-downloaded from Zenodo on first run
│   ├── pubchem_cache/
│   │   └── pubchem_global_cache.json   # optional; created on first PubChem lookup
│   └── modelseed_db/
│       └── modelseed.tsv               # optional; auto-downloaded on first use by analysis_summary.py
└── projects/
    └── targeted_outputs/               # analysis outputs written here
        └── <owner>/
            └── <user>/
                └── <project_name>/
```

---

## Automatic downloads on first run

### Main metatlas DuckDB (~150 MB)

Contains all compounds, reference atlases, and MS2 spectra needed to run the workflow. On the first Docker-mode `run`, `metatlas2.sh` automatically downloads it from Zenodo into `$METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb`.

The download happens transparently before the container starts. Subsequent runs check the local version stamp and skip the download if the database is already current.

### ModelSEED reference table (`modelseed.tsv`)

Used during the analysis summary stage for compound annotation. If `modelseed.tsv` is not already present at `$METATLAS_DATA_DIR/databases/modelseed_db/modelseed.tsv`, `analysis_summary.py` downloads it automatically on first use. No manual action is required.

---

## Placing your raw data files

Raw LCMS data must be in `.h5` format (HDF5, as produced by the NERSC msconvert pipeline). Place them at:

```
$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/
```

Where:
- `<owner>` matches the `GENERAL.owner` field in your config YAML (e.g. `jgi`)
- `<project_name>` matches the `--project` argument exactly

**Project name format:** The project name must have at least 5 underscore-delimited fields. The 5th field (index 4) is used as the short name for log files and output directories. Example:

```
20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX
│         │   │  │      └── field[4]: short name used in log filenames
│         │   │  └── field[3]
│         │   └── field[2]
│         └── field[1]
└── field[0]: date
```

A minimal valid project name: `20260101_JGI_XX_000000_MYPROJECT`

---

## Config file

Start from the example config:

```bash
cp ~/metatlas2/configs/example_configs/analyses/jgi_default_hilicz_config.yaml \
   "${METATLAS_DATA_DIR}/configs/my_analysis_config.yaml"
```

Edit the copy:

1. **`GENERAL.owner`** — set to your owner string (e.g. `jgi`); must match your raw data directory name
2. **Atlas UIDs** — the `uid:` values under `RT_ALIGNMENT` and `TARGETED_ANALYSES` must match atlases that exist in the downloaded main database. Query available atlases with:
   ```bash
   metatlas2.sh get-atlases query --chromatography HILICZ --polarity POS
   ```
3. **`upload_to_gdrive: false`** — keep this `false` unless you have Google Drive credentials configured

The config file can live anywhere on your filesystem — `metatlas2.sh` will add a read-only bind-mount for its directory automatically if it is outside `$METATLAS_DATA_DIR`.

---

## Running the pipeline

```bash
metatlas2.sh run \
    --config  /path/to/my_analysis_config.yaml \
    --project 20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX \
    --rt-align-num 0 \
    --analysis-num 0
```

Add `--dev` to use your local repository source instead of the installed package:

```bash
metatlas2.sh --dev run \
    --config  /path/to/my_analysis_config.yaml \
    --project 20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX \
    --rt-align-num 0 \
    --analysis-num 0
```

The pipeline runs three stages automatically:
1. **Project setup** — creates output directories and project database
2. **RT alignment** — fits a retention-time correction model using QC runs
3. **Auto identification** — extracts MS1/MS2 data, scores MS2 hits, generates curation notebooks

Outputs are written to:
```
$METATLAS_DATA_DIR/projects/targeted_outputs/<owner>/<user>/<project_name>/
```

---

## Manual curation GUI

After the pipeline completes, a Jupyter notebook is generated for each targeted analysis in the output directory. Open these notebooks in JupyterLab to run the interactive curation GUI.

### On a local laptop

1. Start JupyterLab (using the standalone mode or your own Jupyter installation)
2. Open the generated `.ipynb` notebook
3. Run the GUI cell — the Dash app starts and a link is printed:
   ```
   ▶ Open Dash App ↗  →  http://localhost:8050/
   ```
4. Click the link to open the curation GUI in your browser

### On a headless HPC (no display, SSH access)

The container exposes ports 8050–8069 on `127.0.0.1`. Use **VSCode SSH port forwarding** (or `ssh -L`) to access the GUI in your local browser:

**VSCode (recommended):**
1. Connect to the remote machine via VSCode Remote SSH
2. Open the generated notebook in VSCode's Jupyter extension or via the forwarded JupyterLab URL
3. Run the GUI cell — VSCode automatically forwards `localhost:8050` to your local browser
4. The link `http://localhost:8050/` opens directly in your local browser

**Manual SSH tunnel:**
```bash
# On your local machine, in a separate terminal:
ssh -L 8050:localhost:8050 user@remote-hpc
```
Then open `http://localhost:8050/` in your local browser after running the GUI cell on the remote.

---

## Differences from NERSC

| Feature | NERSC (Shifter) | Non-NERSC (Docker) |
|---|---|---|
| Container runtime | Shifter (auto-detected) | Docker (auto-detected) |
| Filesystem mounts | Global NERSC filesystems auto-mounted | `$METATLAS_DATA_DIR` bind-mounted |
| Main database | Already on shared filesystem | Auto-downloaded from Zenodo |
| `submit` subcommand (SLURM) | ✅ Supported | ❌ Not supported (use `run` directly) |
| GUI URL | JupyterHub proxy URL | `http://localhost:8050/` |
| GUI access | JupyterHub browser tab | Local browser via VSCode SSH or `ssh -L` |
| Parallel job submission | `sbatch` via `submit` | Run directly or use your HPC's scheduler |

---

## Troubleshooting

### `METATLAS_DATA_DIR is not set`
Add `export METATLAS_DATA_DIR=/path/to/data` to `~/.bashrc` and run `source ~/.bashrc`.

### `Raw data directory not found`
Check that your `.h5` files are at `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/` and that `<owner>` matches `GENERAL.owner` in your config YAML.

### `Main database not found`
The Zenodo download may have failed. Check your internet connection and that `ZENODO_MAIN_DB_DOI` is set in `scripts/metatlas2.sh`. You can also manually place `metatlas.duckdb` at `$METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb`.

### Atlas UID not found in database
The atlas UIDs in your config YAML must exist in the downloaded main database. Query available atlases:
```bash
metatlas2.sh get-atlases query --chromatography HILICZ
```

### GUI not accessible at `localhost:8050`
- Confirm the pipeline completed (the GUI is launched from the curation notebook, not from `metatlas2.sh run`)
- If on a remote machine, ensure your SSH tunnel or VSCode port forwarding is active
- Try ports 8051–8069 if 8050 is in use (the app auto-selects the first free port)

### `docker: permission denied`
On Linux, add your user to the `docker` group:
```bash
sudo usermod -aG docker $USER
# Log out and back in, then retry
```
