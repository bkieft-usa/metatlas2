# Initial Setup

Complete these steps **once** on any new machine or user account before running any metatlas2 workflow (adding compounds, adding atlases, or running a targeted analysis).

---

## Analyst setup (shared installation)

### 1. Set `METATLAS_DATA_DIR` and add `scripts/` to your PATH

`METATLAS_DATA_DIR` must point to the root of the shared data directory. All workflow paths (raw data, databases, project outputs) are derived from this variable.

```bash
echo 'export METATLAS_DATA_DIR="/global/cfs/cdirs/metatlas"' >> ~/.bashrc
echo 'export PATH="/global/cfs/cdirs/metatlas/tools/metatlas2/scripts:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

The expected directory layout under `METATLAS_DATA_DIR` is:

```
$METATLAS_DATA_DIR/
├── raw_data/                          # Raw LCMS files (.raw, .mzML, .h5) per owner/project
│   └── <owner>/
│       └── <project_name>/
├── databases/
│   ├── main_db/
│   │   └── metatlas.duckdb            # Central compound/atlas database
│   ├── pubchem_cache/
│   │   └── pubchem_global_cache.json  # PubChem metadata cache
│   └── modelseed_db/
│       └── modelseed.tsv              # ModelSEED reference table
└── projects/
    └── targeted_outputs/              # Per-project analysis outputs
        └── <owner>/
            └── <user>/
                └── <project_name>/
```

### 2. Install Jupyter kernel specs

```bash
install_kernels.sh
```

This registers the `metatlas2` (latest image) and `metatlas2-dev` (local source) Jupyter kernels in your `~/.local/share/jupyter/kernels/` directory. Run this once, and again whenever a new image version is released and you want to pin a specific tag.

### Setup is complete!

You can now run any metatlas2 workflow. See the relevant doc for each command:

- [add_compounds_to_db.md](add_compounds_to_db.md) — add compounds to the main database
- [add_atlases_to_db.md](add_atlases_to_db.md) — add reference atlases to the main database
- [run_targeted_analysis.md](run_targeted_analysis.md) — run the full targeted analysis workflow

---

## Administrator setup (shared installation)

Run these steps once on behalf of all analysts. Requires credentials to pull the private container image.

### 1. Clone the repository to a shared location

```bash
git clone https://github.com/bkieft-usa/metatlas2.git /global/cfs/cdirs/metatlas/tools/metatlas2
chgrp -R <shared_group> /global/cfs/cdirs/metatlas/tools/metatlas2
chmod -R g+rX /global/cfs/cdirs/metatlas/tools/metatlas2
```

### 2. Authenticate and pull the container image

```bash
shifterimg login ghcr.io   # enter GitHub username + classic PAT with read:packages scope
shifterimg pull ghcr.io/bkieft-usa/metatlas2:latest
```

### 3. Register a cron job to keep the image current

```bash
*/5 * * * * /global/cfs/cdirs/metatlas/tools/metatlas2/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1
```

Once the shared repo is on analysts' PATH and the image is in the shifter cache, analysts only need to run `install_kernels.sh` once to register their own kernel specs.

---

## Standalone / local development setup

To run metatlas2 on a local machine (without NERSC access), use the standalone Docker mode. See [standalone_dev_environment.md](standalone_dev_environment.md) for full instructions.

```bash
# Clone the repo
git clone https://github.com/bkieft-usa/metatlas2.git ~/metatlas2

# Launch standalone JupyterLab environment
~/metatlas2/scripts/metatlas2.sh --standalone
```
