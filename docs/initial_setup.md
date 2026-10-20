# Initial Setup

Complete these steps **once** on any new NERSC account before running any metatlas2 workflow (adding compounds, adding atlases, or running a targeted analysis).

---

## New analyst setup (NERSC)

These steps walk a brand-new analyst through everything needed to run metatlas2 on NERSC from scratch.

### 1. Set `METATLAS_DATA_DIR` and add `scripts/` to your PATH

`METATLAS_DATA_DIR` must point to the root of the shared metatlas data directory on the NERSC Community File System (CFS). All workflow paths (raw data, databases, project outputs) are derived from this variable.

Add the following lines to your `~/.bashrc` on NERSC:

```bash
echo 'export METATLAS_DATA_DIR="/global/cfs/cdirs/metatlas"' >> ~/.bashrc
echo 'export PATH="/global/cfs/cdirs/metatlas/tools/metatlas2/scripts:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Jupyter kernel specs

The metatlas2 Jupyter kernels must be registered in your personal account before you can open curation notebooks in JupyterHub. Run this command from a NERSC login node terminal:

```bash
install_kernels.sh
```

This registers the `metatlas2` (latest image) and `metatlas2-dev` (local source) Jupyter kernels in your `~/.local/share/jupyter/kernels/` directory. Run this once, and again whenever a new image version is released and you want to pin a specific tag.

> **Note:** `install_kernels.sh` requires `METATLAS_DATA_DIR` to be set (step 2) and `jupyter` to be available. On NERSC, `jupyter` is available in the default Python environment — no module load is needed.

### 3. Access JupyterHub

Curation notebooks are opened and run in NERSC JupyterHub:

1. Go to [https://jupyter.nersc.gov](https://jupyter.nersc.gov) in your browser.
2. Log in with your NERSC credentials (username + password + OTP).
3. Select **Login Node** or **Exclusive CPU node** to start up the interface. 

### Setup is complete!

You can now run any metatlas2 workflow. See the relevant doc for each command:

- [add_compounds_to_db.md](add_compounds_to_db.md) — add compounds to the main database (Note: DO NOT use unless you have run a validated reference standard on the JGI instruments)
- [add_atlases_to_db.md](add_atlases_to_db.md) — add reference atlases to the main database
- [run_targeted_analysis.md](run_targeted_analysis.md) — run the full targeted analysis workflow
- [manual_curation_and_summary.md](manual_curation_and_summary.md) — curate identifications and export results

---

## Administrator setup (shared installation)

Note: STOP HERE if you are not the owner of the metatlas2 GitHub repository.

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

Once the shared repo is on analysts' PATH and the image is in the shifter cache, analysts only need to complete the **New analyst setup** steps above (set env vars, run `install_kernels.sh`) once per account.

---