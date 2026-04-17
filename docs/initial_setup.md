# Initial Setup

Complete these steps **once** on any new machine or user account before running any metatlas2 workflow (adding compounds, adding atlases, or running a targeted analysis).

---

## Analyst setup (shared installation)

### 1. Set `METATLAS_DATA_DIR` and add `scripts/` to your PATH

```bash
echo 'export METATLAS_DATA_DIR="/global/cfs/cdirs/metatlas/"' >> ~/.bashrc
echo 'export PATH="/global/cfs/cdirs/metatlas/tools/metatlas2/scripts:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Jupyter kernel specs

```bash
install_kernels.sh
```

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
```

### 3. Register a cron job to keep the image current

```bash
*/5 * * * * /global/cfs/cdirs/metatlas/tools/metatlas2/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1
```

Once the shared repo is on analysts' PATH and the image is in the shifter cache, analysts only need to run `install_kernels.sh` once to register their own kernel specs.
