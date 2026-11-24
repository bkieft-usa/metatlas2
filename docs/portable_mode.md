# Portable Mode (running metatlas2 off NERSC)

Off NERSC, metatlas2 runs in **portable mode**: `scripts/metatlas2.sh` uses Docker instead of
Shifter. Portable mode has two branches. They share the same code path and differ only in
which data they use:

| | **Local** (analysts) | **Sandbox** (developers) |
|---|---|---|
| Purpose | Run the targeted workflow on your own data | Change the code safely and test it end to end |
| Command | `metatlas2.sh run ...`, `metatlas2.sh jupyter` | `metatlas2.sh --sandbox [--image TAG]` |
| Data | Your `.h5` files + config under `$METATLAS_DATA_DIR` | Sandbox package from Zenodo in `~/.metatlas2-sandbox/` |
| Main database | Copy of the NERSC main DB, downloaded from Zenodo | Built from scratch on every launch |
| Atlases | Already in the main DB, or added with `add-atlases`, selected by `uid` | Added from `sandbox/inputs/` by the sandbox notebook |
| Code | Code in the latest image | Your local clone, mounted live (`--dev`) |
| Image | Always `latest` (`--image` is rejected) | `latest` or any tag given with `--image` |
| Writes back to NERSC | Never | Never |

On NERSC nothing changes: Shifter is detected automatically, and JupyterHub uses the
`metatlas2` and `metatlas2-dev` kernels (see [initial_setup.md](initial_setup.md)).

---

## Prerequisites (both branches)

1. **Docker**, installed and running:
   - macOS / Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop)
     (Windows: enable WSL 2 integration and run everything below inside WSL)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/) or Docker Desktop.
     If you see `docker: permission denied`, run `sudo usermod -aG docker $USER` and log out and back in.

   Check: `docker --version`
2. **Internet access**: used to pull the image and download the Zenodo data.
3. **The `metatlas2.sh` script**:
   ```bash
   git clone https://github.com/biorack/metatlas2.git ~/metatlas2
   echo 'export PATH="${HOME}/metatlas2/scripts:${PATH}"' >> ~/.bashrc   # ~/.zshrc on macOS
   source ~/.bashrc
   ```

The image (`ghcr.io/biorack/metatlas2`) is public, so no registry login is needed. Every
portable run pulls the tag first, so `latest` is always current.

---

## Local: run the workflow on your own data

### 1. Set `METATLAS_DATA_DIR`

This is the single root folder for all your metatlas2 data.

```bash
export METATLAS_DATA_DIR="/path/to/your/metatlas_data"   # add to ~/.bashrc or ~/.zshrc
mkdir -p "${METATLAS_DATA_DIR}"
```

Layout:

```
$METATLAS_DATA_DIR/
├── raw_data/<owner>/<project_name>/*.h5    # your raw data (owner = GENERAL.owner, jgi or egsb)
├── databases/
│   ├── main_db/metatlas.duckdb             # downloaded from Zenodo on first run
│   └── modelseed_db/modelseed.tsv          # downloaded on first use by the summary
└── projects/targeted_outputs/<owner>/<user>/<project_name>/   # outputs
```

### 2. Place the raw data

Raw data must be `.h5` files (as produced by the NERSC conversion pipeline), at
`$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/`:
- `<owner>` must match `GENERAL.owner` in your config. It also selects the filename-parsing
  rules, so it must be `jgi` or `egsb`.
- `<project_name>` must match `--project` exactly. It needs at least 5 underscore-separated
  fields, for example `20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX`.

### 3. The main database

On the first `metatlas2.sh` call (and whenever `ZENODO_MAIN_DB_DOI` in `metatlas2.sh` changes),
the NERSC main DB (about 150 MB) is downloaded to `databases/main_db/metatlas.duckdb`.

> **A new DB version replaces your copy, including any atlases you added.** To keep your
> current copy, add `--skip-db-update` to every command. If no DB exists yet, `--skip-db-update`
> only prints a warning and continues.

### 4. Atlases: use UIDs, not tags

The centralized atlas Google Sheet is not available off NERSC. A config that uses
`ATLAS.tags` stops right away with an error. Instead, use `ATLAS.uid`:

- **An atlas already in the main DB**: find its UID with
  `metatlas2.sh get-atlases query --chromatography HILICZ --polarity POS`.
- **Your own atlas**: add it to your local DB, then copy the UID it prints
  (see [add_atlases_to_db.md](add_atlases_to_db.md)):
  ```bash
  metatlas2.sh add-atlases --config_path /path/to/my_atlases.yaml --custom
  ```
  New compounds can be added the same way with `metatlas2.sh add-compounds --input_csv FILE`.

If a config refers to a UID that isn't in your DB, the workflow stops with an error that names the missing atlas.

`update-tags` (which edits the sheet) and `submit` (SLURM) are NERSC-only.

### 5. Config

Start from an example and edit it:

```bash
mkdir -p "${METATLAS_DATA_DIR}/configs"
cp ~/metatlas2/configs/example_configs/analyses/jgi_default_hilicz_config.yaml \
   "${METATLAS_DATA_DIR}/configs/my_analysis_config.yaml"
```

- `GENERAL.owner`: `jgi` or `egsb`, matching your `raw_data/<owner>/` folder
- every `ATLAS` block: `uid: <atlas uid>` (and `polarity:` under `RT_ALIGNMENT`)
- `upload_to_gdrive: false`, unless you set up Google Drive upload (see below)

The config can live anywhere. If it is outside `$METATLAS_DATA_DIR`, its folder is mounted
read-only into the container automatically.

### 6. Run

```bash
metatlas2.sh run \
    --config  "${METATLAS_DATA_DIR}/configs/my_analysis_config.yaml" \
    --project 20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX \
    --rt-align-num 0 \
    --analysis-num 0
```

This runs project setup, RT alignment and auto-identification, and writes curation notebooks to
`projects/targeted_outputs/<owner>/<user>/<project_name>/RTA<N>/TGA<M>/`.

### 7. Curate and summarize

```bash
metatlas2.sh jupyter            # JupyterLab at http://localhost:8889/lab (--port to change)
```

Open the generated curation notebook and run the GUI cell. Click **Open Dash App**
(`http://localhost:8050/`), curate, then run the summary cell. See
[manual_curation_and_summary.md](manual_curation_and_summary.md).

On a headless remote machine, forward ports 8889 and 8050–8069. VS Code Remote SSH does this
automatically. Otherwise use `ssh -L 8889:localhost:8889 -L 8050:localhost:8050 user@host`.

### Google Drive upload (optional)

Install rclone on your machine and run `rclone config` to create a Google Drive remote. Then set
`GENERAL.gdrive_subfolder: <FOLDER_ID>` and `upload_to_gdrive: true` in the config.
`metatlas2.sh` mounts `~/.config/rclone` into the container when it exists.

---

## Sandbox: develop and test changes

The sandbox runs the **whole** targeted workflow on a small, fixed dataset. On each launch it
builds a fresh database (compounds → atlases → MS/MS refs), then runs project setup → RT
alignment → auto-identification → curation GUI → summary. The code it runs comes from your local clone.

### Developer workflow

0. **Check that the sandbox data package is current.** The Zenodo package holds only the subset
   `.h5` files. Everything else the sandbox uses lives in the repo under `sandbox/inputs/`
   (compounds, atlases, MS/MS refs, config templates), so you change those in your branch like
   any other code. Publish a new package only when the h5 data must change: new runs, or atlas
   m/z or RT windows that the current subset doesn't cover. See
   [Updating the sandbox data package](#updating-the-sandbox-data-package).
1. **Create a branch before editing anything.** The sandbox refuses to start on `main` or a
   detached HEAD:
   ```bash
   cd ~/metatlas2
   git switch main && git pull
   git switch -c my-feature
   ```
2. **Launch:**
   ```bash
   metatlas2.sh --sandbox                 # latest image
   metatlas2.sh --sandbox --image sha-abc1234   # a specific image's Python environment
   ```
   This downloads the package (first time, or when `ZENODO_SANDBOX_DATA_DOI` changes) to
   `~/.metatlas2-sandbox/`, **wipes the sandbox DB, configs and outputs**, and opens JupyterLab
   at `http://localhost:8889/lab/tree/sandbox_workflow.ipynb`. With `--image TAG`, the notebooks
   run on a kernel named `metatlas2-TAG` that uses that image's environment.
3. **Run and iterate.** Run the notebook top to bottom. Part 1 builds the DB (it skips itself if
   the DB already exists), and Part 2 runs the workflow. Edit code in your clone on your machine,
   then use **Kernel → Restart Kernel** and re-run. Your clone is mounted read-only and comes first
   on `PYTHONPATH`, so edits take effect without rebuilding anything. To keep earlier results for
   comparison, change `rt_alignment_number` and/or `analysis_number` before re-running Part 2.
   Every run from this launch is kept until the next `--sandbox` launch.
4. **Curate and summarize.** Open the generated curation notebooks in
   `projects/targeted_outputs/jgi/<user>/<project>/RTA<N>/TGA<M>/`, run the GUI
   (`http://localhost:8050/`), then run the summary.
5. **Commit and open a PR from your machine** (not from inside the container):
   ```bash
   git add -p && git commit
   git push -u origin my-feature
   ```
   Then open a pull request into `main` on GitHub. CI runs the unit and system tests.

### What lives where

```
sandbox/
├── sandbox_workflow.ipynb          # copied to ~/.metatlas2-sandbox/ on each launch
├── inputs/                         # read directly from your clone by the notebook
│   ├── compounds.csv
│   ├── atlases/*.tsv
│   ├── msms_refs.tab
│   ├── modelseed.tsv               # copied into the sandbox DB folder on each launch
│   └── configs/                    # templates; ${...} placeholders are filled by the notebook
│       ├── atlases_config.yaml
│       ├── msms_refs_config.yaml
│       └── analysis_config.yaml    # atlas UIDs are filled in after add-atlases runs
├── prepare_sandbox_data_package.sh # (NERSC) build + publish the h5 package
└── subset_h5.py                    # shrinks h5 files to the atlas windows

~/.metatlas2-sandbox/               # METATLAS_DATA_DIR inside the sandbox
├── raw_data/jgi/20260101_JGI_XX_000000_SANDBOX_test_EXP000_HILICZ_TESTXXXX/*.h5   # from Zenodo
├── sandbox_workflow.ipynb
├── configs/                        # rendered configs (reset each launch)
├── databases/main_db/              # sandbox DB (reset each launch)
└── projects/targeted_outputs/      # outputs (reset each launch)
```

To force a fresh download of the package, delete `~/.metatlas2-sandbox/`.

### Updating the sandbox data package

On NERSC, from your branch:

```bash
export ZENODO_ACCESS_TOKEN=...      # Zenodo token with deposit access
sandbox/prepare_sandbox_data_package.sh                       # build + publish
sandbox/prepare_sandbox_data_package.sh /tmp/pkg --skip-zenodo-upload   # build only, to test
```

The script copies the runs listed at its top from the source production project, shrinks them
with `subset_h5.py` to the windows in `sandbox/inputs/atlases/*.tsv`, publishes a new Zenodo
version, and updates `ZENODO_SANDBOX_DATA_DOI` in `scripts/metatlas2.sh`. **Commit that change**
on your branch so others get the new package.

---

## Troubleshooting

### `METATLAS_DATA_DIR is not set`
Add `export METATLAS_DATA_DIR=/path/to/data` to `~/.bashrc` (or `~/.zshrc`) and re-source it.

### `uses 'tags', but local (non-NERSC) runs cannot access the centralized atlas sheet`
Replace `ATLAS.tags` with `ATLAS.uid` (see [Atlases](#4-atlases-use-uids-not-tags)).

### `Unknown owner '...'. Expected one of: ['jgi', 'egsb']`
`GENERAL.owner` (and the `raw_data/<owner>/` folder) must be `jgi` or `egsb`.

### `Raw data directory not found`
Check that your `.h5` files are at `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/`.

### `Main database not found`
The Zenodo download failed, or you used `--skip-db-update` with no DB present. Re-run without
`--skip-db-update`, or place `metatlas.duckdb` at `$METATLAS_DATA_DIR/databases/main_db/` yourself.

### Sandbox: `the sandbox runs your local clone ..., which is on 'main'`
Create a feature branch first: `git switch -c <branch-name>`.

### Notebook asks to select a kernel
The generated notebooks expect `metatlas2` (latest) or `metatlas2-<tag>`. Launch JupyterLab
with the same image you ran the workflow with (`metatlas2.sh --sandbox --image <tag>`).

### GUI not accessible at `localhost:8050`
- The GUI is started from the curation notebook, not by `metatlas2.sh run`.
- On a remote machine, make sure port forwarding is active.
- If 8050 is busy the app picks the next free port (8051–8069). Use the link the cell prints.
