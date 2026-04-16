# Initial Setup

Complete these steps **once** on any new machine or user account before running any metatlas2 workflow (adding compounds, adding atlases, or running a targeted analysis).

**Prerequisites:** `shifter` and `shifterimg` must be available on the host. On NERSC login/compute nodes both are pre-installed. `jupyter` must also be available on the host (it is used by `install_kernels.sh` only to register the kernel spec). All actual Python execution — both `metatlas2.sh` commands and JupyterLab notebook cells — runs inside the Shifter container image.

---

## 1. Clone the repository

```bash
git clone https://github.com/bkieft-usa/metatlas2.git ~/metatlas2
cd ~/metatlas2
```

The repository provides the `scripts/` directory and documentation. You do **not** need to create a virtualenv or install Python packages.

---

## 2. Set `METATLAS_DATA_DIR`

Define the base directory for all input data (raw files, main database, PubChem cache) in your shell profile so every tool picks it up automatically:

```bash
echo 'export METATLAS_DATA_DIR="/global/cfs/cdirs/metatlas/"' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. Add `scripts/` to your PATH

```bash
echo 'export PATH="${HOME}/metatlas2/scripts:${PATH}"' >> ~/.bashrc
source ~/.bashrc
```

After this, commands like `metatlas2 run ...`, `metatlas2 add-compounds ...`, and `metatlas2 add-atlases ...` are available from any directory.

---

## 4. Pull the container image

The image is public at `ghcr.io/bkieft-usa/metatlas2`. Pull it into the shifter cache:

```bash
pull_latest.sh
```

This runs `shifterimg pull` to load the image into shifter's local cache. To keep the cache current automatically, register it as a cron job:

```bash
*/5 * * * * ~/metatlas2/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1
```

---

## 5. Install Jupyter kernel specs

```bash
install_kernels.sh
```

This registers three kernel specs in JupyterLab:

| Kernel name | Image used | Source code |
|---|---|---|
| `metatlas2` | `latest` | Installed inside the image |
| `metatlas2-dev` | `latest` | Local repo's `metatlas2/` package mounted over the installed copy |
| `metatlas2-<tag>` | `<tag>` | Installed inside the pinned image — registered automatically on first use of `--image <tag>` |

---

## Setup is complete

You can now run any metatlas2 workflow. See the relevant doc for each command:

- [add_compounds_to_db.md](add_compounds_to_db.md) — add compounds to the main database
- [add_atlases_to_db.md](add_atlases_to_db.md) — add reference atlases to the main database
- [run_targeted_analysis.md](run_targeted_analysis.md) — run the full targeted analysis workflow

---

## Wrapper flags (all commands)

The `metatlas2` shell wrapper accepts two flags that apply to every subcommand. These are consumed by the wrapper and not forwarded to the Python module.

| Flag | Default | Description |
|---|---|---|
| `--image TAG` | `latest` | Use a specific image tag instead of the default. Overrides the `METATLAS2_IMAGE_TAG` environment variable. |
| `--dev` | `false` | Mount the local repository source tree over the installed package inside the container. Edits to the working tree take effect immediately without rebuilding the image. |

```bash
# Pin to a specific image tag
metatlas2 --image v1.2.3 run --config analysis.yaml --project MyProject

# Use local source edits (dev mode)
metatlas2 --dev run --config analysis.yaml --project MyProject
```
