#!/bin/bash
# Host-side wrapper: runs metatlas2 repo commands inside a Shifter container.
#
# Usage:
#   metatlas2 [--image TAG] [--dev] run    --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] submit --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] add-compounds --config_path FILE
#   metatlas2 [--image TAG] [--dev] add-atlases   --config_path FILE
#
# Flags (consumed by this script, not forwarded to Python):
#   --image TAG   Use a specific image tag instead of the default (latest).
#                 Overrides the METATLAS2_IMAGE_TAG environment variable.
#   --dev         Mount the local repository source over the installed package,
#                 so edits to the working tree take effect immediately.
#
# Shifter automatically mounts all NERSC GPFS filesystems (home, CFS, scratch)
# inside the container, so no explicit volume flags are needed for data access.
#
# The wrapper handles the submit/sbatch split:
#   submit -> container generates the SLURM script -> host calls sbatch.

set -euo pipefail

IMAGE_REPO="ghcr.io/bkieft-usa/metatlas2"
IMAGE_TAG="${METATLAS2_IMAGE_TAG:-latest}"
DEV_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse wrapper-specific flags; pass everything else through to Python
# ---------------------------------------------------------------------------
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE_TAG="$2"; shift 2 ;;
        --image=*)      IMAGE_TAG="${1#*=}"; shift ;;
        --dev)          DEV_MODE=true; shift ;;
        *)              PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

IMAGE="docker:${IMAGE_REPO}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------
if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
    echo "Error: METATLAS_DATA_DIR is not set." >&2
    echo "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Common shifter flags.
# GPFS paths (home, CFS, scratch) are auto-mounted by shifter -- no -v needed.
# ---------------------------------------------------------------------------
SHIFTER_ARGS=(
    "--image=${IMAGE}"
    "--env=METATLAS2_IMAGE_TAG=${IMAGE_TAG}"
    "--env=METATLAS_DATA_DIR=${METATLAS_DATA_DIR}"
    "--env=HOME=${HOME}"
    "--env=JUPYTERHUB_SERVICE_PREFIX=${JUPYTERHUB_SERVICE_PREFIX:-/}"
    # metatlas2 has no [build-system] in pyproject.toml so it is not installed
    # as a wheel in the venv.  Set PYTHONPATH=/app so Python always finds
    # /app/metatlas2/ regardless of the working directory.
    "--env=PYTHONPATH=/app"
)

# Dev mode: put the local metatlas2/ package first on PYTHONPATH so edits
# take effect immediately.  GPFS is auto-mounted by shifter, so the repo is
# already visible inside the container at the same absolute path.
# Also keep /app as fallback so other metatlas2 subpackages still resolve.
if [[ "${DEV_MODE}" == "true" ]]; then
    SHIFTER_ARGS+=("--env=PYTHONPATH=${REPO_DIR}:/app")
fi

# ---------------------------------------------------------------------------
# Auto-install a Jupyter kernel spec for pinned image tags.
# The two default kernels (metatlas2, metatlas2-dev) reference :latest and
# stay valid across image updates -- they only need to be installed once.
# Pinned tags are registered on first use so the analyst never has to
# run install_kernels.sh --tag manually.
# ---------------------------------------------------------------------------
if [[ "${IMAGE_TAG}" != "latest" ]]; then
    KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/metatlas2-${IMAGE_TAG}"
    if [[ ! -d "${KERNEL_DIR}" ]]; then
        echo "Registering Jupyter kernel spec for ${IMAGE_TAG} ..."
        "${SCRIPT_DIR}/install_kernels.sh" --tag "${IMAGE_TAG}"
    fi
fi

# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------
SUBCOMMAND="${PASSTHROUGH_ARGS[0]:-}"

if [[ "${SUBCOMMAND}" == "submit" ]]; then
    # Generate the SLURM script inside the container (sbatch is not available
    # there), write it to a temp file on the shared /tmp, then submit from host.
    TMPSCRIPT="$(mktemp /tmp/metatlas2_XXXXXX.sh)"
    # shellcheck disable=SC2064
    trap "rm -f '${TMPSCRIPT}'" EXIT

    shifter "${SHIFTER_ARGS[@]}" --entrypoint \
        "${PASSTHROUGH_ARGS[@]}" --script-only --output "${TMPSCRIPT}"

    echo "Slurm script written to: ${TMPSCRIPT}"
    sbatch "${TMPSCRIPT}"

elif [[ "${SUBCOMMAND}" == "add-compounds" || "${SUBCOMMAND}" == "add-atlases" ]]; then
    if [[ "${SUBCOMMAND}" == "add-compounds" ]]; then
        PY_MODULE="metatlas2.add_compounds_to_db"
    else
        PY_MODULE="metatlas2.add_atlases_to_db"
    fi

    # Run python directly, bypassing the default entrypoint.
    # GPFS is auto-mounted read-write by shifter so metatlas.duckdb is writable.
    shifter "${SHIFTER_ARGS[@]}" \
        /app/.venv/bin/python -m "${PY_MODULE}" "${PASSTHROUGH_ARGS[@]:1}"

else
    # run (or any other subcommand): use the container's default entrypoint.
    LOG_TO_STDOUT=false
    for arg in "${PASSTHROUGH_ARGS[@]}"; do
        [[ "$arg" == "--log-to-stdout" ]] && LOG_TO_STDOUT=true && break
    done
    if [[ "${LOG_TO_STDOUT}" == "false" && "${SUBCOMMAND}" == "run" ]]; then
        if [[ "${DEV_MODE}" == "true" ]]; then
            echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=dev)..."
        else
            echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=prod)..."
        fi
    fi

    shifter "${SHIFTER_ARGS[@]}" --entrypoint \
        "${PASSTHROUGH_ARGS[@]}"
fi