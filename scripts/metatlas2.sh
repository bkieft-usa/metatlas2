#!/bin/bash
# Host-side wrapper: runs metatlas2 repo commands inside a Shifter container.
#
# Usage:
#   metatlas2 [--image TAG] [--dev] run    --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] submit --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] add-compounds --config_path FILE
#   metatlas2 [--image TAG] [--dev] add-atlases   --config_path FILE
#   metatlas2 [--image TAG] [--standalone] [--update-data]
#
# Flags (consumed by this script, not forwarded to Python):
#   --image TAG   Use a specific image tag instead of the default (latest).
#                 Overrides the METATLAS2_IMAGE_TAG environment variable.
#   --dev         Mount the local repository source over the installed package,
#                 so edits to the working tree take effect immediately.
#   --standalone  Launch standalone dev environment with JupyterLab notebook.
#                 Downloads dev data if needed to ~/.metatlas2-dev/.
#   --update-data Force re-download of dev data (use with --standalone).
#                 Useful when new Zenodo versions are published.
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
STANDALONE_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Parse wrapper-specific flags; pass everything else through to Python

PASSTHROUGH_ARGS=()
UPDATE_DATA=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE_TAG="$2"; shift 2 ;;
        --image=*)      IMAGE_TAG="${1#*=}"; shift ;;
        --dev)          DEV_MODE=true; shift ;;
        --standalone)   STANDALONE_MODE=true; shift ;;
        --update-data)  UPDATE_DATA=true; shift ;;
        *)              PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

IMAGE="docker:${IMAGE_REPO}:${IMAGE_TAG}"


# Validate required environment variables (skip for standalone mode)
if [[ "${STANDALONE_MODE}" == "false" ]]; then
    if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
        echo "Error: METATLAS_DATA_DIR is not set." >&2
        echo "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it." >&2
        exit 1
    fi
fi


# Standalone mode setup
if [[ "${STANDALONE_MODE}" == "true" ]]; then
    STANDALONE_DIR="${HOME}/.metatlas2-dev"
    ZENODO_DOI="https://doi.org/10.5281/zenodo.20090323"
    TARBALL_NAME="metatlas2-dev-data.tar.gz"
    NOTEBOOK_PATH="/repo/notebooks/standalone_dev_workflow.ipynb"
    VERSION_FILE="${STANDALONE_DIR}/.zenodo_version"
    
    echo "=========================================="
    echo "Metatlas2 Standalone Development Mode"
    echo "=========================================="
    echo ""
    
    # Check if dev data exists and is current version
    NEEDS_DOWNLOAD=false
    if [[ ! -d "${STANDALONE_DIR}" ]]; then
        echo "Dev environment not found at ${STANDALONE_DIR}"
        NEEDS_DOWNLOAD=true
    elif [[ "${UPDATE_DATA}" == "true" ]]; then
        echo "Force update requested (--update-data)"
        NEEDS_DOWNLOAD=true
    elif [[ ! -f "${VERSION_FILE}" ]]; then
        echo "Version file not found, data may be outdated"
        NEEDS_DOWNLOAD=true
    else
        CURRENT_VERSION=$(cat "${VERSION_FILE}" 2>/dev/null || echo "unknown")
        if [[ "${CURRENT_VERSION}" != "${ZENODO_DOI}" ]]; then
            echo "Data version mismatch:"
            echo "  Current: ${CURRENT_VERSION}"
            echo "  Expected: ${ZENODO_DOI}"
            NEEDS_DOWNLOAD=true
        else
            echo "Dev environment found at ${STANDALONE_DIR} (up to date)"
        fi
    fi
    
    if [[ "${NEEDS_DOWNLOAD}" == "true" ]]; then
        echo "Downloading dev data package from Zenodo..."
        echo "  DOI: ${ZENODO_DOI}"
        echo ""
        
        # Remove old data if it exists
        if [[ -d "${STANDALONE_DIR}" ]]; then
            echo "Removing old data..."
            rm -rf "${STANDALONE_DIR}"
        fi
        
        # Download and extract
        TMPDIR=$(mktemp -d)
        trap "rm -rf '${TMPDIR}'" EXIT
        
        echo "Downloading (~500MB, this may take a few minutes)..."
        cd "${TMPDIR}"
        if ! uvx zenodo_get -d "${ZENODO_DOI}"; then
            echo "Error: Failed to download dev data from Zenodo" >&2
            echo "Please manually download from: ${ZENODO_DOI}" >&2
            echo "Or install zenodo_get and try again: pip install zenodo-get" >&2
            exit 1
        fi
        
        # Find the downloaded tarball (zenodo_get downloads with original filename)
        if [[ ! -f "${TMPDIR}/${TARBALL_NAME}" ]]; then
            echo "Error: Expected file ${TARBALL_NAME} not found after download" >&2
            echo "Available files:" >&2
            ls -lh "${TMPDIR}" >&2
            exit 1
        fi
        
        echo "Extracting to ${STANDALONE_DIR}..."
        mkdir -p "${STANDALONE_DIR}"
        if ! tar -xzf "${TMPDIR}/${TARBALL_NAME}" -C "${STANDALONE_DIR}" --strip-components=1; then
            echo "Error: Failed to extract dev data" >&2
            exit 1
        fi
        
        # Save version file
        echo "${ZENODO_DOI}" > "${VERSION_FILE}"
        
        echo "Dev environment setup complete"
        echo ""
    else
        echo ""
    fi
    
    # Override environment for standalone mode
    export METATLAS_DATA_DIR="${STANDALONE_DIR}"
    export METATLAS2_STANDALONE="true"
    
    # Clean up any existing outputs from previous runs
    PROJECTS_DIR="${STANDALONE_DIR}/projects/targeted_outputs"
    if [[ -d "${PROJECTS_DIR}" ]]; then
        echo "Cleaning up previous workflow outputs..."
        rm -rf "${PROJECTS_DIR}"
        echo "  Removed: ${PROJECTS_DIR}"
        echo ""
    fi
    
    # Authenticate with GitHub Container Registry if needed
    echo "Checking container image authentication..."
    
    # Try pulling first to see if already authenticated
    if docker pull "${IMAGE_REPO}:${IMAGE_TAG}" >/dev/null 2>&1; then
        echo "  Image accessible (already authenticated or public)"
    else
        GHCR_TOKEN="${GHCR_TOKEN:-${GITHUB_TOKEN:-}}"
        
        if [[ -n "${GHCR_TOKEN}" ]]; then
            echo "  Using token for automatic authentication..."
            if echo "${GHCR_TOKEN}" | docker login ghcr.io -u "$(whoami)" --password-stdin >/dev/null 2>&1; then
                echo "  Successfully authenticated"
            else
                echo "  Warning: Automatic authentication failed, but continuing..." >&2
                echo "  (Image may already be accessible)" >&2
            fi
        else
            echo "  Warning: No authentication token found, but continuing..." >&2
            echo "  (Image may be public or you're already authenticated)" >&2
        fi
    fi
    echo ""
    
    # Pull the latest image to ensure we have updates (skip if just pulled above)
    echo "Pulling latest container image..."
    if ! docker pull "${IMAGE_REPO}:${IMAGE_TAG}" 2>&1 | grep -q "up to date"; then
        echo "  Image updated"
    fi
    echo ""
    
    # Launch JupyterLab with the standalone notebook
    echo "Launching JupyterLab in Docker container..."
    echo ""
    echo "Opening standalone workflow notebook:"
    echo "   ${NOTEBOOK_PATH}"
    echo ""
    echo "JupyterLab will open in your browser at:"
    echo "   http://localhost:8888"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    echo "=========================================="
    
    # Run JupyterLab in Docker
    # Mount local repo to /repo (not /app) to avoid shadowing the venv in /app/.venv
    # Use PYTHONPATH to prioritize local repo code over installed package
    # Set JUPYTERHUB_SERVICE_PREFIX="/" for local JupyterLab proxy URLs
    # Set working directory to STANDALONE_DIR so relative paths in configs resolve correctly
    # Set USER env var for output path construction
    docker run --rm -it \
        --entrypoint /bin/bash \
        -p 8888:8888 \
        -v "${STANDALONE_DIR}:${STANDALONE_DIR}" \
        -v "${REPO_DIR}:/repo" \
        -e METATLAS_DATA_DIR="${STANDALONE_DIR}" \
        -e METATLAS2_STANDALONE="true" \
        -e JUPYTERHUB_SERVICE_PREFIX="/" \
        -e USER="standalone" \
        -e PYTHONPATH="/repo:/app" \
        -w "${STANDALONE_DIR}" \
        "${IMAGE_REPO}:${IMAGE_TAG}" \
        -c "/app/.venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' /repo/notebooks/standalone_dev_workflow.ipynb"
    
    exit 0
fi


# Common shifter flags.
# GPFS paths (home, CFS, scratch) are auto-mounted by shifter -- no -v needed.
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


# Auto-install a Jupyter kernel spec for pinned image tags.
# The two default kernels (metatlas2, metatlas2-dev) reference :latest and
# stay valid across image updates -- they only need to be installed once.
# Pinned tags are registered on first use so the analyst never has to
# run install_kernels.sh --tag manually.
if [[ "${IMAGE_TAG}" != "latest" ]]; then
    KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/metatlas2-${IMAGE_TAG}"
    if [[ ! -d "${KERNEL_DIR}" ]]; then
        echo "Registering Jupyter kernel spec for ${IMAGE_TAG} ..."
        "${SCRIPT_DIR}/install_kernels.sh" --tag "${IMAGE_TAG}"
    fi
fi


# Subcommand dispatch

SUBCOMMAND="${PASSTHROUGH_ARGS[0]:-}"

if [[ "${SUBCOMMAND}" == "submit" ]]; then
    # Generate the SLURM script inside the container (sbatch is not available
    # there), write it to a temp file on the shared /tmp, then submit from host.
    TMPSCRIPT="$(mktemp /tmp/metatlas2_XXXXXX.sh)"
    # shellcheck disable=SC2064
    trap "rm -f '${TMPSCRIPT}'" EXIT

    shifter "${SHIFTER_ARGS[@]}" --entrypoint \
        "${PASSTHROUGH_ARGS[@]}" --script-only --output "${TMPSCRIPT}"

    sbatch "${TMPSCRIPT}"

elif [[ "${SUBCOMMAND}" == "add-compounds" || "${SUBCOMMAND}" == "add-atlases" ]]; then
    if [[ "${SUBCOMMAND}" == "add-compounds" ]]; then
        PY_MODULE="metatlas2.add_compounds_to_db"
    else
        PY_MODULE="metatlas2.add_atlases_to_db"
    fi

    if [[ "${DEV_MODE}" == "true" ]]; then
        echo "Lauching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=dev)..."
    else
        echo "Lauching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=prod)..."
    fi

    shifter "${SHIFTER_ARGS[@]}" \
        /app/.venv/bin/python -m "${PY_MODULE}" "${PASSTHROUGH_ARGS[@]:1}"

else # run main targeted pipeline
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