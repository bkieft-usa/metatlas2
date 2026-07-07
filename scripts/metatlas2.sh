#!/bin/bash
# Host-side wrapper: runs metatlas2 repo commands inside a Shifter container.
#
# Usage:
#   metatlas2 [--image TAG] [--dev] run    --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] submit --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] add-compounds --config_path FILE
#   metatlas2 [--image TAG] [--dev] add-atlases   --config_path FILE
#   metatlas2 [--image TAG] [--dev] get-atlases   fetch  --atlas_uids UID1,UID2 [--output_path PATH]
#   metatlas2 [--image TAG] [--dev] get-atlases   query [--chromatography X] [--polarity X] [--analysis_type X] [--analysis_name X] [--created_by X]
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
    ZENODO_DOI="https://doi.org/10.5281/zenodo.21250667"
    TARBALL_NAME="metatlas2-dev-data.tar.gz"
    VERSION_FILE="${STANDALONE_DIR}/.zenodo_version"

    echo "=========================================="
    echo "Metatlas2 Standalone Development Mode"
    echo "=========================================="
    echo ""

    # Check for docker and docker compose
    if ! command -v docker &>/dev/null; then
        echo "Error: 'docker' is required but not found on PATH." >&2
        echo "Install Docker Desktop: https://docs.docker.com/get-docker/" >&2
        exit 1
    fi
    if ! docker compose version &>/dev/null; then
        echo "Error: 'docker compose' (v2) is required." >&2
        echo "Update Docker Desktop or install the compose plugin." >&2
        exit 1
    fi

    # ── Data download (unchanged from your original) ───────────────────────────
    NEEDS_DOWNLOAD=false
    if [[ ! -d "${STANDALONE_DIR}" ]]; then
        NEEDS_DOWNLOAD=true
    elif [[ "${UPDATE_DATA}" == "true" ]]; then
        NEEDS_DOWNLOAD=true
    elif [[ ! -f "${VERSION_FILE}" ]]; then
        NEEDS_DOWNLOAD=true
    else
        CURRENT_VERSION=$(cat "${VERSION_FILE}" 2>/dev/null || echo "unknown")
        [[ "${CURRENT_VERSION}" != "${ZENODO_DOI}" ]] && NEEDS_DOWNLOAD=true
    fi

    if [[ "${NEEDS_DOWNLOAD}" == "true" ]]; then
            echo "Downloading dev data from Zenodo..."
            if [[ -d "${STANDALONE_DIR}" ]]; then
                rm -rf "${STANDALONE_DIR}"
            fi
            mkdir -p "${STANDALONE_DIR}"

            docker run --rm \
                --entrypoint /bin/bash \
                -v "${STANDALONE_DIR}:/data" \
                -w /data \
                "${IMAGE_REPO}:${IMAGE_TAG}" \
                -c "/app/.venv/bin/zenodo_get -d '${ZENODO_DOI}' && \
                    tar -xzf '${TARBALL_NAME}' --strip-components=1 && \
                    rm '${TARBALL_NAME}'"

            echo "${ZENODO_DOI}" > "${VERSION_FILE}"
            echo "Dev data setup complete"
            echo ""
        fi

    # ── Clean previous outputs ─────────────────────────────────────────────────
    PROJECTS_DIR="${STANDALONE_DIR}/projects/targeted_outputs"
    if [[ -d "${PROJECTS_DIR}" ]]; then
        echo "Cleaning up previous workflow outputs..."
        rm -rf "${PROJECTS_DIR}"
        echo ""
    fi

    echo "Copying fresh notebook to ${STANDALONE_DIR}..."
    cp "${REPO_DIR}/local/standalone_dev_workflow.ipynb" "${STANDALONE_DIR}/standalone_dev_workflow.ipynb"
    echo ""

    # ── Pull latest image ──────────────────────────────────────────────────────
    echo "Pulling latest container image..."
    docker pull "${IMAGE_REPO}:${IMAGE_TAG}"
    echo ""

    # ── Write a fresh Jupyter config inside a temp dir ────────────────────────
    # This is baked into the container at runtime via JUPYTER_CONFIG_DIR,
    # guaranteeing nothing inside the image can override the bind address.
    JUPYTER_CONFIG_TMPDIR=$(mktemp -d)
    trap "rm -rf '${JUPYTER_CONFIG_TMPDIR}'" EXIT
    cat > "${JUPYTER_CONFIG_TMPDIR}/jupyter_server_config.py" <<'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.IdentityProvider.token = ''
c.ServerApp.password = ''
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
EOF

    # ── Launch via Docker Compose ──────────────────────────────────────────────
    echo "=========================================="
    echo "Launching JupyterLab..."
    echo ""
    echo "Data dir: ${STANDALONE_DIR}"
    echo "Repo dir: ${REPO_DIR} (live edits enabled)"
    echo ""
    echo "Open your browser at:"
    echo "   http://localhost:${STANDALONE_PORT:-8889}/lab"
    echo ""
    echo "Press Ctrl+C to stop"
    echo "=========================================="
    echo ""

    REPO_DIR="${REPO_DIR}" \
    STANDALONE_DIR="${STANDALONE_DIR}" \
    JUPYTER_CONFIG_TMPDIR="${JUPYTER_CONFIG_TMPDIR}" \
    METATLAS2_IMAGE_TAG="${IMAGE_TAG}" \
    STANDALONE_PORT="${STANDALONE_PORT:-8889}" \
        docker compose \
            -f "${REPO_DIR}/local/docker-compose.standalone.yml" \
            up --remove-orphans

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
    echo "Dev mode enabled: using local repo scripts at ${REPO_DIR}/scripts"
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

elif [[ "${SUBCOMMAND}" == "add-compounds" || "${SUBCOMMAND}" == "add-atlases" || "${SUBCOMMAND}" == "get-atlases" ]]; then
    if [[ "${SUBCOMMAND}" == "add-compounds" ]]; then
        PY_MODULE="metatlas2.add_compounds_to_db"
    elif [[ "${SUBCOMMAND}" == "add-atlases" ]]; then
        PY_MODULE="metatlas2.add_atlases_to_db"
    else
        PY_MODULE="metatlas2.get_atlases_from_db"
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