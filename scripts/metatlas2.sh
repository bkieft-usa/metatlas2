#!/bin/bash
# Host-side wrapper: runs metatlas2 repo commands inside a container.
# Supports both Shifter (NERSC/HPC) and Docker (local/other HPC) runtimes,
# detected automatically via PATH lookup.
#
# Usage:
#   metatlas2 [--image TAG] [--dev] run    --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] submit --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] add-compounds  --config_path FILE
#   metatlas2 [--image TAG] [--dev] add-atlases    --config_path FILE
#   metatlas2 [--image TAG] [--dev] add-msms-refs  --config_path FILE
#   metatlas2 [--image TAG] [--dev] get-atlases    fetch  --atlas_uids UID1,UID2 [--output_path PATH]
#   metatlas2 [--image TAG] [--dev] get-msms-refs  [--output_path PATH] [--tab-out] [--inchikeys K1,K2] [--database_filter DB] [--polarity POS]
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
# Runtime detection (automatic, no flag needed):
#   - If 'shifter' is found on PATH, Shifter is used (NERSC/Perlmutter).
#   - Otherwise, if 'docker' is found on PATH, Docker is used.
#   - If neither is found, the script exits with an error.
#
# Docker-specific requirements:
#   - METATLAS_DATA_DIR must be set and the config file must reside somewhere
#     under that directory (e.g. $METATLAS_DATA_DIR/configs/my_config.yaml).
#     This ensures the config is accessible inside the container via the single
#     volume mount of $METATLAS_DATA_DIR.
#   - On first run (or when the database version changes), the main metatlas
#     DuckDB is downloaded automatically from Zenodo into
#     $METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb.
#     To publish a new database version, run scripts/upload_main_db_to_zenodo.sh
#     on NERSC and update ZENODO_MAIN_DB_DOI below.

set -euo pipefail

# Versioned Zenodo DOI for the main metatlas DuckDB.
# Update this value after running scripts/upload_main_db_to_zenodo.sh on NERSC.
# Non-NERSC (Docker) runs will automatically download the DB if the local
# version stamp does not match this DOI.
ZENODO_MAIN_DB_DOI="https://doi.org/10.5281/zenodo.22968291"

IMAGE_REPO="ghcr.io/bkieft-usa/metatlas2"
IMAGE_TAG="${METATLAS2_IMAGE_TAG:-latest}"
DEV_MODE=false
STANDALONE_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

# Runtime detection: shifter (NERSC) takes priority, then docker.
if command -v shifter &>/dev/null; then
    RUNTIME="shifter"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
else
    echo "Error: neither 'shifter' nor 'docker' found on PATH." >&2
    echo "Install Docker (https://docs.docker.com/get-docker/) or run on a system with Shifter." >&2
    exit 1
fi

IMAGE_FULL="${IMAGE_REPO}:${IMAGE_TAG}"

# Validate required environment variables (skip for standalone mode)
if [[ "${STANDALONE_MODE}" == "false" ]]; then
    if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
        echo "Error: METATLAS_DATA_DIR is not set." >&2
        echo "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc (or ~/.bash_profile on macOS) and re-source it." >&2
        exit 1
    fi
fi

EXTRA_CONFIG_MOUNT=""
if [[ "${RUNTIME}" == "docker" && "${STANDALONE_MODE}" == "false" ]]; then
    CONFIG_PATH=""
    for i in "${!PASSTHROUGH_ARGS[@]}"; do
        if [[ "${PASSTHROUGH_ARGS[$i]}" == "--config" ]]; then
            CONFIG_PATH="${PASSTHROUGH_ARGS[$i+1]:-}"
            break
        fi
    done

    if [[ -n "${CONFIG_PATH}" ]]; then
        # Resolve to absolute path
        CONFIG_ABS="$(cd "$(dirname "${CONFIG_PATH}")" 2>/dev/null && pwd)/$(basename "${CONFIG_PATH}")" || CONFIG_ABS="${CONFIG_PATH}"
        DATA_ABS="$(cd "${METATLAS_DATA_DIR}" && pwd)"
        if [[ "${CONFIG_ABS}" != "${DATA_ABS}"* ]]; then
            # Config is outside $METATLAS_DATA_DIR — add a read-only mount of its directory.
            CONFIG_DIR="$(dirname "${CONFIG_ABS}")"
            EXTRA_CONFIG_MOUNT="${CONFIG_DIR}"
            echo "Note: config file is outside \$METATLAS_DATA_DIR; adding read-only mount for ${CONFIG_DIR}"
        fi
    fi
fi

# Standalone mode setup
if [[ "${STANDALONE_MODE}" == "true" ]]; then
    STANDALONE_DIR="${HOME}/.metatlas2-dev"
    ZENODO_DOI="https://doi.org/10.5281/zenodo.22817689"
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

    # Data download
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
                "${IMAGE_FULL}" \
                -c "/app/.venv/bin/zenodo_get -d '${ZENODO_DOI}' && \
                    tar -xzf '${TARBALL_NAME}' --strip-components=1 && \
                    rm '${TARBALL_NAME}'"

            echo "${ZENODO_DOI}" > "${VERSION_FILE}"
            echo "Dev data setup complete"
            echo ""
        fi

    # Clean previous outputs
    PROJECTS_DIR="${STANDALONE_DIR}/projects/targeted_outputs"
    if [[ -d "${PROJECTS_DIR}" ]]; then
        echo "Cleaning up previous workflow outputs..."
        rm -rf "${PROJECTS_DIR}"
        echo ""
    fi

    echo "Copying fresh notebook to ${STANDALONE_DIR}..."
    mkdir -p "${STANDALONE_DIR}/notebooks"
    cp "${REPO_DIR}/local/local_workflow.ipynb" "${STANDALONE_DIR}/notebooks/local_workflow.ipynb"
    cp "${REPO_DIR}/local/parquet_query.ipynb" "${STANDALONE_DIR}/notebooks/parquet_query.ipynb"
    echo ""

    # Pull latest image
    echo "Pulling latest container image..."
    docker pull "${IMAGE_FULL}"
    echo ""

    # Write a fresh Jupyter config inside a temp dir
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

    # Launch via Docker Compose
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

SUBCOMMAND="${PASSTHROUGH_ARGS[0]:-}"

if [[ "${RUNTIME}" == "shifter" ]]; then
    # Shifter (NERSC/Perlmutter): global filesystems are auto-mounted.

    SHIFTER_ARGS=(
        "--image=docker:${IMAGE_FULL}"
        "--env=METATLAS2_IMAGE_TAG=${IMAGE_TAG}"
        "--env=METATLAS_DATA_DIR=${METATLAS_DATA_DIR}"
        "--env=HOME=${HOME}"
        "--env=JUPYTERHUB_SERVICE_PREFIX=${JUPYTERHUB_SERVICE_PREFIX:-/}"
        "--env=PYTHONPATH=/app"
    )

    if [[ "${DEV_MODE}" == "true" ]]; then
        SHIFTER_ARGS+=("--env=PYTHONPATH=${REPO_DIR}:/app")
        echo "Dev mode enabled: using local repo scripts at ${REPO_DIR}/scripts"
    fi

    if [[ "${IMAGE_TAG}" != "latest" ]]; then
        KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/metatlas2-${IMAGE_TAG}"
        if [[ ! -d "${KERNEL_DIR}" ]]; then
            echo "Registering Jupyter kernel spec for ${IMAGE_TAG} ..."
            "${SCRIPT_DIR}/install_kernels.sh" --tag "${IMAGE_TAG}"
        fi
    fi

    if [[ "${SUBCOMMAND}" == "submit" ]]; then
        TMPSCRIPT="$(mktemp /tmp/metatlas2_XXXXXX.sh)"
        # shellcheck disable=SC2064
        trap "rm -f '${TMPSCRIPT}'" EXIT

        shifter "${SHIFTER_ARGS[@]}" --entrypoint \
            "${PASSTHROUGH_ARGS[@]}" --script-only --output "${TMPSCRIPT}"

        sbatch "${TMPSCRIPT}"

    elif [[ "${SUBCOMMAND}" == "add-compounds" || "${SUBCOMMAND}" == "add-atlases" || "${SUBCOMMAND}" == "add-msms-refs" || "${SUBCOMMAND}" == "get-atlases" || "${SUBCOMMAND}" == "get-msms-refs" ]]; then
        if [[ "${SUBCOMMAND}" == "add-compounds" ]]; then
            PY_MODULE="metatlas2.add_compounds_to_db"
        elif [[ "${SUBCOMMAND}" == "add-atlases" ]]; then
            PY_MODULE="metatlas2.add_atlases_to_db"
        elif [[ "${SUBCOMMAND}" == "add-msms-refs" ]]; then
            PY_MODULE="metatlas2.add_msms_refs_to_db"
        elif [[ "${SUBCOMMAND}" == "get-msms-refs" ]]; then
            PY_MODULE="metatlas2.get_msms_refs_from_db"
        else
            PY_MODULE="metatlas2.get_atlases_from_db"
        fi

        if [[ "${DEV_MODE}" == "true" ]]; then
            echo "Launching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=dev)..."
        else
            echo "Launching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=prod)..."
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
                echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=dev, runtime=shifter)..."
            else
                echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=prod, runtime=shifter)..."
            fi
        fi

        shifter "${SHIFTER_ARGS[@]}" --entrypoint \
            "${PASSTHROUGH_ARGS[@]}"
    fi

else
    # Docker (local macOS / non-NERSC HPC): explicit volume mounts needed.

    # Ensure the main metatlas DuckDB is present and up-to-date.
    # Download from Zenodo if missing or if the local version stamp differs
    # from ZENODO_MAIN_DB_DOI.  Skipped when ZENODO_MAIN_DB_DOI is unset
    # (e.g. during development before the first Zenodo upload).
    if [[ -n "${ZENODO_MAIN_DB_DOI}" ]]; then
        MAIN_DB_DIR="${METATLAS_DATA_DIR}/databases/main_db"
        MAIN_DB_PATH="${MAIN_DB_DIR}/metatlas.duckdb"
        MAIN_DB_VERSION_FILE="${MAIN_DB_DIR}/.zenodo_version"

        NEEDS_DB_DOWNLOAD=false
        if [[ ! -f "${MAIN_DB_PATH}" ]]; then
            NEEDS_DB_DOWNLOAD=true
        elif [[ ! -f "${MAIN_DB_VERSION_FILE}" ]]; then
            NEEDS_DB_DOWNLOAD=true
        else
            CURRENT_DB_VERSION=$(cat "${MAIN_DB_VERSION_FILE}" 2>/dev/null || echo "unknown")
            [[ "${CURRENT_DB_VERSION}" != "${ZENODO_MAIN_DB_DOI}" ]] && NEEDS_DB_DOWNLOAD=true
        fi

        if [[ "${NEEDS_DB_DOWNLOAD}" == "true" ]]; then
            echo "Downloading metatlas main database from Zenodo..."
            echo "  DOI: ${ZENODO_MAIN_DB_DOI}"
            echo "  Destination: ${MAIN_DB_PATH}"
            mkdir -p "${MAIN_DB_DIR}"

            docker run --rm \
                --entrypoint /bin/bash \
                -v "${MAIN_DB_DIR}:/db" \
                -w /db \
                "${IMAGE_FULL}" \
                -c "/app/.venv/bin/zenodo_get -d '${ZENODO_MAIN_DB_DOI}'"

            if [[ ! -f "${MAIN_DB_PATH}" ]]; then
                echo "Error: Download completed but metatlas.duckdb not found in ${MAIN_DB_DIR}." >&2
                echo "Check that the Zenodo deposit contains a file named 'metatlas.duckdb'." >&2
                exit 1
            fi

            echo "${ZENODO_MAIN_DB_DOI}" > "${MAIN_DB_VERSION_FILE}"
            echo "Main database download complete."
            echo ""
        else
            echo "Main database is up-to-date (${MAIN_DB_PATH})."
        fi
    fi

    DOCKER_ARGS=(
        "--rm"
        "-e" "METATLAS2_IMAGE_TAG=${IMAGE_TAG}"
        "-e" "METATLAS_DATA_DIR=${METATLAS_DATA_DIR}"
        "-e" "HOME=${HOME}"
        "-e" "USER=${USER:-$(id -un)}"
        "-e" "PYTHONPATH=/app"
        "-v" "${METATLAS_DATA_DIR}:${METATLAS_DATA_DIR}"
        "--user" "$(id -u):$(id -g)"
    )

    # Only forward JUPYTERHUB_SERVICE_PREFIX when it is actually set (i.e. on JupyterHub).
    if [[ -n "${JUPYTERHUB_SERVICE_PREFIX:-}" ]]; then
        DOCKER_ARGS+=("-e" "JUPYTERHUB_SERVICE_PREFIX=${JUPYTERHUB_SERVICE_PREFIX}")
    fi

    # Mount the config directory if it lives outside $METATLAS_DATA_DIR.
    if [[ -n "${EXTRA_CONFIG_MOUNT}" ]]; then
        DOCKER_ARGS+=("-v" "${EXTRA_CONFIG_MOUNT}:${EXTRA_CONFIG_MOUNT}:ro")
    fi

    if [[ "${DEV_MODE}" == "true" ]]; then
        DOCKER_ARGS+=("-e" "PYTHONPATH=${REPO_DIR}:/app")
        DOCKER_ARGS+=("-v" "${REPO_DIR}:${REPO_DIR}:ro")
        echo "Dev mode enabled: using local repo at ${REPO_DIR}"
    fi

    if [[ "${SUBCOMMAND}" == "submit" ]]; then
        echo "Error: 'submit' (SLURM/sbatch) is only supported on NERSC/Shifter systems." >&2
        exit 1

    elif [[ "${SUBCOMMAND}" == "add-compounds" || "${SUBCOMMAND}" == "add-atlases" || "${SUBCOMMAND}" == "add-msms-refs" || "${SUBCOMMAND}" == "get-atlases" || "${SUBCOMMAND}" == "get-msms-refs" ]]; then
        if [[ "${SUBCOMMAND}" == "add-compounds" ]]; then
            PY_MODULE="metatlas2.add_compounds_to_db"
        elif [[ "${SUBCOMMAND}" == "add-atlases" ]]; then
            PY_MODULE="metatlas2.add_atlases_to_db"
        elif [[ "${SUBCOMMAND}" == "add-msms-refs" ]]; then
            PY_MODULE="metatlas2.add_msms_refs_to_db"
        elif [[ "${SUBCOMMAND}" == "get-msms-refs" ]]; then
            PY_MODULE="metatlas2.get_msms_refs_from_db"
        else
            PY_MODULE="metatlas2.get_atlases_from_db"
        fi

        if [[ "${DEV_MODE}" == "true" ]]; then
            echo "Launching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=dev, runtime=docker)..."
        else
            echo "Launching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=prod, runtime=docker)..."
        fi

        docker run "${DOCKER_ARGS[@]}" \
            "${IMAGE_FULL}" \
            /app/.venv/bin/python -m "${PY_MODULE}" "${PASSTHROUGH_ARGS[@]:1}"

    else # run main targeted pipeline
        LOG_TO_STDOUT=false
        for arg in "${PASSTHROUGH_ARGS[@]}"; do
            [[ "$arg" == "--log-to-stdout" ]] && LOG_TO_STDOUT=true && break
        done
        if [[ "${LOG_TO_STDOUT}" == "false" && "${SUBCOMMAND}" == "run" ]]; then
            if [[ "${DEV_MODE}" == "true" ]]; then
                echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=dev, runtime=docker)..."
            else
                echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=prod, runtime=docker)..."
            fi
        fi

        docker run "${DOCKER_ARGS[@]}" \
            -p "127.0.0.1:8050-8069:8050-8069" \
            --entrypoint /app/.venv/bin/python \
            "${IMAGE_FULL}" \
            -m metatlas2.run_targeted_analysis \
            "${PASSTHROUGH_ARGS[@]}"
    fi
fi