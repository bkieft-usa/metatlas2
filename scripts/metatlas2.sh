#!/bin/bash
# Host-side wrapper: runs metatlas2 repo commands inside a container.
# Supports both Shifter (NERSC/HPC) and Docker (local/other HPC) runtimes,
# detected automatically via PATH lookup.
#
# Usage:
#   metatlas2 [--image TAG] [--dev] run    --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] submit --config FILE --project NAME ...
#   metatlas2 [--image TAG] [--dev] jupyter [--port PORT] [--config_path FILE]
#   metatlas2 [--image TAG] [--dev] add-compounds --input_csv FILE [--overwrite_db]
#   metatlas2 [--image TAG] [--dev] add-atlases --config_path FILE [--custom]
#   metatlas2 [--image TAG] [--dev] add-msms-refs --config_path FILE
#   metatlas2 [--image TAG] [--dev] standards-annotation --config_path FILE
#   metatlas2 [--image TAG] [--dev] reverse-rt-align --config_path FILE --annotation_timestamp TIMESTAMP [--skip_rt_alignment]
#   metatlas2 [--image TAG] [--dev] atlas-msms-deposit --config_path FILE --annotation_timestamp TIMESTAMP --alignment_timestamp TIMESTAMP [--dry_run] [--force_add] [--skip_atlas_upload] [--skip_msms_refs_deposit]
#   metatlas2 [--image TAG] [--dev] get-atlases fetch --atlas_uids UID1,UID2 [--output_path PATH]
#   metatlas2 [--image TAG] [--dev] get-atlases query [--chromatography X] [--polarity X] [--analysis_type X] [--analysis_name X] [--created_by X]
#   metatlas2 [--image TAG] [--dev] get-db-table --table-name NAME [--output_path PATH] [--database_path PATH]
#   metatlas2 [--image TAG] [--dev] update-tags {add|remove|retag} --input-csv FILE
#   metatlas2 [--image TAG] [--dev] update-tags retrieve
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
#   --port PORT   JupyterLab port for the 'jupyter' subcommand (default: 8889).
#                 Can also be set via METATLAS2_JUPYTER_PORT env var.
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

ZENODO_MAIN_DB_DOI="https://doi.org/10.5281/zenodo.22968291"
ZENODO_DEV_DATA_DOI="https://doi.org/10.5281/zenodo.23697968"

IMAGE_REPO="ghcr.io/bkieft-usa/metatlas2"
IMAGE_TAG="${METATLAS2_IMAGE_TAG:-latest}"
DEV_MODE=false
STANDALONE_MODE=false
UPDATE_DATA=false
JUPYTER_PORT="${METATLAS2_JUPYTER_PORT:-8889}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PASSTHROUGH_ARGS=()

# ---------------------------------------------------------------------------
# Helpers (defined once, shared by every runtime / subcommand)
# ---------------------------------------------------------------------------
die() { printf '%s\n' "$@" >&2; exit 1; }

# _parse_arg --key ARGS...  -> value of "--key VAL" or "--key=VAL" (empty if absent)
_parse_arg() {
    local key="$1" val=""; shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            "${key}="*) val="${1#*=}"; break ;;
            "${key}")   val="${2:-}"; break ;;
        esac
        shift
    done
    echo "${val}"
}

# _has_flag --flag ARGS...  -> success if the flag is present
_has_flag() {
    local flag="$1" arg; shift
    for arg in "$@"; do [[ "${arg}" == "${flag}" ]] && return 0; done
    return 1
}

# _py_str VAL -> Python literal: None if VAL is empty, else 'VAL'
_py_str() { if [[ -n "$1" ]]; then echo "'$1'"; else echo "None"; fi; }

# _stamp_stale STAMP_FILE DOI -> success if stamp is missing or differs from DOI
_stamp_stale() {
    [[ ! -f "$1" ]] || [[ "$(cat "$1" 2>/dev/null || echo unknown)" != "$2" ]]
}

# _zenodo_get DIR DOI [FOLLOW_UP_CMD] -> download a Zenodo record into DIR via the image
_zenodo_get() {
    docker run --rm \
        --entrypoint /bin/bash \
        -v "$1:/data" \
        -w /data \
        "${IMAGE_FULL}" \
        -c "/app/.venv/bin/zenodo_get -d '$2'${3:+ && $3}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)        IMAGE_TAG="$2"; shift 2 ;;
        --image=*)      IMAGE_TAG="${1#*=}"; shift ;;
        --dev)          DEV_MODE=true; shift ;;
        --standalone)   STANDALONE_MODE=true; shift ;;
        --update-data)  UPDATE_DATA=true; shift ;;
        --port)         JUPYTER_PORT="$2"; shift 2 ;;
        --port=*)       JUPYTER_PORT="${1#*=}"; shift ;;
        *)              PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

SUBCOMMAND="${PASSTHROUGH_ARGS[0]:-}"
SUB_ARGS=("${PASSTHROUGH_ARGS[@]:1}")
IMAGE_FULL="${IMAGE_REPO}:${IMAGE_TAG}"
MODE="prod"; [[ "${DEV_MODE}" == "true" ]] && MODE="dev"

# Runtime detection: shifter (NERSC) takes priority, then docker.
if command -v shifter &>/dev/null; then
    RUNTIME="shifter"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
else
    die "Error: neither 'shifter' nor 'docker' found on PATH." \
        "Install Docker (https://docs.docker.com/get-docker/) or run on a system with Shifter."
fi

# ---------------------------------------------------------------------------
# Standalone mode (always Docker + Compose)
# ---------------------------------------------------------------------------
if [[ "${STANDALONE_MODE}" == "true" ]]; then
    STANDALONE_DIR="${HOME}/.metatlas2-dev"
    STANDALONE_PORT="${STANDALONE_PORT:-8889}"
    TARBALL_NAME="metatlas2-dev-data.tar.gz"
    VERSION_FILE="${STANDALONE_DIR}/.zenodo_version"

    echo "=========================================="
    echo "Metatlas2 Standalone Development Mode"
    echo "=========================================="
    echo ""

    command -v docker &>/dev/null || die \
        "Error: 'docker' is required but not found on PATH." \
        "Install Docker Desktop: https://docs.docker.com/get-docker/"
    docker compose version &>/dev/null || die \
        "Error: 'docker compose' (v2) is required." \
        "Update Docker Desktop or install the compose plugin."

    if [[ ! -d "${STANDALONE_DIR}" || "${UPDATE_DATA}" == "true" ]] \
        || _stamp_stale "${VERSION_FILE}" "${ZENODO_DEV_DATA_DOI}"; then
        echo "Downloading dev data from Zenodo..."
        rm -rf "${STANDALONE_DIR}"
        mkdir -p "${STANDALONE_DIR}"
        _zenodo_get "${STANDALONE_DIR}" "${ZENODO_DEV_DATA_DOI}" \
            "tar -xzf '${TARBALL_NAME}' --strip-components=1 && rm '${TARBALL_NAME}'"
        echo "${ZENODO_DEV_DATA_DOI}" > "${VERSION_FILE}"
        echo "Dev data setup complete"
        echo ""
    fi

    PROJECTS_DIR="${STANDALONE_DIR}/projects/targeted_outputs"
    if [[ -d "${PROJECTS_DIR}" ]]; then
        echo "Cleaning up previous workflow outputs..."
        rm -rf "${PROJECTS_DIR}"
        echo ""
    fi

    echo "Copying fresh notebook to ${STANDALONE_DIR}..."
    mkdir -p "${STANDALONE_DIR}/notebooks"
    cp "${REPO_DIR}/local/local_workflow.ipynb" "${REPO_DIR}/local/parquet_query.ipynb" \
        "${STANDALONE_DIR}/notebooks/"
    echo ""

    echo "Pulling latest container image..."
    docker pull "${IMAGE_FULL}"
    echo ""

    # Fresh Jupyter config in a temp dir
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

    echo "=========================================="
    echo "Launching JupyterLab..."
    echo ""
    echo "Data dir: ${STANDALONE_DIR}"
    echo "Repo dir: ${REPO_DIR} (live edits enabled)"
    echo ""
    echo "Open your browser at:"
    echo "   http://localhost:${STANDALONE_PORT}/lab"
    echo ""
    echo "Press Ctrl+C to stop"
    echo "=========================================="
    echo ""

    REPO_DIR="${REPO_DIR}" \
    STANDALONE_DIR="${STANDALONE_DIR}" \
    JUPYTER_CONFIG_TMPDIR="${JUPYTER_CONFIG_TMPDIR}" \
    METATLAS2_IMAGE_TAG="${IMAGE_TAG}" \
    STANDALONE_PORT="${STANDALONE_PORT}" \
        docker compose \
            -f "${REPO_DIR}/local/docker-compose.standalone.yml" \
            up --remove-orphans
    exit 0
fi

# ---------------------------------------------------------------------------
# Validation (non-standalone)
# ---------------------------------------------------------------------------
[[ -n "${METATLAS_DATA_DIR:-}" ]] || die \
    "Error: METATLAS_DATA_DIR is not set." \
    "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc (or ~/.bash_profile on macOS) and re-source it."

if [[ "${SUBCOMMAND}" == "jupyter" && "${RUNTIME}" == "shifter" ]]; then
    die "Error: 'jupyter' is only supported in Docker mode." \
        "On NERSC, use JupyterHub directly to open curation notebooks."
fi
if [[ "${SUBCOMMAND}" == "submit" && "${RUNTIME}" == "docker" ]]; then
    die "Error: 'submit' (SLURM/sbatch) is only supported on NERSC/Shifter systems."
fi

# Dev mode prepends the repo to PYTHONPATH
PYTHONPATH_VALUE="/app"
[[ "${DEV_MODE}" == "true" ]] && PYTHONPATH_VALUE="${REPO_DIR}:/app"

# ---------------------------------------------------------------------------
# Docker only: locate and absolutize standards config files (including Jupyter);
# add a read-only mount if it lives outside $METATLAS_DATA_DIR.
# ---------------------------------------------------------------------------
EXTRA_CONFIG_MOUNT=""
TAG_INPUT_MOUNT=""
if [[ "${RUNTIME}" == "docker" ]]; then
    IS_STANDARDS=false
    case "${SUBCOMMAND}" in
        standards-annotation|reverse-rt-align|atlas-msms-deposit|jupyter) IS_STANDARDS=true ;;
    esac

    CONFIG_PATH="" CONFIG_IDX="" CONFIG_INLINE=false
    for i in "${!PASSTHROUGH_ARGS[@]}"; do
        arg="${PASSTHROUGH_ARGS[$i]}"
        if [[ "${arg}" == "--config" || ( "${IS_STANDARDS}" == "true" && "${arg}" == "--config_path" ) ]]; then
            CONFIG_PATH="${PASSTHROUGH_ARGS[$i+1]:-}"; CONFIG_IDX=$((i+1)); break
        elif [[ "${IS_STANDARDS}" == "true" && "${arg}" == --config_path=* ]]; then
            CONFIG_PATH="${arg#*=}"; CONFIG_IDX=$i; CONFIG_INLINE=true; break
        fi
    done

    if [[ -n "${CONFIG_PATH}" ]]; then
        if config_dir="$(cd "$(dirname "${CONFIG_PATH}")" 2>/dev/null && pwd)"; then
            CONFIG_ABS="${config_dir}/$(basename "${CONFIG_PATH}")"
        else
            CONFIG_ABS="${CONFIG_PATH}"
        fi
        if [[ "${IS_STANDARDS}" == "true" ]]; then
            if [[ "${CONFIG_INLINE}" == "true" ]]; then
                PASSTHROUGH_ARGS[CONFIG_IDX]="--config_path=${CONFIG_ABS}"
            else
                PASSTHROUGH_ARGS[CONFIG_IDX]="${CONFIG_ABS}"
            fi
        fi
        DATA_ABS="$(cd "${METATLAS_DATA_DIR}" && pwd)"
        if [[ "${CONFIG_ABS}" != "${DATA_ABS}" && "${CONFIG_ABS}" != "${DATA_ABS}/"* ]]; then
            EXTRA_CONFIG_MOUNT="$(dirname "${CONFIG_ABS}")"
            echo "Note: config file is outside \$METATLAS_DATA_DIR; adding read-only mount for ${EXTRA_CONFIG_MOUNT}"
        fi
        SUB_ARGS=("${PASSTHROUGH_ARGS[@]:1}")
    fi

    if [[ "${SUBCOMMAND}" == "update-tags" ]]; then
        for i in "${!SUB_ARGS[@]}"; do
            arg="${SUB_ARGS[$i]}"
            if [[ "${arg}" == "--input-csv" || "${arg}" == "--input_csv" ]]; then
                TAG_INPUT_PATH="${SUB_ARGS[$i+1]:-}"
                TAG_INPUT_IDX=$((i+1))
                TAG_INPUT_PREFIX=""
            elif [[ "${arg}" == --input-csv=* || "${arg}" == --input_csv=* ]]; then
                TAG_INPUT_PATH="${arg#*=}"
                TAG_INPUT_IDX=$i
                TAG_INPUT_PREFIX="${arg%%=*}="
            else
                continue
            fi
            [[ -f "${TAG_INPUT_PATH}" ]] || die "Error: input CSV not found: ${TAG_INPUT_PATH}"
            TAG_INPUT_DIR="$(cd "$(dirname "${TAG_INPUT_PATH}")" && pwd -P)"
            TAG_INPUT_ABS="${TAG_INPUT_DIR}/$(basename "${TAG_INPUT_PATH}")"
            SUB_ARGS[TAG_INPUT_IDX]="${TAG_INPUT_PREFIX}${TAG_INPUT_ABS}"
            DATA_ABS="$(cd "${METATLAS_DATA_DIR}" && pwd -P)"
            if [[ "${TAG_INPUT_ABS}" != "${DATA_ABS}/"* ]]; then
                TAG_INPUT_MOUNT="${TAG_INPUT_ABS}"
            fi
            break
        done
    fi
fi

# ---------------------------------------------------------------------------
# Runtime-specific argument arrays
# ---------------------------------------------------------------------------
if [[ "${RUNTIME}" == "shifter" ]]; then
    # Shifter (NERSC/Perlmutter): global filesystems are auto-mounted.
    SHIFTER_ARGS=(
        "--image=docker:${IMAGE_FULL}"
        "--env=METATLAS2_IMAGE_TAG=${IMAGE_TAG}"
        "--env=METATLAS_DATA_DIR=${METATLAS_DATA_DIR}"
        "--env=HOME=${HOME}"
        "--env=JUPYTERHUB_SERVICE_PREFIX=${JUPYTERHUB_SERVICE_PREFIX:-/}"
        "--env=PYTHONPATH=${PYTHONPATH_VALUE}"
    )
    [[ "${DEV_MODE}" == "true" && "${SUBCOMMAND}" != "update-tags" ]] && echo "Dev mode enabled: using local repo scripts at ${REPO_DIR}/scripts"

    if [[ "${IMAGE_TAG}" != "latest" && "${SUBCOMMAND}" != "update-tags" ]]; then
        KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/metatlas2-${IMAGE_TAG}"
        if [[ ! -d "${KERNEL_DIR}" ]]; then
            echo "Registering Jupyter kernel spec for ${IMAGE_TAG} ..."
            "${SCRIPT_DIR}/install_kernels.sh" --tag "${IMAGE_TAG}"
        fi
    fi
else
    # Docker (local macOS / non-NERSC HPC): ensure main DuckDB is present and current.
    if [[ -n "${ZENODO_MAIN_DB_DOI}" && "${SUBCOMMAND}" != "atlas-msms-deposit" && "${SUBCOMMAND}" != "update-tags" ]]; then
        MAIN_DB_DIR="${METATLAS_DATA_DIR}/databases/main_db"
        MAIN_DB_PATH="${MAIN_DB_DIR}/metatlas.duckdb"

        if [[ ! -f "${MAIN_DB_PATH}" ]] || _stamp_stale "${MAIN_DB_DIR}/.zenodo_version" "${ZENODO_MAIN_DB_DOI}"; then
            echo "Downloading metatlas main database from Zenodo..."
            echo "  DOI: ${ZENODO_MAIN_DB_DOI}"
            echo "  Destination: ${MAIN_DB_PATH}"
            mkdir -p "${MAIN_DB_DIR}"
            _zenodo_get "${MAIN_DB_DIR}" "${ZENODO_MAIN_DB_DOI}"

            [[ -f "${MAIN_DB_PATH}" ]] || die \
                "Error: Download completed but metatlas.duckdb not found in ${MAIN_DB_DIR}." \
                "Check that the Zenodo deposit contains a file named 'metatlas.duckdb'."

            echo "${ZENODO_MAIN_DB_DOI}" > "${MAIN_DB_DIR}/.zenodo_version"
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
        "-e" "PYTHONPATH=${PYTHONPATH_VALUE}"
        "-e" "NUMBA_CACHE_DIR=${METATLAS_DATA_DIR}/.numba_cache"
        "-e" "MPLCONFIGDIR=${METATLAS_DATA_DIR}/.matplotlib_cache"
        "-v" "${METATLAS_DATA_DIR}:${METATLAS_DATA_DIR}"
        "--user" "$(id -u):$(id -g)"
    )
    # Forward JUPYTERHUB_SERVICE_PREFIX only when set (i.e. on JupyterHub).
    [[ -n "${JUPYTERHUB_SERVICE_PREFIX:-}" ]] && \
        DOCKER_ARGS+=("-e" "JUPYTERHUB_SERVICE_PREFIX=${JUPYTERHUB_SERVICE_PREFIX}")
    # Config directory outside $METATLAS_DATA_DIR.
    [[ -n "${EXTRA_CONFIG_MOUNT}" ]] && \
        DOCKER_ARGS+=("-v" "${EXTRA_CONFIG_MOUNT}:${EXTRA_CONFIG_MOUNT}:ro")
    [[ -n "${TAG_INPUT_MOUNT}" ]] && \
        DOCKER_ARGS+=("-v" "${TAG_INPUT_MOUNT}:${TAG_INPUT_MOUNT}:ro")
    # rclone config (Google Drive upload on non-NERSC systems).
    [[ -d "${HOME}/.config/rclone" ]] && \
        DOCKER_ARGS+=("-v" "${HOME}/.config/rclone:${HOME}/.config/rclone:ro")
    if [[ "${SUBCOMMAND}" == "jupyter" ]]; then
        ATLAS_KEY="/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
        [[ ! -f "${ATLAS_KEY}" ]] || DOCKER_ARGS+=("-v" "${ATLAS_KEY}:${ATLAS_KEY}:ro")
    fi
    if [[ "${DEV_MODE}" == "true" ]]; then
        DOCKER_ARGS+=("-v" "${REPO_DIR}:${REPO_DIR}:ro")
        [[ "${SUBCOMMAND}" == "update-tags" ]] || echo "Dev mode enabled: using local repo at ${REPO_DIR}"
    fi
fi

# run_in_container CMD... — run CMD with the image's default entrypoint
run_in_container() {
    if [[ "${RUNTIME}" == "shifter" ]]; then
        shifter "${SHIFTER_ARGS[@]}" "$@"
    else
        docker run "${DOCKER_ARGS[@]}" "${IMAGE_FULL}" "$@"
    fi
}

# ---------------------------------------------------------------------------
# Python call builders for the lightweight "routine" subcommands
# ---------------------------------------------------------------------------
_arg() { _parse_arg "$1" "${SUB_ARGS[@]}"; }   # value of --key in SUB_ARGS
_opt() { _py_str "$(_arg "$1")"; }             # same, as a Python literal (None if empty)

build_py_call() {
    local cfg timestamp ow
    case "${SUBCOMMAND}" in
        standards-annotation)
            cfg="$(_arg --config_path)"
            PY_CALL="from metatlas2.run_standards_annotation import run_standards_annotation; run_standards_annotation('${cfg}')"
            ;;
        add-compounds)
            cfg="$(_arg --input_csv)"
            [[ -n "${cfg}" ]] || die "Error: add-compounds requires --input_csv FILE."
            ow="False"
            if _has_flag --overwrite_db "${SUB_ARGS[@]}"; then ow="True"; fi
            PY_CALL="from metatlas2.add_compounds_to_db import add_compounds_to_db; add_compounds_to_db('${cfg}', overwrite_db=${ow})"
            ;;
        add-atlases)
            custom="False"
            if _has_flag --custom "${SUB_ARGS[@]}"; then custom="True"; fi
            PY_CALL="from metatlas2.add_atlases_to_db import add_atlases_to_db; add_atlases_to_db('$(_arg --config_path)', custom=${custom})"
            ;;
        add-msms-refs)
            PY_CALL="from metatlas2.add_msms_refs_to_db import add_msms_refs_to_db; add_msms_refs_to_db('$(_arg --config_path)')"
            ;;
        get-db-table)
            PY_CALL="from metatlas2.get_db_table import export_table_to_csv; export_table_to_csv('$(_arg --table-name)', output_path=$(_opt --output_path), database_path=$(_opt --database_path))"
            ;;
        get-atlases)
            if [[ "${SUB_ARGS[0]:-}" == "fetch" ]]; then
                PY_CALL="from metatlas2.get_atlases_from_db import get_atlases, _parse_csv_values; get_atlases(atlas_uids=_parse_csv_values('$(_arg --atlas_uids)'), output_path=$(_opt --output_path), database_path=$(_opt --database_path))"
            else
                PY_CALL="from metatlas2.get_atlases_from_db import query_atlases; query_atlases(database_path=$(_opt --database_path), chromatography=$(_opt --chromatography), polarity=$(_opt --polarity), atlas_category=$(_opt --atlas_category), atlas_name_key=$(_opt --atlas_name_key), created_by=$(_opt --created_by))"
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${SUBCOMMAND}" in
    update-tags)
        if [[ "${RUNTIME}" == "docker" ]]; then
            ATLAS_KEY="/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
            [[ ! -f "${ATLAS_KEY}" ]] || DOCKER_ARGS+=("-v" "${ATLAS_KEY}:${ATLAS_KEY}:ro")
        fi
        run_in_container /app/.venv/bin/python -m metatlas2.update_tags "${SUB_ARGS[@]}"
        ;;

    jupyter)  # Docker only (Shifter rejected above)
        echo "=========================================="
        echo "Metatlas2 JupyterLab (portable mode)"
        echo "=========================================="
        echo ""
        echo "Data dir: ${METATLAS_DATA_DIR}"
        echo ""
        echo "Open your browser at:"
        echo "   http://localhost:${JUPYTER_PORT}/lab"
        echo ""
        echo "Press Ctrl+C to stop"
        echo "=========================================="
        echo ""

        docker run --rm -it \
            "${DOCKER_ARGS[@]}" \
            -p "127.0.0.1:${JUPYTER_PORT}:${JUPYTER_PORT}" \
            -p "127.0.0.1:8050-8069:8050-8069" \
            --entrypoint /bin/bash \
            "${IMAGE_FULL}" \
            -c "
                mkdir -p '${METATLAS_DATA_DIR}/.jupyter_data/kernels' \
                && /app/.venv/bin/python -m ipykernel install \
                    --name metatlas2 \
                    --display-name 'metatlas2 (latest)' \
                    --prefix '${METATLAS_DATA_DIR}/.jupyter_data' \
                && JUPYTER_DATA_DIR='${METATLAS_DATA_DIR}/.jupyter_data' \
                   IPYTHONDIR='${METATLAS_DATA_DIR}/.ipython' \
                   /app/.venv/bin/jupyter lab \
                    --ip=0.0.0.0 \
                    --port=${JUPYTER_PORT} \
                    --no-browser \
                    --allow-root \
                    --IdentityProvider.token='' \
                    --ServerApp.password='' \
                    --ServerApp.root_dir='${METATLAS_DATA_DIR}' \
                    --MultiKernelManager.default_kernel_name=metatlas2
            "
        ;;

    submit)  # Shifter only (Docker rejected above)
        TMPSCRIPT="$(mktemp /tmp/metatlas2_XXXXXX.sh)"
        # shellcheck disable=SC2064
        trap "rm -f '${TMPSCRIPT}'" EXIT
        shifter "${SHIFTER_ARGS[@]}" --entrypoint \
            "${PASSTHROUGH_ARGS[@]}" --script-only --output "${TMPSCRIPT}"
        sbatch "${TMPSCRIPT}"
        ;;

    atlas-msms-deposit)
        if [[ "${RUNTIME}" == "docker" ]]; then
            ATLAS_KEY="/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
            if [[ -f "${ATLAS_KEY}" ]] && ! _has_flag --dry_run "${SUB_ARGS[@]}" && ! _has_flag --skip_atlas_upload "${SUB_ARGS[@]}"; then
                DOCKER_ARGS+=("-v" "${ATLAS_KEY}:${ATLAS_KEY}:ro")
            fi
        fi
        run_in_container /app/.venv/bin/python -m metatlas2.run_atlas_and_msms_deposit "${SUB_ARGS[@]}"
        ;;

    reverse-rt-align)
        if [[ "${RUNTIME}" == "docker" ]]; then
            ATLAS_KEY="/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
            [[ ! -f "${ATLAS_KEY}" ]] || DOCKER_ARGS+=("-v" "${ATLAS_KEY}:${ATLAS_KEY}:ro")
        fi
        run_in_container /app/.venv/bin/python -m metatlas2.run_reverse_rt_align "${SUB_ARGS[@]}"
        ;;

    add-compounds|add-atlases|add-msms-refs|get-atlases|get-db-table|standards-annotation)
        echo "Launching metatlas2 container to kick off routine '${SUBCOMMAND}' (tag=${IMAGE_TAG}, mode=${MODE}, runtime=${RUNTIME})..."
        build_py_call
        run_in_container /app/.venv/bin/python -c "${PY_CALL}"
        ;;

    *)  # main targeted pipeline
        if [[ "${SUBCOMMAND}" == "run" ]] && ! _has_flag --log-to-stdout "${PASSTHROUGH_ARGS[@]}"; then
            echo "=-------- Launching metatlas2 container (tag=${IMAGE_TAG}, mode=${MODE}, runtime=${RUNTIME})..."
        fi
        if [[ "${RUNTIME}" == "shifter" ]]; then
            shifter "${SHIFTER_ARGS[@]}" --entrypoint "${PASSTHROUGH_ARGS[@]}"
        else
            docker run "${DOCKER_ARGS[@]}" \
                -p "127.0.0.1:8050-8069:8050-8069" \
                --entrypoint /app/.venv/bin/python \
                "${IMAGE_FULL}" \
                -m metatlas2.run_targeted_analysis \
                "${PASSTHROUGH_ARGS[@]}"
        fi
        ;;
esac