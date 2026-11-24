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
#   metatlas2 [--image TAG] [--dev] get-atlases fetch --atlas_uids UID1,UID2 [--output_path PATH]
#   metatlas2 [--image TAG] [--dev] get-atlases query [--chromatography X] [--polarity X] [--analysis_type X] [--analysis_name X] [--created_by X]
#   metatlas2 [--image TAG] [--dev] get-db-table --table-name NAME [--output_path PATH] [--database_path PATH]
#   metatlas2 [--image TAG] [--dev] update-tags {add|remove|retag} --input-csv FILE
#   metatlas2 [--image TAG] [--dev] update-tags retrieve
#   metatlas2 [--image TAG] [--dev] convert-raw convert {jgi|egsb} [--dry_run] [--max_workers N]
#   metatlas2 [--image TAG] [--dev] convert-raw convert-file PATH.raw
#   metatlas2 [--image TAG] [--dev] lims-sync [--dry_run] [--skip_file_tables] ...
#   metatlas2 [--image TAG] [--dev] untargeted run [--config FILE] [--dry_run] [--direct_input P1,P2] ...
#   metatlas2 [--image TAG] [--dev] untargeted status [--print_recent N] [--direct_input P1,P2]
#   metatlas2 [--image TAG] [--dev] untargeted mirror-raw --mode {hard|soft} --direct_input P1,P2
#   metatlas2 [--image TAG] --sandbox
#
# Flags (consumed by this script, not forwarded to Python):
#   --image TAG   Use a specific image tag instead of the default (latest).
#                 Overrides the METATLAS2_IMAGE_TAG environment variable.
#   --dev         Mount the local repository source over the installed package,
#                 so edits to the working tree take effect immediately.
#   --sandbox     Developer sandbox (Docker only): download the sandbox data package
#                 to ~/.metatlas2-sandbox/, wipe its DB and outputs, and open
#                 JupyterLab with the sandbox workflow notebook and the local repo
#                 mounted (implies --dev). Refuses to start on 'main' or a
#                 detached HEAD; create a feature branch first.
#   --skip-db-update
#                 Docker only: do not download the main DB from Zenodo, even if a
#                 newer version is published (keeps local add-atlases changes).
#   --port PORT   JupyterLab port for the 'jupyter' subcommand (default: 8889).
#                 Can also be set via METATLAS2_JUPYTER_PORT env var.
#
# Runtime detection (automatic, no flag needed):
#   - If 'shifter' is found on PATH, Shifter is used (NERSC/Perlmutter).
#   - Otherwise, if 'docker' is found on PATH, Docker is used.
#   - If neither is found, the script exits with an error.
#
# Docker (portable mode, off NERSC) has two branches:
#   - local (default): your own .h5 files and config under $METATLAS_DATA_DIR,
#     always on the 'latest' image (--image is rejected). On first run (or when
#     ZENODO_MAIN_DB_DOI changes) the main DuckDB is downloaded from Zenodo into
#     $METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb (see --skip-db-update).
#     Atlas 'tags' are not supported off NERSC; use add-atlases + 'uid'.
#   - sandbox (--sandbox): see above. The sandbox data package is published with
#     sandbox/prepare_sandbox_data_package.sh (on NERSC), which updates
#     ZENODO_SANDBOX_DATA_DOI below.
#   To publish a new main DB version, run scripts/upload_main_db_to_zenodo.sh on
#   NERSC and update ZENODO_MAIN_DB_DOI below.

set -euo pipefail

ZENODO_MAIN_DB_DOI="https://doi.org/10.5281/zenodo.22968291"
ZENODO_SANDBOX_DATA_DOI="https://doi.org/10.5281/zenodo.24517305"

IMAGE_REPO="ghcr.io/biorack/metatlas2"
IMAGE_TAG="${METATLAS2_IMAGE_TAG:-latest}"
DEV_MODE=false
SANDBOX_MODE=false
SKIP_DB_UPDATE=false
JUPYTER_DEFAULT_URL=""
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
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp \
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
        --sandbox)      SANDBOX_MODE=true; shift ;;
        --skip-db-update) SKIP_DB_UPDATE=true; shift ;;
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
# Docker image policy and sandbox setup
# ---------------------------------------------------------------------------
if [[ "${RUNTIME}" == "docker" && "${SANDBOX_MODE}" != "true" && "${IMAGE_TAG}" != "latest" ]]; then
    die "Error: --image is only supported with --sandbox when running with Docker." \
        "Local (non-NERSC) runs always use the latest image."
fi

if [[ "${SANDBOX_MODE}" == "true" ]]; then
    [[ "${RUNTIME}" == "docker" ]] || die \
        "Error: --sandbox requires Docker (it is for developing off NERSC)." \
        "On NERSC, use the 'metatlas2 (dev - local repo)' kernel instead."
    [[ -z "${SUBCOMMAND}" ]] || die "Error: --sandbox does not take a subcommand (got '${SUBCOMMAND}')."

    # Edits must happen on a feature branch so they can become a PR.
    BRANCH="$(git -C "${REPO_DIR}" symbolic-ref --short -q HEAD || true)"
    if [[ -z "${BRANCH}" || "${BRANCH}" == "main" ]]; then
        die "Error: the sandbox runs your local clone (${REPO_DIR}), which is on '${BRANCH:-a detached HEAD}'." \
            "Create a feature branch before editing code, then re-run:" \
            "    git -C '${REPO_DIR}' switch -c <your-branch-name>"
    fi

    SANDBOX_DIR="${HOME}/.metatlas2-sandbox"
    SANDBOX_STAMP="${SANDBOX_DIR}/.zenodo_version"
    echo "=========================================="
    echo "Metatlas2 sandbox"
    echo "=========================================="
    echo "Repo:    ${REPO_DIR} (branch '${BRANCH}', mounted read-only)"
    echo "Data:    ${SANDBOX_DIR}"
    echo "Image:   ${IMAGE_FULL}"
    echo "Package: ${ZENODO_SANDBOX_DATA_DOI}"
    echo "         If your change affects the sandbox h5 data, publish a new package first:"
    echo "         sandbox/prepare_sandbox_data_package.sh (on NERSC)."
    echo ""

    if [[ ! -d "${SANDBOX_DIR}/raw_data" ]] || _stamp_stale "${SANDBOX_STAMP}" "${ZENODO_SANDBOX_DATA_DOI}"; then
        echo "Downloading sandbox data package from Zenodo..."
        rm -rf "${SANDBOX_DIR}"
        mkdir -p "${SANDBOX_DIR}"
        _zenodo_get "${SANDBOX_DIR}" "${ZENODO_SANDBOX_DATA_DOI}" \
            "tar -xzf metatlas2-sandbox-data.tar.gz --strip-components=1 && rm metatlas2-sandbox-data.tar.gz"
        echo "${ZENODO_SANDBOX_DATA_DOI}" > "${SANDBOX_STAMP}"
    fi

    echo "Resetting sandbox database and outputs..."
    rm -rf "${SANDBOX_DIR}/databases/main_db" "${SANDBOX_DIR}/projects" "${SANDBOX_DIR}/configs" "${SANDBOX_DIR}/logs"
    mkdir -p "${SANDBOX_DIR}/databases/modelseed_db"
    cp "${REPO_DIR}/sandbox/inputs/modelseed.tsv" "${SANDBOX_DIR}/databases/modelseed_db/modelseed.tsv"

    # Fresh copy of the workflow notebook, pointed at the kernel registered below.
    KERNEL_NAME="metatlas2"; [[ "${IMAGE_TAG}" == "latest" ]] || KERNEL_NAME="metatlas2-${IMAGE_TAG}"
    sed "s/\"name\": \"metatlas2\"/\"name\": \"${KERNEL_NAME}\"/" \
        "${REPO_DIR}/sandbox/sandbox_workflow.ipynb" > "${SANDBOX_DIR}/sandbox_workflow.ipynb"

    # Continue as 'metatlas2.sh --dev jupyter' on the sandbox data dir.
    export METATLAS_DATA_DIR="${SANDBOX_DIR}"
    DEV_MODE=true
    MODE="dev"
    SUBCOMMAND="jupyter"
    PASSTHROUGH_ARGS=("jupyter")
    SUB_ARGS=()
    JUPYTER_DEFAULT_URL="/lab/tree/sandbox_workflow.ipynb"
fi

# ---------------------------------------------------------------------------
# Validation
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
if [[ "${SUBCOMMAND}" == "update-tags" && "${RUNTIME}" == "docker" ]]; then
    die "Error: 'update-tags' edits the centralized atlas sheet, which is only available on NERSC."
fi

# Dev mode prepends the repo to PYTHONPATH
PYTHONPATH_VALUE="/app"
[[ "${DEV_MODE}" == "true" ]] && PYTHONPATH_VALUE="${REPO_DIR}:/app"

# ---------------------------------------------------------------------------
# Docker only: locate and absolutize standards config files (including Jupyter);
# add a read-only mount if it lives outside $METATLAS_DATA_DIR.
# ---------------------------------------------------------------------------
EXTRA_CONFIG_MOUNT=""
if [[ "${RUNTIME}" == "docker" ]]; then
    IS_STANDARDS=false
    case "${SUBCOMMAND}" in
        standards-annotation|jupyter) IS_STANDARDS=true ;;
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
    # Docker (portable mode). Always run the current image: 'docker run' would
    # otherwise reuse a stale cached copy of the tag.
    if ! docker pull -q "${IMAGE_FULL}" >/dev/null; then
        docker image inspect "${IMAGE_FULL}" &>/dev/null \
            || die "Error: could not pull ${IMAGE_FULL} and no local copy exists."
        echo "Warning: could not pull ${IMAGE_FULL}; using the cached local copy." >&2
    fi

    # Local branch: make sure the main DuckDB is present and current. The sandbox
    # builds its own DB from scratch in the workflow notebook.
    if [[ "${SANDBOX_MODE}" != "true" ]]; then
        MAIN_DB_DIR="${METATLAS_DATA_DIR}/databases/main_db"
        MAIN_DB_PATH="${MAIN_DB_DIR}/metatlas.duckdb"

        if [[ "${SKIP_DB_UPDATE}" == "true" ]]; then
            if [[ -f "${MAIN_DB_PATH}" ]]; then
                echo "Skipping main database update check (--skip-db-update)."
            else
                echo "Warning: no main database at ${MAIN_DB_PATH}; skipping download as requested (--skip-db-update)." >&2
            fi
        elif [[ ! -f "${MAIN_DB_PATH}" ]] || _stamp_stale "${MAIN_DB_DIR}/.zenodo_version" "${ZENODO_MAIN_DB_DOI}"; then
            echo "Downloading metatlas main database from Zenodo..."
            echo "  DOI: ${ZENODO_MAIN_DB_DOI}"
            echo "  Destination: ${MAIN_DB_PATH}"
            [[ ! -f "${MAIN_DB_PATH}" ]] || echo "  This replaces your current copy, including any atlases you added. Use --skip-db-update to keep it."
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
        "-e" "METATLAS2_PORTABLE=true"
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
    # rclone config (Google Drive upload on non-NERSC systems).
    [[ -d "${HOME}/.config/rclone" ]] && \
        DOCKER_ARGS+=("-v" "${HOME}/.config/rclone:${HOME}/.config/rclone:ro")
    if [[ "${DEV_MODE}" == "true" ]]; then
        DOCKER_ARGS+=("-v" "${REPO_DIR}:${REPO_DIR}:ro" "-e" "METATLAS2_REPO_DIR=${REPO_DIR}")
        echo "Dev mode enabled: using local repo at ${REPO_DIR}"
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
            PY_CALL="from metatlas2.standards_annotation.run_standards_annotation import run_standards_annotation; run_standards_annotation('${cfg}')"
            ;;
        add-compounds)
            cfg="$(_arg --input_csv)"
            [[ -n "${cfg}" ]] || die "Error: add-compounds requires --input_csv FILE."
            ow="False"
            if _has_flag --overwrite_db "${SUB_ARGS[@]}"; then ow="True"; fi
            PY_CALL="from metatlas2.shared_tools.add_compounds_to_db import add_compounds_to_db; add_compounds_to_db('${cfg}', overwrite_db=${ow})"
            ;;
        add-atlases)
            custom="False"
            if _has_flag --custom "${SUB_ARGS[@]}"; then custom="True"; fi
            PY_CALL="from metatlas2.shared_tools.add_atlases_to_db import add_atlases_to_db; add_atlases_to_db('$(_arg --config_path)', custom=${custom})"
            ;;
        add-msms-refs)
            PY_CALL="from metatlas2.shared_tools.add_msms_refs_to_db import add_msms_refs_to_db; add_msms_refs_to_db('$(_arg --config_path)')"
            ;;
        get-db-table)
            PY_CALL="from metatlas2.shared_tools.get_db_table import export_table_to_csv; export_table_to_csv('$(_arg --table-name)', output_path=$(_opt --output_path), database_path=$(_opt --database_path))"
            ;;
        get-atlases)
            if [[ "${SUB_ARGS[0]:-}" == "fetch" ]]; then
                PY_CALL="from metatlas2.shared_tools.get_atlases_from_db import get_atlases, _parse_csv_values; get_atlases(atlas_uids=_parse_csv_values('$(_arg --atlas_uids)'), output_path=$(_opt --output_path), database_path=$(_opt --database_path))"
            else
                PY_CALL="from metatlas2.shared_tools.get_atlases_from_db import query_atlases; query_atlases(database_path=$(_opt --database_path), chromatography=$(_opt --chromatography), polarity=$(_opt --polarity), atlas_category=$(_opt --atlas_category), atlas_name_key=$(_opt --atlas_name_key), created_by=$(_opt --created_by))"
            fi
            ;;
    esac
}

# _drain_sbatch_queue FILE — submit MZmine jobs queued by the untargeted pipeline.
# Each line: <sbatch file>\t<job-id file>\t<prefix>. Runs on the host because
# sbatch is not available inside the container.
_drain_sbatch_queue() {
    local queue="$1" sbatch_file job_id_file prefix out job_id
    if [[ -s "${queue}" ]]; then
        while IFS=$'\t' read -r sbatch_file job_id_file prefix; do
            [[ -n "${sbatch_file}" ]] || continue
            if out="$(sbatch --parsable "${sbatch_file}" 2>&1)"; then
                job_id="${out%%;*}"
                echo "${prefix}=${job_id}" > "${job_id_file}"
                echo "Submitted MZmine job ${job_id} for ${prefix}"
            else
                echo "ERROR: sbatch failed for ${sbatch_file}: ${out}" >&2
            fi
        done < "${queue}"
    fi
    rm -f "${queue}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${SUBCOMMAND}" in
    untargeted)
        UNT_ARGS=("${SUB_ARGS[@]}")
        QUEUE_FILE=""
        if [[ "${SUB_ARGS[0]:-}" == "run" ]] \
            && ! _has_flag --dry_run "${SUB_ARGS[@]}" && ! _has_flag --dry-run "${SUB_ARGS[@]}" \
            && command -v sbatch &>/dev/null; then
            QUEUE_DIR="${HOME}/.metatlas2/untargeted_sbatch_queue"
            mkdir -p "${QUEUE_DIR}"
            QUEUE_FILE="$(mktemp "${QUEUE_DIR}/queue_XXXXXX.tsv")"
            UNT_ARGS+=("--sbatch_queue_file" "${QUEUE_FILE}")
            [[ "${RUNTIME}" == "docker" ]] && DOCKER_ARGS+=("-v" "${QUEUE_DIR}:${QUEUE_DIR}")
        fi
        set +e
        if [[ "${RUNTIME}" == "shifter" ]]; then
            shifter "${SHIFTER_ARGS[@]}" /app/.venv/bin/python -m metatlas2.untargeted.run_untargeted_pipeline "${UNT_ARGS[@]}"
        else
            docker run "${DOCKER_ARGS[@]}" --entrypoint /app/.venv/bin/python "${IMAGE_FULL}" \
                -m metatlas2.untargeted.run_untargeted_pipeline "${UNT_ARGS[@]}"
        fi
        UNT_RC=$?
        set -e
        # Submit queued MZmine jobs even if a later pipeline step failed.
        [[ -n "${QUEUE_FILE}" ]] && _drain_sbatch_queue "${QUEUE_FILE}"
        exit "${UNT_RC}"
        ;;

    update-tags)  # Shifter only (Docker rejected above)
        run_in_container /app/.venv/bin/python -m metatlas2.shared_tools.update_tags "${SUB_ARGS[@]}"
        ;;

    jupyter)  # Docker only (Shifter rejected above)
        # Kernel name must match what notebook_generator.py writes into the
        # generated curation notebooks: metatlas2 for latest, else metatlas2-<tag>.
        KERNEL_NAME="metatlas2"; [[ "${IMAGE_TAG}" == "latest" ]] || KERNEL_NAME="metatlas2-${IMAGE_TAG}"
        echo "=========================================="
        echo "Metatlas2 JupyterLab (portable mode)"
        echo "=========================================="
        echo ""
        echo "Data dir: ${METATLAS_DATA_DIR}"
        echo "Kernel:   ${KERNEL_NAME} (${IMAGE_FULL})"
        echo ""
        echo "Open your browser at:"
        echo "   http://localhost:${JUPYTER_PORT}${JUPYTER_DEFAULT_URL:-/lab}"
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
                    --name '${KERNEL_NAME}' \
                    --display-name 'metatlas2 (${IMAGE_TAG})' \
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
                    --MultiKernelManager.default_kernel_name='${KERNEL_NAME}' \
                    ${JUPYTER_DEFAULT_URL:+--ServerApp.default_url='${JUPYTER_DEFAULT_URL}'}
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

    convert-raw)
        # raw -> mzML -> h5 converter. The orchestrator runs on the HOST (ThermoRawFileParser
        # has its own shifter image and shifter can't run inside a container) with plain
        # python3.11 (standard library only); the mzML -> h5 step runs in this image.
        [[ "${RUNTIME}" == "shifter" ]] || die "Error: 'convert-raw' is only supported on NERSC/Shifter systems."
        HOST_PY="${METATLAS2_HOST_PYTHON:-/usr/bin/python3.11}"
        [[ -x "${HOST_PY}" ]] || die "Error: host Python not found: ${HOST_PY} (set METATLAS2_HOST_PYTHON)."
        PYTHONPATH="${REPO_DIR}" \
        METATLAS2_IMAGE="docker:${IMAGE_FULL}" \
        METATLAS2_CONTAINER_PYTHONPATH="${PYTHONPATH_VALUE}" \
            exec "${HOST_PY}" -m metatlas2.ingest.run_file_conversion "${SUB_ARGS[@]}"
        ;;

    lims-sync)
        # Sync raw/mzML/h5 files on disk into the LIMS file tables (runs in the container).
        run_in_container /app/.venv/bin/python -m metatlas2.ingest.run_lims_file_sync "${SUB_ARGS[@]}"
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
                -m metatlas2.targeted.run_targeted_analysis \
                "${PASSTHROUGH_ARGS[@]}"
        fi
        ;;
esac