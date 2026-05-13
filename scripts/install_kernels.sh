#!/bin/bash

set -euo pipefail

IMAGE_REPO="ghcr.io/bkieft-usa/metatlas2"
REPO_DIR="/global/cfs/cdirs/metatlas/tools/metatlas2"
EXTRA_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)    EXTRA_TAG="$2"; shift 2 ;;
        --tag=*)  EXTRA_TAG="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

#   dev_mode = "true" mounts the local repo and sets PYTHONPATH
install_kernel() {
    local kernel_name="$1"
    local display_name="$2"
    local image_tag="$3"
    local dev_mode="${4:-false}"
    local data_dir="${METATLAS_DATA_DIR}"

    local tmpdir
    tmpdir="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '${tmpdir}'" RETURN

    python3 - \
        "${kernel_name}" "${display_name}" \
        "${IMAGE_REPO}:${image_tag}" "${image_tag}" \
        "${HOME}" "${dev_mode}" "${REPO_DIR}" \
        "${data_dir}" \
        "${tmpdir}/kernel.json" \
    <<'PYEOF'
import json, os, sys

kernel_name, display_name, image, tag, home, dev_mode, repo_dir, data_dir, json_out = sys.argv[1:]
dev_mode = dev_mode == "true"
real_home = os.path.realpath(home)

# shifter argv — mirrors the working metatlas-targeted kernel pattern.
# --module=none prevents Lmod from loading modules that could shadow the
# container's Python.  Env vars are forwarded explicitly via --env so the
# metatlas2 package knows the data directory and image tag at runtime.
pythonpath = f"{repo_dir}:/app" if dev_mode else "/app"

argv = [
    "shifter",
    "--module=none",
    f"--env=METATLAS2_IMAGE_TAG={tag}",
    f"--env=METATLAS_DATA_DIR={data_dir}",
    f"--env=HOME={home}",
    # metatlas2 is not installed as a proper wheel in the venv (no [build-system]
    # in pyproject.toml), so we must put /app on PYTHONPATH so Python always
    # finds /app/metatlas2/ regardless of the working directory at kernel launch.
    # Dev mode prepends the local repo so edits take effect without a rebuild.
    f"--env=PYTHONPATH={pythonpath}",
    f"--image=docker:{image}",
]

argv += [
    "/app/.venv/bin/python",
    "-m", "ipykernel_launcher",
    "-f", "{connection_file}",
]

spec = {
    "argv": argv,
    "display_name": display_name,
    "language": "python",
    "metadata": {"debugger": False},
}
with open(json_out, "w") as f:
    json.dump(spec, f, indent=2)
PYEOF

    jupyter kernelspec install "${tmpdir}" --user --name "${kernel_name}"
    echo "Installed kernel '${kernel_name}' → ${display_name}"
}

# Validate required environment variables
if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
    echo "Error: METATLAS_DATA_DIR is not set." >&2
    echo "Add 'export METATLAS_DATA_DIR=/path/to/data' to ~/.bashrc and re-source it." >&2
    exit 1
fi

# Always install the two standard kernels
install_kernel "metatlas2"     "metatlas2 (latest)"            "latest" "false"
install_kernel "metatlas2-dev" "metatlas2 (dev - local repo)"  "latest" "true"

# Optionally install a pinned version kernel
if [[ -n "${EXTRA_TAG}" ]]; then
    install_kernel "metatlas2-${EXTRA_TAG}" "metatlas2 (${EXTRA_TAG})" "${EXTRA_TAG}" "false"
fi

echo ""
echo "Installed kernels:"
jupyter kernelspec list
