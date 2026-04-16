#!/bin/bash
# Install Jupyter kernel specs that launch metatlas2 inside a Shifter container.
#
# Usage:
#   ./install_kernels.sh              # installs 'metatlas2' (latest) + 'metatlas2-dev'
#   ./install_kernels.sh --tag v1.2.3 # also installs a pinned 'metatlas2-v1.2.3' kernel
#
# The installed kernels appear in JupyterLab's kernel selector.  Generated
# curation notebooks embed the tag used at analysis time in their kernelspec
# metadata; run this script with --tag to install a matching pinned kernel.

set -euo pipefail

IMAGE_REPO="ghcr.io/bkieft-usa/metatlas2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRA_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)    EXTRA_TAG="$2"; shift 2 ;;
        --tag=*)  EXTRA_TAG="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# install_kernel <kernel_name> <display_name> <image_tag> <dev_mode>
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

    # Use Python to generate kernel.json.
    # kernel.json calls shifter directly with {connection_file} in argv,
    # matching the pattern used by the working metatlas-targeted kernel.
    # shifter automatically mounts GPFS filesystems so no volume flags are needed.
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
argv = [
    "shifter",
    "--module=none",
    "--entrypoint",
    f"--env=METATLAS2_IMAGE_TAG={tag}",
    f"--env=METATLAS_DATA_DIR={data_dir}",
    f"--env=HOME={home}",
    f"--image=docker:{image}",
]

if dev_mode:
    argv.append(f"--volume={repo_dir}/metatlas2:/app/metatlas2:ro")

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
