#!/bin/bash
# Install Jupyter kernel specs that launch metatlas2 inside a Podman container.
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

# ---------------------------------------------------------------------------
# install_kernel <kernel_name> <display_name> <image_tag> <dev_mode>
#   dev_mode = "true" mounts the local repo and sets PYTHONPATH
# ---------------------------------------------------------------------------
install_kernel() {
    local kernel_name="$1"
    local display_name="$2"
    local image_tag="$3"
    local dev_mode="${4:-false}"

    local tmpdir
    tmpdir="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '${tmpdir}'" RETURN

    # Use Python for correct JSON serialisation
    python3 - \
        "${kernel_name}" "${display_name}" \
        "${IMAGE_REPO}:${image_tag}" "${image_tag}" \
        "${HOME}" "${dev_mode}" "${REPO_DIR}" \
        "${tmpdir}/kernel.json" \
    <<'PYEOF'
import json, sys

kernel_name, display_name, image, tag, home, dev_mode, repo_dir, out = sys.argv[1:]
dev_mode = dev_mode == "true"

DATA_MOUNT = (
    "/pscratch/sd/b/bkieft/metatlas_lite_data"
    ":/pscratch/sd/b/bkieft/metatlas_lite_data:ro"
)

argv = [
    "podman", "run", "--rm",
    "--network", "host",
    "-v", DATA_MOUNT,
    "-v", f"{home}:{home}",
]

env = {
    "METATLAS2_IMAGE_TAG": tag,
    "HOME": home,
    "JUPYTERHUB_SERVICE_PREFIX": "/",
}

if dev_mode:
    argv += ["-v", f"{repo_dir}:/dev_repo:ro"]
    env["PYTHONPATH"] = "/dev_repo"

for k, v in env.items():
    argv += ["-e", f"{k}={v}"]

argv += [image, "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]

spec = {
    "argv": argv,
    "display_name": display_name,
    "language": "python",
    "metadata": {"debugger": False},
}
with open(out, "w") as f:
    json.dump(spec, f, indent=2)
PYEOF

    jupyter kernelspec install "${tmpdir}" --user --name "${kernel_name}"
    echo "Installed kernel '${kernel_name}' → ${display_name}"
}

# Always install the two standard kernels
install_kernel "metatlas2"     "metatlas2 (latest)"            "latest" "false"
install_kernel "metatlas2-dev" "metatlas2 (dev – local repo)"  "latest" "true"

# Optionally install a pinned version kernel
if [[ -n "${EXTRA_TAG}" ]]; then
    install_kernel "metatlas2-${EXTRA_TAG}" "metatlas2 (${EXTRA_TAG})" "${EXTRA_TAG}" "false"
fi

echo ""
echo "Installed kernels:"
jupyter kernelspec list
