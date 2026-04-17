#!/bin/bash
# Pull the latest metatlas2 container image into the shifter cache.
# shifter is used by both metatlas2.sh and the JupyterLab kernel.
#
# Register as a cronjob to keep the local image cache current:
#   */5 * * * * /path/to/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1

IMAGE="ghcr.io/bkieft-usa/metatlas2:latest"
echo "[$(date -Iseconds)] Pulling docker:${IMAGE} via shifterimg ..."
OUTPUT="$(shifterimg pull "docker:${IMAGE}" 2>&1)"
echo "${OUTPUT}"
if echo "${OUTPUT}" | grep -q "status: FAILURE"; then
    echo "[$(date -Iseconds)] Pull failed." >&2
    exit 1
fi
echo "[$(date -Iseconds)] Pull complete."
