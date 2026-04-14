#!/bin/bash
# Pull the latest metatlas2 container image.
#
# Register as a cronjob to keep the local image cache current:
#   */5 * * * * /path/to/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1

IMAGE="ghcr.io/bkieft-usa/metatlas2:latest"
echo "[$(date -Iseconds)] Pulling ${IMAGE} ..."
if podman pull "${IMAGE}"; then
    echo "[$(date -Iseconds)] Pull complete."
else
    echo "[$(date -Iseconds)] Pull failed." >&2
    exit 1
fi
