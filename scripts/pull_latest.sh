#!/bin/bash

IMAGE="ghcr.io/bkieft-usa/metatlas2:latest"
echo "[$(date -Iseconds)] Pulling docker:${IMAGE} via shifterimg ..."
OUTPUT="$(shifterimg pull "docker:${IMAGE}" 2>&1)"
echo "${OUTPUT}"
if echo "${OUTPUT}" | grep -q "status: FAILURE"; then
    echo "[$(date -Iseconds)] Pull failed." >&2
    echo "[$(date -Iseconds)] You may need to log in to the container registry with 'shifterimg login docker.io' and your github credentials (classic token)." >&2
    exit 1
fi
echo "[$(date -Iseconds)] Pull complete."
