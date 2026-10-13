#!/bin/bash

IMAGE="ghcr.io/bkieft-usa/metatlas2:latest"
echo "[$(date -Iseconds)] Pulling docker://${IMAGE} via shifterimg ..."
OUTPUT="$(shifterimg pull "docker://${IMAGE}" 2>&1)"
echo "${OUTPUT}"

if echo "${OUTPUT}" | grep -q "status: FAILURE"; then
    echo "[$(date -Iseconds)] Pull failed. Make sure you're logged in with 'shifterimg login ghcr.io' and your github credentials (classic token)." >&2
    exit 1
fi
echo "[$(date -Iseconds)] Pull complete."