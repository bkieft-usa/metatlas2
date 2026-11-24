#!/bin/bash

IMAGE="ghcr.io/biorack/metatlas2:latest"
echo "[$(date -Iseconds)] Pulling docker:${IMAGE} via shifterimg ..."
OUTPUT="$(shifterimg pull "docker:${IMAGE}" 2>&1)"
echo "${OUTPUT}"
if echo "${OUTPUT}" | grep -q "status: FAILURE"; then
    echo "[$(date -Iseconds)] Pull failed." >&2
    echo "[$(date -Iseconds)] The image is private: log in with 'shifterimg login ghcr.io' (GitHub username + classic personal access token with the read:packages scope). Credentials are per NERSC account (~/.udiRoot/.cred)." >&2
    exit 1
fi
echo "[$(date -Iseconds)] Pull complete."
