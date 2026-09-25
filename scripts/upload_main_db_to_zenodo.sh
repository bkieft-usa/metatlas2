#!/bin/bash
# Upload the metatlas main database (metatlas.duckdb) to Zenodo as a new version.
#
# Run this script on NERSC whenever the main database has changed and you want
# non-NERSC users (macOS, other HPC) to receive the updated database on their
# next metatlas2.sh run.
#
# Prerequisites:
#   1. Set ZENODO_ACCESS_TOKEN in your environment (or ~/.bashrc):
#        export ZENODO_ACCESS_TOKEN="your-personal-access-token"
#      Generate one at: https://zenodo.org/account/settings/applications/tokens/new/
#      Required scopes: deposit:write, deposit:actions
#
#   2. Set METATLAS_DATA_DIR in your environment (already required for the workflow):
#        export METATLAS_DATA_DIR=/global/cfs/cdirs/metatlas/data
#
#   3. On first use, set ZENODO_CONCEPT_ID to the concept record ID of the
#      existing Zenodo deposit (the stable ID that never changes across versions).
#      After the first upload, this is printed and also written to
#      $METATLAS_DATA_DIR/databases/main_db/.zenodo_concept_id for future runs.
#      Leave ZENODO_CONCEPT_ID unset to create a brand-new deposit.
#
# After a successful upload this script prints the new deposit DOI.
# Update the ZENODO_MAIN_DB_DOI constant in scripts/metatlas2.sh to that value
# so non-NERSC users automatically download the new version on their next run.
#
# Usage:
#   bash scripts/upload_main_db_to_zenodo.sh
#   ZENODO_CONCEPT_ID=12345 bash scripts/upload_main_db_to_zenodo.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZENODO_API="https://zenodo.org/api"
DB_FILENAME="metatlas.duckdb"

if [[ -z "${ZENODO_ACCESS_TOKEN:-}" ]]; then
    echo "Error: ZENODO_ACCESS_TOKEN is not set." >&2
    echo "Generate a token at https://zenodo.org/account/settings/applications/tokens/new/" >&2
    echo "then: export ZENODO_ACCESS_TOKEN=your-token" >&2
    exit 1
fi

if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
    echo "Error: METATLAS_DATA_DIR is not set." >&2
    exit 1
fi

DB_PATH="${METATLAS_DATA_DIR}/databases/main_db/${DB_FILENAME}"
CONCEPT_ID_FILE="${METATLAS_DATA_DIR}/databases/main_db/.zenodo_concept_id"

if [[ ! -f "${DB_PATH}" ]]; then
    echo "Error: Database file not found: ${DB_PATH}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve concept record ID (stable across versions)
# ---------------------------------------------------------------------------
if [[ -n "${ZENODO_CONCEPT_ID:-}" ]]; then
    CONCEPT_ID="${ZENODO_CONCEPT_ID}"
elif [[ -f "${CONCEPT_ID_FILE}" ]]; then
    CONCEPT_ID=$(cat "${CONCEPT_ID_FILE}")
    echo "Using concept ID from ${CONCEPT_ID_FILE}: ${CONCEPT_ID}"
else
    CONCEPT_ID=""
fi

# ---------------------------------------------------------------------------
# Create a new deposit (or new version of an existing deposit)
# ---------------------------------------------------------------------------
if [[ -n "${CONCEPT_ID}" ]]; then
    echo "Creating new version of Zenodo deposit (concept ID: ${CONCEPT_ID})..."

    # Find the latest published record ID for this concept via the deposit API.
    # The /records/{concept_id} endpoint returns the latest published record.
    LATEST_RECORD=$(curl -s \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
        "${ZENODO_API}/records/${CONCEPT_ID}" \
    )
    LATEST_ID=$(echo "${LATEST_RECORD}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "  Latest published record ID: ${LATEST_ID}"

    # Create a new version draft via the deposit API (not the /records/ API).
    NEW_VERSION=$(curl -s -X POST \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
        "${ZENODO_API}/deposit/depositions/${LATEST_ID}/actions/newversion" \
    )
    # The newversion action returns the *parent* record; the new draft ID is in links.latest_draft.
    DRAFT_URL=$(echo "${NEW_VERSION}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
url = d.get('links', {}).get('latest_draft')
if not url:
    raise SystemExit('latest_draft URL not found. Response: ' + json.dumps(d))
print(url)
")
    # Extract the numeric deposit ID from the draft URL.
    DEPOSIT_ID=$(basename "${DRAFT_URL}")
    echo "New version draft created: deposit ID ${DEPOSIT_ID}"

    # No need to delete old files — the bucket PUT below overwrites by filename.

else
    echo "Creating new Zenodo deposit..."
    METADATA=$(python3 -c "
import json
print(json.dumps({
    'metadata': {
        'title': 'Metatlas2 Main Database',
        'upload_type': 'dataset',
        'description': (
            'The metatlas2 main DuckDB database containing compounds, atlases, '
            'and project registry for the metatlas2 targeted metabolomics workflow. '
            'Downloaded automatically by metatlas2.sh when running outside NERSC.'
        ),
        'creators': [{'name': 'Metatlas2 Team', 'affiliation': 'Lawrence Berkeley National Laboratory'}],
        'access_right': 'open',
        'license': 'cc-by-4.0',
        'keywords': ['metabolomics', 'metatlas', 'database', 'targeted analysis'],
    }
}))
")
    NEW_DEPOSIT=$(curl -s -X POST \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "${METADATA}" \
        "${ZENODO_API}/deposit/depositions" \
    )
    DEPOSIT_ID=$(echo "${NEW_DEPOSIT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "New deposit created: ID ${DEPOSIT_ID}"
fi

# ---------------------------------------------------------------------------
# Upload the database file via the S3-compatible bucket API.
#
# The legacy /files endpoint has an nginx body-size limit (~100 MB) that
# rejects large files with HTTP 413.  The bucket API streams directly to
# Zenodo's object store and has no such limit.
# ---------------------------------------------------------------------------
DB_SIZE=$(du -sh "${DB_PATH}" | cut -f1)
echo "Uploading ${DB_FILENAME} (${DB_SIZE}) to deposit ${DEPOSIT_ID}..."

# Retrieve the deposit record to get the bucket URL.
DEPOSIT_META=$(curl -s \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    "${ZENODO_API}/deposit/depositions/${DEPOSIT_ID}" \
)
BUCKET_URL=$(echo "${DEPOSIT_META}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
links = d.get('links', {})
bucket = links.get('bucket')
if not bucket:
    raise SystemExit('bucket URL not found in deposit metadata: ' + json.dumps(links))
print(bucket)
")
echo "  Bucket URL: ${BUCKET_URL}"

# Stream the file directly to the bucket (no nginx body-size limit).
UPLOAD_RESPONSE=$(curl -s -X PUT \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --upload-file "${DB_PATH}" \
    "${BUCKET_URL}/${DB_FILENAME}" \
)

UPLOAD_KEY=$(echo "${UPLOAD_RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('key','ERROR'))" 2>/dev/null || echo "ERROR")
if [[ "${UPLOAD_KEY}" == "ERROR" ]]; then
    echo "Error: Upload failed. Response:" >&2
    echo "${UPLOAD_RESPONSE}" >&2
    exit 1
fi
echo "Upload complete (key: ${UPLOAD_KEY})"

# ---------------------------------------------------------------------------
# Publish the deposit
# ---------------------------------------------------------------------------
echo "Publishing deposit ${DEPOSIT_ID}..."
PUBLISH_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    "${ZENODO_API}/deposit/depositions/${DEPOSIT_ID}/actions/publish" \
)

NEW_DOI=$(echo "${PUBLISH_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['doi'])")
NEW_CONCEPT_ID=$(echo "${PUBLISH_RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['conceptrecid'])")

# ---------------------------------------------------------------------------
# Save concept ID for future runs
# ---------------------------------------------------------------------------
echo "${NEW_CONCEPT_ID}" > "${CONCEPT_ID_FILE}"

# ---------------------------------------------------------------------------
# Print instructions
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "  New DOI:        https://doi.org/${NEW_DOI}"
echo "  Concept ID:     ${NEW_CONCEPT_ID}  (saved to ${CONCEPT_ID_FILE})"
echo ""
echo "ACTION REQUIRED: Update ZENODO_MAIN_DB_DOI in scripts/metatlas2.sh:"
echo ""
echo "  ZENODO_MAIN_DB_DOI=\"https://doi.org/${NEW_DOI}\""
echo ""
echo "Commit and push this change so non-NERSC users automatically"
echo "download the new database on their next metatlas2.sh run."
echo "=========================================="
