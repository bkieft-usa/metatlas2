#!/bin/bash
#
# Build (and publish) the sandbox data package used by `metatlas2.sh --sandbox`.
# Run on NERSC: it copies a handful of runs from a production project, shrinks
# them to the sandbox atlas windows, tars them, and publishes a new Zenodo
# version, then updates ZENODO_SANDBOX_DATA_DOI in scripts/metatlas2.sh.
#
# The package holds ONLY the h5 files. Everything else the sandbox uses
# (compounds, atlases, MS/MS refs, configs) lives in sandbox/inputs/ in the repo,
# so changing those does not need a new package -- unless you change atlas m/z or
# RT windows, which decide what h5 data is kept.
#
# Usage:
#   sandbox/prepare_sandbox_data_package.sh [output_dir] [--keep-existing-archive] [--no-h5-subset] [--skip-zenodo-upload]
#
# Options:
#   --keep-existing-archive  Skip package creation and only upload the existing tarball
#   --no-h5-subset           Keep the full h5 files (subsetting is on by default)
#   --skip-zenodo-upload     Build the tarball only (useful for testing)
#
# Requires: METATLAS_DATA_DIR, the repo .venv (uv sync), and ZENODO_ACCESS_TOKEN to publish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -x "${REPO_DIR}/.venv/bin/python" ]]; then
    echo "Error: ${REPO_DIR}/.venv not found. Run 'uv sync' in the repo first." >&2
    exit 1
fi
PYTHON="${REPO_DIR}/.venv/bin/python"

KEEP_EXISTING=false
SUBSET_H5=true
SKIP_ZENODO_UPLOAD=false
OUTPUT_DIR=""
for arg in "$@"; do
    case "$arg" in
        --keep-existing-archive) KEEP_EXISTING=true ;;
        --no-h5-subset)          SUBSET_H5=false ;;
        --skip-zenodo-upload)    SKIP_ZENODO_UPLOAD=true ;;
        *)                       OUTPUT_DIR="$arg" ;;
    esac
done

if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
    echo "Error: METATLAS_DATA_DIR is not set." >&2
    exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-${METATLAS_DATA_DIR}/databases/sandbox_data}"
PACKAGE_NAME="metatlas2-sandbox-data"

# Production project the sandbox runs are copied from.
SOURCE_PROJECT="20260311_JGI_AE_511825_SorghAnth_final_EXP120B_HILICZ_USHXG03401"
SOURCE_BASE="${METATLAS_DATA_DIR}/raw_data/jgi/${SOURCE_PROJECT}"

# Must match GENERAL.owner in sandbox/inputs/configs/analysis_config.yaml (it selects the
# filename-parsing rules, so it must be jgi or egsb) and
# project_name in sandbox/sandbox_workflow.ipynb.
SANDBOX_OWNER="jgi"
SANDBOX_PROJECT="20260101_JGI_XX_000000_SANDBOX_test_EXP000_HILICZ_TESTXXXX"

QC_RUNS=(
    "Run196" 
    "Run150" 
    "Run58"
)
EXPERIMENTAL_RUNS=(
    "Run159" # NEG_MS2_56_T1-256534-8-Tr-RE_1
    "Run162" # NEG_MS2_57_T1-256534-8-Tr-RE_2
    "Run183" # NEG_MS2_58_T1-256534-8-Tr-RE_3
    "Run158" # POS_MS2_56_T1-256534-8-Tr-RE_1
    "Run161" # POS_MS2_57_T1-256534-8-Tr-RE_2
    "Run182" # POS_MS2_58_T1-256534-8-Tr-RE_3
    "Run153" # NEG_MS2_51_T1-256534-8-WT-RE_1
    "Run174" # NEG_MS2_52_T1-256534-8-WT-RE_2
    "Run186" # NEG_MS2_53_T1-256534-8-WT-RE_3
    "Run152" # POS_MS2_51_T1-256534-8-WT-RE_1
    "Run173" # POS_MS2_52_T1-256534-8-WT-RE_2
    "Run185" # POS_MS2_53_T1-256534-8-WT-RE_3
)

echo "========================================"
echo "Metatlas2 sandbox package"
echo "========================================"
echo "Source project: ${SOURCE_PROJECT}"
echo "Output:         ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
echo ""

if [[ "$KEEP_EXISTING" == true ]]; then
    [[ -f "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" ]] || {
        echo "Error: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz not found; drop --keep-existing-archive." >&2
        exit 1
    }
    echo "Using existing tarball (--keep-existing-archive)."
else
    [[ -d "${SOURCE_BASE}" ]] || { echo "Error: source project not found: ${SOURCE_BASE}" >&2; exit 1; }

    H5_DIR="${OUTPUT_DIR}/${PACKAGE_NAME}/raw_data/${SANDBOX_OWNER}/${SANDBOX_PROJECT}"
    rm -rf "${OUTPUT_DIR:?}/${PACKAGE_NAME}"
    mkdir -p "${H5_DIR}"

    echo "Copying h5 files..."
    MISSING=0
    for run in "${QC_RUNS[@]}" "${EXPERIMENTAL_RUNS[@]}"; do
        shopt -s nullglob
        run_files=("${SOURCE_BASE}/"*"_${run}".h5)
        shopt -u nullglob
        if [[ ${#run_files[@]} -eq 0 ]]; then
            echo "  ${run}: MISSING"
            MISSING=$((MISSING + 1))
            continue
        fi
        cp "${run_files[@]}" "${H5_DIR}/"
        echo "  ${run}: ${#run_files[@]} file(s)"
    done
    [[ ${MISSING} -eq 0 ]] || echo "Warning: ${MISSING} run(s) not found; adjust the run lists in this script."

    if [[ "$SUBSET_H5" == true ]]; then
        echo ""
        echo "Subsetting h5 files to the sandbox atlas windows..."
        "${PYTHON}" "${SCRIPT_DIR}/subset_h5.py" "${H5_DIR}" "${SCRIPT_DIR}"/inputs/atlases/*.tsv
    fi

    echo ""
    echo "Creating tarball..."
    tar -C "${OUTPUT_DIR}" -czf "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
    echo "   Compressed size: $(du -h "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" | cut -f1)"
fi

# Ensure we have the tarball size for upload step
if [[ -z "${TARBALL_SIZE:-}" ]]; then
    TARBALL_SIZE=$(du -h "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" | cut -f1)
fi

echo ""
if [[ "$SKIP_ZENODO_UPLOAD" == true ]]; then
    echo "Skipping Zenodo upload as requested."
    echo "Package location: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz (size: ${TARBALL_SIZE})"
    exit 0
fi

echo ""
echo "Uploading to Zenodo..."

# Check for Zenodo access token
if [[ -z "${ZENODO_ACCESS_TOKEN:-}" ]]; then
    echo "Warning: ZENODO_ACCESS_TOKEN not set. Skipping Zenodo upload."
    echo "Package location: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
    exit 0
fi

ZENODO_RECORD_ID="20172505"
VERSION_TAG="v$(date +%Y%m%d)"
TARBALL_PATH="${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"

echo "   Retrieving latest version of record ${ZENODO_RECORD_ID}..."

# First, get the published record to find the latest version's deposition ID
RECORD_RESPONSE=$(curl -s "https://zenodo.org/api/records/${ZENODO_RECORD_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# Extract the record_id (this is the deposition ID we need)
LATEST_DEPOSITION_ID=$(echo "$RECORD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null)

if [[ -z "$LATEST_DEPOSITION_ID" ]]; then
    echo "Error: Could not retrieve record. Response:"
    echo "$RECORD_RESPONSE"
    exit 1
fi

echo "   Latest deposition ID: ${LATEST_DEPOSITION_ID}"

# Check for any existing unpublished drafts by querying the deposition directly
echo "   Checking for existing drafts..."
DEPOSITION_CHECK=$(curl -s "https://zenodo.org/api/deposit/depositions/${LATEST_DEPOSITION_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# If the deposition has a 'newversion' link, there's no draft. If not, there might be a draft already.
HAS_NEWVERSION=$(echo "$DEPOSITION_CHECK" | python3 -c "import sys, json; d=json.load(sys.stdin); print('yes' if 'newversion' in d.get('links', {}) else 'no')" 2>/dev/null)

if [[ "$HAS_NEWVERSION" == "no" ]]; then
    # No newversion link means a draft already exists - need to find and delete it
    echo "   Found existing draft version, searching for it..."
    
    # Get all versions of this record
    CONCEPT_RECORD=$(curl -s "https://zenodo.org/api/records/${ZENODO_RECORD_ID}/versions" \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")
    
    # Find the draft deposition ID
    DRAFT_DEPOSITION_ID=$(echo "$CONCEPT_RECORD" | python3 -c "
import sys, json
hits = json.load(sys.stdin).get('hits', {}).get('hits', [])
for hit in hits:
    if hit.get('links', {}).get('self', '').startswith('https://zenodo.org/api/deposit/'):
        print(hit.get('id', ''))
        break
" 2>/dev/null)
    
    if [[ -n "$DRAFT_DEPOSITION_ID" ]] && [[ "$DRAFT_DEPOSITION_ID" != "$LATEST_DEPOSITION_ID" ]]; then
        echo "   Deleting draft deposition ${DRAFT_DEPOSITION_ID}..."
        curl -s -X DELETE "https://zenodo.org/api/deposit/depositions/${DRAFT_DEPOSITION_ID}" \
            -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" > /dev/null
        sleep 2
    fi
fi

echo "   Creating new version..."

# Create new version using the correct deposition ID
NEW_VERSION_RESPONSE=$(curl -s -X POST \
    "https://zenodo.org/api/deposit/depositions/${LATEST_DEPOSITION_ID}/actions/newversion" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# Extract the latest_draft URL
LATEST_DRAFT_URL=$(echo "$NEW_VERSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['links']['latest_draft'])" 2>/dev/null)

if [[ -z "$LATEST_DRAFT_URL" ]]; then
    echo "Error: Failed to create new version. Response:"
    echo "$NEW_VERSION_RESPONSE"
    exit 1
fi

echo "   New draft created: $LATEST_DRAFT_URL"

# Get the new deposition details
NEW_DEPOSITION=$(curl -s "$LATEST_DRAFT_URL" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

NEW_DEPOSITION_ID=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
BUCKET_URL=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; print(json.load(sys.stdin)['links']['bucket'])")

echo "   New deposition ID: ${NEW_DEPOSITION_ID}"

# Delete old files from the new version
echo "   Removing old files from draft..."
OLD_FILES=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; [print(f['id']) for f in json.load(sys.stdin)['files']]")

for file_id in $OLD_FILES; do
    curl -s -X DELETE \
        "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}/files/${file_id}" \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" > /dev/null
done

# Upload new tarball
echo "   Uploading ${PACKAGE_NAME}.tar.gz (${TARBALL_SIZE})..."
UPLOAD_RESPONSE=$(curl -s \
    -X PUT "${BUCKET_URL}/${PACKAGE_NAME}.tar.gz" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${TARBALL_PATH}")

UPLOAD_KEY=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('key', 'ERROR'))" 2>/dev/null)

if [[ "$UPLOAD_KEY" == "ERROR" ]] || [[ -z "$UPLOAD_KEY" ]]; then
    echo "Error: Upload failed. Response:"
    echo "$UPLOAD_RESPONSE"
    exit 1
fi

echo "   Upload successful: ${UPLOAD_KEY}"

# Update metadata with all required fields
echo "   Updating metadata with version ${VERSION_TAG}..."
METADATA_UPDATE=$(cat <<EOF
{
    "metadata": {
        "title": "Metatlas2 Sandbox Data",
        "upload_type": "dataset",
        "description": "Subset LC-MS/MS h5 files used by the metatlas2 developer sandbox (metatlas2.sh --sandbox).",
        "creators": [
            {
                "name": "Kieft, Brandon",
                "orcid": "0000-0003-2458-9844"
            }
        ],
        "version": "${VERSION_TAG}",
        "publication_date": "$(date +%Y-%m-%d)",
        "related_identifiers": [
            {
                "identifier": "https://github.com/biorack/metatlas2",
                "relation": "isSupplementTo",
                "resource_type": "software"
            }
        ]
    }
}
EOF
)

curl -s -X PUT \
    "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$METADATA_UPDATE" > /dev/null

# Publish the new version
echo "   Publishing new version..."
PUBLISH_RESPONSE=$(curl -s -X POST \
    "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}/actions/publish" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

NEW_DOI=$(echo "$PUBLISH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('doi', 'ERROR'))" 2>/dev/null)

if [[ "$NEW_DOI" == "ERROR" ]] || [[ -z "$NEW_DOI" ]]; then
    echo "Error: Publish failed. Response:"
    echo "$PUBLISH_RESPONSE"
    exit 1
fi

# Point scripts/metatlas2.sh at the newly published sandbox package.
METATLAS2_SCRIPT="${REPO_DIR}/scripts/metatlas2.sh"
if [[ -f "${METATLAS2_SCRIPT}" ]]; then
    DOI_URL="${NEW_DOI}"
    if [[ "${DOI_URL}" != https://doi.org/* ]]; then
        DOI_URL="https://doi.org/${DOI_URL}"
    fi

    DOI_PATTERN='^ZENODO_SANDBOX_DATA_DOI="[^"]*"'
    if grep -qE "${DOI_PATTERN}" "${METATLAS2_SCRIPT}"; then
        sed -i -E "s|${DOI_PATTERN}|ZENODO_SANDBOX_DATA_DOI=\"${DOI_URL}\"|" "${METATLAS2_SCRIPT}"
        echo "Updated scripts/metatlas2.sh with DOI: ${DOI_URL}"
        echo "Commit that change on your branch so sandbox users get the new package."
    else
        echo "Warning: Could not find ZENODO_SANDBOX_DATA_DOI line in scripts/metatlas2.sh" >&2
    fi
else
    echo "Warning: scripts/metatlas2.sh not found at ${METATLAS2_SCRIPT}" >&2
fi

echo ""
echo "========================================"
echo "SUCCESS!"
echo "========================================"
echo "New version published: ${VERSION_TAG}"
echo "DOI: ${NEW_DOI}"
echo "Local package: ${TARBALL_PATH}"
echo ""