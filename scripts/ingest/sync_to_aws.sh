#!/bin/bash
# Archive raw data (.raw files and */other/*) to S3 Glacier with `aws s3 sync`.
# Ported from metatlas-repo/utils/sync_to_aws.sh. Same source, destination, filters and
# log file. Lines are timestamped like moreutils `ts` ("Nov 23 09:50:04 ..."), which the
# AWS email report parses, without needing `ts` installed.
#
# Usage: sync_to_aws.sh <jgi|egsb>
# Env:   AWS_CLI (default: aws), RAW_DATA_DIR, S3_RAW_DATA (default s3://northen-archive/raw_data),
#        AWS_SYNC_LOG_DIR (default /global/cfs/cdirs/m2650/aws_s3_sync_log)

usage() {
  >&2 echo "ERROR: Usage: $0 jgi|egsb"
  exit 128
}

[ "$#" -ne "1" ] && usage
! { [ "$1" = "jgi" ] || [ "$1" = "egsb" ]; } && usage

nersc_raw_data="${RAW_DATA_DIR:-/global/cfs/cdirs/metatlas/raw_data}"
s3_raw_data="${S3_RAW_DATA:-s3://northen-archive/raw_data}"
log="${AWS_SYNC_LOG_DIR:-/global/cfs/cdirs/m2650/aws_s3_sync_log}/${1}.log"

_ts() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%b %d %H:%M:%S')" "$line"; done; }

"${AWS_CLI:-aws}" s3 sync --storage-class GLACIER \
            --color off \
            --no-progress \
            --exclude '*' \
            --include '*.raw' \
            --include '*/other/*' \
            "${nersc_raw_data}/${1}" \
            "${s3_raw_data}/${1}" \
            2>&1 \
| _ts \
| tee -a "${log}"
