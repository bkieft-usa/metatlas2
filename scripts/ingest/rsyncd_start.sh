#!/bin/bash
# Start the rsync daemon on dtn04.nersc.gov that receives raw data from metabo-dtn
# (scripts/ingest/file_transfer_sync.sh) and writes it into raw_data/{jgi,egsb}.
# Ported from metatlas-repo/utils/rsyncd.start.sh; it now reads the config that sits
# next to this script (rsyncd.<hostname>.conf) instead of the legacy repo.
#
# msdata's crontab on dtn04.nersc.gov (restarts the daemon if it isn't running):
#   */5 * * * * /bin/flock -w 0 /tmp/msdata.rsyncd.lock /global/cfs/cdirs/metatlas/tools/metatlas2/scripts/ingest/rsyncd_start.sh
#
# If the daemon is already running, the new one can't bind the port and exits;
# that's expected. The daemon re-reads its config on every connection, so config
# edits pulled into the checkout take effect without a restart. Changing the
# config *path* (e.g. moving checkouts) needs the running daemon killed first.

set -euo pipefail

umask 027
machine="$(uname -n)"
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
config="${script_dir}/rsyncd.${machine}.conf"
[[ -f "${config}" ]] || { echo "rsyncd config not found: ${config}" >&2; exit 1; }
sg metatlas "rsync --daemon --config ${config} --port=39653"
