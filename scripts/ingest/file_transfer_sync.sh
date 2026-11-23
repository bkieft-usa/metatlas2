#!/bin/bash
# Push raw data from JGI storage to NERSC (rsync daemon on dtn04 -> metatlas raw_data).
#
# Runs from cron as svc-metabolomics@lbl.gov on metabo-dtn.jgi.lbl.gov, from a
# checkout of this repo (replaces the hand-copied ~/bin/sync.sh). Crontab:
#
#   */5 * * * * cd $HOME/metatlas2 && git pull --quiet >> $HOME/logs/git_pull.log 2>&1
#   */5 * * * * $HOME/metatlas2/scripts/ingest/file_transfer_sync.sh jgi
#   */5 * * * * $HOME/metatlas2/scripts/ingest/file_transfer_sync.sh egsb
#   48 2 * * * /usr/sbin/logrotate --state $HOME/logs/logrotate.status $HOME/logs/logrotate.conf
#
# What is transferred (unchanged from sync.sh):
#   - per project directory (one level below <vol>/): *.raw files,
#     anything under */other/ (any case of "Other"), and NO_UNTARGETED.TXT opt-out files;
#   - everything else, deeper subdirectories, $RECYCLE.BIN/, filechecks/ and
#     Robo_copy_test/ are skipped; empty directories are not created.
#
# Logs (unchanged locations): rsync's own transfer log in $LOG_DIR/<vol>.log, and
# rsync stdout/stderr in $LOG_DIR/<vol>.error.log, each line prefixed with a
# timestamp like moreutils `ts` ("Nov 23 09:50:04 ..."), without needing `ts`.
# A flock per volume keeps runs from overlapping; a run that finds the lock held
# exits quietly.
#
# Usage: file_transfer_sync.sh <jgi|egsb>
# Env overrides (defaults match the legacy script):
#   NETAPP_MOUNT  (/net/storage.jgi.lbl.gov/vol_metabolomics)
#   RSYNCD_USER   (msdata)          DEST_HOST (dtn04.nersc.gov)   DEST_PORT (39653)
#   PASSWORD_FILE ($HOME/msdata_pw) LOG_DIR   ($HOME/logs)        RSYNC (rsync)

netapp_mount="${NETAPP_MOUNT:-/net/storage.jgi.lbl.gov/vol_metabolomics}"
rsyncd_user="${RSYNCD_USER:-msdata}"
dest_host="${DEST_HOST:-dtn04.nersc.gov}"
dest_port="${DEST_PORT:-39653}"
password_file="${PASSWORD_FILE:-${HOME}/msdata_pw}"
log_dir="${LOG_DIR:-${HOME}/logs}"

vol="${1:-}"

if [ "$#" -ne 1 ] || { [ "$vol" != "jgi" ] && [ "$vol" != "egsb" ]; }; then
    >&2 echo "Usage $0: vol_name"
    >&2 echo "    where vol_name is one of 'jgi' or 'egsb'"
    exit 128
fi

src="${netapp_mount}/${vol}/"
dest="rsync://${rsyncd_user}@${dest_host}:${dest_port}/${vol}/"
log_file="${log_dir}/${vol}.log"
log_error_file="${log_dir}/${vol}.error.log"
lock_file="/tmp/msdata.${vol}.lockfile"

mkdir -p "${log_dir}"

# Timestamp each line like moreutils `ts` (default format "%b %d %H:%M:%S").
_ts() { while IFS= read -r line; do printf '%s %s\n' "$(date '+%b %d %H:%M:%S')" "$line"; done; }

flock -n "$lock_file" stdbuf -oL \
   "${RSYNC:-rsync}" \
      --archive \
      --no-perms \
      --no-group \
      --chmod=ugo=rwX \
      --prune-empty-dirs \
      "--exclude=\$RECYCLE.BIN/" \
      "--exclude=filechecks/" \
      "--exclude=Robo_copy_test/" \
      "--include=*/[Oo]ther/***" \
      "--exclude=*/*/" \
      "--include=*/" \
      "--include=*.raw" \
      "--include=NO_UNTARGETED.TXT" \
      "--exclude=*" \
      "--password-file=$password_file" \
      "--log-file=${log_file}" \
      "$src" "$dest" 2>&1 | _ts >> "$log_error_file"
