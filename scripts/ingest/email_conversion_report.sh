#!/bin/bash
# Email the weekly file-conversion report for jgi or egsb.
#
# Usage: email_conversion_report.sh <jgi|egsb> <days>
# Env:   METATLAS2_REPORT_SENDER   optional mailx -R reply-to address
#        REPORT_RECIPIENTS_DIR     (default /global/cfs/cdirs/metatlas/raw_data);
#                                  recipients are read from <dir>/email_<jgi|egsb>_reports
set -euo pipefail
IFS=$'\n\t'

if [ "$#" -ne 2 ] || { [ "$1" != "jgi" ] && [ "$1" != "egsb" ]; }; then
    >&2 printf "Usage: %s search_directory days\n" "$0"
    >&2 printf "    search_directory is one of 'jgi' or 'egsb'\n"
    >&2 printf "    days is how many days back to report on\n"
    exit 128
fi

printf "%s INFO: generating conversion report for %s covering %s days...\n" "$(date '+%b %d %H:%M:%S')" "$1" "$2"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RECIPIENTS_DIR="${REPORT_RECIPIENTS_DIR:-/global/cfs/cdirs/metatlas/raw_data}"
addresses="$(cat "${RECIPIENTS_DIR}/email_${1}_reports")"
sender_args=()
[[ -n "${METATLAS2_REPORT_SENDER:-}" ]] && sender_args=(-R "${METATLAS2_REPORT_SENDER}")

summary_line=$( "${SCRIPT_DIR}/conversion_report.sh" "$1" "$2" 2>&1 1>/dev/null | head -n1 )
report_body=$( "${SCRIPT_DIR}/conversion_report.sh" "$1" "$2" 2>/dev/null )

eval "$summary_line"          # creates $converted $failed $pending $in_progress
subject="${1} conversion report: ${converted} succeeded, ${failed} failed, ${pending} pending, ${in_progress} in-progress"
# shellcheck disable=SC2086
printf "%s\n" "$report_body" | mailx -s "$subject" "${sender_args[@]}" ${addresses}

printf "%s INFO: report generation and emailing complete\n" "$(date '+%b %d %H:%M:%S')"
