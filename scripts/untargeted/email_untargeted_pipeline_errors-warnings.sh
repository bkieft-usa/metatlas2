#!/bin/bash
# Email the warnings/errors report for the last N pipeline runs.
#
# Usage: email_untargeted_pipeline_errors-warnings.sh <cycles>
# Env:   UNTARGETED_REPORT_SENDER   optional mailx -R reply-to address
#        UNTARGETED_ERROR_RECIPIENTS_FILE (default: /global/cfs/cdirs/metatlas/raw_data/email_untargeted_errors-warnings)
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <cycles>"
    echo "Where <cycles> is the number of pipeline runs to look back"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RECIPIENTS_FILE="${UNTARGETED_ERROR_RECIPIENTS_FILE:-/global/cfs/cdirs/metatlas/raw_data/email_untargeted_errors-warnings}"
addresses="$(cat "${RECIPIENTS_FILE}")"
sender_args=()
[[ -n "${UNTARGETED_REPORT_SENDER:-}" ]] && sender_args=(-R "${UNTARGETED_REPORT_SENDER}")

# shellcheck disable=SC2086
"${SCRIPT_DIR}/untargeted_warning-error_report.sh" "$1" | \
  mailx -S sendwait -s "untargeted pipeline warnings and errors report" "${sender_args[@]}" ${addresses}

printf "INFO: report generation and emailing complete\n"
