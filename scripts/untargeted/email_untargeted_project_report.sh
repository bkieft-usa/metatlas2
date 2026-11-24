#!/bin/bash
# Email the per-department project completion report.
#
# Usage: email_untargeted_project_report.sh <timeback_days> <department: jgi|eb>
# Env:   UNTARGETED_REPORT_SENDER   optional mailx -R reply-to address
#        UNTARGETED_REPORT_RECIPIENTS_DIR (default: /global/cfs/cdirs/metatlas/raw_data);
#        recipients are read from <dir>/email_untargeted_reports_<department>
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <timeback> <department>"
    echo "Where <timeback> is the number of days to look back and <department> is the department to filter on ('jgi' or 'eb')"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RECIPIENTS_DIR="${UNTARGETED_REPORT_RECIPIENTS_DIR:-/global/cfs/cdirs/metatlas/raw_data}"
addresses="$(cat "${RECIPIENTS_DIR}/email_untargeted_reports_$2")"
sender_args=()
[[ -n "${UNTARGETED_REPORT_SENDER:-}" ]] && sender_args=(-R "${UNTARGETED_REPORT_SENDER}")

# shellcheck disable=SC2086
"${SCRIPT_DIR}/untargeted_project_summary_report.sh" "$1" "$2" | \
  mailx -S sendwait -s "untargeted projects completion report" "${sender_args[@]}" ${addresses}

printf "INFO: report generation and emailing complete\n"
