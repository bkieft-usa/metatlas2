#!/bin/bash
# Print WARNING/ERROR/CRITICAL/Traceback lines from the last N untargeted pipeline runs.
# Runs are delimited by the "Successfully started" line the pipeline logs at startup.
#
# Usage: untargeted_warning-error_report.sh <cycles>
# Env:   UNTARGETED_LOG_FILE (default: /global/cfs/cdirs/m2650/untargeted_logs/untargeted_pipeline.log)

LOG_FILE="${UNTARGETED_LOG_FILE:-/global/cfs/cdirs/m2650/untargeted_logs/untargeted_pipeline.log}"

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log file does not exist: ${LOG_FILE}"
    exit 1
fi

if [ $# -ne 1 ]; then
    echo "Usage: $0 <cycles>"
    echo "Where <cycles> is the number of pipeline runs to look back"
    exit 1
fi

cycles=$1

printf 'Error and warning report for untargeted pipeline.\n\t- Run on %s\n\n' "$(date)"
printf 'Checked the last %s untargeted pipeline runs.\n\n' "$cycles"

errors=$(awk -v cycles="$cycles" '
/Successfully started/ { positions[++count] = NR }
{ lines[NR] = $0 }
END {
    if (count >= cycles) {
        start = positions[count - (cycles - 1)]
        for (i = start + 1; i <= NR; i++) {
            print lines[i]
        }
    }
}
' "${LOG_FILE}" | grep -E "WARNING|CRITICAL|ERROR|Traceback")

if [ -z "$errors" ]; then
    printf "There were no warnings or errors!\n\n"
else
    printf 'Errors or warnings that occurred during last untargeted pipeline cycle:\n\n'
    printf '%s\n\n' "$errors"
    printf 'Check log here: %s\n\n' "${LOG_FILE}"
fi
