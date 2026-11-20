#!/bin/bash
# Summarise untargeted projects (errored / running / completed+uploaded) for one department
# over the last <timeback> days, based on `metatlas2.sh untargeted status`.
#
# Usage: untargeted_project_summary_report.sh <timeback_days> <department: jgi|eb>
# Env:   METATLAS2_WRAPPER (default: ../metatlas2.sh next to this script)
#        METATLAS2_WRAPPER_ARGS (e.g. "--image v1.2.3"; default empty)

if [ $# -ne 2 ]; then
    echo "Usage: $0 <timeback> <department>"
    echo "Where <timeback> is the number of days to look back and <department> is the department to filter on ('jgi' or 'eb')"
    exit 1
fi

timeback=$1
department=$2
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WRAPPER="${METATLAS2_WRAPPER:-${SCRIPT_DIR}/../metatlas2.sh}"

printf '%s completion report for untargeted pipeline.\n\t- Run on %s\n\n' "${department^^}" "$(date +"%Y-%m-%d")"

temp_status_table=$(mktemp)
temp_status_table_subset=$(mktemp)
temp_sorted_file=$(mktemp)
temp_timestamps=$(mktemp)
temp_projects_subset=$(mktemp)
temp_projects_subset_filtered=$(mktemp)
trap 'rm -f "$temp_status_table" "$temp_status_table_subset" "$temp_sorted_file" "$temp_timestamps" "$temp_projects_subset" "$temp_projects_subset_filtered"' EXIT

# Get the status of all recent projects (stdout is the tab-separated table only)
# shellcheck disable=SC2086
"${WRAPPER}" ${METATLAS2_WRAPPER_ARGS:-} untargeted status --print_recent 100 > "$temp_status_table"

# Only keep lines with project statuses (they start with the Modified timestamp)
grep -E '^[0-9]+' "$temp_status_table" > "$temp_status_table_subset"
sort -k1,1 "$temp_status_table_subset" > "$temp_sorted_file"
cut -f1 "$temp_sorted_file" > "$temp_timestamps"

current_timestamp=$(date -Iseconds)
target_timestamp=$(date -d "$current_timestamp - $timeback days" +"%Y-%m-%d %H:%M:%S")

# Find the status timestamp closest to the target date
closest_timestamp=""
closest_diff=9999999999
while IFS= read -r timestamp; do
    current_diff=$(($(date -d "$timestamp" +%s) - $(date -d "$target_timestamp" +%s)))
    current_diff=${current_diff#-}
    if [ "$current_diff" -lt "$closest_diff" ]; then
        closest_diff=$current_diff
        closest_timestamp=$timestamp
    fi
done < "$temp_timestamps"

# Projects modified since the closest timestamp
awk -v ts="$closest_timestamp" '
    $0 ~ ts { found = 1; next }
    found { print }
' "$temp_sorted_file" > "$temp_projects_subset"

# Keep the requested department (project field 2)
awk -F'_' -v dept="$department" 'tolower($2) == tolower(dept)' "$temp_projects_subset" > "$temp_projects_subset_filtered"

errors=$(grep -c "09 error" "$temp_projects_subset_filtered")
errored_projects=$(grep "09 error" "$temp_projects_subset_filtered" | cut -f2 | sed 's/^/\t- /1')
runnings=$(grep -cE "04 running|01 initiation" "$temp_projects_subset_filtered")
running_projects=$(grep -E "04 running|01 initiation" "$temp_projects_subset_filtered" | cut -f2 | sed 's/^/\t- /1')
success_pattern=$'07 complete\t07 complete\t07 complete\t07 complete\t08 uploaded'
successes=$(grep -cF "$success_pattern" "$temp_projects_subset_filtered")
successful_projects=$(grep -F "$success_pattern" "$temp_projects_subset_filtered" | cut -f2,8 | rev | sed 's/ /\t/1' | rev | awk -F'\t' '{print $1" (Uploaded on "$2")"}' | sed 's/^/\t- /1')

printf 'Status of %s untargeted pipline runs in the last %s days\n' "${department^^}" "$timeback"
printf '\nProjects unsuccessfully ended with error status: %s\n' "$errors"
if [ "$errors" -gt 0 ]; then
    printf '%s\n' "$errored_projects"
fi
printf '\nProjects currently running: %s\n' "$runnings"
if [ "$runnings" -gt 0 ]; then
    printf '%s\n' "$running_projects"
fi
printf '\nProjects successfully completed and uploaded: %s\n' "$successes"
if [ "$successes" -gt 0 ]; then
    printf '%s\n' "$successful_projects"
fi
