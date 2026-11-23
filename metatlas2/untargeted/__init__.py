"""Untargeted metabolomics pipeline (MZmine -> GNPS2 FBMN -> zip/Google Drive).

Ported from the legacy ``metatlas/untargeted`` scripts. Project state is
tracked in the LabKey LIMS list ``untargeted_tasks``; see
``docs/untargeted_workflow.md`` for an end-to-end description.

Modules
-------
run_untargeted_pipeline  CLI entry point (run, status, mirror-raw, write-config).
config            Default settings, YAML loading and CLI overrides.
status            Status vocabulary, per-polarity path helpers, LIMS dataframe filters.
lims              LabKey client (lazy) plus an in-memory stand-in for tests/dry runs.
project_setup     Step 1: find new projects and write MZmine inputs.
mzmine            MZmine batch/sbatch files, Step 2 (status) and Step 3 (queue jobs).
feature_filtering Feature counts and background-filtered peak-height tables.
gnps2             GNPS2 HTTP/SFTP client, FBMN/Everything Bagel params, MGF helpers.
packaging         Step 7: zip results and upload to Google Drive.
pipeline          Orchestrates Steps 1-7 (plus Steps 4-6 implementations).
status_report     Tabular status report used by the email scripts.
"""
