# Untargeted Workflow

The untargeted pipeline turns new LC-MS/MS projects into feature tables, molecular networks and a zip archive on Google Drive. It does this without any per-project input:

1. Find new projects in LIMS.
2. Run MZmine feature finding on Slurm.
3. Run Feature-Based Molecular Networking (FBMN) and Everything Bagel on GNPS2.
4. Download the results.
5. Zip and upload.

It is a port of the legacy `metatlas/untargeted` scripts, which lived in `/global/common/software/m2650/metatlas-repo/metatlas/untargeted/`. Output files, LIMS columns and status values are unchanged, so the new pipeline can take over the same LIMS list and directories. [Differences from the legacy pipeline](#differences-from-the-legacy-pipeline) lists everything that changed.

---

## Running it

```bash
# Normal scheduled run (all steps)
metatlas2.sh untargeted run

# Read-only rehearsal: reads LIMS/GNPS2/disk, changes nothing
metatlas2.sh untargeted run --dry_run --log_to_stdout

# Re-run specific projects (bypasses status filters)
metatlas2.sh untargeted run --direct_input PROJ_A,PROJ_B --skip_steps Sync

# Project status table (used by the weekly email)
metatlas2.sh untargeted status --print_recent 50

# Mirror raw mzML files to GNPS2 by hand
metatlas2.sh untargeted mirror-raw --direct_input PROJ_A --mode soft     # or --mode hard

# Write the default config to edit
metatlas2.sh untargeted write-config my_untargeted.yaml
```

`metatlas2.sh untargeted …` runs `python -m metatlas2.untargeted.run_untargeted_pipeline …` in the metatlas2 container. Without the wrapper, for example from the repo venv on a login node, run that same module command. When there is no queue file and `sbatch` is on `PATH`, MZmine jobs are submitted directly.

### Scheduling (scrontab)

The legacy job ran every 6 hours as `msdata`. The equivalent entry looks like this:

```bash
#SCRON -A m2650 -q cron -t 02:00:00 --job-name run_untargeted_pipeline_v2
#SCRON -o /global/cfs/cdirs/m2650/untargeted_logs/untargeted_pipeline_scron.out
45 */6 * * * METATLAS_DATA_DIR=/global/cfs/cdirs/metatlas /path/to/metatlas2/scripts/metatlas2.sh untargeted run
```

The pipeline writes its own log to `PATHS.log_file`; the scron `-o` file only captures wrapper output and `sbatch` submissions. `metatlas2.sh` requires `METATLAS_DATA_DIR` to be set.

The email reports live in `scripts/untargeted/`:

| Script | Legacy schedule | What it does |
|---|---|---|
| `email_untargeted_pipeline_errors-warnings.sh <runs>` | daily, `5` | Mails WARNING/ERROR/CRITICAL/Traceback lines from the last N runs |
| `email_untargeted_project_report.sh <days> <jgi\|eb>` | weekly, `7 jgi` and `7 eb` | Mails counts of errored / running / completed-and-uploaded projects |

Both read their recipients from the same files as before (`/global/cfs/cdirs/metatlas/raw_data/email_untargeted_*`). Set `UNTARGETED_REPORT_SENDER` to give mailx a reply-to address.

---

## The seven steps

Each run executes all steps in order (minus `--skip_steps`). A project normally needs about three runs to finish, because the steps only *submit* work and the next run picks up the result. A step that raises an exception is logged with its traceback and the remaining steps still run; the run then exits non-zero.

| # | `--skip_steps` name | What happens |
|---|---|---|
| 1 | `Sync` | Find new projects in LIMS `lcmsrun_plus`, write MZmine inputs, insert rows into `untargeted_tasks` |
| 2 | `MZmine_Status` | Finished MZmine jobs become `07 complete`, with feature, control and MS/MS counts and background-filtered tables. Dead jobs become `09 error` |
| 3 | `MZmine_Submit` | Queue MZmine sbatch jobs (new projects; own errored jobs with retries left) |
| 4 | `FBMN_Status` | Poll GNPS2 task status pages |
| 5 | `FBMN_Submit` | Mirror MZmine results and raw mzML to GNPS2 over SFTP, strip the calibrant from the MGF, submit FBMN and Everything Bagel |
| 6 | `FBMN_Download` | Download the network `.graphml`, library hits `.tsv` and the GNPS2 link |
| 7 | `Zip_and_Upload` | Zip both polarities flat with the GNPS2 guide, shorten file names, and upload with rclone |

### Step 1: which projects are picked up

A raw-data directory (LIMS `parent_dir`) becomes a task when all of these hold:

- it has `POS`/`NEG` files in `lcmsrun_plus`;
- it is not already in `untargeted_tasks`;
- its newest file is more than 3 hours old;
- at least one file name contains `MS2`/`MSMS` but not `_MS1_`;
- there is no opt-out file (`NO_UNTARGETED.TXT` or `nountargeted.txt`, any case) in the raw-data directory. The legacy pipeline only checked `nountargeted.txt`, but the data transfer only copies `NO_UNTARGETED.TXT`, so the opt-out never took effect before.

The MZmine file list per polarity excludes:

- `QC`, `InjBl` and `ISTD` sample groups (field 12);
- names containing `Blank`, unless `skip_blank_filter` is set.

The MZmine template is chosen from the instrument field (field 6):

| Instrument field contains | Template |
|---|---|
| `iqx`, `idx`, or no match | `IQX-mzmine-3.7.2-batchparams.xml` (LIMS row 5) |
| `exp`, `exploris`, `qe` | `mzmine-3.7.2-batchparams.xml` (LIMS row 2) |

Use `--custom_mzmine_batch_params` (or `MZMINE.custom_batch_params`) to set one template for all projects.

### Status values (LIMS `untargeted_tasks`)

| Value | Meaning |
|---|---|
| `01 initiation` | MZmine inputs written, not yet submitted |
| `04 running` | MZmine queued/running, or FBMN submitted |
| `07 complete` | Outputs verified |
| `09 error` | Failed (or polarity had a pre-existing directory in Step 1) |
| `12 not relevant` | No files or not enough MS/MS for this polarity |
| `13 waiting` | FBMN waiting for MZmine |
| `08 uploaded` / `14 not uploaded` | Only in the status report (derived from the Drive listing) |

### Files per project polarity

Everything lives in `{output_dir}/{project}_{positive|negative}/`, with names prefixed `{project}_{polarity}`:

| Suffix | Written by |
|---|---|
| `_metadata.tab`, `_batch-params.xml`, `_filelist.txt`, `_mzmine-sbatch.sbatch`, `_mzmine.sh` | Step 1 |
| `_mzmine-queued.txt` *(new)* | Step 3. One epoch per queueing; also counts retries |
| `_mzmine-job-id.txt` | Host wrapper after `sbatch`. Format `{prefix}={jobid}`, as before |
| `_mzmine-status.txt` *(new)* | The sbatch script: `RUNNING <jobid> <epoch>`, then `DONE <epoch>` or `FAILED <rc> <epoch>` |
| `-mzmine.out/.err`, `.mgf`, `_quant.csv`, `_peak-height.csv` | MZmine |
| `_peak-height-filtered-*.csv`, `_peak-height-filtering-summary.csv` | Step 2 |
| `_gnps2-fbmn-task.txt`, `_gnps2-eb-task.txt` | Step 5 |
| `_gnps2-page-link.txt`, `_gnps2-fbmn-network.graphml`, `_gnps2-fbmn-library-results.tsv` | Step 6 |

The zip is written to `{download_dir}/{project}.zip`. The two new marker files are not included in it.

---

## How Slurm works from the container

`sbatch` and `squeue` don't exist inside the metatlas2 container, so the work is split between the container and the host:

1. In the container, Step 3 sets LIMS to `04 running` and appends `<sbatch>\t<job-id file>\t<prefix>` to a queue file. `metatlas2.sh` creates that file under `~/.metatlas2/untargeted_sbatch_queue/` and passes it with `--sbatch_queue_file`.
2. After the container exits, the wrapper runs `sbatch --parsable` on the host for each queued job and writes `_mzmine-job-id.txt`. It does this even if the container failed.
3. MZmine itself runs directly on the compute node from `/global/common/software/m2650/mzmine_parameters`, not in a container. The sbatch script records `RUNNING`/`DONE`/`FAILED` in `_mzmine-status.txt`.
4. Step 2 decides whether a `04 running` job whose outputs are missing is dead. This replaces `squeue`:

   | Markers | Verdict |
   |---|---|
   | Status file ends in `FAILED` or `DONE` | Dead |
   | `RUNNING` for longer than the sbatch time limit plus `status_grace_hours` | Dead (killed) |
   | Queued, but no job-id file after `status_grace_hours` | Dead (`sbatch` never ran) |
   | Job-id file but no `RUNNING` after `max_pending_hours` | Dead |
   | Job submitted by the legacy pipeline (job-id file only) | Dead once the file is older than `max_pending_hours` |

5. When Step 3 is automatic (no `--direct_input`), it resubmits `09 error` jobs with `retry_time` (6 h) up to `max_retries` times. It only does this for jobs that have a `_mzmine-queued.txt` marker. Projects that errored under the legacy pipeline still need `--direct_input`, so switching pipelines does not trigger a wave of resubmissions.

---

## Configuration

All defaults are built into `metatlas2/untargeted/config.py`. `configs/untargeted/untargeted_pipeline.yaml` is an annotated copy of them (a unit test keeps the two identical). Pass `--config FILE` with any subset of these sections:

| Section | Notable keys |
|---|---|
| `PATHS` | `raw_data_dir`, `output_dir`, `download_dir`, `log_file`, `raw_data_subdir`, `fallback_raw_data_subdirs`, `group` |
| `LIMS` | `server`, `container`, `api_key_file`, `tasks_table`, `files_table` |
| `PROJECT_SELECTION` | `min_file_age_hours`, `validate_names`, `skip_blank_filter`, `background_designator` |
| `MZMINE` | `executable`, `batch_file_dir`, `template_rules`, `custom_batch_params`, `slurm.*`, `auto_resubmit_errors`, `max_retries`, `status_grace_hours`, `max_pending_hours` |
| `FILTERING` | `background_ratio`, `zero_value`, HILIC/C18 solvent front/end |
| `GNPS2` | `username`, `credentials_file`, SFTP host/port, mirroring and Everything Bagel switches, `calibrant_mz`, `fbmn_params`, `everything_bagel_params` |
| `GDRIVE` | `upload`, `remote`, `add_documentation`, `doc_name`, `abridged_filenames`, `min_features` |

CLI flags override the config. The legacy flag names still work, for example `--background_ratio`, `--no_gdrive_upload`, `--skip_mirror_raw_data` and `--overwrite_fbmn`. Run `metatlas2.sh untargeted run --help` for the full list.

Credentials stay in their existing files and are only referenced by path:

| Credential | Location |
|---|---|
| LabKey key | `/global/cfs/cdirs/metatlas/labkey_user.txt` |
| GNPS2 password | `/global/cfs/cdirs/metatlas/gnps2/gnps2_bpbowen.txt` |
| Google Drive remote | The rclone config of the user running the pipeline |

### Dry runs

`--dry_run` is strictly read-only. It reads LIMS, the filesystem and the GNPS2 status pages. It does **not**:

- write to LIMS;
- queue or submit Slurm jobs;
- mirror files to or submit jobs on GNPS2;
- modify MGF files;
- download, zip or upload anything;
- create project directories.

Everything it would have done is logged with a `[dry-run]` prefix, so it is safe to point at production paths.

---

## Differences from the legacy pipeline

**Bug fixes**

- Errored MZmine jobs are resubmitted automatically. Legacy Step 3 only looked at projects created by Step 1 in the same run. The fix is limited by the retry cap and only applies to jobs queued by this pipeline.
- `12 not relevant` changes made in the error branches of Steps 3 and 5 are now written to LIMS. The legacy code set them in memory only.
- `status --update_lims_table` works (it crashed with a TypeError) and is a real flag. It was `type=bool`, so any value counted as True.
- A missing task ID in a GNPS2 response is logged instead of raising `KeyError`.

**Behaviour changes**

- Slurm is handled by the host-side queue and the marker files described [above](#how-slurm-works-from-the-container), instead of `squeue`.
- A failing step no longer aborts the rest of the run.
- A dry-run mode exists.
- Steps 3 and 5 never resubmit a polarity that is `04 running`, unless you pass `--direct_input` together with `--overwrite_mzmine`.
- `--project_tag` and `--fps_files_only` were dropped. Use separate configs and `--custom_mzmine_batch_params` instead.
- Zips are built with Python `zipfile` (deflate-compressed) instead of `zip -rj`. The abridged rename happens in memory instead of in `/tmp`.
- The log format is the metatlas2 one (`time | logger | LEVEL | message`). The "Successfully started/completed" markers that the email script relies on are unchanged.
- Group ownership failures are logged at INFO, as before.
- The email sender address is no longer hardcoded.

**Legacy quirks kept on purpose**

These are marked `LEGACY QUIRK` in the code. They shape existing outputs, so they are left for a deliberate decision:

- `_metadata.tab` puts the background sample groups in constant columns named `CONTROL`, `CASE` and `ATTRIBUTE_media`, whatever they contain. It also lists QC, InjBl and ISTD files that are not in the MZmine file list.
- In the background filter:
  - "sample" columns include `row ID`, `row m/z` and `row retention time`;
  - the first feature is always kept in filtered tables;
  - empty tables drop the `row ID` header;
  - JGI projects never get empty tables written.
- The calibrant filter removes *any* MGF line containing a number within 5 ppm of 202.0777 (positive) or 202.0788 (negative), including header lines.
- Everything Bagel tasks are submitted but never polled or downloaded.

### Verified parity

These checks were run on real production data:

| Check | Data | Result |
|---|---|---|
| Background filtering and counts, new vs. legacy function | One EB HILIC project (all 6 tables) and one JGI C18 project (zero replacement) | Byte-identical to the legacy function and to the files in production; about 5x faster |
| Step 1 inputs | Real `lcmsrun_plus` rows for the same two projects | `_batch-params.xml` byte-identical; `_metadata.tab` and `_filelist.txt` same contents, rows in a different order |
| Full `--dry_run` | Production LIMS and directories | Completed with no warnings; nothing on disk changed |

---

## Migration checklist

The legacy scrontab jobs keep running until they are switched off. Don't run both pipelines against the same LIMS list at the same time.

1. Build and pull an image that contains this code. The image needs the new `labkey` and `paramiko` dependencies, so `--dev` with an older image is not enough.
2. As `msdata`, run `metatlas2.sh untargeted run --dry_run --log_to_stdout` and `metatlas2.sh untargeted status`. Check that the LabKey key, GNPS2 credentials and the `ben_lbl_gdrive` rclone remote are all accessible.
3. Optionally, rehearse against a copy of the LIMS list: `LIMS.tasks_table` in a config, or `--lims_tasks_table`.
4. Comment out the legacy `run_untargeted_pipeline` scron entry and add the new one.
5. Point the email scron entries at `scripts/untargeted/`.
6. The LIMS file sync and the raw-file converter feed this pipeline (LIMS `lcmsrun_plus` and the mzML files). Both are now ported as `metatlas2.sh lims-sync` and `metatlas2.sh convert-raw`; see [raw_data_ingest.md](raw_data_ingest.md).

---

## Code map

| Module | Role |
|---|---|
| `untargeted/run_untargeted_pipeline.py` | CLI (`run`, `status`, `mirror-raw`, `write-config`) |
| `untargeted/config.py` | Defaults, YAML loading, CLI overrides |
| `untargeted/context.py` | `RunContext`: config, LIMS backend, flags, clock |
| `untargeted/status.py` | `TaskStatus`, `PolarityTask` (file layout), LIMS dataframe filters |
| `untargeted/lims.py` | `LimsClient` (lazy LabKey) and `InMemoryLims` (tests) |
| `untargeted/project_setup.py` | Step 1 |
| `untargeted/mzmine.py` | Templates, sbatch, job markers, Steps 2–3 |
| `untargeted/feature_filtering.py` | Counts and background-filtered peak-height tables |
| `untargeted/gnps2.py` | `Gnps2Client` (HTTP + SFTP), FBMN/EB parameters, MGF helpers |
| `untargeted/fbmn.py` | Steps 4–6 |
| `untargeted/packaging.py` | Step 7 |
| `untargeted/pipeline.py` | Step orchestration and log markers |
| `untargeted/status_report.py` | Status table for `status` and the email scripts |

Reused metatlas2 code:

- `file_and_project_format.get_name_field` / `validate_file_name_minimal` / `parent_dir_num_fields`. These replace the legacy `validate_filenames`.
- `gdrive_upload._rclone_copy` / `rclone_list` / `rclone_run`.
- `logging_config`.
