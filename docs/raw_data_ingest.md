# Raw Data Ingest: File Conversion, LIMS Sync, Archiving

Five scheduled jobs turn instrument `.raw` files into data the rest of metatlas2 can use:

| Job | What it does | Runs where | Legacy equivalent |
|---|---|---|---|
| `scripts/ingest/file_transfer_sync.sh {jgi\|egsb}` | Pushes `.raw` files from JGI storage to `raw_data` at NERSC | `metabo-dtn` VM (cron) | `~/bin/sync.sh` on the VM |
| `metatlas2.sh convert-raw convert {jgi\|egsb}` | `.raw` → `.mzML` → `.h5` next to the raw file | Host Python + shifter | `utils/convert_files.sh` → `raw_to_h5.sh` |
| `metatlas2.sh lims-sync` | Records raw/mzML/h5 files on disk in the LIMS file tables and links them per run | metatlas2 container | `metatlas/io/update_lcmsfiles_in_lims.py` |
| `scripts/ingest/sync_to_aws.sh {jgi\|egsb}` | Archives `.raw` files to S3 Glacier | Host (`aws` CLI) | `utils/sync_to_aws.sh` |
| `scripts/ingest/email_*_report.sh` | Weekly conversion and AWS sync emails | Host (`mailx`) | `utils/email_conversion_report.sh`, `email_aws_report.sh` |

The untargeted pipeline ([untargeted_workflow.md](untargeted_workflow.md)) depends on the transfer, the converter and the LIMS sync. It finds new projects through LIMS `lcmsrun_plus`, which the sync keeps current, and it reads the mzML files the converter writes.

---

## Data transfer (JGI storage → NERSC)

How a raw file gets from the instrument to NERSC:

1. **Instrument PC → JGI storage.** `sync_data.ps1` (in the infrastructure_automation repo) runs on each instrument PC and robocopies its data to `\\storage.jgi.lbl.gov\metabolomics\<dept>`.
2. **JGI storage → NERSC.** `scripts/ingest/file_transfer_sync.sh` runs every 5 minutes on `metabo-dtn.jgi.lbl.gov` as `svc-metabolomics@lbl.gov`. It rsyncs `/net/storage.jgi.lbl.gov/vol_metabolomics/<vol>/` to the rsync daemon on `dtn04.nersc.gov:39653`, which writes into `raw_data/<vol>/`.

What the transfer sends:

- **Included:** in each project directory, the `*.raw` files, everything under `*/[Oo]ther/`, and `NO_UNTARGETED.TXT`.
- **Excluded:** everything else, deeper subdirectories, `$RECYCLE.BIN/`, `filechecks/` and `Robo_copy_test/`.

This is identical to the old `~/bin/sync.sh`, except that output lines are timestamped without needing `ts`.

| Log file on the VM | Contents |
|---|---|
| `~/logs/<vol>.log` | rsync's own transfer log |
| `~/logs/<vol>.error.log` | rsync output and errors |

The script uses the same lock file as before (`/tmp/msdata.<vol>.lockfile`), so the old and new entries can't run at the same time during the switch.

To set up the VM, as `svc-metabolomics@lbl.gov`:

```bash
git clone https://github.com/biorack/metatlas2.git ~/metatlas2      # public repo, no credentials needed
crontab -e
```

The new crontab:

```bash
*/5 * * * * cd $HOME/metatlas2 && git pull --quiet --ff-only >> $HOME/logs/git_pull.log 2>&1
*/5 * * * * $HOME/metatlas2/scripts/ingest/file_transfer_sync.sh jgi
*/5 * * * * $HOME/metatlas2/scripts/ingest/file_transfer_sync.sh egsb
48 2 * * * /usr/sbin/logrotate --state $HOME/logs/logrotate.status $HOME/logs/logrotate.conf
```

The clone follows `main`, so the script has to be merged there first. Once the new entries have run cleanly, remove `~/bin/sync.sh` and the unused `~/metatlas` checkout.

**Receiving end (dtn04).** The rsync daemon that receives the transfer runs as `msdata` on `dtn04.nersc.gov`.

| File | Purpose |
|---|---|
| `scripts/ingest/rsyncd_start.sh` | Starts the daemon |
| `scripts/ingest/rsyncd.dtn04.nersc.gov.conf` | Daemon config: modules `jgi` and `egsb` → `raw_data/<vol>`, user `msdata`, port 39653 |

`msdata`'s crontab on dtn04 restarts the daemon if it isn't running:

```bash
*/5 * * * * /bin/flock -w 0 /tmp/msdata.rsyncd.lock /global/cfs/cdirs/metatlas/tools/metatlas2/scripts/ingest/rsyncd_start.sh
```

The daemon re-reads its config on every connection, so edits to the config in `/global/cfs/cdirs/metatlas/tools/metatlas2` take effect immediately; review them before saving. The config path, however, is fixed when the daemon starts. After changing the crontab path, kill the running daemon so cron restarts it from the new location:

```bash
pkill -u msdata -f 'rsync --daemon'
```

The daemon log `/global/cfs/cdirs/m2650/rsyncd/logs/rsyncd.dtn04.nersc.gov.log` is rotated nightly by the `rotate_rsyncd_logs` scron job, using `scripts/ingest/logrotate.rsyncd.conf`. The daemon keeps the log open, so rotation uses `copytruncate`.

**Opt-out files.** A project can opt out of the untargeted pipeline with `NO_UNTARGETED.TXT`, which is the only spelling the transfer copies. The legacy pipeline only looked for `nountargeted.txt`, so the opt-out never worked: 53 projects in `raw_data` have the file and were processed anyway. metatlas2 now accepts either name in any case. This applies to new projects only.

---

## File conversion

```bash
metatlas2.sh convert-raw convert jgi                 # convert everything pending under raw_data/jgi
metatlas2.sh convert-raw convert egsb --dry_run      # list what would be converted
metatlas2.sh convert-raw convert jgi --max_workers 4 --log_to_stdout
metatlas2.sh convert-raw convert-file /path/to/X.raw # one file, ignoring the scan
```

### How it runs

ThermoRawFileParser (TRFP) has its own shifter image, and shifter can't be started from inside a container. So the orchestrator (`metatlas2/ingest/conversion.py`) runs on the host. It uses `/usr/bin/python3.11` and only the standard library, so it works for any account (for example `msdata`) without a virtualenv. Override the interpreter with `METATLAS2_HOST_PYTHON`. The code comes from the repository checkout that `metatlas2.sh` lives in.

For each file:

1. Create `X.progress` (group `metatlas`, mode 640). This file is the per-file log.
2. Validate the file name: at least 16 fields, polarity in field 9.
3. Run ThermoRawFileParser 1.4.2 through shifter with `-f=1` to write `X.mzML`. The image digest is the same as before.
4. Write `X.h5` with `python -m metatlas2.ingest.h5_writer`, inside the metatlas2 image (`--image` tag; `--dev` uses the working tree).
5. On success, delete `X.progress`. On failure, rename it to `X.progress` → `X.failed`. A `.failed` file is never retried. If the h5 step failed, the mzML is moved to the same path under `conversion_failures/`.

A file is picked up when `raw_data/<subdir>/<project>/X.raw` exists, exactly two levels deep, and there is neither `X.h5` nor `X.failed`. To retry a failed file, delete its `.failed`.

### What's the same, what changed

The `.h5` format and contents are unchanged. This was checked on production files:

- new h5 output equals the legacy `mzml_to_hdf` output table by table and value by value;
- it also equals the `.h5` files already in `raw_data`;
- `format_version=5` and `metatlas_version='0.7'` are written as before, plus a new `writer` attribute.

Changes:

- **Speed:**
  - The TRFP image is pulled once per run, not once per file.
  - File-name validation runs in-process. The old pipeline started a container per file for this.
  - The h5 writer handles peaks as numpy arrays and appends them in batches: about 10–35% faster on the same file.
  - Files run in parallel (`--max_workers`, default = CPUs allocated, at most 8).
- **Safer:**
  - `.h5` is written as `X.h5.partial` and renamed when complete, so a killed run can't leave a truncated `.h5` that counts as done.
  - SIGTERM (for example the scron time limit) cancels running conversions and removes their partial outputs.
  - A lock file stops two runs on the same subdirectory from overlapping.
- **Bug fix:** a missing or empty mzML after TRFP now produces `.failed`. The old script exited without it, so the file was retried every 5 minutes forever.
- **Logs:** the run log (`/global/cfs/cdirs/m2650/file_converter_logs/<subdir>.log`, same file as before) now has:
  - a scan summary (raw / already converted / previously failed / to convert, per project);
  - one line per file with stage timings and h5 row counts;
  - one line per failure with the stage and reason;
  - a final summary.

  TRFP's verbose output stays in the per-file `.progress`/`.failed` files. Their lines look like `2026-11-23 09:57:09.136131, ERROR, message`, which the conversion email still parses.
- **No legacy DB record:** the old converter also inserted an `LcmsRun` into the legacy metatlas MySQL database. That is gone.

### Scheduling

```bash
#SCRON --job-name jgi_converter
#SCRON -q cron
#SCRON -A m2650
#SCRON -t 24:00:00
#SCRON -c 4
#SCRON -o /global/cfs/cdirs/m2650/file_converter_logs/scrontab_jgi-%j.log
#SCRON --open-mode=append
*/5 * * * * METATLAS_DATA_DIR=/global/cfs/cdirs/metatlas /path/to/metatlas2/scripts/metatlas2.sh convert-raw convert jgi
```

Use the same entry with `egsb` for the second subdirectory. `-c` sets the number of parallel conversions. The scron `-o` file only gets wrapper output; the detailed log goes to `file_converter_logs/<subdir>.log`, which the existing logrotate config already rotates.

---

## LIMS file sync

```bash
metatlas2.sh lims-sync                     # all three steps
metatlas2.sh lims-sync --dry_run --log_to_stdout
metatlas2.sh lims-sync --skip_lcmsrun_links
```

Equivalent to `update_lcmsfiles_in_lims.py --update_file_tables True --update_lcmsrun_names True --update_lcmsrun_items True`:

1. **File tables.** Every `*.raw`, `*.mzML` and `*.h5` under `raw_data` that isn't yet in `raw_file` / `mzml_file` / `hdf5_file` is inserted. Each row gets `filename`, `basename`, `file_type`, `timeepoch` (mtime) and `name`.
   - Matching is case-insensitive, at any depth, and includes directories named `*.raw` (Waters), the same as `find -iname`.
   - `name` is the path relative to `raw_data` with the extensions `raw`, `mzML`, `h5`, `d`, `tab`, `gz` and `pactolus` removed (case-sensitive).
   - Rows are never deleted.
2. **lcmsrun names.** Every file `name` without an `lcmsrun` row gets one.
3. **lcmsrun links.** `lcmsrun.raw_file` / `mzml_file` / `hdf5_file` are set to the `Key` of the file row with the same `name`.

Changes:

- **Disk scan:** one walk of `raw_data` collects all three file types. The old script ran three `find` passes. On production the walk found the same 313,845 / 438,447 / 438,454 files with identical mtimes, in 5.8 min instead of 9.4 min.
- **Queries:** each LIMS table is fetched once and re-fetched only after inserts; there is no `UNION` query.
- **Deterministic names:** names keep the original dot order. The old code joined name pieces through a `set`, so names with internal dots, like `n.__234.a`, could come out scrambled and never link across file types. Names without internal dots are unchanged.
- **Stable links:** when several file rows share a name, an existing valid link is kept; otherwise the newest row wins. The old code picked whichever row LabKey returned first, which could change between runs.
- **Truncation guard:** a LIMS read that hits `--lims_max_rows` (default 5,000,000) aborts the run. Without it, missing rows would look like new files and be inserted as duplicates.
- **Logs:**
  - scan and per-table counts with timings, including rows only in LIMS and names matching several file rows;
  - a sample of inserted files;
  - a final summary.

  The log file is `/global/cfs/cdirs/m2650/untargeted_logs/lims_update.log`, same as before.

Scheduling, every 30 minutes as before:

```bash
#SCRON --job-name untargeted_lims_update
#SCRON -q cron
#SCRON -A m2650
#SCRON -t 00:30:00
#SCRON -o /global/cfs/cdirs/m2650/untargeted_logs/lims_update_scron.log
#SCRON --open-mode=append
*/30 * * * * METATLAS_DATA_DIR=/global/cfs/cdirs/metatlas /path/to/metatlas2/scripts/metatlas2.sh lims-sync
```

The full scan takes about 6 minutes, well inside the 30-minute limit.

---

## Reports and AWS archive

All scripts are in `scripts/ingest/` and keep their arguments, files and output formats. The sender address is no longer hardcoded: set `METATLAS2_REPORT_SENDER` to give mailx a reply-to.

| Script | Schedule (legacy) | Notes |
|---|---|---|
| `email_conversion_report.sh <jgi\|egsb> <days>` | Tue 15:00 | Uses `conversion_report.sh`, which counts `.h5` / `.failed` / `.progress` files and extracts their `ERROR` lines. |
| `sync_to_aws.sh <jgi\|egsb>` | hourly | `aws s3 sync --storage-class GLACIER` of `*.raw` and `*/other/*` to `s3://northen-archive/raw_data/<subdir>`. Lines are timestamped like moreutils `ts` without needing it installed. Set `AWS_CLI` if `aws` is not on `PATH`. |
| `email_aws_report.sh <jgi\|egsb> <days>` | Tue 15:00 | Summarises `upload:` / `error:` lines from the sync log. |

Environment overrides:

| Variable | Overrides |
|---|---|
| `RAW_DATA_DIR` | Raw data root |
| `REPORT_RECIPIENTS_DIR` | Directory holding the `email_<subdir>_reports` files |
| `AWS_SYNC_LOG_DIR` | AWS sync log directory |
| `S3_RAW_DATA` | S3 archive destination |

---

## Migration checklist

1. Build and pull an image that contains `metatlas2/ingest/`. The mzML → h5 step runs in the image, so until then use `--dev`.
2. Run both jobs once as `msdata` and check the output:
   - `metatlas2.sh convert-raw convert jgi --dry_run --log_to_stdout`
   - `metatlas2.sh lims-sync --dry_run --log_to_stdout`
3. Swap the scrontab entries (`jgi_converter`, `egsb_converter`, `untargeted_lims_update`, the AWS sync and the report jobs) one at a time. Leave the old entry commented out until the new one has run cleanly.
4. The new converter and the old one must not run at the same time on the same subdirectory. They don't share a lock.
